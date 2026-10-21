# -*- coding: utf-8 -*-
"""编译 TMLR 稿件：pdflatex -> bibtex -> pdflatex -> pdflatex。

用法：
    python paper/build_pdf.py            # 正常编译
    python paper/build_pdf.py --clean    # 编译后删除 aux 文件
输出：
    paper/latex/main.pdf
并打印摘要：页数、未定义引用/交叉引用、overfull/underfull 数量、LaTeX 错误行。

为什么用 Python 而不是 .ps1：PowerShell 对含全角标点/嵌套引号的脚本解析很脆，
本仓库其余工具也都是 Python。
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXDIR = os.path.join(ROOT, "paper", "latex")
MAIN = "main.tex"
JOB = os.path.splitext(MAIN)[0]


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    local = os.environ.get("LOCALAPPDATA", "")
    appdata = os.environ.get("APPDATA", "")
    cands = [
        os.path.join(local, "Programs", "MiKTeX", "miktex", "bin", "x64", name + ".exe"),
        os.path.join(appdata, "MiKTeX", "miktex", "bin", "x64", name + ".exe"),
        r"C:\texlive\2025\bin\windows" + "\\" + name + ".exe",
        r"C:\texlive\2024\bin\windows" + "\\" + name + ".exe",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def run(cmd, cwd, logfile):
    """执行并把 stdout/stderr 重定向到文件（不用管道，避免受限沙箱下 EPERM）。"""
    with io.open(logfile, "w", encoding="utf-8", errors="replace") as fh:
        code = subprocess.call(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    return code


def main():
    global MAIN, JOB
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--main", default=MAIN)
    args = ap.parse_args()

    MAIN = args.main
    JOB = os.path.splitext(MAIN)[0]

    pdflatex = find_tool("pdflatex")
    bibtex = find_tool("bibtex")
    if not pdflatex:
        print("FAILED: pdflatex not found (install MiKTeX or TeX Live)")
        return 127
    print("pdflatex :", pdflatex)
    print("bibtex   :", bibtex or "(not found; citations will not resolve)")

    out = os.path.join(TEXDIR, "_build_stdout.txt")
    passes = [("pass 1: pdflatex", [pdflatex, "-interaction=nonstopmode",
                                    "-file-line-error", MAIN])]
    if bibtex:
        passes.append(("pass 2: bibtex", [bibtex, JOB]))
    for i in (3, 4):
        passes.append(("pass %d: pdflatex" % i,
                       [pdflatex, "-interaction=nonstopmode", "-file-line-error", MAIN]))

    for label, cmd in passes:
        code = run(cmd, TEXDIR, out)
        print("--- %-18s exit=%d" % (label, code))

    log = os.path.join(TEXDIR, JOB + ".log")
    pdf = os.path.join(TEXDIR, JOB + ".pdf")
    print("\n=== summary ===")
    if not os.path.exists(pdf):
        print("FAILED: %s not produced" % pdf)
        if os.path.exists(log):
            txt = io.open(log, encoding="utf-8", errors="replace").read()
            bad = [l for l in txt.splitlines() if l.startswith("!") or re.match(r"^\S+\.tex:\d+:", l)]
            print("--- first errors ---")
            for l in bad[:30]:
                print("   ", l)
        return 1

    size = os.path.getsize(pdf)
    mtime = os.path.getmtime(pdf)
    import time
    print("PDF      : %s (%d bytes, %s)" % (pdf, size,
          time.strftime("%H:%M:%S", time.localtime(mtime))))

    if os.path.exists(log):
        txt = io.open(log, encoding="utf-8", errors="replace").read()
        m = re.search(r"Output written on .*\((\d+) pages?, (\d+) bytes\)", txt)
        if m:
            print("pages    : %s" % m.group(1))
        undef_cite = len(re.findall(r"Citation `[^']*' .*undefined", txt))
        undef_ref = len(re.findall(r"Reference `[^']*' .*undefined", txt))
        overfull = len(re.findall(r"Overfull \\hbox", txt))
        underfull = len(re.findall(r"Underfull \\hbox", txt))
        errors = len(re.findall(r"(?m)^! ", txt))
        print("undefined citations : %d" % undef_cite)
        print("undefined refs      : %d" % undef_ref)
        print("overfull hbox       : %d" % overfull)
        print("underfull hbox      : %d" % underfull)
        print("LaTeX errors (! ...): %d" % errors)
        if undef_cite or undef_ref or errors:
            print("\n--- undefined / errors ---")
            for l in txt.splitlines():
                if re.search(r"Citation .*undefined|Reference .*undefined", l) or l.startswith("! "):
                    print("   ", l[:160])
        if overfull:
            print("\n--- worst overfull boxes ---")
            for l in re.findall(r"Overfull \\hbox[^\n]*", txt)[:5]:
                print("   ", l[:160])

    if args.clean:
        # 保留 .bbl：投稿源码包需要它（无 BibTeX 环境也能编译）
        for ext in (".aux", ".log", ".out", ".blg", ".toc"):
            p = os.path.join(TEXDIR, JOB + ext)
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(out):
            os.remove(out)
        print("\ncleaned aux files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
