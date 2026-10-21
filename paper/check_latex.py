# -*- coding: utf-8 -*-
"""LaTeX 结构守卫（论文合稿层，与 verify_numbers.py 的数字守卫互补）。

用法：python paper/check_latex.py
检查：
  1. 每个 \\begin{env} 都有配对的 \\end{env}；
  2. 每行的 $ 数量为偶数（捕捉坏掉的数学模式，如被 Tab 吃掉的 \\text）；
  3. 所有 \\cite* 的 key 都在 references.bib 里（无未定义引用）；
  4. references.bib 里没有从未被引用的条目；
  5. 所有 \\ref/\\eqref 的目标 \\label 都已定义（含 \\input 进来的子文件）；
  6. \\input 的子文件都存在。
退出码非 0 表示有失败项。
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX = os.path.join(HERE, "latex")
MAIN = os.path.join(LATEX, "main.tex")
BIB = os.path.join(LATEX, "references.bib")

problems = []


def read(p):
    return io.open(p, encoding="utf-8").read()


def main():
    src = read(MAIN)
    inputs = re.findall(r"\\input\{([^}]+)\}", src)
    body = src
    for name in inputs:
        p = os.path.join(LATEX, name if name.endswith(".tex") else name + ".tex")
        if not os.path.exists(p):
            problems.append("missing \\input target: %s" % name)
            continue
        body += "\n" + read(p)

    # 1. 环境配对
    begins = re.findall(r"\\begin\{(\w+\*?)\}", body)
    ends = re.findall(r"\\end\{(\w+\*?)\}", body)
    for env in sorted(set(begins) | set(ends)):
        nb, ne = begins.count(env), ends.count(env)
        if nb != ne:
            problems.append("environment %s: %d begin / %d end" % (env, nb, ne))
    print("environments: %d begin / %d end, %d distinct"
          % (len(begins), len(ends), len(set(begins))))

    # 2. 每行 $ 奇偶（含 \\$ 转义已排除）
    bad_lines = []
    for i, line in enumerate(body.splitlines(), 1):
        stripped = re.sub(r"\\\$", "", line)
        if stripped.count("$") % 2:
            bad_lines.append(i)
    if bad_lines:
        problems.append("odd number of $ on lines: %s" % bad_lines[:10])
    print("math delimiters: %d, odd-parity lines: %d"
          % (body.count("$"), len(bad_lines)))

    # 2b. 可疑转义残留（\\n / \\t 后面不是字母）
    artefacts = []
    for i, line in enumerate(body.splitlines(), 1):
        for m in re.finditer(r"\\[nt](?![A-Za-z])", line):
            artefacts.append("%d:%s" % (i, m.group(0)))
    if artefacts:
        problems.append("literal escape artefacts: %s" % artefacts[:8])
    print("escape artefacts: %d" % len(artefacts))

    # 2c. 行内裸 %（会静默注释掉该行剩余内容）
    pct = []
    for i, line in enumerate(body.splitlines(), 1):
        m = re.search(r"(?<!\\)%", line)
        if m and line[:m.start()].strip():
            pct.append(i)
    if pct:
        problems.append("bare %% mid-line (would comment out the rest): %s" % pct[:8])
    print("bare mid-line %%: %d" % len(pct))

    # 3./4. 引用 ↔ bib
    used = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", body):
        used |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    defined = set(re.findall(r"@\w+\{([^,]+),", read(BIB)))
    missing = sorted(used - defined)
    uncited = sorted(defined - used)
    if missing:
        problems.append("cited but not in bib: %s" % missing)
    if uncited:
        problems.append("in bib but never cited: %s" % uncited)
    print("citations: %d used, %d defined, %d missing, %d uncited"
          % (len(used), len(defined), len(missing), len(uncited)))

    # 5. \\ref ↔ \\label
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set()
    for m in re.finditer(r"\\(?:ref|eqref|autoref|Cref|cref)\{([^}]+)\}", body):
        refs |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    dangling = sorted(refs - labels)
    orphan = sorted(labels - refs)
    if dangling:
        problems.append("\\ref without \\label: %s" % dangling)
    if orphan:
        print("  (warning) \\label never referenced: %s" % orphan)
    print("labels: %d defined, %d referenced, %d dangling, %d unreferenced"
          % (len(labels), len(refs), len(dangling), len(orphan)))

    print()
    if problems:
        print("FAIL (%d):" % len(problems))
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("PASS: LaTeX structure clean")


if __name__ == "__main__":
    main()
