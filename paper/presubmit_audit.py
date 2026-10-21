# -*- coding: utf-8 -*-
"""投稿前终检：手写交叉引用 / 陈旧编号 / 生成物一致性。

M5 重排改了章节顺序、标题、贡献编号与一个附录的去留，而 `\\ref` 会自动更新、
**手写的 "§N" / "Table N" / "C#" 不会**。本脚本专查这一类，并复核各类自述计数。

用法：python -X utf8 paper/presubmit_audit.py
"""
import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "paper", "latex", "main.tex")
t = io.open(MAIN, encoding="utf-8").read()
pdf_txt = os.path.join(ROOT, "paper", "latex", "main.txt")
pdf = io.open(pdf_txt, encoding="utf-8", errors="replace").read() \
    if os.path.exists(pdf_txt) else ""

issues = []


def check(cond, label, detail=""):
    print(("  OK  " if cond else "  !!  ") + label + (f"   {detail}" if detail else ""))
    if not cond:
        issues.append(label + " " + detail)


print("=== 1. 手写交叉引用（\\ref 之外的） ===")
# 手写 §N 或 Section N
hand = re.findall(r"(?:\\S|§|Section~?)\s*(\d+(?:\.\d+)?)", t)
print(f"  手写章节引用共 {len(hand)} 处: {sorted(set(hand))}")
# 当前章节号
secs = re.findall(r"\\section\{([^}]*)\}", t)
num = {}
n = 0
for name in secs:
    if "appendix" in t[:t.index("\\section{" + name + "}")][-200:].lower():
        pass
    n += 1
    num[name] = n
appendix_start = t.index("\\appendix")
body_secs = [s for s in secs if t.index("\\section{" + s + "}") < appendix_start]
for i, s in enumerate(body_secs, 1):
    print(f"    §{i} = {s[:56]}")
# 检查引用的编号是否越界（正文最多 len(body_secs)）
bad = [h for h in set(hand) if h.split(".")[0].isdigit()
       and int(h.split(".")[0]) > len(body_secs)]
check(not bad, "手写章节号未越界", f"越界: {bad}" if bad else "")

print("\n=== 2. 生成物与 \\input 的一致性 ===")
for f in ("provenance.tex", "tab_arms.tex", "tab_c3_gating.tex", "tab_criteria.tex"):
    used = f"\\input{{{f[:-4]}}}" in t
    exists = os.path.exists(os.path.join(ROOT, "paper", "latex", f))
    # 被 input 的生成物里不应再有 \section（否则会与手写节重复）
    dup = False
    if exists:
        body = io.open(os.path.join(ROOT, "paper", "latex", f), encoding="utf-8").read()
        dup = "\\section{" in body
    check(used and exists and not dup, f"{f}: 被引用 / 存在 / 无重复 \\section",
          f"used={used} exists={exists} has_section={dup}")

print("\n=== 3. 贡献编号连续性 ===")
cl = re.findall(r"\\textbf\{(C\d) --- ", t)
print(f"  贡献标签: {cl}")
check(cl == [f"C{i}" for i in range(1, len(cl) + 1)], "C1..CN 连续无重复")
# 正文里引用的 C# 必须存在
refs = sorted(set(re.findall(r"\bC(\d)\b", t)))
bad_c = [r for r in refs if f"C{r}" not in cl]
check(not bad_c, "正文引用的 C# 都已定义", f"未定义: {bad_c}" if bad_c else "")

print("\n=== 4. 自述计数 ===")
def count(pattern):
    return len(re.findall(pattern, t))
preregs = len([f for f in os.listdir(os.path.join(ROOT, "experiments"))
               if f.startswith("prereg_") and f.endswith(".json")])
check(preregs == 12, f"预注册文件 = {preregs}", "")
check("twelve preregistration" in t or "Twelve preregistration" in t,
      "正文写 twelve preregistrations")
# 溯源条数：从 JSON 读
import json
pv = json.load(io.open(os.path.join(ROOT, "analysis_output",
                                    "number_provenance_check.json"), encoding="utf-8"))
nchk = len(pv["checks"])
in_text = sorted(set(int(x) for x in re.findall(r"compares (\d+) of them", t)
                     + re.findall(r"plus (\d+) number-provenance", t)
                     + re.findall(r"status: (\d+)/\d+ PASS", t)))
check(in_text == [nchk], f"溯源计数一致 = {nchk}", f"正文出现: {in_text}")
mft = json.load(io.open(os.path.join(ROOT, "analysis_output",
                                     "checkpoints_manifest.json"), encoding="utf-8"))
check(len(mft["checkpoints"]) == 289 and len(mft["roots"]) == 17,
      "checkpoint 清单 = 289 / 17 roots")
check("SHA-256 digest" in t and "SHA-256 hashes" not in t,
      "摘要哈希说法为 16-character digest 而非 hashes")

print("\n=== 5. 无残留的旧自称 ===")
for bad_s in ("app:corrector", "§3.4", "see §6.5", "the rest of the paper",
              "Computation-Class Fidelity"):
    hit = bad_s in t
    check(not hit, f"无残留: {bad_s!r}")

print("\n=== 6. 外部检查 ===")
for script, label in (("paper/check_latex.py", "check_latex"),
                      ("paper/verify_numbers.py", "verify_numbers")):
    r = subprocess.run([sys.executable, "-X", "utf8",
                        os.path.join(ROOT, script)],
                       capture_output=True, text=True, cwd=ROOT)
    last = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:] or [""]
    ok = r.returncode == 0
    check(ok, f"{label} 退出码 0", last[0][:60])
    if not ok:
        print("      ", r.stdout[-500:])

print("\n" + "=" * 60)
if issues:
    print(f"发现 {len(issues)} 个问题：")
    for i in issues:
        print("  -", i)
else:
    print("终检通过：无问题。")
