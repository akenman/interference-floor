# -*- coding: utf-8 -*-
"""judge_pair 单元测试：崩溃口径冲突检测的各分支。"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from build_e1a_report import judge_pair, _verdict_of

THR = 0.05


def ci(d, lo, hi, n=10, se=0.05, tc=1.96):
    return (d, se, lo, hi, n, tc)


cases = [
    # (名称, ci_st, ci_al, 期望)
    ("两口径同 PASS",      ci(0.10, 0.06, 0.14), ci(0.12, 0.08, 0.16), "PASS"),
    ("两口径同 FAIL",      ci(0.00, -0.04, 0.04), ci(0.02, -0.02, 0.049), "FAIL"),
    ("两口径同 UNDECIDED", ci(0.05, 0.02, 0.08), ci(0.05, 0.02, 0.08), "UNDECIDED"),
    # 关键：冲突
    ("剔崩溃FAIL/全量PASS", ci(0.043, 0.02, 0.049), ci(0.078, 0.06, 0.10), "DISAGREE"),
    ("剔崩溃PASS/全量FAIL", ci(0.09, 0.06, 0.12), ci(0.04, 0.02, 0.049), "DISAGREE"),
    # 主口径缺失（退用全量，不算冲突）
    ("主口径缺失+全量PASS", None, ci(0.09, 0.06, 0.12), "PASS"),
    ("主口径缺失+全量UNDECIDED", None, ci(0.05, 0.02, 0.08), "UNDECIDED"),
    # 一口径已定、另一口径未定 → 结论依赖口径，同样拒绝单一判定
    ("剔崩溃UNDECIDED/全量FAIL", ci(0.05, 0.02, 0.06), ci(0.00, -0.04, 0.049), "DISAGREE"),
    ("剔崩溃PASS/全量UNDECIDED", ci(0.09, 0.06, 0.13), ci(0.05, 0.02, 0.07), "DISAGREE"),
    # 两口径都缺失
    ("两口径均缺失",       None, None, "UNDECIDED"),
    # nan CI
    ("剔崩溃nan/全量FAIL", ci(0.02, float("nan"), float("nan")), ci(0.00, -0.04, 0.04), "FAIL"),
]

fails = 0
print(f"{'用例':<26}{'期望':<12}{'实际':<12}{'v_st':<11}{'v_al':<11}结果")
print("-" * 84)
for name, a, b, want in cases:
    got, vs, va = judge_pair(a, b, THR)
    ok = got == want
    fails += (not ok)
    print(f"{name:<26}{want:<12}{got:<12}{vs:<11}{va:<11}{'OK' if ok else '[FAIL] FAIL'}")

print("\n_verdict_of 边界:")
for label, c, want in [
    ("lo>=thr PASS", ci(0.1, 0.05, 0.15), "PASS"),
    ("hi<thr FAIL", ci(0.0, -0.03, 0.049), "FAIL"),
    ("hi==thr UNDECIDED", ci(0.05, 0.0, 0.05), "UNDECIDED"),
    ("None UNDECIDED", None, "UNDECIDED"),
]:
    got = _verdict_of(c, THR)
    ok = got == want
    fails += (not ok)
    print(f"  {label:<22} 期望 {want:<11} 实际 {got:<11} {'OK' if ok else '[FAIL] FAIL'}")

print(f"\n{'全部通过 [OK]' if fails == 0 else f'{fails} 个失败 [FAIL]'}")
sys.exit(1 if fails else 0)
