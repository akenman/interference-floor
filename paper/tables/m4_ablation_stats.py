# -*- coding: utf-8 -*-
"""M4 统计：E5 消融的多 seed 汇总（闭环 k=5, px）。

定义：delta = 1 - frozen/full，**delta > 0 表示 frozen 更好**（闭环误差更低）。
用法：python -X utf8 paper/tables/m4_ablation_stats.py
输出：analysis_output/m4_ablation_stats.json
"""
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AO = os.path.join(ROOT, "analysis_output")

d = json.load(open(os.path.join(AO, "e5_multiseed.json"), encoding="utf-8"))
out = {}
print("=== E5 消融 4 seed 汇总（闭环 k=5, px；delta>0 表示 frozen 更好）===\n")
for cond in ("g1125", "drag"):
    fulls, fros, mism = [], [], []
    print(f"[{cond}]")
    for s in sorted(d, key=int):
        r = d[s][cond]
        v = r["variants"]
        f, g = v["full"]["roll"], v["frozen_match"]["roll"]
        fulls.append(f); fros.append(g)
        line = f"  s{s}: prior={r['prior_one']:.4f}  full={f:.4f}  frozen_match={g:.4f}"
        if "frozen_mis" in v:
            mism.append(v["frozen_mis"]["roll"])
            line += f"  frozen_mis={v['frozen_mis']['roll']:.4f}"
        print(line)
    # delta = full/frozen - 1 > 0  <=>  frozen 的闭环误差更低（frozen 更好）
    delta = [f / g - 1 for f, g in zip(fulls, fros)]
    n = len(delta)
    mean, sd = st.mean(delta), (st.stdev(delta) if n > 1 else 0.0)
    se = sd / (n ** 0.5) if n else 0.0
    t = mean / se if se > 0 else float("inf")
    win = sum(x > 0 for x in delta)   # delta>0 = frozen 更好
    print(f"  full mean={st.mean(fulls):.4f}  frozen mean={st.mean(fros):.4f}")
    print(f"  frozen 相对 full 改善 = {100*mean:.1f}% (sd {100*sd:.1f}%, se {100*se:.1f}%, "
          f"t={abs(t):.2f})  frozen 更好 {win}/{n}")
    rec = dict(full_mean=st.mean(fulls), frozen_mean=st.mean(fros), n=n,
               delta_mean=mean, delta_sd=sd, delta_se=se, t=t, win=win)
    if mism:
        dm = [f / m - 1 for m, f in zip(mism, fulls)]
        rec["frozen_mis_mean"] = st.mean(mism)
        rec["frozen_mis_delta_mean"] = st.mean(dm)
        rec["frozen_mis_win"] = sum(x > 0 for x in dm)
        print(f"  frozen_mis mean={st.mean(mism):.4f}  比 full 改善 "
              f"{100*st.mean(dm):.1f}%  更好 {rec['frozen_mis_win']}/{n}")
    out[cond] = rec
    print()

# ---- 控制条件：seed 间变异 ----
mo = [d[s]["matched_full"]["hybrid_one"] for s in sorted(d, key=int)]
mr = [d[s]["matched_full"]["hybrid_roll_k5"] for s in sorted(d, key=int)]
cv = lambda x: 100 * st.stdev(x) / st.mean(x)
print("=== 控制条件 matched 的 seed 间变异 ===")
print(f"  单步 : {[round(x,4) for x in mo]}  mean={st.mean(mo):.4f}  CV={cv(mo):.1f}%")
print(f"  闭环k5: {[round(x,4) for x in mr]}  mean={st.mean(mr):.4f}  CV={cv(mr):.1f}%")
print("  论文自报同配置重训的差：单步 0.5%、闭环 k2 0.2%、五步 15%")
out["matched_seed_variation"] = dict(
    one_step_cv_pct=cv(mo), roll5_cv_pct=cv(mr),
    one_step_values=mo, roll5_values=mr,
    paper_retrain_one_step_pct=0.5, paper_retrain_roll5_pct=15.0)
print(f"  -> 换 seed 的单步 CV={cv(mo):.1f}%，是重训 0.5% 的 "
      f"{cv(mo)/0.5:.0f} 倍；闭环 k5 的 seed CV={cv(mr):.1f}%")
print("     注意：论文的 15% 五步重现差来自一次重训，而这里同指标的 seed CV 只有 "
      f"{cv(mr):.1f}%——说明那个 15% 不能当作统一阈值。")

with open(os.path.join(AO, "m4_ablation_stats.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n[save] analysis_output/m4_ablation_stats.json")
