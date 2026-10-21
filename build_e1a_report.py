# -*- coding: utf-8 -*-
"""E1a 结果汇总 — 解析 V*_s*.log，统计每个变体的 best val、崩溃率，评估预注册判据。

用法:
    python build_e1a_report.py --logs logs-corrected \
        --out analysis_output/E1a_corrected_report.json \
        --plan-normal 42-51 --plan-nogua 42-44

崩溃定义: 训练过程中曾出现 val one-step < 5px（收敛良好），随后某次验证 val > 10px
（发散），或日志含 [EARLY-STOP] collapse=1。仅 patience 早停（无 collapse 标记）
不算崩溃。best checkpoint 在崩溃前已保存，数值仍有效。

统计口径: 判据以「同 seed 配对差值」为主（变体与 V3 共享 seed），配对 t 分布 95% CI；
标准差为样本标准差（ddof=1）。完整性依据显式计划表（--plan-*）判定。
"""
import argparse, json, os, re, glob, sys
from collections import defaultdict
import numpy as np

try:
    from scipy import stats as _st
except Exception:          # 无 scipy 时退化为正态近似
    _st = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

VARIANTS = {
    "V1": "仅初爻（接触轴）",
    "V2a": "初爻 + 中爻（砍上爻）",
    "V2b": "初爻 + 上爻（砍中爻）",
    "V2c": "中爻 + 上爻（砍初爻）",
    "V3": "三爻全（= ours-base）",
    "V4": "三爻 + 能量轴",
    "Vfree": "三爻全 + 自由 MoE 门控",
    # --- 解耦消融臂（E1a-D）：容量固定 64 专家，只变「监督轴」与「门控可见轴」---
    "A1x64": "解耦: 监督1轴(初爻) + 门控只读初爻, 64专家",
    "A2x64": "解耦: 监督2轴(初+中) + 门控只读该2轴, 64专家",
    "A3g1": "解耦: 监督3轴(全) + 门控只读初爻, 64专家",
    "A3x64": "解耦对照: 与 V3 逐位等价（回归校验用，分析以 V3 为基准）",
}

# 解耦臂基准 = V3（容量同为 64 专家）。A3x64 与 V3 等价，若单独跑过可用它校验。
DECOUPLED_BASE = "V3"

RE_VAL = re.compile(r"\[val\]\s*one-step[^0-9]*([0-9]+\.[0-9]+)")
RE_EARLY = re.compile(r"\[EARLY-STOP\][^\n]*?val=([0-9.]+)px[^\n]*?best=([0-9.]+)px")
RE_COLLAPSE = re.compile(r"\[EARLY-STOP\]\s+collapse=1")
RE_OK = re.compile(r"\[OK\][^\n]*?([0-9]+\.[0-9]+)px")


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "ignore")


def _finite(xs):
    return [float(x) for x in xs if x is not None and np.isfinite(x)]


def _mean(xs):
    xs = _finite(xs)
    return float(np.mean(xs)) if xs else float("nan")


def _sd(xs):
    """样本标准差（ddof=1）。n<2 时为 nan —— n=1 不存在离散度。"""
    xs = _finite(xs)
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else float("nan")


def _tcrit95(n):
    """双侧 95% t 分位数；n<2 或无 scipy 时退回 1.96。"""
    if n is None or n < 2 or _st is None:
        return 1.96
    return float(_st.t.ppf(0.975, n - 1))


def _verdict_of(ci, thr):
    """由 (delta, se, lo, hi, n, tc) 判 PASS/FAIL/UNDECIDED；ci 不可用则 UNDECIDED。"""
    if ci is None or not np.isfinite(ci[2]) or not np.isfinite(ci[3]):
        return "UNDECIDED"
    lo, hi = ci[2], ci[3]
    if lo >= thr:
        return "PASS"
    if hi < thr:
        return "FAIL"
    return "UNDECIDED"


def judge_pair(ci_st, ci_al, thr):
    """综合两种崩溃处理口径给出判定。

    只有当**两种口径都可用且判定不一致**时才返回 DISAGREE：
      * 主口径不可用（公共 seed 不足、CI 为 nan）→ 属「口径缺失」，退用另一口径，
        不误报为冲突；
      * 两口径都可用却给出不同 PASS/FAIL → 结论随崩溃处理方式翻转，拒绝给单一判定。
    """
    usable = lambda c: c is not None and np.isfinite(c[2]) and np.isfinite(c[3])
    vs, va = _verdict_of(ci_st, thr), _verdict_of(ci_al, thr)
    if usable(ci_st) and usable(ci_al) and vs != va:
        return "DISAGREE", vs, va
    if usable(ci_st):
        return vs, vs, va
    if usable(ci_al):
        return va, vs, va
    return "UNDECIDED", vs, va


def parse_log(path):
    text = read_text(path)
    vals = [float(x) for x in RE_VAL.findall(text)]
    best = min(vals) if vals else None
    best_epoch = (vals.index(best) + 1) * 2 if best is not None else None  # 每 2 epoch 验证
    early = RE_EARLY.search(text)
    collapse = bool(RE_COLLAPSE.search(text))
    if not collapse and vals and best is not None and best < 5.0:
        # 出现 >10 的验证值（且在 best 之后）
        after = vals[vals.index(best):]
        collapse = any(v > 10.0 for v in after)
    final = vals[-1] if vals else None
    ok = RE_OK.search(text)
    done = bool(ok)
    name = os.path.basename(path)[:-4]
    m = re.match(r"(.+)_s(\d+)(_nogua)?$", name)
    variant = m.group(1) if m else name
    seed = int(m.group(2)) if m else -1
    no_gua = bool(m.group(3)) if m else False
    return dict(variant=variant, seed=seed, no_gua=no_gua, best=best,
                best_epoch=best_epoch, final=final, collapse=collapse,
                done=done, n_vals=len(vals), log=path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="analysis_output/E1a_report.json")
    ap.add_argument("--plan-normal", default="42-51",
                    help="计划中的常态臂 seed 列表/区间，逗号分隔，如 42-51")
    ap.add_argument("--plan-nogua", default="42-44",
                    help="计划中的 no_gua 臂 seed 列表/区间")
    ap.add_argument("--plan-normal-variants",
                    default="V1,V2a,V2b,V3,Vfree",
                    help="按 --plan-normal 种子数纳入计划的常态臂（逗号分隔）。"
                         "预注册只把这 5 臂扩到 42-51（见 seed_expansion）。"
                         "跑 E1a-D 解耦批次时传 A1x64,A2x64,A3g1,A3x64，"
                         "否则完整性判定只针对当前批次")
    ap.add_argument("--plan-fixed-variants", default="V2c,V4",
                    help="仍停留在 --plan-fixed-seeds 的常态臂（逗号分隔）。"
                         "若并入 --plan-normal-variants 会因永远不扩种子而"
                         "永远报 missing（2026-09-27 实证）")
    ap.add_argument("--plan-fixed-seeds", default="42-44",
                    help="上述固定臂的计划种子列表/区间")
    args = ap.parse_args()

    def parse_seeds(spec):
        out = set()
        for tok in str(spec).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                lo, hi = tok.split("-", 1)
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(tok))
        return sorted(out)

    # 未完成（尚无 [OK]）的 run：其 best 只是训练中途值、且不可与已完成 run 配对，
    # 必须从所有统计中剔除，否则会污染均值与配对 CI（2026-09-27 实证：V3_s45 进行中
    # 曾把 V3 由 n=3 虚增到 n=4 并与已完成的 seed45 配对）。
    runs_all = []
    # 同时匹配 V*（V1/V3/...）与 A*（解耦臂 A1x64/A2x64/A3g1/A3x64）—— Stage D
    # 的解耦判据依赖 A 臂，漏掉会导致 criteria_decoupled_exploratory 为空（2026-09-28 实证）。
    log_paths = (sorted(glob.glob(os.path.join(args.logs, "V*_s*.log")))
                 + sorted(glob.glob(os.path.join(args.logs, "A*_s*.log"))))
    for p in log_paths:
        if "_err" in p:
            continue
        r = parse_log(p)
        if r["variant"] in VARIANTS and r["best"] is not None:
            runs_all.append(r)
    runs_incomplete = [r for r in runs_all if not r["done"]]
    runs = [r for r in runs_all if r["done"]]

    by_key = defaultdict(list)
    for r in runs:
        key = r["variant"] + ("_nogua" if r["no_gua"] else "")
        by_key[key].append(r)

    # 展示顺序：每个变体的常态臂，紧跟其 no_gua 对照臂
    order = []
    for v in VARIANTS:
        if v in by_key:
            order.append(v)
        if (v + "_nogua") in by_key:
            order.append(v + "_nogua")

    print(f"\n=== E1a 汇总（已完成 {len(runs)} runs"
          + (f"，进行中 {len(runs_incomplete)} 未计入" if runs_incomplete else "")
          + "）===\n")
    hdr = (f"{'臂':<14}{'n':>3}{'崩溃':>5}{'稳定n':>7}"
           f"{'best均值':>10}{'±std':>8}{'稳定均值':>10}{'完成':>5}")
    print(hdr); print("-" * len(hdr))
    summary = {}
    for key in order:
        rs = by_key[key]
        bests = [r["best"] for r in rs if r["best"] is not None]
        stable = [r["best"] for r in rs if r["best"] is not None
                  and not r["collapse"]]
        ncol = sum(1 for r in rs if r["collapse"])
        ndone = sum(1 for r in rs if r["done"])
        m, s = _mean(bests), _sd(bests)
        ms = _mean(stable)
        print(f"{key:<14}{len(rs):>3}{ncol:>5}{len(stable):>7}"
              f"{m:>10.3f}{s:>8.3f}{ms:>10.3f}{ndone:>5}")
        summary[key] = dict(n=len(rs), mean=m, std=s,
                            mean_stable=ms,
                            std_stable=_sd(stable),
                            n_stable=len(stable),
                            min=float(min(bests)) if bests else float("nan"),
                            n_collapse=int(ncol), n_done=int(ndone),
                            collapse_rate=float(ncol / len(rs)) if rs else float("nan"),
                            seeds={r["seed"]: r["best"] for r in rs},
                            seeds_stable={r["seed"]: r["best"] for r in rs
                                          if not r["collapse"]},
                            # 配对统计需要逐 seed 的完成/崩溃状态
                            detail={r["seed"]: dict(best=r["best"],
                                                    collapse=bool(r["collapse"]),
                                                    done=bool(r["done"]),
                                                    best_epoch=r["best_epoch"])
                                    for r in rs})

    # 门控增益：no_gua 应比常态差（RMSE 更大）；增益 = (nogua - 常态)/常态
    gating = {}
    print("\n门控增益（no_gua − 常态，正=门控有用；均用剔崩溃稳定均值）:")
    for v in VARIANTS:
        if v in summary and (v + "_nogua") in summary:
            a, b = summary[v], summary[v + "_nogua"]
            if not (np.isfinite(a["mean_stable"]) and np.isfinite(b["mean_stable"])):
                continue
            g = (b["mean_stable"] - a["mean_stable"]) / a["mean_stable"]
            ga = (b["mean"] - a["mean"]) / a["mean"]
            gating[v] = dict(normal=a["mean_stable"], nogua=b["mean_stable"],
                             gain=g, gain_all=ga,
                             n_normal=a["n_stable"], n_nogua=b["n_stable"])
            flag = "  ← 门控无益（no_gua 更准）" if g < 0 else ""
            print(f"  {v:<7} 常态={a['mean_stable']:.3f}(n={a['n_stable']}) "
                  f"no_gua={b['mean_stable']:.3f}(n={b['n_stable']}) "
                  f"增益={g:+.1%}  [全量 {ga:+.1%}]{flag}")

    # 判据评估（需 V3）。预注册口径为 one-step RMSE 相对 V3。
    # 崩溃处理规则（本报告固化）：崩溃 run 从均值剔除，单独计崩溃率；
    # 若剔除后某臂无稳定 run，则退回全量均值并标注。
    crit = {}
    def rel(a, b):
        return (a - b) / b if b else float("nan")

    def unpaired_delta(v, v3key="mean_stable"):
        """非配对口径（各臂独立均值 + 独立标准误），仅作对照。"""
        a, base = summary[v], summary["V3"]
        x, y = a[v3key], base[v3key]
        if not (np.isfinite(x) and np.isfinite(y)):
            a, base = summary[v], summary["V3"]
            x, y = a["mean"], base["mean"]
            fallback = True
        else:
            fallback = False
        d = rel(x, y)
        nx = a["n_stable"] if not fallback else a["n"]
        ny = base["n_stable"] if not fallback else base["n"]
        sx = a["std_stable"] if not fallback else a["std"]
        sy = base["std_stable"] if not fallback else base["std"]
        se = float(np.sqrt(sx ** 2 / max(nx, 1) + sy ** 2 / max(ny, 1))) / y \
            if y and np.isfinite(sx) and np.isfinite(sy) else float("nan")
        return d, se, fallback, nx, ny

    def paired_delta(v, base_key="V3", require_stable=True):
        """配对口径：变体与 V3 共享 seed，按同 seed 相对差值求均值 + t 分布 CI。

        require_stable=True  → 预注册主口径：只用「两臂都完成且都未崩溃」的公共 seed。
                               剔除单臂崩溃会破坏配对，否则崩溃 run 会算作对方优势。
        require_stable=False → 敏感性口径：纳入崩溃 run 的 best.pt。
                               依据实证（2026-09-27）：4 个崩溃 run 的 best.pt
                               其 val 恰等于 best（0.793/0.908/0.928/1.032，均 <5px），
                               即权重是崩溃前保存的、数据可用；崩溃反映的是
                               优化不稳定，而非模型不可用。
                               仅当主口径因公共 seed 不足而无法给出 CI 时使用，并标注。
        """
        A, B = summary[v]["detail"], summary[base_key]["detail"]
        common = sorted(set(A) & set(B))
        pairs, used = [], []
        for s in common:
            ra, rb = A[s], B[s]
            if not (ra["done"] and rb["done"]):
                continue
            if require_stable and (ra["collapse"] or rb["collapse"]):
                continue
            if not (np.isfinite(ra["best"]) and np.isfinite(rb["best"])) or not rb["best"]:
                continue
            pairs.append((ra["best"] - rb["best"]) / rb["best"])
            used.append(s)
        n = len(pairs)
        if n < 2:          # n=1 无离散度，CI 无意义
            return None
        d = float(np.mean(pairs))
        se = _sd(pairs) / np.sqrt(n)
        return dict(delta=d, se=se, n=n, seeds=used, pairs=pairs,
                    tcrit=_tcrit95(n), require_stable=require_stable)

    if "V3" in summary:
        v3 = summary["V3"]["mean_stable"]
        if not np.isfinite(v3):
            v3 = summary["V3"]["mean"]
        print(f"\n判据（V3 稳定基准 {v3:.3f}px，n={summary['V3']['n_stable']}）"
              f"  [配对 t 分布 CI 为主口径]")
        for k, vv, thr in [("E1a-1", "V1", 0.05), ("E1a-2", "V2a", 0.05),
                           ("E1a-3", "V2b", 0.05), ("E1a-4", "Vfree", 0.03)]:
            if vv not in summary:
                continue
            ud, use, ufallback, unx, uny = unpaired_delta(vv)

            # 同时算两种崩溃处理口径，若结论不一致则拒绝给单一 PASS/FAIL
            pd_st = paired_delta(vv, require_stable=True)
            pd_al = paired_delta(vv, require_stable=False)

            def as_ci(pd_):
                if pd_ is None or not np.isfinite(pd_["se"]):
                    return None
                d, se, tc, n = pd_["delta"], pd_["se"], pd_["tcrit"], pd_["n"]
                return d, se, d - tc * se, d + tc * se, n, tc

            ci_st, ci_al = as_ci(pd_st), as_ci(pd_al)
            usable = lambda c: c is not None and np.isfinite(c[2]) and np.isfinite(c[3])

            if usable(ci_st):
                d, se, lo, hi, n, tc = ci_st
                method = "paired-t"
                pd_ = pd_st
            elif usable(ci_al):
                d, se, lo, hi, n, tc = ci_al
                method = "paired-t[all-records(含崩溃run的best.pt)]"
                pd_ = pd_al
            else:
                # 无可用公共 seed（多为 n=1 或全崩溃）→ 退回非配对正态近似
                d, se = ud, use
                lo, hi = (d - 1.96 * se, d + 1.96 * se) if np.isfinite(se) \
                    else (float("nan"), float("nan"))
                n, tc, method, pd_ = None, 1.96, "unpaired-normal", {}

            verdict, v_st, v_al = judge_pair(ci_st, ci_al, thr)

            crit[k] = dict(variant=vv, V3=float(v3), delta=float(d),
                           se=float(se), ci95=[float(lo), float(hi)],
                           threshold=thr, passed=bool(d >= thr),
                           verdict=verdict,
                           method=method,
                           n_paired=pd_.get("n"), paired_seeds=pd_.get("seeds"),
                           paired_deltas=pd_.get("pairs"),
                           stable_only=pd_.get("require_stable", True),
                           verdict_stable=v_st, verdict_all=v_al,
                           ci95_stable=(None if not usable(ci_st)
                                        else [float(ci_st[2]), float(ci_st[3])]),
                           ci95_all=(None if not usable(ci_al)
                                     else [float(ci_al[2]), float(ci_al[3])]),
                           unpaired_delta=float(ud), unpaired_se=float(use),
                           unpaired_n=[int(unx), int(uny)],
                           stable_fallback=bool(ufallback))
            mark = {"PASS": "PASS", "FAIL": "FAIL", "UNDECIDED": "≈不确定",
                    "DISAGREE": "[WARN]口径敏感"}[verdict]
            extra = f" 配对n={pd_.get('n')} seeds={pd_.get('seeds')}" \
                if pd_.get("n") else " [无公共seed→非配对]"
            if verdict == "DISAGREE":
                extra += (f" [WARN] 剔崩溃[{v_st}] vs 全量[{v_al}]"
                          f" → 结论随崩溃口径翻转，不给单一判定")
            elif method.startswith("paired-t[all-records"):
                extra += " [主口径seed不足→退用全量配对]"
            print(f"  {k} {vv:<6} Δ={d:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] "
                  f"门槛≥{thr} [{mark}]{extra}")


    # ---------------- 解耦消融判据（探索性，事后新增，不并入原预注册） ----------------
    # 固定 64 专家容量，基准 V3：
    #   E1a-D1  A1x64 vs V3 —— 固定容量下，"只监督初爻"是否显著更差？（爻信息量）
    #   E1a-D2  A3g1  vs V3 —— 门控读全部 3 轴 vs 只读初爻，差多少？（门控信息通路）
    #   E1a-D3  A2x64 vs V3 —— 中间点，用于判断是否存在单调趋势
    crit_d = {}
    if DECOUPLED_BASE in summary:
        base_d = summary[DECOUPLED_BASE]["mean_stable"]
        if not np.isfinite(base_d):
            base_d = summary[DECOUPLED_BASE]["mean"]
        print(f"\n解耦判据（探索性，基准 {DECOUPLED_BASE}={base_d:.3f}px, "
              f"固定 64 专家容量）:")
        for k, vv, thr in [("E1a-D1", "A1x64", 0.05), ("E1a-D2", "A3g1", 0.03),
                           ("E1a-D3", "A2x64", 0.05)]:
            if vv not in summary:
                continue
            pd_ = paired_delta(vv, base_key=DECOUPLED_BASE, require_stable=True)
            fb_used = None
            if pd_ is None or not np.isfinite(pd_.get("se", float("nan"))):
                pd_all = paired_delta(vv, base_key=DECOUPLED_BASE,
                                      require_stable=False)
                if pd_all is not None and np.isfinite(pd_all["se"]):
                    pd_, fb_used = pd_all, "all-records"
            if pd_ is None or not np.isfinite(pd_.get("se", float("nan"))):
                print(f"  {k} {vv:<7} 数据不足（缺少两臂都完成且可配对的 ≥2 个公共 seed）")
                continue
            d, se, n, tc = pd_["delta"], pd_["se"], pd_["n"], pd_["tcrit"]
            lo, hi = d - tc * se, d + tc * se
            verdict = ("PASS" if lo >= thr else
                       "FAIL" if hi < thr else "UNDECIDED")
            crit_d[k] = dict(variant=vv, base=DECOUPLED_BASE,
                             base_value=float(base_d), delta=float(d), se=float(se),
                             ci95=[float(lo), float(hi)], threshold=thr,
                             verdict=verdict, n_paired=n, paired_seeds=pd_["seeds"],
                             paired_deltas=pd_["pairs"],
                             method=("paired-t" if fb_used is None
                                     else f"paired-t[{fb_used}]"),
                             stable_only=pd_.get("require_stable", True))
            mark = {"PASS": "PASS", "FAIL": "FAIL", "UNDECIDED": "≈不确定"}[verdict]
            print(f"  {k} {vv:<7} Δ={d:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] "
                  f"门槛≥{thr} [{mark}] 配对n={n} seeds={pd_['seeds']}"
                  + (f" [{fb_used}]" if fb_used else ""))

    # 崩溃率 vs 轴数
    axes = {"V1": 1, "V2a": 2, "V2b": 2, "V2c": 2, "V3": 3, "V4": 4, "Vfree": 3,
            "A1x64": 1, "A2x64": 2, "A3g1": 3, "A3x64": 3}

    # 指标有效性标注：V2c 无初爻监督，但 rollout 硬编码用 yao_log[:,0:2]（初爻）
    # 做接触反馈（model_h.py:183），故其 rollout/yao_acc 不可信；one-step 仍有效。
    # V2c 未进入任何预注册判据，故无需重训，仅在此显式标注避免误用。
    metric_validity = {
        "V2c": dict(one_step="valid",
                    rollout="INVALID（rollout 用无监督的初爻 logits 做接触反馈）",
                    yao_acc="INVALID（爻标签失衡 + 无初爻监督）"),
    }
    print("\n指标有效性:")
    for k, vv in metric_validity.items():
        if k in summary:
            print(f"  {k:<6} one-step={vv['one_step']}  rollout={vv['rollout']}")
    print("\n崩溃率 vs 轴数:")
    for v, n_ax in sorted(axes.items(), key=lambda x: x[1]):
        if v in summary:
            print(f"  {v:<6} 轴数={n_ax}  崩溃率={summary[v]['collapse_rate']:.0%} "
                  f"(n={summary[v]['n']})")

    # 保留已有预注册信息（并补入崩溃处理规则）
    pre = None
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out, encoding="utf-8"))
            pre = old.get("pre_registration")
        except Exception:
            pass
    if pre is None and os.path.exists("analysis_output/E1a_report.json"):
        try:
            pre = json.load(open("analysis_output/E1a_report.json",
                                 encoding="utf-8")).get("pre_registration")
        except Exception:
            pass
    if isinstance(pre, dict):
        pre["crash_rule"] = (
            "崩溃 run（best<5px 后 val>10px 或含 [EARLY-STOP] collapse=1）"
            "从均值剔除，单独计崩溃率；判据用剔崩溃后的稳定均值与稳定 CI95。"
            "若某臂无稳定 run，退回全量均值并标注 stable_fallback=True。")
        pre["crash_rule_sensitivity"] = (
            "2026-09-27 实证：已核验的全部 6 个崩溃 run（V1_s44_nogua/V2a_s47/"
            "V2b_s43/V2b_s44/V2b_s45/V3_s43_nogua）的 best.pt 其 val 恰等于 best"
            "（0.793/1.126/0.908/0.928/1.267/1.032，均<5px），证明权重为崩溃前保存、"
            "数据可用；而对应 checkpoint_latest 已发散至 11–288px，故评测必须用 "
            "best.pt。因此在主口径（剔崩溃）因公共 seed 不足而无法给出 CI 时，"
            "自动退用「配对+全量（含崩溃 run 的 best）」作为敏感性口径，"
            "并在 method 字段显式标注 —— 不改变预注册主口径。"
            "另：各臂崩溃率不等（V1 0/10、V2a 1/10、V2b 3/10）且崩溃 run 的 best "
            "系统性更差，剔除造成差异化选择偏差；故当两种口径判定不一致时"
            "（judge_pair），判据标为 DISAGREE 而不给单一 PASS/FAIL。")
        pre["seed_expansion"] = (
            "2026-09-27 起将 V1/V2a/V2b/Vfree/V3 常态臂扩至 seeds 42-51 "
            "（各 10 seeds），以使 5% 门槛在 95% CI 下可判定。")

    seeds_seen = sorted({r["seed"] for r in runs})
    seeds_incomplete = sorted({r["seed"] for r in runs_incomplete})

    # 计划表：显式列出「应有」的 run，完整性据此判定（而非按已解析 run 反推）
    plan_norm_seeds = parse_seeds(args.plan_normal)
    plan_nogua_seeds = parse_seeds(args.plan_nogua)
    plan_norm_vars = [v.strip() for v in args.plan_normal_variants.split(",")
                      if v.strip() in VARIANTS]
    plan_fixed_seeds = parse_seeds(args.plan_fixed_seeds)
    plan_fixed_vars = [v.strip() for v in args.plan_fixed_variants.split(",")
                       if v.strip() in VARIANTS and v.strip() not in plan_norm_vars]
    planned = [(v, s, False) for v in plan_norm_vars for s in plan_norm_seeds]
    # 未扩种子但仍属实验一部分的臂（V2c/V4），按其实际种子数计入计划，
    # 否则会永远报 missing、使 status 恒为 INCOMPLETE。
    planned += [(v, s, False) for v in plan_fixed_vars for s in plan_fixed_seeds]
    planned += [(v, s, True) for v in summary
                if v.endswith("_nogua") for s in plan_nogua_seeds]
    # 两边统一用「臂名」（no_gua 臂带 _nogua 后缀）比对：planned 的 no_gua 项用的是
    # summary 键 "V1_nogua"，而 r["variant"] 是 "V1" —— 若不统一，18 个 no_gua run
    # 会被全部误报为 missing（2026-09-27 实证：completed 61 + missing 44 > 计划 88）。
    have = {(r["variant"] + ("_nogua" if r["no_gua"] else ""), r["seed"], r["no_gua"])
            for r in runs_all}

    def log_name(v, s, ng):
        # 计划键的 no_gua 项已带 _nogua 后缀（"V1_nogua"），还原成 run_e1a 的
        # 命名 "V1_s42_nogua"
        base = v[:-6] if (ng and v.endswith("_nogua")) else v
        return f"{base}_s{s}" + ("_nogua" if ng else "")

    # 「已启动但尚无 val 行」的 run（日志刚建、首个验证未出，约 1–2 分钟窗口）
    # 既不在 runs_all 也不该算 missing —— 否则在该窗口内查询会被误报为未跑。
    #
    # 完整性按**计划**逐条归类（完成 / 进行中 / 缺失），三者之和恒等于计划数。
    # 注意：不能拿「全部已完成 run 数」去和「计划数」比 —— 当用子集计划表查看某个
    # 批次（如 Stage D 只列 A 臂）时两者不同域，会误报账目不符（2026-09-28 实证）。
    missing, started_no_val = [], []
    n_planned_done = 0
    for p in planned:
        if (p[0], p[1], p[2]) in have:
            # have 来自 runs_all（含未完成）；仅当该 run 真的 done 才计完成
            key_done = any((r["variant"] + ("_nogua" if r["no_gua"] else ""),
                            r["seed"], r["no_gua"]) == p and r["done"]
                           for r in runs_all)
            if key_done:
                n_planned_done += 1
            continue
        nm = log_name(*p)
        if os.path.exists(os.path.join(args.logs, nm + ".log")):
            started_no_val.append(nm)
        else:
            missing.append(p)
    expected_runs = len(planned)
    # 进行中 = 计划内已启动（有日志）但未 done 的
    n_incomplete = (expected_runs - n_planned_done - len(missing))
    all_done = (n_planned_done == expected_runs)
    n_missing = len(missing)

    # 自检：完成 + 进行中 + 缺失 恒等于计划数（按计划归类，与批次视图无关）。
    assert n_planned_done + n_incomplete + n_missing == expected_runs, (
        f"完整性账目不符：完成 {n_planned_done} + 进行中 {n_incomplete} "
        f"+ 缺失 {n_missing} != 计划 {expected_runs}")

    if n_missing or n_incomplete:
        print(f"\n[完整性] 计划 {expected_runs} runs；"
              f"缺失 {n_missing}，已解析但未完成 {n_incomplete}")
        for v, s, ng in missing[:12]:
            print(f"   缺失: {v}_s{s}")
        if n_missing > 12:
            print(f"   ... 另有 {n_missing - 12} 个缺失")
    out = dict(
        experiment="E1a",
        title="爻轴消融（Yao-Axis Ablation）",
        status=("COMPLETE (corrected protocol)" if all_done
                else f"INCOMPLETE (corrected protocol, missing={n_missing},"
                     f" running/unfinished={n_incomplete})"),
        expected_runs=expected_runs,
        completed_runs=n_planned_done,
        # 计划内、已解析（有指标）但未 [OK] 的 run —— 过滤掉计划外的批次
        incomplete_runs=[{k: v for k, v in r.items() if k != "log"}
                         for r in runs_incomplete
                         if (r["variant"] + ("_nogua" if r["no_gua"] else ""),
                             r["seed"], r["no_gua"]) in set(planned)],
        started_no_metrics=started_no_val,
        incomplete_seeds=seeds_incomplete,
        missing_runs=[f"{v}_s{s}" for v, s, ng in missing],
        plan=dict(normal_variants=plan_norm_vars,
                  normal_seeds=plan_norm_seeds,
                  fixed_variants=plan_fixed_vars,
                  fixed_seeds=plan_fixed_seeds,
                  nogua_seeds=plan_nogua_seeds),
        seeds=seeds_seen,
        pre_registration=pre,
        variants=VARIANTS,
        summary=summary,
        gating_gain=gating,
        criteria=crit,
        criteria_decoupled_exploratory=crit_d,
        metric_validity=metric_validity,
        runs=[{k: v for k, v in r.items() if k != "log"} for r in runs],
        caveats=[
            "爻标签类别失衡：上爻(obj0) 恒为 0、初爻正例率约 5%，"
            "故 yao_acc 跨变体不可比，报告中不使用该指标。",
            "轴数消融为弱消融：yao_head 恒为 6 维、gua_gate 恒吃全部 3 轴 logits，"
            "被移除的轴仍经 state loss 获得梯度；且 n_gua=4^n_axes 与轴数共变，"
            "爻语义与门控容量不可分离。",
            "V2c 的 rollout 用初爻(yao_log[:,0:2]) 做接触反馈，"
            "而 V2c 无初爻监督，其结果不可信。",
            "训练用 scheduled sampling(升至0.5) 而验证 ss_prob=0，"
            "one-step 指标不受影响，rollout 指标存在 train/eval 不一致。",
            "corrected 协议 windows_per_epoch=60000×60 轮 ≈ 7 个全量 epoch，"
            "优于旧协议 60 个全量 epoch 的原因尚未解释。",
            "评测必须加载 best.pt：崩溃 run 的 checkpoint_latest 已发散至 13–275px，"
            "而 best.pt 保持崩溃前最优（<5px）。两者的 best 字段相同，"
            "勿以 best 字段判断权重是否发散。",
            "崩溃剔除存在差异化选择偏差：各臂崩溃率不同（V1 0/10、V2a 1/10、"
            "V2b 3/10），且崩溃 run 的 best 系统性更差（V2b 崩溃 best 均值 1.034 "
            "vs 稳定 0.750），故剔除对各臂偏袒程度不等，会扭曲跨臂比较。"
            "报告同时给出剔崩溃与全量两种口径，判据仅在两口径判定一致时才给 "
            "PASS/FAIL，否则标 DISAGREE（口径敏感）。",
        ],
        note="崩溃定义: best<5px 后出现 val>10px，或日志含 [EARLY-STOP] collapse=1"
             "（patience 早停但未崩溃不计）。"
             "判据口径：one-step RMSE 相对 V3；主口径为同 seed 配对差值 + t 分布 95% CI，"
             "无公共 seed 时退回非配对正态近似并标注 method；CI 跨门槛判 UNDECIDED。",
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[report] {args.out}")


if __name__ == "__main__":
    main()
