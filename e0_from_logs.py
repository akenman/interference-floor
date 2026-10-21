"""E0 日志 → 结构化报告（从 logs/*.log 抽取已完成的 arm×seed 结果）。
用途：E0 训练中途即可产出 interim report，供 build_e0_figures.py 出图。
"""
import glob, json, os, re
import numpy as np

LOG_DIR = "logs"
OUT = "analysis_output/E0_report.json"

# [OK] <arm> seed=<s> 最佳 val one-step = <v>px (epoch <e>)
DONE = re.compile(
    r"\[OK\]\s+(\S+)\s+seed=(\d+)\s+最佳\s+val one-step\s*=\s*([0-9.]+)px"
    r"(?:\s*\(epoch\s+(\d+)\))?\s*$")
VAL = re.compile(r"\[val\]\s+one-step\s+模型\s+([0-9.]+)\s*\|")
ROLL = re.compile(r"\[val\]\s+rollout\s*=\s*\[([0-9.,\s]+)\]")


def parse_log(path):
    """返回 (arm, seed, {one_step, rollout, yao_acc, prior, zero_vel, epoch, status})"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    lines = [l.rstrip("\r") for l in txt.splitlines() if l.strip()]
    name = os.path.basename(path).replace(".log", "")
    if "_s" not in name:
        return None, None, None
    arm, seed = name.rsplit("_s", 1)

    info = {"one_step": None, "rollout": None, "yao_acc": None,
            "prior": None, "zero_vel": None, "epoch": None, "status": "running"}
    for ln in lines:
        m = DONE.search(ln)
        if m:
            info["one_step"] = float(m.group(3))
            info["epoch"] = int(m.group(4)) if m.group(4) else None
            info["status"] = "done"
            continue
        if info["status"] == "done":
            continue
        m = VAL.search(ln)
        if m:
            info["one_step"] = float(m.group(1))
        # 逐 epoch 的 rollout 行：[val] rollout = [k1, k2, ...]
        m = ROLL.search(ln)
        if m:
            info["rollout"] = [float(x) for x in m.group(1).split(",")]
    return arm, seed, info


def main():
    arms = {}
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        if not os.path.getsize(path):
            continue
        arm, seed, info = parse_log(path)
        if info["one_step"] is None:
            continue
        arms.setdefault(arm, {})[seed] = info

    report = {
        "experiment": "E0",
        "title": "统一主表（Table 1）— interim（由日志抽取，非正式）",
        "note": "RTX 5060 8GB；held-out = 300 episodes（seed=42 90/10）；"
                "20000 随机窗口，pos_rmse 位置 4 维 feat_weights 加权。",
        "holdout_note": "held-out = 300 episodes (seed=42 90/10)",
        "arms": {},
        "gua_rmse_base": {},
        "gua_occupancy_base": {},
        "summary": {},
        "status": f"INTERIM — {sum(len(v) for v in arms.values())} 个 arm×seed 有数据",
        "findings": [],
        "bugs_found": [],
        "limitations": [],
    }

    for arm in sorted(arms):
        seeds = arms[arm]
        recs = []
        for s in sorted(seeds):
            i = seeds[s]
            rec = {"seed": s}
            for k in ("one_step", "yao_acc", "prior", "zero_vel"):
                if i.get(k) is not None:
                    rec[k] = round(i[k], 4)
            if i.get("rollout"):
                rec["rollout"] = {f"k={k}": round(v, 3)
                                  for k, v in enumerate(i["rollout"], 1)}
            rec["epoch"] = i["epoch"]
            rec["status"] = i["status"]
            recs.append(rec)
        report["arms"][arm] = recs

        vals = [r["one_step"] for r in recs if "one_step" in r]
        if len(vals) >= 2:
            m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
            print(f"{arm:14s} n={len(vals)}  mean={m:.3f} std={s:.3f} "
                  f"cv={100*s/m:.1f}%  {[round(r['one_step'],3) for r in recs]}")
        elif vals:
            print(f"{arm:14s} n=1  {vals[0]:.3f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[ok] {OUT}")


if __name__ == "__main__":
    main()