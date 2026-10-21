"""E0 统一主表（Table 1）生成器 — 读 checkpoints-E0 全部臂×种子，输出主表 + JSON。

等 E0 跑完后执行：python build_e0_table.py [--out analysis_output/E0_report.json]
"""
import argparse, json, os, sys, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_common import load_checkpoint, validate, E0Data
from layout_h import OBJ_DIMS, POS_IDX

states = np.load("trajectories_hinge.npz")["states"].astype(np.float64)
contact = np.load("trajectories_hinge.npz")["contact"]
actions = np.load("trajectories_hinge.npz")["actions"]
yao = np.load("trajectories_hinge.npz")["yao"]
N = states.shape[0]

rng = np.random.RandomState(42)
segs = np.arange(N); rng.shuffle(segs)
val_eps = segs[:max(1, int(N * 0.1))]
tr_eps = segs[int(N * 0.1):]
obj_flat = states[tr_eps].reshape(-1, 22).astype(np.float64)
mean_obj = obj_flat.mean(0).astype(np.float32)
std_obj = obj_flat.std(0).astype(np.float32)
val_ds = E0Data(states, contact, actions, yao, val_eps, L=8, K=5,
                mean_obj=mean_obj, std_obj=std_obj)


def load_all(arm_dir):
    pts = sorted(glob.glob(os.path.join(arm_dir, "best.pt")))
    out = []
    for p in pts:
        try:
            model, arm, ck = load_checkpoint(p, torch.device("cpu"))
            model.eval()
            m = validate(model, val_ds, torch.device("cpu"), K=5, max_windows=20000)
            out.append(dict(path=p, arm=arm, seed=ck.get("seed"), epoch=ck.get("epoch"),
                            **m))
        except Exception as e:
            print(f"  WARN {p}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis_output/E0_report.json")
    args = ap.parse_args()

    arms = {}
    for d in sorted(glob.glob("checkpoints-E0/*")):
        if not os.path.isdir(d):
            continue
        arm = os.path.basename(d).rsplit("_s", 1)[0]
        arms.setdefault(arm, []).append(d)

    print(f"[build] {len(arms)} 臂, {sum(len(v) for v in arms.values())} 个 checkpoint")
    table = {}
    for arm, dirs in sorted(arms.items()):
        rows = []
        for d in dirs:
            rows += load_all(d)
        if rows:
            table[arm] = rows
            print(f"  {arm:<14} n={len(rows)}  "
                  f"one_step={np.mean([r['one_step_model'] for r in rows]):.3f} "
                  f"±{np.std([r['one_step_model'] for r in rows]):.3f}  "
                  f"prior={np.mean([r['one_step_prior'] for r in rows]):.3f}")

    # 主表（Table 1 风格）
    print("\n" + "=" * 78)
    print("E0 统一主表（Table 1）— held-out 20000 窗口，pos_rmse (px)")
    print("=" * 78)
    hdr = f"{'arm':<14}{'n':>3}{'model':>9}{'prior':>9}{'zero_vel':>9}{'yao_acc':>8}"
    print(hdr); print("-" * len(hdr))
    for arm in sorted(table):
        r = table[arm]
        print(f"{arm:<14}{len(r):>3}"
              f"{np.mean([x['one_step_model'] for x in r]):9.3f}"
              f"±{np.std([x['one_step_model'] for x in r]):6.3f}  "
              f"{np.mean([x['one_step_prior'] for x in r]):9.3f}"
              f"{np.mean([x['one_step_zero_vel'] for x in r]):9.3f}"
              f"{np.mean([x['yao_acc'] for x in r]):8.3f}")

    # 卦级汇总
    print("\n卦级 one-step RMSE（先验臂）：")
    if "ours_base" in table:
        g = []
        for r in table["ours_base"]:
            row = [r["gua_rmse"].get(k) for k in range(8)]
            row = [v if v is not None and v == v else np.nan for v in row]
            g.append(row)
        g = np.array(g)
        if g.size:
            names = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]
            for i, n in enumerate(names):
                print(f"  {n}: {g[:, i].mean():.3f} ± {g[:, i].std():.3f}  "
                      f"(n={(~np.isnan(g[:, i])).sum()})")

    out = dict(arms=table,
               pos_dims=POS_IDX,
               note="held-out = seed=42 90/10 划分中的 10%；pos_rmse 位置 4 维 feat_weights 加权")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[report] {args.out}")


if __name__ == "__main__":
    main()