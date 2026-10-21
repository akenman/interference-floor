# -*- coding: utf-8 -*-
"""
E1b 评估脚本 (eval_e1b.py)
==================================================
对 E1b 三臂 × 3 种子的 checkpoint 做统一评估：
  - 整体 one-step 位置 RMSE（位置 4 维，E0 协议）
  - 8 卦各自的 one-step 位置 RMSE（带样本数）
  - rollout k=1..5、爻准确率

用法：
  python eval_e1b.py --pattern "checkpoints-e1b/*/best.pt" --output analysis_output/E1b_raw.json

输出 JSON 结构：
  { "runs": [ {checkpoint, variant, seed, params, overall, gua_err, gua_count, rollout, yao_acc} ],
    "arms": { variant: { overall: {mean, std}, gua_err: {gua: {mean,std}}, ...} } }
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, NUM_OBJ, PER_OBJ,
                      DT as DT_H, feat_weights as feat_weights_h)
from model_h_e1b import PhaseCapacityWorldModelH
from model_h import WorldModelH

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS
POS_IDX = [1, 2, 12, 13]
GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]


def gua_idx_from_yao(yao_t):
    bits = yao_t[:, [1, 3, 5]].long()
    return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]


def pos_rmse(pred_raw, tgt_raw):
    err = (pred_raw[:, POS_IDX] - tgt_raw[:, POS_IDX]) ** 2
    return float(np.sqrt(err.mean()))


def per_gua_rmse(pred_raw, tgt_raw, gua_idx, n_gua=8):
    errs = np.zeros(n_gua, dtype=np.float64)
    counts = np.zeros(n_gua, dtype=np.int64)
    e = np.sqrt(((pred_raw[:, POS_IDX] - tgt_raw[:, POS_IDX]) ** 2).mean(axis=1))
    for g in range(n_gua):
        m = gua_idx == g
        if m.any():
            errs[g] = float(e[m].mean())
            counts[g] = int(m.sum())
    return errs, counts


class HingeData:
    def __init__(self, states, contact, actions, yao, episodes, L, K,
                 mean_obj, std_obj):
        self.L, self.K = L, K
        self.mean_obj, self.std_obj = mean_obj, std_obj
        episodes = np.asarray(episodes)
        T = states.shape[1]
        E = len(episodes)
        self.obj_norm = ((states[episodes] - mean_obj) / (std_obj + 1e-8)).astype(np.float32)
        self.act_norm = actions[episodes].astype(np.float32)
        self.contact = contact[episodes].astype(np.float32)
        self.yaos = yao[episodes]
        n_win = T - L - K + 1
        self.ep_idx = np.repeat(np.arange(E), n_win)
        self.win_start = np.tile(np.arange(n_win), E)

    def __len__(self):
        return len(self.ep_idx)

    def gather(self, idx):
        ep = self.ep_idx[idx]
        i0 = self.win_start[idx]
        L, K = self.L, self.K
        B = len(idx)
        off = np.arange(L + K)
        frm = self.obj_norm[ep[:, None], i0[:, None] + off[None, :]]
        ct = self.contact[ep[:, None], i0[:, None] + off[None, :]]
        def assemble(obj_seg, ct_seg, n):
            return np.concatenate([obj_seg, ct_seg], axis=2)
        x = assemble(frm[:, :L], ct[:, :L], L)
        x_next = assemble(frm[:, 1:L + 1], ct[:, 1:L + 1], L)
        x_fut = assemble(frm[:, L:L + K], ct[:, L:L + K], K)
        a0 = self.act_norm[ep, i0 + L]
        af = self.act_norm[ep[:, None], i0[:, None] + L + np.arange(K)[None, :]]
        prior_n = self._prior_np(frm[:, L - 1], a0)
        d_norm = frm[:, L] - prior_n
        y_raw = frm[:, L] * (self.std_obj + 1e-8) + self.mean_obj
        s_last = frm[:, L - 1] * (self.std_obj + 1e-8) + self.mean_obj
        yao_t = self.yaos[ep, i0 + L].reshape(B, 6)
        ct_t = self.contact[ep, i0 + L]
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return dict(x=to(x), x_next=to(x_next), a0=to(a0), af=to(af),
                    d_norm=to(d_norm), y_raw=to(y_raw), yao=to(yao_t),
                    s_last=to(s_last), ct=to(ct_t), x_fut=to(x_fut))

    def _prior_np(self, obj_norm_batch, a_norm):
        from layout_h import L1, L2, M1, M2, TAU_MAX
        raw = obj_norm_batch * (self.std_obj + 1e-8) + self.mean_obj
        b1, b2 = 0, 11
        th1 = np.arctan2(raw[:, b1 + 3], raw[:, b1 + 4])
        th2 = np.arctan2(raw[:, b2 + 3], raw[:, b2 + 4])
        w1 = raw[:, b1 + 7]
        w2 = raw[:, b2 + 7]
        tau = a_norm[:, 0] * TAU_MAX
        c2 = np.cos(th2); s2 = np.sin(th2)
        M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
        M12 = M2 * L2**2 + M2 * L1 * L2 * c2
        M22 = M2 * L2**2
        h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
              + (M1 + M2) * 900.0 * L1 * np.sin(th1)
              + M2 * 900.0 * L2 * np.sin(th1 + th2))
        h2 = (M2 * L1 * L2 * w1**2 * s2 + M2 * 900.0 * L2 * np.sin(th1 + th2)
              - tau)
        det = M11 * M22 - M12**2
        det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9, det)
        a1 = (M22 * (-h1) - M12 * (-h2)) / det
        a2 = (-M12 * (-h1) + M11 * (-h2)) / det
        w1n = w1 + a1 * 0.05
        w2n = w2 + a2 * 0.05
        th1n = th1 + w1n * 0.05
        th2n = th2 + w2n * 0.05
        out = raw.copy()
        x1 = L1 * np.sin(th1n); y1 = -L1 * np.cos(th1n)
        x2 = x1 + L2 * np.sin(th1n + th2n); y2 = y1 - L2 * np.cos(th1n + th2n)
        vx1 = L1 * w1n * np.cos(th1n); vy1 = L1 * w1n * np.sin(th1n)
        vx2 = vx1 + L2 * (w1n + w2n) * np.cos(th1n + th2n)
        vy2 = vy1 + L2 * (w1n + w2n) * np.sin(th1n + th2n)
        out[:, b1 + 1], out[:, b1 + 2] = x1, y1
        out[:, b2 + 1], out[:, b2 + 2] = x2, y2
        out[:, b1 + 3], out[:, b1 + 4] = np.sin(th1n), np.cos(th1n)
        out[:, b2 + 3], out[:, b2 + 4] = np.sin(th2n), np.cos(th2n)
        out[:, b1 + 5], out[:, b1 + 6] = vx1, vy1
        out[:, b2 + 5], out[:, b2 + 6] = vx2, vy2
        out[:, b1 + 7], out[:, b2 + 7] = w1n, w2n
        return (out - self.mean_obj) / (self.std_obj + 1e-8)


def build_model(config, device):
    cfg = dict(config)
    cls = PhaseCapacityWorldModelH if cfg.get("variant") == "phase" else WorldModelH
    kwargs = {k: v for k, v in cfg.items() if k in
              ("input_dim", "obj_dim", "action_dim", "hidden_dim",
               "num_prototypes", "yao_dim", "sub_head_dim")}
    kwargs.setdefault("input_dim", INPUT_DIM)
    kwargs.setdefault("obj_dim", OBJ_DIMS)
    kwargs.setdefault("action_dim", ACTION_DIM)
    kwargs.setdefault("yao_dim", 6)
    model = cls(**kwargs).to(device)
    return model


@torch.no_grad()
def evaluate(model, ds, device, K, max_windows):
    model.eval()
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    bs = 1024
    pred_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    prior_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    pred_roll = [np.zeros((M, OBJ_DIMS), dtype=np.float32) for _ in range(K)]
    yao_hits = 0
    yao_n = 0
    gua_idx_all = np.zeros(M, dtype=np.int64)
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        ab = b["a0"].to(device)
        h, c = model.encode_sequence(xb)
        out = model.forward_state(h, ab) if isinstance(model, PhaseCapacityWorldModelH) \
            else model.predict_state(h, ab)
        delta = out.float().cpu().numpy()
        prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()
        pred_one[sl] = (prior_n + delta) * (ds.std_obj + 1e-8) + ds.mean_obj
        prior_one[sl] = prior_n * (ds.std_obj + 1e-8) + ds.mean_obj
        yao_log = model.predict_yao(h)
        yp = (torch.sigmoid(yao_log.float()) > 0.5).cpu().numpy()
        yt = b["yao"].numpy()
        yao_hits += (yp == yt).mean() * len(idx)
        yao_n += len(idx)
        afb = b["af"].to(device)
        xf = b["x_fut"].to(device)
        abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=0.0)
        for k in range(K):
            pred_roll[k][sl] = abs_preds[k].float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj
        gua_idx_all[sl] = gua_idx_from_yao(b["yao"]).numpy()

    frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
    tgt = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj
    rollout_tgt = [
        ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L + k]
        * (ds.std_obj + 1e-8) + ds.mean_obj
        for k in range(K)
    ]
    s_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1] \
        * (ds.std_obj + 1e-8) + ds.mean_obj

    gua_err, gua_cnt = per_gua_rmse(pred_one, tgt, gua_idx_all)

    return {
        "one_step_model": pos_rmse(pred_one, tgt),
        "one_step_prior": pos_rmse(prior_one, tgt),
        "one_step_zero_vel": pos_rmse(s_last, tgt),
        "rollout": {f"k={k + 1}": pos_rmse(pred_roll[k], rollout_tgt[k]) for k in range(K)},
        "yao_acc": float(yao_hits / max(yao_n, 1)),
        "gua_err": {f"g{g} ({GUA_NAMES[g]})": float(gua_err[g]) for g in range(8)},
        "gua_count": {f"g{g}": int(gua_cnt[g]) for g in range(8)},
    }


def main():
    parser = argparse.ArgumentParser(description="E1b 统一评估")
    parser.add_argument("--pattern", type=str, default="checkpoints-e1b/*/best.pt")
    parser.add_argument("--data_path", type=str, default="experiments/trajectories_hinge_heldout.npz")
    parser.add_argument("--val_windows", type=int, default=0, help="0 = 全部验证窗口")
    parser.add_argument("--output", type=str, default="analysis_output/E1b_raw.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        sys.exit(f"[error] 未找到匹配的检查点: {args.pattern}")

    d = np.load(args.data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    val_eps = np.arange(N)[:max(1, int(N * 0.1))]
    print(f"评估数据: {args.data_path} | 验证片段: {len(val_eps)}")

    runs = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        cfg = dict(ck.get("config", {}))
        variant = cfg.get("variant", "uniform")
        seed = ck.get("args", {}).get("seed", -1)
        model = build_model(cfg, device)
        ns = ck.get("norm_stats", {})
        if "mean_obj" in ns:
            model.set_stats(ns["mean_obj"], ns["std_obj"])
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.eval()

        if "mean_obj" not in ns or "std_obj" not in ns:
            raise ValueError(f"checkpoint 缺少训练集归一化统计: {p}")
        mean_obj = np.asarray(ns["mean_obj"], dtype=np.float32)
        std_obj = np.asarray(ns["std_obj"], dtype=np.float32)
        ds = HingeData(states, contact, actions, yao, val_eps, 8, 5, mean_obj, std_obj)

        m = evaluate(model, ds, device, 5, args.val_windows if args.val_windows else None)
        params = sum(pr.numel() for pr in model.parameters())

        print(f"[{variant} seed={seed}] one-step={m['one_step_model']:.3f} "
              f"(先验 {m['one_step_prior']:.3f}) | "
              + " ".join(f"{k}={v:.3f}" for k, v in m["gua_err"].items()))
        runs.append(dict(
            checkpoint=p, variant=variant, seed=seed, params=params,
            overall=m["one_step_model"], prior=m["one_step_prior"],
            zero_vel=m["one_step_zero_vel"], rollout=m["rollout"],
            yao_acc=m["yao_acc"], gua_err=m["gua_err"], gua_count=m["gua_count"],
        ))

    # ---- 按臂汇总（mean ± std）---------------------------------------
    arms = {}
    for r in runs:
        arms.setdefault(r["variant"], []).append(r)
    summary = {}
    for variant, rs in arms.items():
        n = len(rs)
        overall = np.array([r["overall"] for r in rs])
        yao = np.array([r["yao_acc"] for r in rs])
        rollout = {k: np.array([r["rollout"][k] for r in rs]) for k in rs[0]["rollout"]}
        gua = {k: np.array([r["gua_err"][k] for r in rs]) for k in rs[0]["gua_err"]}
        cnt = {k: np.array([r["gua_count"][k] for r in rs]) for k in rs[0]["gua_count"]}
        summary[variant] = dict(
            n_seeds=n,
            overall_mean=float(overall.mean()), overall_std=float(overall.std()),
            yao_acc_mean=float(yao.mean()), yao_acc_std=float(yao.std()),
            rollout_mean={k: float(v.mean()) for k, v in rollout.items()},
            rollout_std={k: float(v.std()) for k, v in rollout.items()},
            gua_err_mean={k: float(v.mean()) for k, v in gua.items()},
            gua_err_std={k: float(v.std()) for k, v in gua.items()},
            gua_count_mean={k: float(v.mean()) for k, v in cnt.items()},
        )

    out = dict(runs=runs, arms=summary)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 评估 {len(runs)} 个检查点 → {args.output}")
    for variant, s in summary.items():
        print(f"  {variant}: one-step = {s['overall_mean']:.3f} ± {s['overall_std']:.3f}")


if __name__ == "__main__":
    main()
