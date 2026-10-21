# -*- coding: utf-8 -*-
"""
观象世界模型 v3 训练脚本 (train_v3.py)
==================================================
与旧 train.py 的关系：**不共用代码路径**。旧脚本是 18 个补丁的载体（v2 布局、
12+ 项损失），保留作历史对照；v3 按"减法"原则另起干净管线：

  - 数据：trajectories_v3.npz（layout.py 组装 116 维），episode 级 90/10 划分，
    归一化统计只取训练轨迹，统计量随 checkpoint 保存（评估复现无忧）
  - 内存：惰性窗口数据源（预归一化数组 + 向量化 fancy-indexing gather），
    5 万条全量 ~1.9GB 常驻，无逐窗 Python 开销、无 DataLoader 拷贝
  - 模型：model_v3.WorldModelV3（GRU 记忆 + 残差参数化状态主头 +
    三爻头 + 原型辅助层）
  - 损失：5 项（state/yao/proto/commit/usage），权重全部可从命令行消融
  - 模型选择：验证集 one-step 位置 RMSE（纯预测力，无 health score 启发式）
  - 验证报告：每 epoch 与零速度/含重力惯性基线同台对比，按接触 regime 分层

用法：
  python train_v3.py --data_path trajectories_v3.npz --epochs 40 --windows_per_epoch 2000000
  python train_v3.py --data_path experiments/trajectories_v3_pilot.npz --epochs 30
"""

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from layout import (OBJ_DIMS, OBS_DIMS, CONTACT_DIMS, ASSEMBLED_DIM, ACTION_DIM,
                    DT_V3, IMPULSE_MAX_V3, NUM_OBJ, PER_OBJ, F as LF,
                    obj_pos_dims, feat_weights, yao_labels, physics_prior)
import physics_prior_spec_v3
from model_v3 import WorldModelV3, V3Loss, GatedWorldModelV3

GRAVITY_Y = -900.0
POS_DIMS_OBJ = obj_pos_dims()   # 物体段(33)内的位置维度


def parse_args():
    p = argparse.ArgumentParser(description="观象世界模型 v3 训练")
    p.add_argument("--data_path", type=str, default="trajectories_v3.npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=8, help="输入历史窗口长度 L")
    p.add_argument("--future_steps", type=int, default=3, help="闭环 rollout 步数 K")
    p.add_argument("--num_prototypes", type=int, default=32)
    p.add_argument("--num_coarse", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w_state", type=float, default=1.0)
    p.add_argument("--w_yao", type=float, default=0.1)
    p.add_argument("--w_proto", type=float, default=0.05)
    p.add_argument("--w_commit", type=float, default=0.01)
    p.add_argument("--w_usage", type=float, default=0.02)
    p.add_argument("--w_roll", type=float, default=0.3,
                   help="闭环 rollout 损失权重（scheduled sampling，0=关闭）")
    p.add_argument("--w_cal", type=float, default=0.2,
                   help="G2 校准损失权重（高斯 NLL，异方差 σ 头）")
    p.add_argument("--noise_aug", type=float, default=0.0,
                   help="G1 观测噪声增强上限（归一化空间逐批 U(0, noise_aug)；0=关闭）")
    p.add_argument("--ss_max", type=float, default=0.5,
                   help="scheduled sampling 教师强制概率上限（E1 实验=1.0）")
    p.add_argument("--use_gua", action="store_true",
                   help="M2b：状态头加入构造性卦象条件化（三爻位模式嵌入）")
    p.add_argument("--gua_emb_dim", type=int, default=16)
    p.add_argument("--windows_per_epoch", type=int, default=0,
                   help="每 epoch 采样的窗口数上限（0=全部；全量 5 万条约 855 万窗，"
                        "建议 200 万以控制单 epoch 时长）")
    p.add_argument("--val_windows", type=int, default=20000,
                   help="验证采样窗口数上限（0=全部）")
    p.add_argument("--no_amp", action="store_true", help="禁用 bf16 自动混合精度")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints-v3")
    p.add_argument("--prior_spec", type=str, default="parabola005",
                   choices=sorted(physics_prior_spec_v3.SPECS),
                   help="先验积分器档位（SLOT-1b 阶梯）。默认 parabola005 = legacy "
                        "闭式抛物线（逐位兼容历史 checkpoint）；训练期残差目标与"
                        "评估期 model.prior_norm 必须同档")
    p.add_argument("--w_ff", type=float, default=1.0,
                   help="SLOT-1c regime 重加权：末输入帧与目标帧均无接触的窗口，"
                        "状态损失权重（1.0=关闭，历史行为）")
    p.add_argument("--w_ff_roll", type=float, default=1.0,
                   help="SLOT-1e 闭环 regime 重加权：同一 win_ff 掩码施加在 rollout 项上"
                        "（1.0=关闭，历史行为逐位不变；与 --w_ff 独立登记）")
    p.add_argument("--w_ct_roll", type=float, default=0.0,
                   help="SLOT-1f：闭环各步接触位 BCE 权重（0=关闭，历史行为）。把 regime "
                        "指示器训在它被使用的闭环步分布上（单步爻损失只覆盖 k=1）")
    p.add_argument("--ct_pos_weight", type=float, default=1.0,
                   help="SLOT-1f：接触位 BCE 的正类权重（<1 = 降低'接触'类偏置；"
                        "单元类比约 60:40）")
    p.add_argument("--gate", action="store_true",
                   help="SLOT-1c：使用 GatedWorldModelV3（静默门控头）")
    p.add_argument("--gate_supervise", action="store_true",
                   help="SLOT-1d：门监督——BCE(g_logits, 1−win_ff) 教门开合"
                        "（需 --gate；SLOT-1c 的端到端门在梯度饥饿下无效）")
    p.add_argument("--w_gate_bce", type=float, default=1.0,
                   help="门监督 BCE 权重（--gate_supervise 时生效）")
    p.add_argument("--log_dir", type=str, default="runs/world_model_v3")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ============================================================
# 数据（惰性窗口数据源）
# ============================================================

def load_dataset(path):
    d = np.load(path)
    return d["states"], d["obstacles"], d["contact"], d["actions"]


def split_episodes(n, seed=42, val_ratio=0.1):
    rng = np.random.RandomState(seed)
    segs = np.arange(n)
    rng.shuffle(segs)
    val_count = max(1, int(n * val_ratio))
    return segs[val_count:], segs[:val_count]


def norm_stats(states, obstacles, episodes):
    """逐维统计只取训练轨迹（物体段 + 障碍段）。"""
    obj = states[episodes].reshape(-1, OBJ_DIMS).astype(np.float64)
    obs = obstacles[episodes].astype(np.float64)
    return (obj.mean(0).astype(np.float32), obj.std(0).astype(np.float32),
            obs.mean(0).astype(np.float32), obs.std(0).astype(np.float32))


class V3Data:
    """惰性窗口数据源。

    预归一化数组常驻内存（5 万条约 1.9GB），窗口用向量化 fancy-indexing 现场组装：
      窗口 i：输入帧 [i, i+L)，预测目标帧 i+L（残差 = 帧 i+L − 帧 i+L−1），
      rollout 目标帧 [i+L, i+L+K)，动作 [i+L, i+L+K)（净冲量归一化）。
    """

    def __init__(self, states, obstacles, contact, actions, episodes, L, K,
                 mean_obj, std_obj, mean_obs, std_obs, prior_spec=None):
        self.L, self.K = L, K
        self.mean_obj, self.std_obj = mean_obj, std_obj
        # SLOT-1b：先验档位（physics_prior_spec_v3.SPECS）；默认 legacy 闭式（逐位兼容）
        self.prior_spec = prior_spec or "parabola005"
        episodes = np.asarray(episodes)
        T = states.shape[1]
        E = len(episodes)
        print(f"    [数据] 预处理 {E} 条片段（归一化 + 三爻标签）...")
        self.obj_norm = ((states[episodes] - mean_obj) / (std_obj + 1e-8)).astype(np.float32)
        self.obs_norm = ((obstacles[episodes] - mean_obs) / (std_obs + 1e-8)).astype(np.float32)
        self.act_norm = (actions[episodes] / IMPULSE_MAX_V3).astype(np.float32)
        self.contact = contact[episodes].astype(np.float32)

        yaos = np.zeros((E, T, 3, NUM_OBJ), dtype=np.int8)
        for ei, ep in enumerate(episodes):
            yaos[ei] = yao_labels(states[ep], contact[ep].astype(np.int8), obstacles[ep])
        self.yaos = yaos

        n_win = T - L - K + 1
        self.ep_idx = np.repeat(np.arange(E), n_win)
        self.win_start = np.tile(np.arange(n_win), E)
        # SLOT-1c：窗口级 regime 旗（末输入帧与目标帧均无接触）——regime 重加权用
        cf_bool = self.contact.max(axis=2) > 0.5                     # (E,T)
        c_last = cf_bool[:, L - 1:L - 1 + n_win]
        c_tgt = cf_bool[:, L:L + n_win]
        self.win_ff = ((~c_last) & (~c_tgt)).astype(np.float32)      # (E,n_win)

    def __len__(self):
        return len(self.ep_idx)

    def gather(self, idx):
        """idx: int 数组 → 批量窗口张量（全部 CPU float32，调用方 .to(device)）。

        M1d：残差目标 = 真值 − 无接触物理先验（惯性+重力+动作冲量，layout.physics_prior），
        网络只学接触/碰撞修正；同时返回未来真值帧与目标（闭环 rollout 训练用）。
        """
        ep = self.ep_idx[idx]
        i0 = self.win_start[idx]
        L, K = self.L, self.K
        B = len(idx)
        off = np.arange(L + K)

        frm_obj = self.obj_norm[ep[:, None], i0[:, None] + off[None, :]]   # (B, L+K, 33)
        frm_ct = self.contact[ep[:, None], i0[:, None] + off[None, :]]     # (B, L+K, 3)
        obs_ep = self.obs_norm[ep]                                         # (B, 80)

        def assemble(obj_seg, ct_seg, n):
            return np.concatenate([obj_seg,
                                   np.broadcast_to(obs_ep[:, None, :], (B, n, OBS_DIMS)),
                                   ct_seg], axis=2)

        x = assemble(frm_obj[:, :L], frm_ct[:, :L], L)
        x_next = assemble(frm_obj[:, 1:L + 1], frm_ct[:, 1:L + 1], L)
        x_fut = assemble(frm_obj[:, L:L + K], frm_ct[:, L:L + K], K)
        a0 = self.act_norm[ep, i0 + L]
        af = self.act_norm[ep[:, None], i0[:, None] + L + np.arange(K)[None, :]]
        y_raw = frm_obj[:, L] * (self.std_obj + 1e-8) + self.mean_obj
        s_last = frm_obj[:, L - 1] * (self.std_obj + 1e-8) + self.mean_obj
        # 物理先验（原始单位）→ 残差目标与先验的归一化形式
        # （SLOT-1b：档位由 --prior_spec 决定；默认 parabola005 = layout.physics_prior 逐位兼容）
        prior_raw = physics_prior_spec_v3.prior_raw_np(
            self.prior_spec, s_last, a0 * IMPULSE_MAX_V3)
        d_norm = (y_raw - prior_raw) / (self.std_obj + 1e-8)
        prior_norm = (prior_raw - self.mean_obj) / (self.std_obj + 1e-8)
        y_fut_norm = frm_obj[:, L:L + K]
        yao_t = self.yaos[ep, i0 + L].reshape(B, 9)
        ct_t = self.contact[ep, i0 + L]
        # M2b：最后输入帧的窗口级卦索引（三爻均值 0.5 阈值 → 3-bit → 0..7）
        yao_last = self.yaos[ep, i0 + L - 1]                    # (B, 3轴, 3物体)
        gua_bits = (yao_last.mean(axis=2) > 0.5).astype(np.int64)
        gua_idx = gua_bits[:, 0] * 4 + gua_bits[:, 1] * 2 + gua_bits[:, 2]
        # G2-v2：输入窗口内各转移动作（创新量特征用）
        a_win = self.act_norm[ep[:, None], i0[:, None] + 1 + np.arange(L - 1)[None, :]]

        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return dict(x=to(x), x_next=to(x_next), a0=to(a0), af=to(af),
                    d_norm=to(d_norm), prior_norm=to(prior_norm),
                    y_raw=to(y_raw), y_fut_norm=to(y_fut_norm), x_fut=to(x_fut),
                    yao=to(yao_t), s_last=to(s_last), ct=to(ct_t),
                    a_win=to(a_win),
                    w_ff=to(self.win_ff[ep, i0]),
                    gua=torch.from_numpy(np.ascontiguousarray(gua_idx, dtype=np.int64)))


# ============================================================
# 评估
# ============================================================

def pos_rmse(pred_raw, target_raw):
    err = (pred_raw[:, POS_DIMS_OBJ] - target_raw[:, POS_DIMS_OBJ]) ** 2
    return float(np.sqrt(err.mean()))


def inertial_pred(s_last_raw, k=0):
    """含重力抛物线外推（τ=(k+1)·DT）。s_last_raw: (M, 33) 原始单位。"""
    tau = (k + 1) * DT_V3
    out = s_last_raw.copy()
    for o in range(NUM_OBJ):
        b = o * PER_OBJ
        out[:, b + LF["x"]] = s_last_raw[:, b + LF["x"]] + s_last_raw[:, b + LF["vx"]] * tau
        out[:, b + LF["y"]] = s_last_raw[:, b + LF["y"]] + s_last_raw[:, b + LF["vy"]] * tau \
            + 0.5 * GRAVITY_Y * tau * tau
        out[:, b + LF["vx"]] = s_last_raw[:, b + LF["vx"]]
        out[:, b + LF["vy"]] = s_last_raw[:, b + LF["vy"]] + GRAVITY_Y * tau
    return out


@torch.no_grad()
def validate(model, ds, device, K, max_windows):
    """验证：one-step / rollout 位置 RMSE（与基线同台，按接触 regime 分层）+ 爻准确率。"""
    model.eval()
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))

    M = len(idx_all)
    bs = 2048
    pred_one_raw = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    pred_roll_raw = [np.zeros((M, OBJ_DIMS), dtype=np.float32) for _ in range(K)]
    yao_hits = np.zeros(3)
    yao_cnt = np.zeros(3)
    assign_sum = torch.zeros(model.num_prototypes, device=device)
    # M2b 增量：卦级诊断（物体级三爻位模式 0..7，idx=4初+2中+上）
    tri_exact = 0
    tri_total = 0
    gua_err_sq = np.zeros(8)
    gua_n = np.zeros(8, dtype=np.int64)
    gua_dist = np.zeros(8, dtype=np.int64)
    GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]
    # G2 校准指标
    cov1 = 0.0
    cov1_n = 0
    sigma_pos_sum = 0.0

    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb, xb2 = b["x"].to(device), b["x_next"].to(device)
        ab, afb = b["a0"].to(device), b["af"].to(device)
        obb = xb[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS].contiguous()

        h, c = model.encode_sequence(xb)
        fine, _ = model.evidence(c)
        assign_sum += torch.bincount(fine.argmax(-1), minlength=model.num_prototypes).float()

        delta = model.predict_state(h, ab, b["gua"].to(device)).float().cpu().numpy()
        prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()
        pred_one_raw[sl] = (prior_n + delta) * (ds.std_obj + 1e-8) + ds.mean_obj

        # G2 校准：±1σ 覆盖率（位置维，归一化空间）与平均 σ
        if hasattr(model, "sigma_head"):
            innov_v = model.innovation_feats(xb, b["a_win"].to(device)) if model.use_innov else None
            logvar = model.predict_sigma(h, ab, innov_v, b["gua"].to(device)).float().cpu().numpy()
            sig = np.exp(0.5 * np.clip(logvar, -8.0, 8.0))
            err_norm = np.abs(b["d_norm"].numpy() - delta)
            cov1 += float((err_norm[:, POS_DIMS_OBJ] <= sig[:, POS_DIMS_OBJ]).sum())
            cov1_n += int(err_norm[:, POS_DIMS_OBJ].size)
            sigma_pos_sum += float(sig[:, POS_DIMS_OBJ].mean()) * len(idx)

        yao_log = model.predict_yao(h)
        yao_pred = (torch.sigmoid(yao_log.float()) > 0.5).cpu().numpy()
        yao_t = b["yao"].numpy()
        for axis in range(3):
            yao_hits[axis] += (yao_pred[:, axis * 3:(axis + 1) * 3] ==
                               yao_t[:, axis * 3:(axis + 1) * 3]).mean() * len(idx)
            yao_cnt[axis] += len(idx)

        # M2b 增量：卦级诊断
        tri_true = (yao_t[:, 0:3] * 4 + yao_t[:, 3:6] * 2 + yao_t[:, 6:9]).astype(np.int64)
        yp64 = yao_pred.astype(np.int64)
        tri_pred = (yp64[:, 0:3] * 4 + yp64[:, 3:6] * 2 + yp64[:, 6:9])
        tri_exact += int((tri_pred == tri_true).sum())
        tri_total += int(tri_true.size)
        y_raw_b = b["y_raw"].numpy()
        for o in range(NUM_OBJ):
            b_ = o * PER_OBJ
            e = ((pred_one_raw[sl][:, b_ + 1:b_ + 3] - y_raw_b[:, b_ + 1:b_ + 3]) ** 2).mean(axis=1)
            for g in range(8):
                m = tri_true[:, o] == g
                if m.any():
                    gua_err_sq[g] += float(e[m].sum())
                    gua_n[g] += int(m.sum())
        gua_dist += np.bincount(b["gua"].numpy(), minlength=8)

        # 闭环 rollout（纯闭环，无真值反馈；先验+残差直接给绝对状态）
        abs_preds, _ = model.rollout(xb, afb, obb, gua0=b["gua"].to(device))
        for k in range(K):
            pred_roll_raw[k][sl] = (abs_preds[k].float().cpu().numpy()
                                    * (ds.std_obj + 1e-8) + ds.mean_obj)

    ct_np = ds.contact[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L].astype(bool)
    contact_mask = ct_np.any(axis=1)
    air_mask = ~contact_mask
    # y_raw 需要重算（gather 之外）
    frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
    tgt_raw = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj
    s_last_raw = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1] \
        * (ds.std_obj + 1e-8) + ds.mean_obj
    # 动爻分层：最后输入帧与目标帧的卦象是否变化（相态转移窗口）
    yao_last = ds.yaos[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1].mean(axis=2)
    yao_tgt = ds.yaos[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L].mean(axis=2)
    dongyi_mask = ((yao_last > 0.5) != (yao_tgt > 0.5)).any(axis=1)

    def strat(pred_raw, tgt):
        out = {"all": pos_rmse(pred_raw, tgt)}
        if contact_mask.any():
            out["接触"] = pos_rmse(pred_raw[contact_mask], tgt[contact_mask])
        if air_mask.any():
            out["腾空"] = pos_rmse(pred_raw[air_mask], tgt[air_mask])
        if dongyi_mask.any():
            out["动爻"] = pos_rmse(pred_raw[dongyi_mask], tgt[dongyi_mask])
        if (~dongyi_mask).any():
            out["静爻"] = pos_rmse(pred_raw[~dongyi_mask], tgt[~dongyi_mask])
        return out

    # [bug 修复 2026-08-29] rollout 第 k 步必须与第 k 个未来帧（i+L+k）比对。
    # 惰性重写版曾把所有 k 的预测都对 k=1 目标帧比——制造了"rollout 退化"伪影。
    frm_fut = [ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L + k]
               * (ds.std_obj + 1e-8) + ds.mean_obj for k in range(K)]

    p_proto = assign_sum / assign_sum.sum().clamp(min=1)
    h_proto = -(p_proto.clamp(min=1e-10) * p_proto.clamp(min=1e-10).log()).sum().item()

    metrics = {
        "one_step": {
            "模型": strat(pred_one_raw, tgt_raw),
            "零速度基线": strat(s_last_raw, tgt_raw),
            "惯性外推(含重力)": strat(inertial_pred(s_last_raw), tgt_raw),
        },
        "rollout": {f"k={k + 1}": {
            "模型": strat(pred_roll_raw[k], frm_fut[k]),
            "零速度基线": strat(s_last_raw, frm_fut[k]),
            "惯性外推(含重力)": strat(inertial_pred(s_last_raw, k), frm_fut[k]),
        } for k in range(K)},
        "yao_acc": {name: float(h / max(c, 1)) for name, h, c in
                    zip(["初爻接触", "中爻能量", "上爻趋离"], yao_hits, yao_cnt)},
        "trigram_exact_acc": float(tri_exact / max(tri_total, 1)),
        "gua_pos_rmse": [float(math.sqrt(gua_err_sq[g] / gua_n[g])) if gua_n[g] > 0 else None
                         for g in range(8)],
        "gua_distribution": (gua_dist / max(gua_dist.sum(), 1)).tolist(),
        "gua_coverage": int((gua_dist > 0).sum()),
        "sigma_cov1": float(cov1 / max(cov1_n, 1)),
        "sigma_pos_mean": float(sigma_pos_sum / max(M, 1)),
        "proto_ppl": float(math.exp(min(h_proto, 20.0))),
    }
    model.train()
    return metrics


def print_val(metrics):
    os_rows = metrics["one_step"]
    print(f"  [val] one-step 位置RMSE(px): " +
          " | ".join(f"{k} 全体{v['all']:.3f}" for k, v in os_rows.items()))
    if "接触" in os_rows["模型"]:
        print(f"        分层: " + " | ".join(
            f"{k} 接触{v.get('接触', float('nan')):.3f}/腾空{v.get('腾空', float('nan')):.3f}"
            f"/动爻{v.get('动爻', float('nan')):.3f}/静爻{v.get('静爻', float('nan')):.3f}"
            for k, v in os_rows.items()))
    for k, row in metrics["rollout"].items():
        print(f"  [val] rollout {k}: 模型 {row['模型']['all']:.3f} | "
              f"零速度 {row['零速度基线']['all']:.3f} | 惯性 {row['惯性外推(含重力)']['all']:.3f}")
    print("  [val] 爻准确率: " + " | ".join(f"{k} {v:.3f}" for k, v in metrics["yao_acc"].items())
          + f" | 原型PPL {metrics['proto_ppl']:.1f}")
    if "sigma_cov1" in metrics and metrics["sigma_cov1"] > 0:
        print(f"  [val] G2 校准: ±1σ 覆盖率 {metrics['sigma_cov1']:.3f}（理想≈0.68）"
              f" | 平均σ(位置,归一化) {metrics['sigma_pos_mean']:.4f}")
    gr = metrics.get("gua_pos_rmse")
    if gr:
        GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]
        line = " | ".join(f"{GUA_NAMES[g]} {gr[g]:.2f}" if gr[g] is not None
                          else f"{GUA_NAMES[g]} -" for g in range(8))
        print(f"  [val] 卦级位置RMSE(px): {line}")
        print(f"  [val] 卦精确匹配率(物体级三爻全对): {metrics['trigram_exact_acc']:.3f} "
              f"| 卦覆盖 {metrics['gua_coverage']}/8 "
              f"| 分布 {['%.2f' % p for p in metrics['gua_distribution']]}")


# ============================================================
# 主流程
# ============================================================

def rollout_ff_weight(w_ff_roll, win_ff):
    """SLOT-1e：闭环 rollout 损失的逐样本 regime 权重。

    w_ff_roll == 1.0 → None（调用方不加权 ⇒ 与历史行为逐位一致）；
    否则 1 + (w_ff_roll−1)·win_ff（win_ff=1 的自由飞行窗口上权 w_ff_roll）。
    """
    if w_ff_roll == 1.0:
        return None
    return 1.0 + (w_ff_roll - 1.0) * win_ff


def ct_roll_loss(yao_preds, x_fut, pos_weight=1.0):
    """SLOT-1f：闭环各步接触位的 BCE（把 regime 指示器训在它被使用的步分布上）。

    yao_preds: rollout 返回的 K 个 (B, 9) 三爻 logits（前 3 位 = 接触位）；
    x_fut:     (B, K, input_dim) 未来真值帧，接触位在最后 3 维（V3Data 组装约定）；
    pos_weight:'接触' 正类权重（单元类比约 60:40，<1 可降低接触偏置）。
    """
    terms = []
    for k, lg in enumerate(yao_preds):
        terms.append(F.binary_cross_entropy_with_logits(
            lg[:, 0:3].float(), x_fut[:, k, -3:].float(),
            pos_weight=torch.as_tensor([pos_weight], device=lg.device)))
    return torch.stack(terms).mean()


def main():
    args = parse_args()
    if args.gate_supervise and not args.gate:
        raise SystemExit("--gate_supervise 需要 --gate（门监督作用于 GatedWorldModelV3）")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    L, K = args.seq_len, args.future_steps
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print("=" * 68)
    print("观象世界模型 v3 训练（减法管线）")
    print(f"  数据: {args.data_path} | 设备: {device} | L={L} K={K} | AMP={use_amp}")
    print("=" * 68)

    states, obstacles, contact, actions = load_dataset(args.data_path)
    N = states.shape[0]
    train_eps, val_eps = split_episodes(N, args.seed)
    print(f"  片段: 训练 {len(train_eps)} / 验证 {len(val_eps)}")
    mean_obj, std_obj, mean_obs, std_obs = norm_stats(states, obstacles, train_eps)

    train_ds = V3Data(states, obstacles, contact, actions, train_eps, L, K,
                      mean_obj, std_obj, mean_obs, std_obs, prior_spec=args.prior_spec)
    val_ds = V3Data(states, obstacles, contact, actions, val_eps, L, K,
                    mean_obj, std_obj, mean_obs, std_obs, prior_spec=args.prior_spec)
    print(f"  窗口: 训练 {len(train_ds):,} / 验证 {len(val_ds):,}"
          f" | prior_spec={args.prior_spec}")

    physics_prior_spec_v3.assert_dir_not_foreign_v3(
        args.checkpoint_dir, args.prior_spec, args.seed)

    model_cls = GatedWorldModelV3 if args.gate else WorldModelV3
    model = model_cls(input_dim=ASSEMBLED_DIM, obj_dim=OBJ_DIMS,
                      action_dim=ACTION_DIM, hidden_dim=args.hidden_dim,
                      num_prototypes=args.num_prototypes,
                      num_coarse=args.num_coarse,
                      tau=DT_V3, gravity=-900.0, impulse_max=IMPULSE_MAX_V3,
                      use_gua=args.use_gua, gua_emb_dim=args.gua_emb_dim,
                      prior_spec=args.prior_spec).to(device)
    model.set_stats(mean_obj, std_obj)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = V3Loss(args.w_state, args.w_yao, args.w_proto,
                       args.w_commit, args.w_usage, w_cal=args.w_cal)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fw_np = feat_weights()[:OBJ_DIMS].copy()
    fw_np = fw_np / fw_np.mean()
    feat_w_t = torch.from_numpy(fw_np).to(device)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.log_dir)
    except ImportError:
        writer = None

    best_rmse = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    n_train = len(train_ds)
    wpe = args.windows_per_epoch if args.windows_per_epoch > 0 else n_train

    import time
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        agg, nb, t_data, t_compute = {}, 0, 0.0, 0.0
        t_epoch = time.time()
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            t0 = time.time()
            batch = train_ds.gather(idx)
            xb = batch["x"].to(device, non_blocking=True)
            xb2 = batch["x_next"].to(device, non_blocking=True)
            ab = batch["a0"].to(device, non_blocking=True)
            afb = batch["af"].to(device, non_blocking=True)
            db = batch["d_norm"].to(device, non_blocking=True)
            yaob = batch["yao"].to(device, non_blocking=True)
            guab = batch["gua"].to(device, non_blocking=True)
            awin = batch["a_win"].to(device, non_blocking=True)
            # G1 观测噪声增强：只污染输入帧（位置/速度/障碍通道高斯；接触通道翻转），
            # 目标保持干净——模型学会"噪观测 → 干净动力学"，σ 头学会从历史不自洽推断不确定度
            if args.noise_aug > 0:
                sigma = args.noise_aug * torch.rand(xb.size(0), 1, 1, device=device)
                xb = torch.cat([xb[..., :-3] + torch.randn_like(xb[..., :-3]) * sigma,
                                torch.abs(xb[..., -3:] -
                                          (torch.rand(xb.size(0), xb.size(1), 3, device=device)
                                           < sigma * 2.0).float())], dim=-1)
            t_data += time.time() - t0

            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                h, c = model.encode_sequence(xb)
                fine, coarse = model.evidence(c)
                _, c_next = model.encode_sequence(xb2)
                fine_next, _ = model.evidence(c_next)
                innov = model.innovation_feats(xb, awin) if model.use_innov else None
                sw = None
                if args.w_ff != 1.0:
                    sw = 1.0 + (args.w_ff - 1.0) * batch["w_ff"].to(device)
                loss, parts, _ = criterion(model, h, c, fine, coarse, ab, db, yaob,
                                           fine_next, feat_w_t, gua=guab, innov=innov,
                                           sample_w=sw)
                if args.gate_supervise:
                    # SLOT-1d：门监督——win_ff=1(FF)→门关(g 目标 0)；win_ff=0(接触)→门开
                    g_logits = model.gate_logits(h, ab)
                    g_tgt = 1.0 - batch["w_ff"].to(device)
                    gate_bce = F.binary_cross_entropy_with_logits(
                        g_logits, g_tgt)
                    loss = loss + args.w_gate_bce * gate_bce
                    parts["gate_bce"] = gate_bce.item()
                # 闭环 rollout 损失（scheduled sampling：真值帧按概率替换自身预测）
                if args.w_roll > 0:
                    obb = xb[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS].contiguous()
                    xf = batch["x_fut"].to(device, non_blocking=True)
                    yf = batch["y_fut_norm"].to(device, non_blocking=True)
                    ss_prob = min(args.ss_max, 0.05 * (epoch + 1))
                    abs_preds, yao_preds_roll = model.rollout(
                        xb, afb, obb, x_fut_gt=xf, ss_prob=ss_prob, gua0=guab)
                    # SLOT-1e：regime 重加权延伸到闭环项（默认 1.0 = 历史行为逐位不变）
                    sw_r = rollout_ff_weight(args.w_ff_roll, batch["w_ff"].to(device))
                    roll_terms = []
                    for k in range(K):
                        hub = F.huber_loss(abs_preds[k].float(), yf[:, k], reduction="none")
                        per = (hub * feat_w_t.view(1, -1)).mean(dim=1)
                        if sw_r is not None:
                            per = per * sw_r
                        roll_terms.append((0.7 ** k) * per.mean())
                    roll_loss = torch.stack(roll_terms).sum()
                    loss = loss + args.w_roll * roll_loss
                    parts = dict(parts, roll=roll_loss.item())
                    # SLOT-1f：闭环步分布上的接触位监督（默认 0 = 历史行为不变）
                    if args.w_ct_roll > 0:
                        ct_loss = ct_roll_loss(yao_preds_roll, xf,
                                               pos_weight=args.ct_pos_weight)
                        loss = loss + args.w_ct_roll * ct_loss
                        parts["ct_roll"] = ct_loss.item()
            if not torch.isfinite(loss):
                print(f"    [警告] batch {j} 损失非有限，跳过")
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.ema_prototypes(c.detach().float(), fine.argmax(-1).detach(),
                                 coarse.argmax(-1).detach())
            t_compute += time.time() - t0
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        sched.step()
        dt_ep = time.time() - t_epoch
        msg = " ".join(f"{k}={v / max(nb, 1):.4f}" for k, v in agg.items())
        print(f"Epoch {epoch + 1}/{args.epochs} [{msg}] "
              f"({dt_ep:.0f}s, 数据 {t_data:.0f}s + 计算 {t_compute:.0f}s)")
        if writer:
            for k, v in agg.items():
                writer.add_scalar(f"train/{k}", v / max(nb, 1), epoch)

        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            metrics = validate(model, val_ds, device, K, args.val_windows)
            print_val(metrics)
            rmse = metrics["one_step"]["模型"]["all"]
            if writer:
                writer.add_scalar("val/pos_rmse", rmse, epoch)
                writer.add_scalar("val/yao_contact_acc", metrics["yao_acc"]["初爻接触"], epoch)
            ckpt = dict(model_state_dict=model.state_dict(),
                        config=dict(input_dim=ASSEMBLED_DIM, obj_dim=OBJ_DIMS,
                                    action_dim=ACTION_DIM, hidden_dim=args.hidden_dim,
                                    num_prototypes=args.num_prototypes,
                                    num_coarse=args.num_coarse,
                                    tau=DT_V3, gravity=-900.0,
                                    impulse_max=IMPULSE_MAX_V3,
                                    use_gua=args.use_gua,
                                    gua_emb_dim=args.gua_emb_dim,
                                    prior_spec=args.prior_spec,
                                    gate=args.gate),
                        norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj,
                                        mean_obs=mean_obs, std_obs=std_obs),
                        args=vars(args), epoch=epoch, val_metrics=metrics)
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
                print(f"  [保存最佳] val one-step 位置RMSE = {rmse:.3f}px")

    if writer:
        writer.close()
    print(f"\n[OK] 训练完成，最佳验证位置 RMSE: {best_rmse:.3f}px")
    print(f"     checkpoints: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
