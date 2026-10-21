# -*- coding: utf-8 -*-
"""
E0 公共管线 (e0_common.py) — 数据 + 评估，训练与评估共享同一实现
==================================================
避免训练/评估协议分歧（M1e 勘误教训：评估协议必须与训练期验证完全一致）。
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, DT as DT_H,
                      feat_weights as feat_weights_h)
import model_e0
from physics_prior_spec import DEFAULT_SPEC, prior_raw_np

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS   # 25
POS_IDX = [1, 2, 12, 13]
RESIDUAL_ARMS = {"ours_base", "ours_gua", "ours_yao", "ours_sigma"}


# ---------------------------------------------------------------- 数据
class E0Data:
    """惰性窗口数据源。assembly = [obj(22) | contact(3)] = 25 维。
    目标 = 绝对状态 − 先验（残差）；noprior/strong_mlp 的损失另取绝对目标。

    prior_spec：先验积分器档位（physics_prior_spec.SPECS）。**训练期构造残差目标
    用的先验必须与评估期 model.prior_norm 用的是同一个 spec**，否则口径静默错配
    （tests/test_prior_spec.py 对此有断言）。"""

    def __init__(self, states, contact, actions, yao, episodes, L, K,
                 mean_obj, std_obj, prior_spec=DEFAULT_SPEC):
        self.L, self.K = L, K
        self.prior_spec = prior_spec
        self.mean_obj, self.std_obj = mean_obj, std_obj
        episodes = np.asarray(episodes)
        T = states.shape[1]
        self.obj_norm = ((states[episodes] - mean_obj) / (std_obj + 1e-8)).astype(np.float32)
        self.act_norm = actions[episodes].astype(np.float32)
        self.contact = contact[episodes].astype(np.float32)
        self.yaos = yao[episodes]
        n_win = T - L - K + 1
        self.ep_idx = np.repeat(np.arange(len(episodes)), n_win)
        self.win_start = np.tile(np.arange(n_win), len(episodes))

    def __len__(self):
        return len(self.ep_idx)

    def gather(self, idx):
        ep = self.ep_idx[idx]
        i0 = self.win_start[idx]
        L, K = self.L, self.K
        off = np.arange(L + K)
        frm = self.obj_norm[ep[:, None], i0[:, None] + off[None, :]]
        ct = self.contact[ep[:, None], i0[:, None] + off[None, :]]

        def assemble(o, c):
            return np.concatenate([o, c], axis=2)

        x = assemble(frm[:, :L], ct[:, :L])
        x_next = assemble(frm[:, 1:L + 1], ct[:, 1:L + 1])
        x_fut = assemble(frm[:, L:L + K], ct[:, L:L + K])
        a0 = self.act_norm[ep, i0 + L]
        af = self.act_norm[ep[:, None], i0[:, None] + L + np.arange(K)[None, :]]
        prior_n = self._prior_np(frm[:, L - 1], a0)
        d_norm = frm[:, L] - prior_n                       # 残差目标
        s_last = frm[:, L - 1] * (self.std_obj + 1e-8) + self.mean_obj
        yao_t = self.yaos[ep, i0 + L].reshape(-1, 6)
        ct_t = self.contact[ep, i0 + L]

        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return dict(x=to(x), x_next=to(x_next), a0=to(a0), af=to(af),
                    d_norm=to(d_norm), yao=to(yao_t), ct=to(ct_t), x_fut=to(x_fut))

    def _prior_np(self, obj_norm_batch, a_norm):
        """残差目标用的先验——**唯一来源**：physics_prior_spec.prior_raw_np。
        （旧版此处是第三份手写半隐式欧拉实现，与 model_h/e0 的先验存在漂移风险。）"""
        raw = obj_norm_batch * (self.std_obj + 1e-8) + self.mean_obj
        out = prior_raw_np(self.prior_spec, raw, a_norm[:, 0])
        return (out - self.mean_obj) / (self.std_obj + 1e-8)


def make_datasets(data_path, L, K, seed, prior_spec=DEFAULT_SPEC):
    d = np.load(data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    rng = np.random.RandomState(seed)
    segs = np.arange(N); rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]
    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)
    kw = dict(L=L, K=K, mean_obj=mean_obj, std_obj=std_obj, prior_spec=prior_spec)
    train = E0Data(states, contact, actions, yao, tr_eps, **kw)
    val = E0Data(states, contact, actions, yao, val_eps, **kw)
    return train, val, mean_obj, std_obj


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def validate(model, ds, device, K, max_windows=30000):
    """统一评估协议（与训练期验证同一实现）：
      one-step RMSE（位置 4 维，feat_weights 加权）
      rollout k=1..K（闭环，接触反馈 = 爻头初爻软概率）
      爻准确率 / 先验单步误差 / 零速度基线
    返回 dict，另附 per-gua one-step RMSE 与 gua occupancy。"""
    model.eval()
    idx_all = np.arange(len(ds))
    if len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    bs = 2048
    pred_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    prior_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    pred_roll = [np.zeros((M, OBJ_DIMS), dtype=np.float32) for _ in range(K)]
    yao_hits = yao_n = 0
    # 卦级误差分解：累加平方误差（而非原始预测值），否则负坐标会使 sqrt 出 NaN
    gua_err2 = np.zeros((8, OBJ_DIMS), dtype=np.float64)
    gua_cnt = np.zeros(8, dtype=np.int64)
    # 先验判定改为显式声明（physics_prior 类属性）：
    # 旧判据 hasattr(model,"prior_norm") 会把 StrongMLP 的零占位 prior_norm 当成
    # 物理先验，于是把「数据集均值基线 87.9px」写进 Table 1 的「先验」列。
    has_prior = getattr(model, "physics_prior", False)

    # 目标帧在循环前算好，卦级分解需要在循环内拿到逐窗误差
    tgt_all = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L] \
        * (ds.std_obj + 1e-8) + ds.mean_obj

    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device); ab = b["a0"].to(device)
        h, c = model.encode_sequence(xb)
        if hasattr(model, "forward_state"):
            delta, _ = model.forward_state(h, ab)
        else:
            delta = (model.predict_state(h, ab) if hasattr(model, "predict_state")
                     else model.predict(h, ab))
        if has_prior:
            prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()
            pred_one[sl] = (prior_n + delta.float().cpu().numpy()) * (ds.std_obj + 1e-8) + ds.mean_obj
            prior_one[sl] = prior_n * (ds.std_obj + 1e-8) + ds.mean_obj
        else:
            pred_one[sl] = delta.float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj
        yao_log = model.predict_yao(h)
        yp = (torch.sigmoid(yao_log.float()) > 0.5).cpu().numpy()
        yt = b["yao"].numpy()
        yao_hits += (yp == yt).mean() * len(idx)
        yao_n += len(idx)

        # 卦级分解：每物体 8 卦 = 3 爻二进制编码
        # 用真值爻标签（b["yao"]）而非模型预测，否则相位误判会污染误差解剖
        yao_t = b["yao"].numpy().astype(np.int64)      # (B, 6) 真值；int64 才能做位运算
        g0 = (yao_t[:, 0] << 2) | (yao_t[:, 1] << 1) | yao_t[:, 2]
        g1 = (yao_t[:, 3] << 2) | (yao_t[:, 4] << 1) | yao_t[:, 5]
        err = pred_one[sl] - tgt_all[sl]             # (B, OBJ_DIMS) 逐窗误差
        for gi in (g0, g1):
            for g in range(8):
                m = gi == g
                if m.any():
                    gua_cnt[g] += int(m.sum())
                    gua_err2[g] += (err[m] ** 2).sum(0)

        afb = b["af"].to(device); xf = b["x_fut"].to(device)
        abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=0.0)
        for k in range(K):
            pred_roll[k][sl] = abs_preds[k].float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj

    frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
    tgt = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj
    rollout_tgt = [
        ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L + k]
        * (ds.std_obj + 1e-8) + ds.mean_obj
        for k in range(K)
    ]
    s_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1] \
        * (ds.std_obj + 1e-8) + ds.mean_obj

    def pos_rmse(p, t, weighted=True):
        err = (p[:, POS_IDX] - t[:, POS_IDX]) ** 2
        if weighted:
            w = feat_weights_h(OBJ_DIMS)[POS_IDX]
            err = err * w
        return float(np.sqrt(err.mean()))

    # 卦级 one-step RMSE（位置加权）
    gua_rmse = {}
    for g in range(8):
        if gua_cnt[g] > 0:
            mean_err2 = gua_err2[g] / gua_cnt[g]
            gua_rmse[g] = float(np.sqrt((mean_err2[POS_IDX] * feat_weights_h(OBJ_DIMS)[POS_IDX]).mean()))
        else:
            gua_rmse[g] = float("nan")
    total_gua = int(gua_cnt.sum())
    gua_occupancy = {g: float(gua_cnt[g] / total_gua) if total_gua else 0.0 for g in range(8)}

    metrics = dict(one_step_model=pos_rmse(pred_one, tgt),
                   one_step_prior=pos_rmse(prior_one, tgt) if has_prior else float("nan"),
                   one_step_zero_vel=pos_rmse(s_last, tgt),
                   rollout={f"k={k + 1}": pos_rmse(pred_roll[k], rollout_tgt[k]) for k in range(K)},
                   yao_acc=float(yao_hits / max(yao_n, 1)),
                   gua_rmse=gua_rmse, gua_occupancy=gua_occupancy,
                   n_windows=M)
    model.train()
    return metrics


def load_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arm = ck["config"]["arm"]
    model = model_e0.make_model(arm)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.set_stats(ck["norm_stats"]["mean_obj"], ck["norm_stats"]["std_obj"])
    return model.to(device), arm, ck


def pos_rmse_raw(pred, tgt, weighted=True):
    err = (pred[:, POS_IDX] - tgt[:, POS_IDX]) ** 2
    if weighted:
        err = err * feat_weights_h(OBJ_DIMS)[POS_IDX]
    return float(np.sqrt(err.mean()))
