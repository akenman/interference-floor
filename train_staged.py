# -*- coding: utf-8 -*-
"""
分阶段优化 v2 (train_staged.py) — T15 修复实验（干净重写版）
==================================================
T15 发现：物理参数与残差头同时梯度优化 → 两阶段干扰（17.8px，纯学习 0.44px）。
第一版实现的工程根因（本版逐条修掉）：
  ① Phase A 用 Adam 辨识：欧拉步对 g/M/L 高度非线性，log 空间梯度剧烈震荡
     ——5 个标量不需要梯度，无导数数值搜索更适定；
  ② Phase B 按 WorldModelH 富接口调用而 OursParamModel 是精简版——接口错位
     （富接口兼容方法已补进 OursParamModel）。
冒烟取证追加发现的两个更深根因（2026-09-02）：
  ③ OursParamModel.prior_step 从未反归一化——归一化输入被当物理单位使用
     （atan2 吃归一化 sin/cos、ω 用归一化值当 rad/s），先验输出为垃圾。
     **T15 的 ours_param 17.8px 实质运行在该坏先验上，"两阶段干扰"结论
     存疑**——本版增设 Phase D 对照臂（修复后的先验 + 联合优化）重检 T15；
  ④ 常数维（presence/type/size/mass，std≈0）归一化后量级 ~1e8-1e9，
     float32 无法往返，以 1.75e8 支配 huber——NM 目标面被淹没而平坦化。
     本版损失/目标限定在 14 个动力学维度。

本版方案：
  Phase A 物理辨识：scipy Nelder-Mead（log 空间 + bounds），目标 = 先验在固定
    8192 窗口样本上的 feat-加权 huber（确定性目标面，无梯度、无震荡）。
    默认偏移起点（诚实的系统辨识测试）；--init truth 复现 T15 起点。
  Phase B 残差训练：冻结物理，只训 GRU+残差头——与 benchmark_surpass 的
    ours_param 臂同协议（单项 state 损失 + noise_aug 0.02），
    保证与 T15 的 17.8px 严格可比（唯一差异 = 优化调度）。
  Phase C 联合微调：lr 1e-5（含物理参数），逐 5 epoch 监控物理漂移。
  Phase D 对照臂：修复后的先验 + 联合优化从偏移起点（T15 机制重检）。
  Phase E 对照臂：同预算 noprior（S3 的严格基线，消除历史参照的预算差）。

预注册判决（对照 T15 同时优化 17.82px / noprior 0.436px）：
  S1: Phase A 后（偏移起点），先验单步位置 MAE < 3.0px（系统辨识成功）
  S2: Phase B 后，one-step RMSE < 1.0px（解耦消除非平稳残差目标）
  S3: 最终 one-step RMSE < 0.436px（先验+残差 > 纯学习）
  三条全过 → 分阶段优化解决 T15，架构优势在正确训练下成立。

用法：
  python train_staged.py                        # 全量判决（偏移起点辨识）
  python train_staged.py --init truth           # T15 同起点对照
  python train_staged.py --smoke                # 管线冒烟（几分钟内跑通）
"""

import os
import math
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from layout_h import OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, feat_weights as feat_weights_h
from train_hinge import HingeData
from benchmark_surpass import OursParamModel
from core_model import NoPriorModel

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS
POS_IDX = [1, 2, 12, 13]

# 动力学维度掩码：剔除 presence/type/size/mass（std≈0，归一化量级 ~1e8-1e9，
# float32 无法往返，曾以 1.75e8 支配 huber——NM 目标面平坦化与 loss 爆炸的根因）
CONST_DIMS = [0, 8, 9, 10, 11, 19, 20, 21]
DYN_IDX = [i for i in range(OBJ_DIMS) if i not in CONST_DIMS]

# 物理真值（layout_h 常量，用于辨识精度报告——不进入任何训练路径）
TRUTH = {"g": 900.0, "M1": 2.0, "M2": 1.0, "L1": 80.0, "L2": 80.0}
# 偏移起点（±33~50%：真实"不知道物理参数"的系统辨识场景）
OFF_INIT = {"g": 1200.0, "M1": 3.0, "M2": 1.5, "L1": 110.0, "L2": 110.0}
# log 空间 bounds（与 prior_step 内部 clamp 一致）
LOG_BOUNDS = [(math.log(500), math.log(1500)),
              (math.log(0.1), math.log(10)),
              (math.log(0.1), math.log(10)),
              (math.log(20), math.log(200)),
              (math.log(20), math.log(200))]
P_KEYS = ["g", "M1", "M2", "L1", "L2"]


def parse_args():
    ap = argparse.ArgumentParser(description="分阶段优化 v2：NM 物理辨识 → 冻结残差 → 联合微调")
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--init", choices=["off", "truth"], default="off",
                    help="Phase A 起点：off=偏移初值（系统辨识判决），truth=T15 同起点")
    ap.add_argument("--epochs_b", type=int, default=60)
    ap.add_argument("--epochs_c", type=int, default=20)
    ap.add_argument("--lr_b", type=float, default=3e-4)
    ap.add_argument("--lr_c", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--future_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--windows_per_epoch", type=int, default=200000)
    ap.add_argument("--noise_aug", type=float, default=0.02,
                    help="与 benchmark_surpass ours_param 臂一致（严格可比）")
    ap.add_argument("--ident_windows", type=int, default=8192)
    ap.add_argument("--nm_maxiter", type=int, default=800)
    ap.add_argument("--eval_windows", type=int, default=5000)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints-staged")
    ap.add_argument("--output_report", type=str, default="analysis_output/staged_report.json")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：全部预算缩到最小")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def eval_onestep(model, ds, device, max_windows, with_residual=True, use_prior=True):
    """单步位置 RMSE(px)，与 benchmark_surpass.eval_model 同协议。"""
    model.eval()
    idx_all = np.arange(len(ds))
    if len(idx_all) > max_windows:
        idx_all = np.sort(np.random.RandomState(0).choice(idx_all, max_windows, replace=False))
    errs = []
    for j in range(0, len(idx_all), 2048):
        b = ds.gather(idx_all[j:j + 2048])
        xb = b["x"].to(device)
        ab = b["a0"].to(device)
        if use_prior:
            pred_n = model.prior_step(xb[:, -1, :OBJ_DIMS], ab)
            if with_residual:
                h = model.encode(xb)
                pred_n = pred_n + model.predict(h, ab)
        else:
            h = model.encode(xb)
            pred_n = model.predict(h, ab)
        tgt = b["x_fut"][:, 0, :OBJ_DIMS].numpy()
        e = np.abs(pred_n.float().cpu().numpy()[:, POS_IDX] - tgt[:, POS_IDX]).mean(axis=1)
        errs.append(e)
    pos_std = float(np.mean(ds.std_obj[POS_IDX]))
    return float(np.sqrt((np.concatenate(errs) ** 2).mean())) * pos_std


def phase_a_identify(model, ds, args, device, loss_w, x0_phys):
    """Nelder-Mead 物理辨识（无梯度）。返回 (res, 辨识参数 dict)。"""
    rng = np.random.RandomState(123)   # 固定辨识样本 → 确定性目标面
    id_idx = np.sort(rng.choice(len(ds), size=min(args.ident_windows, len(ds)), replace=False))
    batches = []
    for j in range(0, len(id_idx), 2048):
        b = ds.gather(id_idx[j:j + 2048])
        batches.append((b["x"][:, -1, :OBJ_DIMS].to(device),
                        b["a0"].to(device),
                        b["x_fut"][:, 0, :OBJ_DIMS].to(device)))

    n_eval = [0]

    def objective(log_x):
        model.set_physics(*(float(v) for v in np.exp(log_x)))
        with torch.no_grad():
            tot, n = 0.0, 0
            for xn, an, tn in batches:
                prior_n = model.prior_step(xn, an)
                loss = (F.huber_loss(prior_n[:, DYN_IDX], tn[:, DYN_IDX],
                                     reduction="none") * loss_w.view(1, -1)).mean()
                tot += loss.item() * len(tn)
                n += len(tn)
        n_eval[0] += 1
        return tot / max(n, 1)

    x0 = [math.log(x0_phys[k]) for k in P_KEYS]
    res = minimize(objective, x0, method="Nelder-Mead", bounds=LOG_BOUNDS,
                   options=dict(maxiter=args.nm_maxiter, xatol=5e-4, fatol=1e-8))
    params = {k: float(v) for k, v in zip(P_KEYS, np.exp(res.x))}
    model.set_physics(**params)
    return res, params, n_eval[0]


def train_residual(model, train_ds, args, device, loss_w, epochs, lr,
                   train_physics, use_prior=True):
    """Phase B/C/D/E 共用训练循环（差异：物理是否解冻 + 是否使用先验）。"""
    physics = ([model.log_g, model.log_M1, model.log_M2, model.log_L1, model.log_L2]
               if hasattr(model, "log_g") else [])
    for p in model.parameters():
        p.requires_grad = True
    if not train_physics:
        for p in physics:
            p.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    n_train = len(train_ds)
    wpe = min(args.windows_per_epoch, n_train)

    import time
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        agg, nb, t0 = {}, 0, time.time()
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            b = train_ds.gather(idx)
            xb = b["x"].to(device, non_blocking=True)
            ab = b["a0"].to(device, non_blocking=True)
            if args.noise_aug > 0:
                s = args.noise_aug * torch.rand(xb.size(0), 1, 1, device=device)
                xb = xb + torch.randn_like(xb) * s
            tgt = b["x_fut"][:, 0, :OBJ_DIMS].to(device, non_blocking=True)

            h = model.encode(xb)
            if use_prior:
                prior_n = model.prior_step(xb[:, -1, :OBJ_DIMS], ab)
                pred = prior_n + model.predict(h, ab)
            else:
                pred = model.predict(h, ab)
            loss = (F.huber_loss(pred[:, DYN_IDX], tgt[:, DYN_IDX], reduction="none")
                    * loss_w.view(1, -1)).mean()
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if any(not torch.isfinite(p.grad.sum())
                   for p in model.parameters() if p.grad is not None):
                optimizer.zero_grad()
                continue
            optimizer.step()
            agg["state"] = agg.get("state", 0.0) + loss.item()
            nb += 1
        sched.step()
        msg = " ".join(f"{k}={v / max(nb, 1):.4f}" for k, v in agg.items())
        if train_physics:
            phys = model.get_params()
            phys_s = (f"g={phys['g']:.0f} M1={phys['M1']:.2f} M2={phys['M2']:.2f} "
                      f"L1={phys['L1']:.1f} L2={phys['L2']:.1f}")
        elif hasattr(model, "log_g"):
            phys_s = "（物理冻结）"
        else:
            phys_s = "（无先验）"
        print(f"  epoch {epoch + 1}/{epochs} [{msg}] {phys_s} ({time.time() - t0:.0f}s)")


def main():
    args = parse_args()
    if args.smoke:
        args.epochs_b, args.epochs_c = 2, 1
        args.windows_per_epoch = 4096
        args.ident_windows = 2048
        args.nm_maxiter = 40
        args.eval_windows = 2000
    device = torch.device(args.device)
    L, K = args.seq_len, args.future_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 68)
    print("分阶段优化 v2：NM 物理辨识 → 冻结残差 → 联合微调（T15 修复判决）")
    print(f"  数据: {args.data_path} | 设备: {device} | Phase A 起点: {args.init}")
    print("=" * 68)

    d = np.load(args.data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    rng = np.random.RandomState(args.seed)
    segs = np.arange(N)
    rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]
    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)

    train_ds = HingeData(states, contact, actions, yao, tr_eps, L, K, mean_obj, std_obj)
    val_ds = HingeData(states, contact, actions, yao, val_eps, L, K, mean_obj, std_obj)
    print(f"  数据: 训练 {len(tr_eps)} / 验证 {len(val_eps)} episodes | "
          f"窗口 {len(train_ds):,} / {len(val_ds):,}")

    fw = feat_weights_h(OBJ_DIMS)[DYN_IDX]
    fw = (fw / fw.mean()).astype(np.float32)
    loss_w = torch.from_numpy(fw).to(device)

    model = OursParamModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim).to(device)
    model.set_stats(mean_obj, std_obj)

    report = {"init": args.init, "smoke": args.smoke, "seed": args.seed,
              "truth": TRUTH, "noprior_ref": 0.436, "t15_joint_opt": 17.82}

    # ================================================================
    # Phase A: 物理辨识（Nelder-Mead，无梯度）
    # ================================================================
    x0_phys = dict(TRUTH) if args.init == "truth" else dict(OFF_INIT)
    print(f"\n[Phase A] Nelder-Mead 物理辨识（起点 g={x0_phys['g']:.0f} "
          f"M1={x0_phys['M1']:.1f} M2={x0_phys['M2']:.1f} L1={x0_phys['L1']:.0f} "
          f"L2={x0_phys['L2']:.0f}）")
    res, params, n_eval = phase_a_identify(model, train_ds, args, device, loss_w, x0_phys)
    delta = {k: (params[k] - TRUTH[k]) / TRUTH[k] * 100 for k in P_KEYS}
    print(f"  NM 收敛: {res.message} | 目标评估 {n_eval} 次 (nit={res.nit})")
    print(f"  辨识结果: " + " ".join(f"{k}={params[k]:.2f}" for k in P_KEYS))
    print(f"  相对真值偏差: " + " ".join(f"{k}:{delta[k]:+.1f}%" for k in P_KEYS))
    prior_mae = eval_onestep(model, val_ds, device, args.eval_windows, with_residual=False)
    s1_pass = prior_mae < 3.0
    print(f"  [Phase A 结果] 先验单步位置 MAE = {prior_mae:.3f}px"
          f"（S1: <3.0 → {'[OK]' if s1_pass else '[FAIL]'}）")
    report["phase_a"] = {"params": params, "delta_pct": delta,
                         "prior_mae_px": prior_mae, "n_eval": n_eval,
                         "nit": int(res.nit), "s1": bool(s1_pass)}

    # ================================================================
    # Phase B: 残差训练（冻结物理）
    # ================================================================
    print(f"\n[Phase B] 残差训练（冻结物理，{args.epochs_b} epochs，"
          f"协议 = benchmark_surpass ours_param 臂）")
    train_residual(model, train_ds, args, device, loss_w,
                   args.epochs_b, args.lr_b, train_physics=False)
    rmse_b = eval_onestep(model, val_ds, device, args.eval_windows, with_residual=True)
    s2_pass = rmse_b < 1.0
    print(f"  [Phase B 结果] one-step RMSE = {rmse_b:.3f}px"
          f"（S2: <1.0 → {'[OK]' if s2_pass else '[FAIL]'}；T15 同时优化 = 17.82px）")
    report["phase_b"] = {"rmse_px": rmse_b, "s2": bool(s2_pass)}

    # ================================================================
    # Phase C: 联合微调（低学习率，含物理）
    # ================================================================
    print(f"\n[Phase C] 联合微调（lr={args.lr_c}，{args.epochs_c} epochs，物理解冻）")
    train_residual(model, train_ds, args, device, loss_w,
                   args.epochs_c, args.lr_c, train_physics=True)
    rmse_c = eval_onestep(model, val_ds, device, args.eval_windows, with_residual=True)
    s3_pass = rmse_c < report["noprior_ref"]
    print(f"  [Phase C 结果] one-step RMSE = {rmse_c:.3f}px"
          f"（S3: <{report['noprior_ref']} → {'[OK]' if s3_pass else '[FAIL]'}；noprior 基线 0.436px）")
    report["phase_c"] = {"rmse_px": rmse_c, "s3": bool(s3_pass),
                         "physics_final": model.get_params()}

    # ================================================================
    # Phase D: 对照臂——修复后的先验 + 联合优化（T15 重检）
    # ================================================================
    print(f"\n[Phase D] 对照臂：联合优化从偏移起点（修复后的先验，T15 机制重检，"
          f"{args.epochs_b} epochs，lr={args.lr_b}）")
    model_d = OursParamModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim).to(device)
    model_d.set_stats(mean_obj, std_obj)
    model_d.set_physics(**OFF_INIT)
    train_residual(model_d, train_ds, args, device, loss_w,
                   args.epochs_b, args.lr_b, train_physics=True)
    rmse_d = eval_onestep(model_d, val_ds, device, args.eval_windows, with_residual=True)
    print(f"  [Phase D 结果] one-step RMSE = {rmse_d:.3f}px | "
          f"物理: {model_d.get_params()}")
    d_joint_artifact = (rmse_d < 5.0)   # 联合优化在修复后不再崩 → T15 系伪影
    print(f"  [Phase D 判读] 联合优化{'未崩（T15 的劣化系坏先验伪影）' if d_joint_artifact else '仍显著劣化（两阶段干扰机制真实存在）'}")
    report["phase_d_joint_control"] = {"rmse_px": rmse_d,
                                       "physics_final": model_d.get_params(),
                                       "t15_was_artifact": bool(d_joint_artifact)}

    # ================================================================
    # Phase E: 同预算 noprior 对照（S3 的严格基线）
    # ================================================================
    print(f"\n[Phase E] 同预算 noprior 对照（同数据/损失掩码/{args.epochs_b} epochs/"
          f"噪声 0.02，无先验）")
    model_e = NoPriorModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim).to(device)
    train_residual(model_e, train_ds, args, device, loss_w,
                   args.epochs_b, args.lr_b, train_physics=False, use_prior=False)
    rmse_e = eval_onestep(model_e, val_ds, device, args.eval_windows, with_residual=True,
                          use_prior=False)
    print(f"  [Phase E 结果] one-step RMSE = {rmse_e:.3f}px"
          f"（同预算 noprior；历史参照：surpass@1000ep 0.436 / arms@30ep 0.405）")
    report["phase_e_noprior_matched"] = {"rmse_px": rmse_e}

    # ================================================================
    # 保存与汇总
    # ================================================================
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(dict(model_state_dict=model.state_dict(),
                    config=dict(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS,
                                action_dim=ACTION_DIM, hidden_dim=args.hidden_dim),
                    norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj),
                    physics_params=model.get_params(),
                    args=vars(args), val_metrics=dict(
                        prior_mae_a=prior_mae, rmse_b=rmse_b, rmse_c=rmse_c)),
               os.path.join(args.checkpoint_dir, "best.pt"))
    report["criteria"] = {"S1": bool(s1_pass), "S2": bool(s2_pass), "S3": bool(s3_pass)}
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print("分阶段优化 vs 同时优化（T15 对照）")
    print("=" * 68)
    print(f"  T15 同时优化:   17.82px（先验+残差一起学）")
    print(f"  Phase A 先验:   {prior_mae:.3f}px（NM 辨识后，起点 {args.init}）")
    print(f"  Phase B 残差:   {rmse_b:.3f}px（冻结物理后）")
    print(f"  Phase C 联合:   {rmse_c:.3f}px（微调后）")
    print(f"  Phase D 对照:   {rmse_d:.3f}px（修复后先验 + 联合优化，偏移起点）")
    print(f"  Phase E 对照:   {rmse_e:.3f}px（同预算 noprior，严格基线）")
    print(f"  NoPrior 历史参照: 0.436px（surpass@1000ep）/ 0.405px（arms@30ep）")
    print(f"\n  S1 (辨识 <3px):     {'[OK]' if s1_pass else '[FAIL]'} ({prior_mae:.3f})")
    print(f"  S2 (残差 <1px):     {'[OK]' if s2_pass else '[FAIL]'} ({rmse_b:.3f})")
    print(f"  S3 (超越纯学习):    {'[OK]' if s3_pass else '[FAIL]'} "
          f"({rmse_c:.3f} vs 预注册 0.436 | vs 同预算 noprior {rmse_e:.3f})")
    all_pass = s1_pass and s2_pass and s3_pass
    print(f"\n{'[OK] 分阶段优化解决 T15 — 架构优势在正确训练下成立' if all_pass else '仍有未通过项（如实记录）'}")
    print(f"[OK] 报告: {args.output_report} | checkpoint: {args.checkpoint_dir}/best.pt")
    print("=" * 68)


if __name__ == "__main__":
    main()
