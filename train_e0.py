# -*- coding: utf-8 -*-
"""
E0 统一训练 (train_e0.py) — 9 臂 × 3 种子
==================================================
协议（全部臂统一，见 RESEARCH_PROGRAM.md §4.4）：
  epochs=60, batch_size=256, lr=1e-3 (AdamW, wd=1e-5)
  w_state=1.0 (Huber δ=1.0), w_yao=0.1 (BCE), w_proto=0.05 (KL),
  w_commit=0.01, w_usage=0.02;  A6 额外 w_cal=0.2 (NLL, 均值解耦)
  scheduled sampling: ss_max=0.5, 线性升温 20 epochs
  AMP bf16 / 梯度裁剪 1.0 / CosineAnnealingLR
  数据 seed=42，训练种子偏移 {0,1,2} → 实际 {42,43,44}

用法：
  python train_e0.py --arm ours_base --seed 42
  python train_e0.py --arm ours_base --all-seeds   # 串行 42/43/44
调度：python run_e0.py [--max-jobs N]  （全部臂 × 种子）
"""

import os, math, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e0_common import E0Data, make_datasets, validate, INPUT_DIM, RESIDUAL_ARMS
from layout_h import feat_weights as feat_weights_h
from physics_prior_spec import DEFAULT_SPEC, SPECS
import model_e0

OBJ_DIMS = 22


# ---------------------------------------------------------------- 损失
class ResidualLoss(nn.Module):
    """A3/A4/A5/A6 共用：状态残差 + 爻 + 原型 + 校准（仅 A6）。"""

    def __init__(self, arm, w_state=1.0, w_yao=0.1, w_proto=0.05, w_commit=0.01,
                 w_usage=0.02, w_cal=0.0):
        super().__init__()
        self.w = dict(state=w_state, yao=w_yao, proto=w_proto, commit=w_commit,
                      usage=w_usage, cal=w_cal)
        self.has_yao = arm in ("ours_base", "ours_gua", "ours_sigma")
        self.has_proto = arm in ("ours_base", "ours_gua")
        self.has_cal = (arm == "ours_sigma") and w_cal > 0

    def forward(self, model, h, c, a0, d_norm, yao_target, feat_w):
        if hasattr(model, "forward_state"):
            delta, _ = model.forward_state(h, a0)
        else:
            delta = model.predict_state(h, a0)
        huber = F.huber_loss(delta, d_norm, reduction="none")
        state_loss = (huber * feat_w.view(1, -1)).mean()
        parts = dict(state=state_loss.item())

        yao_loss = proto_loss = commit_loss = usage_loss = cal_loss = torch.tensor(0.0, device=h.device)

        if self.has_yao:
            yao_log = model.predict_yao(h)
            yao_loss = F.binary_cross_entropy_with_logits(yao_log, yao_target.float(), reduction="mean")
            parts["yao"] = yao_loss.item()
            fine = model.evidence(c)
            if self.has_proto:
                K = model.num_prototypes
                prior_p = model.predict_proto(h, a0)
                target_dist = F.softmax(fine.detach(), dim=-1)
                proto_loss = F.kl_div(F.log_softmax(prior_p, dim=-1), target_dist, reduction="batchmean")
                assigned = model.prototypes.detach()[fine.argmax(dim=-1)]
                commit_loss = (1.0 - F.cosine_similarity(assigned, c, dim=-1)).mean()
                assignment = F.softmax(fine.float(), dim=-1)
                batch_avg = assignment.mean(dim=0).clamp(min=1e-8)
                usage_loss = (math.log(K) - (-(batch_avg * batch_avg.clamp(min=1e-8).log()).sum())) / math.log(K)
                parts.update(proto=proto_loss.item(), commit=commit_loss.item(), usage=usage_loss.item())
            if self.has_cal:
                sigma = model.predict_sigma(h, a0)
                cal_loss = (0.5 * (-2 * torch.log(sigma) + (d_norm / sigma) ** 2)).mean()
                parts["cal"] = cal_loss.item()

        total = (self.w["state"] * state_loss + self.w["yao"] * yao_loss
                 + self.w["proto"] * proto_loss + self.w["commit"] * commit_loss
                 + self.w["usage"] * usage_loss + self.w["cal"] * cal_loss)
        return total, parts


class AbsoluteLoss(nn.Module):
    """A2/A8 共用：直接输出绝对状态（无先验）。"""

    def __init__(self, w_state=1.0):
        super().__init__()
        self.w_state = w_state

    def forward(self, model, h, a0, target, feat_w):
        pred = model.predict(h, a0) if hasattr(model, "predict") else model.predict_state(h, a0)
        huber = F.huber_loss(pred, target, reduction="none")
        return self.w_state * (huber * feat_w.view(1, -1)).mean(), dict(state=0.0)


def jepa_loss(model, xb, a0, target, feat_w):
    """A7 JEPA：latent 预测（EMA target）+ linear probe 解码状态。"""
    latent_pred = model.forward_latent(xb, a0)
    with torch.no_grad():
        target_lat = model.forward_target(xb)
    latent_loss = F.mse_loss(latent_pred, target_lat.detach())
    probe_pred = model.predict_state(latent_pred, a0)
    probe_loss = (F.huber_loss(probe_pred, target, reduction="none") * feat_w.view(1, -1)).mean()
    return latent_loss + probe_loss, dict(latent=latent_loss.item(), probe=probe_loss.item())


def resolve_checkpoint_dir(arm, seed, prior_spec=DEFAULT_SPEC, explicit=None):
    """检查点目录的唯一解析规则（与 run_e0.task_name 保持同一约定）。

    事故（2026-10-01）：本函数此前是内联的 `args.checkpoint_dir = f"checkpoints-E0/{arm}_s{seed}"`，
    **无条件覆盖** run_e0.py 传来的 --checkpoint_dir → P0-a 的 3 个先验档位 × 3 seed
    全部写进同一批目录，互相覆盖并覆盖了 E0 原始 ours_base@euler005 检查点。
    现规则：显式传入优先；否则按档位命名（默认档位保持历史名）。
    """
    if explicit:
        return explicit
    name = (f"{arm}_s{seed}" if prior_spec == DEFAULT_SPEC
            else f"{arm}@{prior_spec}_s{seed}")
    return f"checkpoints-E0/{name}"


def assert_dir_not_foreign(ckpt_dir, arm, seed, prior_spec, w_roll=None, lr=None,
                           detach_prior=None, windows_per_epoch=None):
    """覆盖保护：目录里已有 best.pt 且其记录的配方字段（arm/seed/prior_spec +
    给定时的 w_roll/lr/detach_prior/windows_per_epoch）与本进程不一致 → 拒训。
    这道保护拦住了 2026-10-01 的"3 档先验写进同一目录"事故，并在此扩展到
    "换了训练配方却写进同一目录"（P0-b 的闭环配方必须与单步配方分开存放）。"""
    pt = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(pt):
        return
    try:
        ck = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception as e:                     # noqa: BLE001
        print(f"[warn] 无法读取已有 {pt}（{e}），跳过覆盖保护", flush=True)
        return
    old = ck.get("args", {}) or {}
    o_arm = old.get("arm")
    if o_arm is None:                          # 更早的 checkpoint 没有 args，无法判定
        return
    now = [arm, seed, prior_spec]
    oldt = [o_arm, old.get("seed"), old.get("prior_spec") or DEFAULT_SPEC]
    for name, cur, cast in (("w_roll", w_roll, lambda x: round(float(x), 6)),
                            ("lr", lr, lambda x: round(float(x), 9)),
                            ("detach_prior", detach_prior, bool),
                            ("windows_per_epoch", windows_per_epoch, int)):
        if cur is None:
            continue
        if name == "lr":
            oldv = old.get(name) if old.get(name) is not None else 1e-3
        elif name == "w_roll":
            oldv = old.get(name) if old.get(name) is not None else 0.0
        else:
            oldv = old.get(name)
        now.append(cast(cur))
        oldt.append(cast(oldv) if oldv is not None else None)
    if tuple(now) != tuple(oldt):
        raise SystemExit(
            f"[拒训] {ckpt_dir} 里已有 arm={o_arm} seed={old.get('seed')} "
            f"prior_spec={old.get('prior_spec') or DEFAULT_SPEC} w_roll={old.get('w_roll')} "
            f"lr={old.get('lr')} detach_prior={old.get('detach_prior')} "
            f"wpe={old.get('windows_per_epoch')} 的 checkpoint，与本进程 "
            f"arm={arm} seed={seed} prior_spec={prior_spec} w_roll={w_roll} lr={lr} "
            f"detach_prior={detach_prior} wpe={windows_per_epoch} 不一致。\n"
            f"       若确实要覆盖，请显式指定另一个 --checkpoint_dir，或先移走旧目录。")


# ---------------------------------------------------------------- 主流程
def train_one(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_ds, val_ds, mean_obj, std_obj = make_datasets(
        args.data_path, args.seq_len, args.future_steps, args.seed,
        prior_spec=args.prior_spec)
    model = model_e0.make_model(args.arm, prior_spec=args.prior_spec).to(device)
    model.set_stats(mean_obj, std_obj)
    # 闭环训练的成本控制开关（见 model_h.WorldModelH.rollout）；普通属性，不进 state_dict
    if hasattr(model, "rollout_detach_prior"):
        model.rollout_detach_prior = bool(args.detach_prior)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.arm == "jepa":
        criterion = None
    elif args.arm in RESIDUAL_ARMS:
        criterion = ResidualLoss(args.arm, w_state=args.w_state, w_yao=args.w_yao,
                                 w_proto=args.w_proto, w_commit=args.w_commit,
                                 w_usage=args.w_usage, w_cal=args.w_cal)
    else:
        criterion = AbsoluteLoss(w_state=args.w_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    fw = torch.from_numpy(feat_weights_h(OBJ_DIMS)).to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    wpe = args.windows_per_epoch if args.windows_per_epoch > 0 else len(train_ds)
    use_amp = device.type == "cuda" and not args.no_amp
    best = float("inf"); best_epoch = -1

    print(f"[{args.arm}] seed={args.seed} 参数量 {n_params:,} (可训练 {trainable:,}) | "
          f"窗口 {len(train_ds):,}/{len(val_ds):,} | device={device}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(train_ds))[:wpe]
        agg, nb = {}, 0
        t0e = time.time()

        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            batch = train_ds.gather(idx)
            xb = batch["x"].to(device, non_blocking=True)
            ab = batch["a0"].to(device, non_blocking=True)
            afb = batch["af"].to(device, non_blocking=True)
            db = batch["d_norm"].to(device, non_blocking=True)
            yaob = batch["yao"].to(device, non_blocking=True)
            xf = batch["x_fut"].to(device, non_blocking=True)

            ss = min(args.ss_max, args.ss_max * epoch / args.ss_warmup) if args.ss_warmup else args.ss_max
            target_next = batch["x_next"][:, -1, :OBJ_DIMS].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                h, c = model.encode_sequence(xb)
                if args.arm == "jepa":
                    loss, parts = jepa_loss(model, xb, ab, target_next, fw)
                elif args.arm in RESIDUAL_ARMS:
                    loss, parts = criterion(model, h, c, ab, db, yaob, fw)
                else:
                    loss, parts = criterion(model, h, ab, target_next, fw)

                if args.w_roll > 0:
                    abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=ss)
                    roll = sum((0.7 ** k) * F.huber_loss(
                        abs_preds[k].float().clamp(-50, 50),
                        xf[:, k, :OBJ_DIMS].float(), reduction="mean")
                        for k in range(args.future_steps)) / args.future_steps
                    loss = loss + args.w_roll * roll
                    parts["roll"] = roll.item()

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if any(not torch.isfinite(p.grad.sum()) for p in model.parameters() if p.grad is not None):
                optimizer.zero_grad()
                continue
            optimizer.step()
            if args.arm == "jepa":
                # JEPA 的目标编码器必须 EMA 同步在线编码器，否则 target 是随机初始化的
                # 死目标 → latent loss 永不收敛（之前掉过这个坑）
                model._ema_update()
            if hasattr(model, "prototypes") and args.arm != "jepa":
                assign = model.evidence(c).argmax(dim=-1) if args.arm != "jepa" \
                    else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
                model.ema_prototypes(c.detach().float(), assign)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1

        sched.step()
        msg = " ".join(f"{k}={v / max(nb, 1):.4f}" for k, v in agg.items())
        print(f"  Epoch {epoch + 1}/{args.epochs} [{msg}] ({time.time() - t0e:.0f}s)", flush=True)

        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            m = validate(model, val_ds, device, args.future_steps, 30000)
            print(f"    [val] one-step 模型 {m['one_step_model']:.3f} | 先验 {m['one_step_prior']:.3f}"
                  f" | 零速 {m['one_step_zero_vel']:.3f} px | yao {m['yao_acc']:.3f}", flush=True)
            ckpt = dict(model_state_dict=model.state_dict(),
                        config=dict(arm=args.arm, hidden_dim=args.hidden_dim,
                                    input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=1,
                                    seed=args.seed, epochs=args.epochs),
                        norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj),
                        args=vars(args), epoch=epoch, val_metrics=m)
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))
            if m["one_step_model"] < best:
                best = m["one_step_model"]; best_epoch = epoch
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
                print(f"    [保存最佳] val one-step = {best:.3f}px", flush=True)

    print(f"[OK] {args.arm} seed={args.seed} 最佳 val one-step = {best:.3f}px (epoch {best_epoch + 1})", flush=True)
    return best


def main():
    p = argparse.ArgumentParser(description="E0 统一训练（9 臂 × 3 种子）")
    p.add_argument("--arm", type=str, required=True, choices=list(model_e0.ARM_SPECS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-seeds", action="store_true", help="串行跑种子 42/43/44")
    p.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--future_steps", type=int, default=5)
    p.add_argument("--w_state", type=float, default=1.0)
    p.add_argument("--w_yao", type=float, default=0.1)
    p.add_argument("--w_proto", type=float, default=0.05)
    p.add_argument("--w_commit", type=float, default=0.01)
    p.add_argument("--w_usage", type=float, default=0.02)
    p.add_argument("--w_cal", type=float, default=0.0)
    p.add_argument("--w_roll", type=float, default=0.0)
    p.add_argument("--prior_spec", type=str, default=DEFAULT_SPEC,
                   choices=sorted(SPECS),
                   help="先验积分器档位（physics_prior_spec.SPECS）。"
                        "训练期残差目标与评估期先验必须同源，本参数写入 checkpoint args")
    p.add_argument("--ss_max", type=float, default=0.5)
    p.add_argument("--ss_warmup", type=int, default=20)
    p.add_argument("--windows_per_epoch", type=int, default=0,
                   help="每 epoch 抽多少训练窗口（0=全量）。P0-b 用 150000 控成本，"
                        "两种配方必须取同一值，否则比较被'每 epoch 数据量'混淆")
    p.add_argument("--detach_prior", action="store_true",
                   help="闭环 rollout 里把物理先验的输入 detach（数值不变、省 2× 时间）。"
                        "仅训练时生效；属于配方的一部分，写入 checkpoint args")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="显式指定检查点目录；不给则由 resolve_checkpoint_dir 按 "
                        "arm/seed/prior_spec 生成（默认档位沿用历史名）")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    spec = model_e0.ARM_SPECS[args.arm]
    if not spec["trainable"]:
        print(f"[SKIP] {args.arm} 无训练参数（解析基线），无需训练")
        return
    if args.hidden_dim is not None:
        spec["hidden"] = args.hidden_dim
    if args.all_seeds:
        for s in (42, 43, 44):
            args.seed = s
            args.checkpoint_dir = resolve_checkpoint_dir(
                args.arm, s, args.prior_spec, args.checkpoint_dir)
            assert_dir_not_foreign(args.checkpoint_dir, args.arm, s, args.prior_spec,
                                   args.w_roll, args.lr, args.detach_prior,
                                   args.windows_per_epoch)
            train_one(args)
    else:
        args.checkpoint_dir = resolve_checkpoint_dir(
            args.arm, args.seed, args.prior_spec, args.checkpoint_dir)
        assert_dir_not_foreign(args.checkpoint_dir, args.arm, args.seed, args.prior_spec,
                               args.w_roll, args.lr, args.detach_prior,
                               args.windows_per_epoch)
        train_one(args)


if __name__ == "__main__":
    main()
