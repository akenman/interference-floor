# -*- coding: utf-8 -*-
"""M5 全可观测对照：把摆世界力矩日程约束到 0.05 s 采样网格。

动机（审稿 M5）：精确档残余下限 0.038 px 被归因于"生成器在 0.1 s 过渡期内每 0.01 s
更新 τ，而动作标签每 0.05 s 只记录末步值 ⇒ 区间内 τ 日程不可观测"。本脚本做构造性
对照：同协议生成"每区间内 τ 恒定、仅在采样边界跳变"的数据（标签完全决定施加力矩），
先验侧应退化为 float32 存储精度，训练臂随之应到达机器级下限。

口径：
  --gen    网格对齐日程生成 trajectories_hinge_obsgrid.npz（3000 集，seed 42；
           与 generate_hinge 同 RNG 消费模式：level/ uniform / choice 逐次对应）；
  --prior  先验侧对照（rk4001x5，与 build_torque_granularity.py 同窗口协议）；
  --eval   训练臂评估（make_datasets(seed=42) 自有 val 划分；e0_common.validate +
           FF 窗 head/target 读数，见 build_ladder_floor_r1.py）。
输出并入 analysis_output/observability_control.json。

用法：
  python -X utf8 tmp_m5_observability_control.py --gen
  python -X utf8 tmp_m5_observability_control.py --prior
  python -X utf8 tmp_m5_observability_control.py --eval \
      --ckpt checkpoints-M5/ours_base@rk4001x5_s42/best.pt
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import generate_hinge as G
from layout_h import OBJ_DIMS, feat_weights, yao_labels_h
from physics_prior_spec import prior_raw_np

OUT_NPZ = os.path.join(ROOT, "trajectories_hinge_obsgrid.npz")
OUT_JSON = os.path.join(ROOT, "analysis_output", "observability_control.json")
L, K, N_WIN = 8, 5, 30000
POS = [1, 2, 12, 13]
W_POS = feat_weights(OBJ_DIMS)[POS]
SAMPLE = G.SAMPLE_INTERVAL


def make_tau_schedule_grid(rng, n_phys_steps):
    """generate_hinge.make_tau_schedule 的网格对齐变体：hold 四舍五入到整数个
    0.05 s 采样间隔，档位在边界瞬时切换（无区间内变化，τ 严格分段常数）。"""
    tau = np.zeros(n_phys_steps, dtype=np.float64)
    sample_dt = G.PHYSICS_DT * SAMPLE
    i = 0
    rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
    while i < n_phys_steps:
        hold_n = max(1, int(round(rng.uniform(0.25, 0.75) / sample_dt)))
        level = float(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
        hold = hold_n * SAMPLE
        for j in range(min(hold, n_phys_steps - i)):
            tau[i + j] = level
        i += hold
    return np.clip(tau, -1, 1) * G.TAU_MAX


def _merge(key, payload):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    out = {}
    if os.path.exists(OUT_JSON):
        with io.open(OUT_JSON, encoding="utf-8") as f:
            out = json.load(f)
    out[key] = payload
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save]", OUT_JSON, "::", key)


def gen(episodes=3000, seed=42):
    rng_chk = np.random.default_rng(123)
    tau_chk = make_tau_schedule_grid(rng_chk, G.SEQ_LEN * SAMPLE)
    assert np.allclose(tau_chk.reshape(G.SEQ_LEN, SAMPLE).std(axis=1), 0.0), \
        "schedule not constant within sample intervals"

    # 仅替换日程函数；run_episode 其余逐位复用（RNG 消费模式与原版一致）
    G.make_tau_schedule = make_tau_schedule_grid
    rng = np.random.default_rng(seed)
    print(f"grid-aligned generation: {episodes} episodes, seed {seed}")
    t0 = time.time()
    all_states = np.zeros((episodes, G.SEQ_LEN, OBJ_DIMS), dtype=np.float32)
    all_contact = np.zeros((episodes, G.SEQ_LEN, G.CONTACT_DIMS), dtype=np.float32)
    all_actions = np.zeros((episodes, G.SEQ_LEN, G.ACTION_DIM), dtype=np.float32)
    all_targets = np.zeros((episodes, 2), dtype=np.float32)
    for ep in range(episodes):
        target = (rng.uniform(-120, 120), rng.uniform(-140, -40))
        all_targets[ep] = target
        st, ct, ac = G.run_episode(rng, target)
        all_states[ep] = st
        all_contact[ep] = ct
        all_actions[ep] = ac
        if (ep + 1) % 1000 == 0:
            print(f"  [{(ep + 1) / episodes * 100:5.1f}%] {ep + 1:,}")
    yao = np.zeros((episodes, G.SEQ_LEN, 3, G.NUM_OBJ), dtype=np.int8)
    for ep in range(episodes):
        yao[ep] = yao_labels_h(all_states[ep], all_contact[ep], all_targets[ep])
    np.savez_compressed(OUT_NPZ, states=all_states, contact=all_contact,
                        actions=all_actions, yao=yao, targets=all_targets)
    meta = {
        "version": "hinge-v1-obsgrid",
        "episodes": episodes, "seq_length": G.SEQ_LEN, "sample_rate_hz": 20.0,
        "schedule": "piecewise-constant on the 0.05 s sample grid (instant switch)",
        "physics": {"tau_max": G.TAU_MAX, "integrator": "RK4@0.01s", "dt_sample": G.DT},
        "seed": seed,
    }
    with io.open(OUT_NPZ.replace(".npz", "_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    mb = os.path.getsize(OUT_NPZ) / (1024 ** 2)
    print(f"[OK] {OUT_NPZ} ({mb:.1f} MB, {time.time() - t0:.0f} s)")
    print(f"     手触地帧占比: {all_contact[:, :, 0].mean():.1%} | "
          f"靶区命中帧: {all_contact[:, :, 1].mean():.1%} | "
          f"高速警戒帧: {all_contact[:, :, 2].mean():.1%}")


def _windows(states, actions):
    n_ep = states.shape[0]
    rng = np.random.RandomState(42)
    segs = np.arange(n_ep)
    rng.shuffle(segs)
    val_eps = np.sort(segs[:max(1, int(n_ep * 0.1))])
    S, A = states[val_eps], actions[val_eps]
    n_win = S.shape[1] - L - K + 1
    ep_i = np.repeat(np.arange(len(val_eps)), n_win)
    w0 = np.tile(np.arange(n_win), len(val_eps))
    idx = np.arange(len(ep_i))
    if len(idx) > N_WIN:
        idx = np.sort(np.random.RandomState(0).choice(idx, N_WIN, replace=False))
    ep_i, w0 = ep_i[idx], w0[idx]
    last = S[ep_i, w0 + L - 1].astype(np.float64)
    tgt1 = S[ep_i, w0 + L][:, :OBJ_DIMS].astype(np.float64)
    a0 = A[ep_i, w0 + L].astype(np.float64).reshape(-1)
    a_next = A[ep_i, w0 + L + 1].astype(np.float64).reshape(-1)
    return val_eps, last, tgt1, a0, a_next


def _pos_rmse(pred, tgt, mask=None):
    e2 = ((pred - tgt)[:, POS] ** 2) * W_POS
    return float(np.sqrt(e2.mean(axis=1)[mask].mean() if mask is not None else e2.mean()))


def prior():
    d = np.load(OUT_NPZ, allow_pickle=True)
    val_eps, last, tgt1, a0, a_next = _windows(d["states"], d["actions"])
    ramp = np.abs(a_next - a0) > 1e-6
    pred = prior_raw_np("rk4001x5", last, a0)
    rw = np.sqrt((((pred - tgt1)[:, POS]) ** 2 * W_POS).mean(axis=1))
    ref = {}
    refp = os.path.join(ROOT, "analysis_output", "torque_granularity.json")
    if os.path.exists(refp):
        with io.open(refp, encoding="utf-8") as f:
            g = json.load(f)
        ref = dict(source="analysis_output/torque_granularity.json",
                   exact_tier=g["tiers"]["rk4001x5"],
                   exact_tier_gap_px=g["exact_tier_gap_px"])
    out = dict(
        note=("Full-observability control: torque schedule constant within each 0.05 s "
              "sample interval, so the action label determines the applied torque. "
              "The exact-tier prior should be limited only by float32 storage."),
        generator=dict(file=os.path.basename(OUT_NPZ), episodes=int(d["states"].shape[0]),
                       schedule="piecewise-constant on the 0.05 s sample grid"),
        protocol=dict(val_episodes=int(len(val_eps)), window_seed=0,
                      n_windows=int(len(last)), L=L, K=K, pos_idx=POS,
                      prior_impl="physics_prior_spec.prior_raw_np"),
        prior_new=dict(one_step=_pos_rmse(pred, tgt1),
                       one_step_da0=_pos_rmse(pred, tgt1, ~ramp),
                       one_step_dane0=_pos_rmse(pred, tgt1, ramp),
                       ramp_fraction=float(ramp.mean()),
                       per_window_px=dict(median=float(np.median(rw)),
                                          p99=float(np.percentile(rw, 99)),
                                          max=float(rw.max())),
                       n_over_1e_3=int((rw > 1e-3).sum())),
        prior_original_reference=ref,
    )
    assert out["prior_new"]["one_step"] < 1e-3, out["prior_new"]
    _merge("prior_control", out)
    print(f"  prior on obsgrid data: one-step={out['prior_new']['one_step']:.3e} px "
          f"(old exact-tier floor: {ref.get('exact_tier', {}).get('one_step', float('nan')):.4f} px) "
          f"| max={out['prior_new']['per_window_px']['max']:.3e}")


def eval_arm(ckpt):
    import torch
    from e0_common import make_datasets, validate
    from model_e0 import make_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, mo, so = make_datasets(OUT_NPZ, L=L, K=K, seed=42,
                                             prior_spec="rk4001x5")
    model = make_model("ours_base", prior_spec="rk4001x5").to(device)
    ck = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.set_stats(mo, so)
    model.eval()

    m = validate(model, val_ds, device, K=K, max_windows=N_WIN)
    idx = np.arange(len(val_ds))
    if len(idx) > N_WIN:
        idx = np.sort(np.random.RandomState(0).choice(idx, N_WIN, replace=False))
    ep, i0 = val_ds.ep_idx[idx], val_ds.win_start[idx]
    ff = ((val_ds.contact[ep, i0 + L - 1].max(axis=1) < 0.5)
          & (val_ds.contact[ep, i0 + L].max(axis=1) < 0.5))
    sub = idx[ff]
    batch = val_ds.gather(sub)
    with torch.no_grad():
        h, _ = model.encode_sequence(batch["x"].to(device))
        delta = model.predict_state(h, batch["a0"].to(device)).float().cpu().numpy()
    target = batch["d_norm"].numpy()
    std_px = np.asarray(val_ds.std_obj)[POS]

    d_px = delta[:, POS] * std_px
    t_px = target[:, POS] * std_px
    out = dict(
        checkpoint=ckpt, best_epoch=ck.get("epoch"), device=device,
        validate=dict(one_step_model=m["one_step_model"],
                      one_step_prior=m["one_step_prior"],
                      one_step_zero_vel=m["one_step_zero_vel"],
                      rollout_k5=m["rollout"]["k=5"], yao_acc=m["yao_acc"],
                      n_windows=m["n_windows"]),
        ff_windows=dict(
            n=int(len(sub)),
            head_px=float(np.sqrt((d_px ** 2).mean())),
            target_px=float(np.sqrt((t_px ** 2).mean())),
        ),
    )
    _merge("trained_arm", out)
    print(f"  trained arm: one-step={m['one_step_model']:.3e} px "
          f"(prior {m['one_step_prior']:.3e}) | FF head={out['ff_windows']['head_px']:.3e} px "
          f"target={out['ff_windows']['target_px']:.3e} px | n_FF={out['ff_windows']['n']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt", type=str,
                    default="checkpoints-M5/ours_base@rk4001x5_s42/best.pt")
    a = ap.parse_args()
    if a.gen:
        gen()
    if a.prior:
        prior()
    if a.eval:
        eval_arm(a.ckpt)
