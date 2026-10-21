# -*- coding: utf-8 -*-
"""
伪真实退化注入 (make_pseudo_real.py) — 微缩 sim-to-real 的域随机化方案
==================================================
从仿真数据生成"伪真实"数据：注入真实世界与仿真的全部已知差异。
这是机器人学的标准做法（domain randomization）——不等真实视频，
系统性地测试架构对每一种 sim-to-real 效应的鲁棒性。

注入的退化效应（可单独开关，逐一归因）：
  E1 跟踪抖动：高斯位置噪声（CV 标记检测的典型精度 ~1-3px）
  E2 帧丢失：随机丢帧 + 线性插值（30fps 视频的常见问题）
  E3 运动模糊：时间平滑（相机曝光期间的运动积分）
  E4 量化：位置舍入到像素网格（~0.1cm/px）
  E5 空气阻力：速度衰减 ∝ v²（真实世界效应）
  E6 关节摩擦：角速度衰减 ∝ ω（真实轴承受的摩擦）
  E7 离群点：随机错误检测（CV 偶发误检）
  E8 透视畸变：非正交拍摄的非线性缩放

预注册判决（对照干净数据基线）：
  PR-1 优雅退化（T1b-real）：伪真实数据上 model ≤ prior × 2
  PR-2 逐效应归因：每个效应单独注入时的误差增量排序
  PR-3 残差补偿存活：模型仍优于先验（在伪真实数据上）
  PR-4 最恶劣组合：全部效应同时注入的误差上限

用法：
  python make_pseudo_real.py --input trajectories_hinge.npz --output pseudo_real.npz
  python make_pseudo_real.py --input trajectories_hinge.npz --effect ablation  # 逐效应
"""

import argparse
import os

import numpy as np

from layout_h import OBJ_DIMS, L1, L2, M1, M2

POS_ALL = [1, 2, 5, 6, 12, 13, 16, 17]  # 位置+速度维
POS_XY = [1, 2, 12, 13]                  # 纯位置维


def inject_tracking_jitter(states, sigma_px=2.0):
    """E1: CV 跟踪抖动——高斯位置噪声（速度由噪声位置差分重算）。

    取证修复（2026-09-02）：原实现 `noisy[1:, b+vdim]` 把特征索引错用在
    (N, T, 22) 的时间轴上——实际写的是"第 5/6/16/17 个时间步的整行"，
    且 RHS 变成跨 episode 差分÷0.05（presence 被写 0、sin 被写 ±40、
    x 被写 ±3200），是 v2 报告 85.7px"伪真实崩塌"的全部根因。
    """
    noisy = states.copy()   # (N, T, 22)
    N, T = noisy.shape[:2]
    noise = np.random.randn(N, T, 1) * sigma_px   # (N, T, 1)
    for o in range(2):
        b = o * 11
        noisy[:, :, b + 1] += noise[:, :, 0]   # x
        noisy[:, :, b + 2] += noise[:, :, 0]   # y
    for o in range(2):
        b = o * 11
        for vdim, pdim in [(5, 1), (6, 2)]:
            noisy[1:, :, b + vdim] = (noisy[1:, :, b + pdim]
                                      - noisy[:-1, :, b + pdim]) / 0.05
            noisy[0, :, b + vdim] = noisy[1, :, b + vdim]
    return noisy


def inject_frame_drop(states, drop_rate=0.05):
    """E2: 帧丢失 + 前向填充。"""
    out = states.copy()
    N, T = out.shape[:2]
    for e in range(N):
        for t in range(1, T):
            if np.random.rand() < drop_rate:
                out[e, t] = out[e, t - 1]
    return out


def inject_motion_blur(states, kernel_size=3):
    """E3: 运动模糊 = 时间平滑。"""
    from scipy.ndimage import uniform_filter1d
    out = states.copy()
    for dim in POS_ALL:
        out[:, :, dim] = uniform_filter1d(states[:, :, dim], kernel_size, axis=1)
    return out


def inject_quantization(states, px_per_unit=0.5):
    """E4: 位置量化到像素网格。"""
    out = states.copy()
    for dim in POS_XY:
        out[:, :, dim] = np.round(out[:, :, dim] / px_per_unit) * px_per_unit
    return out


def inject_air_drag(states, drag_coef=0.005):
    """E5: 空气阻力——速度衰减。"""
    out = states.copy()
    dt = 0.05
    for e in range(states.shape[0]):
        for t in range(1, states.shape[1]):
            for o in range(2):
                b = o * 11
                vx, vy = out[e, t, b + 5], out[e, t, b + 6]
                speed = np.hypot(vx, vy)
                decay = max(1.0 - drag_coef * speed * dt, 0.9)
                out[e, t, b + 5] = vx * decay
                out[e, t, b + 6] = vy * decay
    return out


def inject_joint_friction(states, friction_coef=0.01):
    """E6: 关节摩擦——角速度衰减。"""
    out = states.copy()
    dt = 0.05
    for e in range(states.shape[0]):
        for t in range(1, states.shape[1]):
            for b in [0, 11]:
                w = out[e, t, b + 7]
                out[e, t, b + 7] = w * max(1.0 - friction_coef * abs(w) * dt, 0.95)
    return out


def inject_outliers(states, outlier_rate=0.01, outlier_mag=30.0):
    """E7: 离群点——随机位置跳变。"""
    out = states.copy()
    N, T = out.shape[:2]
    n_out = int(N * T * outlier_rate)
    eps_i = np.random.randint(0, N, n_out)
    ts_i = np.random.randint(0, T, n_out)
    objs_i = np.random.randint(0, 2, n_out)
    for e_i, t_i, o_i in zip(eps_i, ts_i, objs_i):
        b = o_i * 11
        out[e_i, t_i, b + 1] += np.random.randn() * outlier_mag
        out[e_i, t_i, b + 2] += np.random.randn() * outlier_mag
    return out


def inject_perspective(states, strength=0.001):
    """E8: 透视畸变——x 随 y 非线性缩放。"""
    out = states.copy()
    for e in range(states.shape[0]):
        for o in range(2):
            b = o * 11
            for t in range(states.shape[1]):
                y_factor = 1.0 + strength * out[e, t, b + 2]
                out[e, t, b + 1] *= y_factor
    return out


def inject_realistic_omega(states, sigma_px=2.0):
    """E9: 真实角速度观测（T13-real）——θ/ω 不再是特权真值。

    真实 CV 管线只能看到标记位置：θ̂ 由肘/手标记矢量 atan2 推出，
    ω̂ = 环绕差分/0.05（含量化混叠与回归均值误差，T13 的仿真预演）。
    覆写两物体的 sin/cos/ω 通道；x/y 不动（与 E1 正交，可组合）。
    """
    out = states.copy()
    N, T = out.shape[:2]
    n1 = np.random.randn(N, T, 2) * sigma_px   # 肘标记噪声
    n2 = np.random.randn(N, T, 2) * sigma_px   # 手标记噪声
    x1 = out[:, :, 1] + n1[:, :, 0]; y1 = out[:, :, 2] + n1[:, :, 1]
    x2 = out[:, :, 12] + n2[:, :, 0]; y2 = out[:, :, 13] + n2[:, :, 1]
    th1 = np.arctan2(x1, -y1)                  # 连杆1 角（相对竖直向下）
    th12 = np.arctan2(x2 - x1, -(y2 - y1))     # 连杆2 绝对角
    th2 = th12 - th1                           # 肘角（与 layout_h 语义一致）

    def _wrap(dth):
        return (dth + np.pi) % (2 * np.pi) - np.pi

    w1 = np.zeros_like(th1); w2 = np.zeros_like(th2)
    w1[:, 1:] = _wrap(th1[:, 1:] - th1[:, :-1]) / 0.05
    w2[:, 1:] = _wrap(th2[:, 1:] - th2[:, :-1]) / 0.05
    w1[:, 0] = w1[:, 1]; w2[:, 0] = w2[:, 1]
    out[:, :, 3] = np.sin(th1); out[:, :, 4] = np.cos(th1); out[:, :, 7] = w1
    out[:, :, 14] = np.sin(th2); out[:, :, 15] = np.cos(th2); out[:, :, 18] = w2
    return out


EFFECTS = {
    "E1_tracking_jitter": lambda s: inject_tracking_jitter(s, 2.0),
    "E2_frame_drop": lambda s: inject_frame_drop(s, 0.05),
    "E3_motion_blur": lambda s: inject_motion_blur(s, 3),
    "E4_quantization": lambda s: inject_quantization(s, 0.5),
    "E5_air_drag": lambda s: inject_air_drag(s, 0.005),
    "E6_joint_friction": lambda s: inject_joint_friction(s, 0.01),
    "E7_outliers": lambda s: inject_outliers(s, 0.01, 30.0),
    "E8_perspective": lambda s: inject_perspective(s, 0.001),
    "E9_realistic_omega": lambda s: inject_realistic_omega(s, 2.0),
}


def main():
    ap = argparse.ArgumentParser(description="伪真实退化注入")
    ap.add_argument("--input", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--output", type=str, default="pseudo_real_hinge.npz")
    ap.add_argument("--mode", type=str, default="all",
                    choices=["all", "ablation"], help="all=全效应 / ablation=逐效应单独输出")
    ap.add_argument("--episodes", type=int, default=500,
                    help="从输入取前 N 条（控制文件大小）")
    args = ap.parse_args()

    d = np.load(args.input)
    states_clean = d["states"][:args.episodes]
    print(f"输入: {states_clean.shape}")

    if args.mode == "all":
        # 全效应同时注入（最恶劣组合）
        np.random.seed(42)
        states_pr = states_clean.copy()
        for name, fn in EFFECTS.items():
            states_pr = fn(states_pr)
            print(f"  注入 {name}")
        np.savez_compressed(args.output, states=states_pr,
                            contact=d["contact"][:args.episodes],
                            actions=d["actions"][:args.episodes],
                            yao=d["yao"][:args.episodes])
        print(f"\n[OK] 伪真实数据（全效应）保存至 {args.output}")

        # 量化退化
        pos_err = np.abs(states_pr[:, :, POS_XY] - states_clean[:, :, POS_XY]).mean()
        print(f"  位置退化 MAE: {pos_err:.2f}px（纯净→伪真实的注入量）")

    else:
        # 逐效应消融
        out_dir = os.path.dirname(args.output) or "."
        for name, fn in EFFECTS.items():
            np.random.seed(42)
            states_pr = fn(states_clean.copy())
            out_path = os.path.join(out_dir, f"pseudo_{name}.npz")
            np.savez_compressed(out_path, states=states_pr,
                                contact=d["contact"][:args.episodes],
                                actions=d["actions"][:args.episodes],
                                yao=d["yao"][:args.episodes])
            pos_err = np.abs(states_pr[:, :, POS_XY] - states_clean[:, :, POS_XY]).mean()
            print(f"  {name}: MAE {pos_err:.2f}px → {out_path}")


if __name__ == "__main__":
    main()
