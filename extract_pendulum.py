# -*- coding: utf-8 -*-
"""
真实摆轨迹提取 (extract_pendulum.py)
==================================================
从手机视频提取双连杆摆的两个标记（红=肘，蓝=手）的位置轨迹。

方法：HSV 颜色空间阈值分割 → 质心提取 → 像素-物理比例标定 → 20Hz 重采样。
支持单摆模式（只跟踪蓝色手端标记）。

用法：
  python extract_pendulum.py --video real_swing.mp4 --output real_traj.npz
  python extract_pendulum.py --video real_swing.mp4 --single_pendulum
  python extract_pendulum.py --video real_swing.mp4 --calib_length 60.0  # 两标记间距 cm
"""

import os
import argparse
import numpy as np

try:
    import cv2
    HAS_CV = True
except ImportError:
    HAS_CV = False


def hsv_centroid(frame, hsv_lo, hsv_hi):
    """单帧 HSV 阈值分割的最大连通域质心。返回 (x, y) 像素坐标或 None。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:  # 面积阈值（噪声）
        return None
    M = cv2.moments(largest)
    if M["m00"] < 1:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


# 默认 HSV 范围（红色/蓝色标记，可通过 --hsv_lo/--hsv_hi 微调）
RED_LO = [0, 100, 100];   RED_HI = [10, 255, 255]
RED_LO2 = [170, 100, 100]; RED_HI2 = [180, 255, 255]  # 红色跨 H=0
BLUE_LO = [100, 100, 100]; BLUE_HI = [130, 255, 255]


def track_red(frame):
    r1 = hsv_centroid(frame, RED_LO, RED_HI)
    if r1 is not None:
        return r1
    return hsv_centroid(frame, RED_LO2, RED_HI2)


def track_blue(frame):
    return hsv_centroid(frame, BLUE_LO, BLUE_HI)


def main():
    ap = argparse.ArgumentParser(description="真实摆轨迹提取")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--output", type=str, default="real_traj.npz")
    ap.add_argument("--single_pendulum", action="store_true",
                    help="单摆模式：只跟踪蓝色手端")
    ap.add_argument("--calib_length", type=float, default=60.0,
                    help="两标记间距的物理长度（cm），用于像素标定")
    ap.add_argument("--fps_target", type=float, default=20.0,
                    help="输出轨迹的目标采样率")
    ap.add_argument("--hsv_tune", action="store_true",
                    help="交互式 HSV 调参窗口")
    args = ap.parse_args()

    if not HAS_CV:
        print("[错误] 需要 opencv-python：pip install opencv-python")
        return
    if not os.path.exists(args.video):
        print(f"[错误] 视频不存在: {args.video}")
        return

    cap = cv2.VideoCapture(args.video)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {args.video} | {fps_video:.1f} fps | {total_frames} 帧")

    # HSV 调参模式
    if args.hsv_tune:
        hsv_tune_window(cap)
        return

    # ---- 逐帧跟踪 ----
    red_track = []   # (frame_idx, x_px, y_px)
    blue_track = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not args.single_pendulum:
            r = track_red(frame)
            if r is not None:
                red_track.append((fi, r[0], r[1]))
        b = track_blue(frame)
        if b is not None:
            blue_track.append((fi, b[0], b[1]))
        fi += 1
    cap.release()
    print(f"跟踪: 红标记 {len(red_track)} 帧 | 蓝标记 {len(blue_track)} 帧")

    # ---- 像素-物理比例标定 ----
    if not args.single_pendulum and len(red_track) > 10 and len(blue_track) > 10:
        # 找同一帧的红蓝位置差
        red_dict = {int(f): (x, y) for f, x, y in red_track}
        dists = []
        for f, bx, by in blue_track[:200]:
            if int(f) in red_dict:
                rx, ry = red_dict[int(f)]
                dists.append(np.hypot(bx - rx, by - ry))
        if dists:
            calib_dist_px = float(np.median(dists))
            px_per_cm = calib_dist_px / args.calib_length
            print(f"标定: 标记间距 {calib_dist_px:.1f}px = {args.calib_length}cm"
                  f" → {px_per_cm:.2f} px/cm")
        else:
            px_per_cm = 1.0
            print("[警告] 标定失败，使用 1px=1cm")
    else:
        px_per_cm = 1.0

    # ---- 重采样到目标 fps ----
    # 用蓝标记（手端）为主轨迹，红标记（肘）在双摆模式下
    dt_out = 1.0 / args.fps_target
    t_video_end = total_frames / fps_video
    t_out = np.arange(0, t_video_end, dt_out)

    def resample(track):
        """视频帧坐标 → 时间坐标 → 均匀重采样（cm 单位）。"""
        if len(track) < 10:
            return None, None
        arr = np.array(track)   # (M, 3): frame_idx, x, y
        t_in = arr[:, 0] / fps_video
        x_cm = arr[:, 1] / px_per_cm
        y_cm = arr[:, 2] / px_per_cm
        # 线性插值到均匀时间网格
        x_out = np.interp(t_out, t_in, x_cm)
        y_out = np.interp(t_out, t_in, y_cm)
        # 缺失帧填充（前向填充）
        valid = (arr[:, 0] > 0)
        if not valid.all():
            # 用前向填充的值替换插值边界
            pass
        return x_out, y_out

    if args.single_pendulum:
        hx, hy = resample(blue_track)
        if hx is None:
            print("[错误] 跟踪数据不足")
            return
        # 单摆：物体0 = 手端（从固定点），物体1 = 零填充
        # 转换为 layout_h 的 22 维状态（位置+速度+角度）
        T = len(hx)
        states_out = np.zeros((1, T, 22), dtype=np.float32)
        for t in range(T):
            states_out[0, t, 0] = 1.0     # presence
            states_out[0, t, 1] = hx[t]   # x (cm→保持 cm 单位)
            states_out[0, t, 2] = hy[t]
            states_out[0, t, 10] = M2     # mass
            states_out[0, t, 8] = 1       # type=ball
            if t > 0:
                dt = t_out[t] - t_out[t-1]
                if dt > 1e-6:
                    states_out[0, t, 5] = (hx[t] - hx[t-1]) / dt
                    states_out[0, t, 6] = (hy[t] - hy[t-1]) / dt
        # 速度平滑
        from scipy.ndimage import uniform_filter1d
        states_out[0, :, 5] = uniform_filter1d(states_out[0, :, 5], 3)
        states_out[0, :, 6] = uniform_filter1d(states_out[0, :, 6], 3)
    else:
        # 双摆模式：肘+手 → 物体0=肘、物体1=手
        rx_dict = {int(f): (x, y) for f, x, y in red_track}
        T = 0
        frames = []
        for t_idx, t in enumerate(t_out):
            fr = int(t * fps_video)
            if fr in rx_dict:
                rx, ry = rx_dict[fr]
                # 找最近的蓝标记
                bx_best, by_best, best_d = None, None, 1e9
                for bf, bx, by in blue_track:
                    if abs(bf - fr) <= 2:
                        d = (bx - rx) ** 2 + (by - ry) ** 2
                        if d < best_d:
                            best_d = d; bx_best, by_best = bx, by
                if bx_best is not None:
                    frames.append((rx, ry, bx_best, by_best, t))
        if len(frames) < 20:
            print("[错误] 双摆跟踪数据不足")
            return
        T = len(frames)
        states_out = np.zeros((1, T, 22), dtype=np.float32)
        for t_i in range(T):
            rx, ry, bx, by, t = frames[t_i]
            dt_prev = frames[t_i][4] - frames[t_i-1][4] if t_i > 0 else 0.05
            states_out[0, t_i, 0] = 1.0
            states_out[0, t_i, 1] = rx
            states_out[0, t_i, 2] = ry
            if t_i > 0:
                dt_p = frames[t_i][4] - frames[t_i-1][4]
                if dt_p > 1e-6:
                    states_out[0, t_i, 5] = (rx - frames[t_i-1][0]) / dt_p
                    states_out[0, t_i, 6] = (ry - frames[t_i-1][1]) / dt_p
                    states_out[0, t_i, 16] = (bx - frames[t_i-1][2]) / dt_p
                    states_out[0, t_i, 17] = (by - frames[t_i-1][3]) / dt_p
            states_out[0, t_i, 12] = bx
            states_out[0, t_i, 13] = by
            states_out[0, t_i, 16] = 0  # 手端 vx（差分回填）
            states_out[0, t_i, 17] = 0
            states_out[0, t_i, 8] = 1
            states_out[0, t_i, 9] = 30.0  # size
            states_out[0, t_i, 10] = 2.0  # mass
            states_out[0, t_i, 19] = 2.0
            states_out[0, t_i, 20] = 30.0
        # 手端速度差分
        for t_i in range(1, T):
            dt_p = frames[t_i][4] - frames[t_i-1][4]
            if dt_p > 1e-6:
                states_out[0, t_i, 16] = (frames[t_i][2] - frames[t_i-1][2]) / dt_p
                states_out[0, t_i, 17] = (frames[t_i][3] - frames[t_i-1][3]) / dt_p

    np.savez_compressed(args.output, states=states_out,
                        fps=args.fps_target, px_per_cm=px_per_cm)
    print(f"\n[OK] 真实轨迹保存至 {args.output} ({T} 帧 @ {args.fps_target}Hz)")
    print(f"     轨迹时长: {T / args.fps_target:.1f}s")


def hsv_tune_window(cap):
    """交互式 HSV 调参（帮助用户确定标记的 HSV 范围）。"""
    cv2.namedWindow("HSV Tune")
    cv2.createTrackbar("H_lo", "HSV Tune", 0, 180, lambda x: None)
    cv2.createTrackbar("H_hi", "HSV Tune", 180, 180, lambda x: None)
    cv2.createTrackbar("S_lo", "HSV Tune", 100, 255, lambda x: None)
    cv2.createTrackbar("V_lo", "HSV Tune", 100, 255, lambda x: None)
    print("调参窗口已打开。调整滑块直到标记被正确高亮，然后按 ESC 退出并记录数值。")
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        h_lo = cv2.getTrackbarPos("H_lo", "HSV Tune")
        h_hi = cv2.getTrackbarPos("H_hi", "HSV Tune")
        s_lo = cv2.getTrackbarPos("S_lo", "HSV Tune")
        v_lo = cv2.getTrackbarPos("V_lo", "HSV Tune")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]),
                           np.array([h_hi, 255, 255]))
        overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
        cv2.putText(overlay, f"H:{h_lo}-{h_hi} S_lo:{s_lo} V_lo:{v_lo}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("HSV Tune", overlay)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"请将 HSV 范围写入命令行参数并重新运行")


if __name__ == "__main__":
    main()
