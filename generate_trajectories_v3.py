# -*- coding: utf-8 -*-
"""
2D物理模拟轨迹数据生成器 v3（几何可观测版）
==================================================
相对 v2 的四项结构性修复（对应 ROADMAP R4/R7）：
  1. 几何可观测：障碍物以 8 槽位矩阵进入数据（v2 中障碍完全不可见，碰撞因果不可学）；
     物体携带 presence / type / size / mass 属性（质量未知曾使动作效果被随机化 8 倍）。
  2. 动作=净冲量：记录采样间隔内的 ΣF·dt（v2 只记采样末步的力，正确性依赖
     reverse_steps 恰为采样间隔倍数这一巧合）。
  3. 20Hz 采样：v2 的 10Hz 下每步位移 25px≈1-3 身位，碰撞相位不可观测；
     20Hz 将位移减半。.physics_dt 不变（0.01s），工程上与引擎精度解耦。
  4. 真实接触标签（初爻）：由 pymunk 仲裁器逐物理步记录每物体是否发生接触，
     作为三爻辅助监督的引擎真值（v2 只能从速度差分反推，是代理标签）。

场景生成逻辑复用 v2（generate_scene），本文件只重写"记录什么"。

输出 trajectories_v3.npz:
    states    (N, T, 33)   物体段（layout.V3 布局，原始单位）
    obstacles (N, 80)      障碍槽位矩阵（片段级常量，组装时广播到每帧）
    contact   (N, T, 3)    接触真值 uint8（初爻）
    actions   (N, T, 6)    净冲量 ΣF·dt（原始单位，归一化常数见 layout.IMPULSE_MAX_V3）

组装为模型输入：assemble_frame(obj, obstacles, contact) → (116,)，见 layout.py
"""

import os
import json
import math
import random
import argparse
import numpy as np

from generate_trajectories import (
    generate_scene, _add_walls, _add_ball, _add_box,
    _add_segment_obstacle, _add_polygon_obstacle, _add_ramp_obstacle,
    _add_circle_pin_obstacle,
    SEED, SPACE_WIDTH, SPACE_HEIGHT, DURATION, PHYSICS_DT,
    VELOCITY_RANGE, ANGULAR_VEL_RANGE, BODY_MASS_RANGE,
)
import pymunk

from layout import (PER_OBJ, NUM_OBJ, OBJ_DIMS, OBS_SLOTS, OBS_PER_SLOT,
                    OBS_DIMS, CONTACT_DIMS, OBJ_FIELDS, F,
                    TYPE_BALL, TYPE_BOX, SAMPLE_INTERVAL_V3,
                    IMPULSE_MAX_V3, DT_V3, obj_base)

SEQ_LENGTH_V3 = int(DURATION / PHYSICS_DT / SAMPLE_INTERVAL_V3)   # 200 @ 20Hz
TOTAL_STEPS = int(DURATION / PHYSICS_DT)                          # 1000


# ============================================================
# 障碍槽位矩阵：从"引擎里实际存在的 shape"反向编码（而非场景配置），
# 这样 ramp 端点裁剪、多边形顶点扰动等内部处理都如实反映。
# ============================================================

def build_obstacle_matrix(scene_config, obstacle_shapes):
    mat = np.zeros((OBS_SLOTS, OBS_PER_SLOT), dtype=np.float32)
    obs_types = [o["type"] for o in scene_config["obstacles"]]
    for s, shape in enumerate(obstacle_shapes[:OBS_SLOTS]):
        cfg_type = obs_types[s] if s < len(obs_types) else "segment"
        if isinstance(shape, pymunk.Segment):
            a, b = shape.a, shape.b
            dx, dy = b.x - a.x, b.y - a.y
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            mat[s, 0] = 1.0                       # is_segment
            if cfg_type == "ramp":
                mat[s, 2] = 1.0                   # is_ramp 标签（几何同 segment）
            mat[s, 4] = a.x                       # cx 锚点
            mat[s, 5] = a.y                       # cy
            mat[s, 6] = dy / length               # dir_sin
            mat[s, 7] = dx / length               # dir_cos
            mat[s, 8] = length                    # p_len
        elif isinstance(shape, pymunk.Poly):
            verts = list(shape.get_vertices())
            n = len(verts)
            cx = sum(v.x for v in verts) / n
            cy = sum(v.y for v in verts) / n
            r = max(math.hypot(v.x - cx, v.y - cy) for v in verts)
            mat[s, 1] = 1.0                       # is_polygon
            mat[s, 4], mat[s, 5] = cx, cy
            mat[s, 8] = r                         # 外接圆半径
            mat[s, 9] = n / 6.0                   # 顶点数归一化
        elif isinstance(shape, pymunk.Circle):
            mat[s, 3] = 1.0                       # is_circle
            mat[s, 4] = shape.body.position.x
            mat[s, 5] = shape.body.position.y
            mat[s, 9] = shape.radius              # p_rad
    return mat


# ============================================================
# 接触检测（pymunk 6/7 API 自适应）
#   pymunk 7.x: space._get_arbiters() 返回原始仲裁器，直接暴露 body_a/body_b/count
#   pymunk 6.x: space.arbitraries 返回 Arbiter 包装对象（.shapes/.contacts）
# ============================================================

def _iter_touching_bodies(space):
    """产出当前步所有处于接触的刚体（Python 包装对象，可直接 id() 匹配）。

    pymunk 7.x 的原始仲裁器暴露的是 cffi 层 body，id() 与 Python 包装对象不同，
    必须经 pymunk.Arbiter 包装层转回 .shapes 再取 .body。
    """
    if hasattr(space, "_get_arbiters"):          # pymunk >= 7
        for raw in space._get_arbiters():
            if getattr(raw, "count", 0) <= 0:
                continue
            wrapper = pymunk.Arbiter(raw, space)
            for shp in wrapper.shapes:
                yield shp.body
    elif hasattr(space, "arbitraries"):          # pymunk 6
        for arb in space.arbitraries:
            if not arb.contacts:
                continue
            for shp in arb.shapes:
                yield shp.body


# ============================================================
# 物理模拟 v3
# ============================================================

def run_simulation_v3(scene_config):
    """执行模拟并记录 v3 五元组：物体状态 / 障碍矩阵 / 接触真值 / 净冲量动作。"""
    space = pymunk.Space()
    space.gravity = (0, -900)
    _add_walls(space)

    dynamic_bodies = []
    for obj_cfg in scene_config["objects"]:
        if obj_cfg["type"] == "ball":
            body, shape = _add_ball(space, obj_cfg["position"][0], obj_cfg["position"][1],
                                    obj_cfg["radius"], obj_cfg["mass"])
        else:
            body, shape = _add_box(space, obj_cfg["position"][0], obj_cfg["position"][1],
                                   obj_cfg["size"], obj_cfg["mass"])
        body.angle = obj_cfg["angle"]
        body.velocity = obj_cfg["velocity"]
        body.angular_velocity = obj_cfg["angular_velocity"]
        dynamic_bodies.append((body, obj_cfg))

    # 静态障碍（收集实际 shape 用于槽位编码）
    obstacle_shapes = []
    for obs_cfg in scene_config["obstacles"]:
        if obs_cfg["type"] == "segment":
            obstacle_shapes.append(_add_segment_obstacle(space, obs_cfg["a"], obs_cfg["b"], obs_cfg["radius"]))
        elif obs_cfg["type"] == "polygon":
            obstacle_shapes.append(_add_polygon_obstacle(space, obs_cfg["vertices"]))
        elif obs_cfg["type"] == "ramp":
            obstacle_shapes.append(_add_ramp_obstacle(space, obs_cfg["x"], obs_cfg["y"],
                                                      obs_cfg["length"], obs_cfg["angle_deg"]))
        elif obs_cfg["type"] == "circle_pin":
            obstacle_shapes.append(_add_circle_pin_obstacle(space, obs_cfg["x"], obs_cfg["y"], obs_cfg["radius"]))

    obstacle_mat = build_obstacle_matrix(scene_config, obstacle_shapes)

    body_index = {id(b): i for i, (b, _) in enumerate(dynamic_bodies)}
    force_list = scene_config.get("forces", [])
    reverse_steps = scene_config.get("reverse_steps", 0)
    force_signs = [1] * len(force_list)

    states = np.zeros((SEQ_LENGTH_V3, OBJ_DIMS), dtype=np.float32)
    contact = np.zeros((SEQ_LENGTH_V3, CONTACT_DIMS), dtype=np.uint8)
    actions = np.zeros((SEQ_LENGTH_V3, NUM_OBJ * 2), dtype=np.float32)
    impulse_acc = np.zeros(NUM_OBJ * 2, dtype=np.float64)

    sample_idx = 0
    for step in range(TOTAL_STEPS):
        if reverse_steps > 0 and step > 0 and step % reverse_steps == 0:
            force_signs = [-s for s in force_signs]

        for fi, force_cfg in enumerate(force_list):
            obj_idx = force_cfg["object_index"]
            if obj_idx < len(dynamic_bodies):
                fx_base, fy_base = force_cfg["force"]
                sign = force_signs[fi]
                fx, fy = fx_base * sign, fy_base * sign
                body = dynamic_bodies[obj_idx][0]
                body.apply_force_at_world_point((fx, fy), body.position)
                impulse_acc[obj_idx * 2 + 0] += fx * PHYSICS_DT
                impulse_acc[obj_idx * 2 + 1] += fy * PHYSICS_DT

        space.step(PHYSICS_DT)

        # 初爻真值：本物理步内每个动态物体是否处于接触（含墙/障碍/物体间）。
        # 语义：contact[t] = 产生状态 t 的那个采样间隔内发生过接触——
        # 在采样时刻之前扫描到的接触都写入"即将落盘的 sample_idx 槽位"，天然是并集。
        for body in _iter_touching_bodies(space):
            idx = body_index.get(id(body))
            if idx is not None:
                contact[sample_idx, idx] = 1

        if (step + 1) % SAMPLE_INTERVAL_V3 == 0:
            # 物体状态（原始单位）
            for i, (body, cfg) in enumerate(dynamic_bodies):
                b = obj_base(i)
                states[sample_idx, b + F["presence"]] = 1.0
                states[sample_idx, b + F["x"]] = body.position.x
                states[sample_idx, b + F["y"]] = body.position.y
                states[sample_idx, b + F["sin_t"]] = math.sin(body.angle)
                states[sample_idx, b + F["cos_t"]] = math.cos(body.angle)
                states[sample_idx, b + F["vx"]] = body.velocity.x
                states[sample_idx, b + F["vy"]] = body.velocity.y
                states[sample_idx, b + F["w"]] = body.angular_velocity
                states[sample_idx, b + F["type"]] = (TYPE_BALL if cfg["type"] == "ball"
                                                     else TYPE_BOX)
                states[sample_idx, b + F["size"]] = (cfg["radius"] if cfg["type"] == "ball"
                                                     else cfg["size"])
                states[sample_idx, b + F["mass"]] = cfg["mass"]
            # 动作 = 本采样间隔的净冲量
            actions[sample_idx] = impulse_acc
            impulse_acc[:] = 0.0
            sample_idx += 1

    return states, obstacle_mat.flatten(), contact, actions


def main():
    parser = argparse.ArgumentParser(description="2D 物理轨迹数据生成器 v3（几何可观测版）")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default="trajectories_v3.npz")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    N = args.episodes
    all_states = np.zeros((N, SEQ_LENGTH_V3, OBJ_DIMS), dtype=np.float32)
    all_obstacles = np.zeros((N, OBS_DIMS), dtype=np.float32)
    all_contact = np.zeros((N, SEQ_LENGTH_V3, CONTACT_DIMS), dtype=np.uint8)
    all_actions = np.zeros((N, SEQ_LENGTH_V3, NUM_OBJ * 2), dtype=np.float32)

    print("=" * 60)
    print("2D 物理轨迹数据生成器 v3（几何可观测版）")
    print(f"  片段数: {N:,} | 序列: {SEQ_LENGTH_V3} 步 @ {1.0/DT_V3:.0f}Hz")
    print(f"  物体段 {OBJ_DIMS} + 障碍 {OBS_DIMS} + 接触 {CONTACT_DIMS} = 组装 {OBJ_DIMS+OBS_DIMS+CONTACT_DIMS} 维")
    print(f"  动作: 净冲量 ΣF·dt (归一化常数 {IMPULSE_MAX_V3})")
    print("=" * 60)

    for ep in range(N):
        scene_config = generate_scene()
        st, ob, ct, ac = run_simulation_v3(scene_config)
        all_states[ep] = st
        all_obstacles[ep] = ob
        all_contact[ep] = ct
        all_actions[ep] = ac
        if (ep + 1) % 500 == 0:
            print(f"  [{(ep + 1) / N * 100:5.1f}%] {ep + 1:,}/{N:,}")

    np.savez_compressed(args.output, states=all_states, obstacles=all_obstacles,
                        contact=all_contact, actions=all_actions)
    size_mb = os.path.getsize(args.output) / (1024 ** 2)
    print(f"\n[OK] v3 数据集已保存至 {args.output} ({size_mb:.1f} MB)")
    print(f"     states {all_states.shape} | obstacles {all_obstacles.shape} | "
          f"contact {all_contact.shape} | actions {all_actions.shape}")

    contact_rate = float(all_contact.mean())
    print(f"     接触帧占比（初爻=1 的比例）: {contact_rate:.1%}")

    metadata = {
        "version": "v3",
        "description": "2D物理模拟轨迹数据集 v3（几何可观测 + 净冲量动作 + 20Hz + 接触真值）",
        "episodes": N,
        "seq_length": SEQ_LENGTH_V3,
        "sample_rate_hz": 1.0 / DT_V3,
        "physics_dt": PHYSICS_DT,
        "layout": {
            "per_obj": PER_OBJ,
            "obj_fields": OBJ_FIELDS,
            "obj_dims": OBJ_DIMS,
            "obstacle_slots": OBS_SLOTS,
            "obstacle_per_slot": OBS_PER_SLOT,
            "obstacle_dims": OBS_DIMS,
            "contact_dims": CONTACT_DIMS,
            "assembled_dim": OBJ_DIMS + OBS_DIMS + CONTACT_DIMS,
            "note": "模型输入 = assemble_frame(obj, obstacles, contact)，见 layout.py",
        },
        "action": {
            "dim": NUM_OBJ * 2,
            "format": "[impulse_x, impulse_y] × 3，采样间隔内净冲量 ΣF·dt",
            "norm_constant": IMPULSE_MAX_V3,
        },
        "contact_definition": "pymunk 仲裁器在采样间隔内任一物理步存在接触点即记 1（含墙/障碍/物体间）",
        "known_approximations": [
            "多边形障碍仅编码质心+外接圆半径+顶点数（顶点级形状留 v4）",
            "接触真值依赖 arb.contacts 非空判定，静态压紧接触可能偶发漏记",
        ],
        "gravity": [0, -900],
        "random_seed": args.seed,
    }
    meta_path = args.output.replace(".npz", "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"     元数据已保存至 {meta_path}")


if __name__ == "__main__":
    main()
