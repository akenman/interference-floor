# -*- coding: utf-8 -*-
"""
2D 推球入靶任务环境 (env_task.py)
==================================================
首席评审意见 #1 的落地：给世界模型接上"消费者"。
交互式任务环境，复用 v3 场景生成与状态布局（与训练分布一致）：
  观测 = v3 组装帧三部分：物体段(33) / 障碍槽位(80) / 接触真值(3)
  动作 = 归一化净冲量 (3,2) ∈ [-1,1]，语义与训练数据一致：
         恒力 F = a·FORCE_MAX(500N) 作用于一个间隔 τ=0.05s（5 个物理子步）
         → 冲量 = F·τ，归一化 = a（与 layout.IMPULSE_MAX_V3 自洽）
任务：控制物体 0 进入半径 target_radius 的靶区（任意时刻进入即成功）。
"""

import math
import random
import numpy as np
import pymunk

from generate_trajectories import generate_scene, _add_walls, _add_ball, _add_box
from generate_trajectories_v3 import (_add_segment_obstacle, _add_polygon_obstacle,
                                      _add_ramp_obstacle, _add_circle_pin_obstacle,
                                      build_obstacle_matrix, _iter_touching_bodies)
from layout import (OBJ_DIMS, OBS_DIMS, CONTACT_DIMS, PER_OBJ, F as LF,
                    TYPE_BALL, TYPE_BOX, PHYSICS_DT, DT_V3, FORCE_MAX)

SUBSTEPS = int(round(DT_V3 / PHYSICS_DT))   # 5 个物理子步 / 决策间隔


class PushTargetEnv:
    """单物体靶向控制任务。动作只通过槽位语义作用于全部物体（规划器自行决定）。"""

    def __init__(self, tau=DT_V3, force_max=FORCE_MAX, target_radius=30.0,
                 max_steps=60, space_w=600, space_h=400):
        self.tau, self.force_max = tau, force_max
        self.target_radius, self.max_steps = target_radius, max_steps
        self.space_w, self.space_h = space_w, space_h

    def reset(self, scene_config=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.scene = scene_config if scene_config is not None else generate_scene()
        self.space = pymunk.Space()
        self.space.gravity = (0, -900)
        _add_walls(self.space)

        self.bodies = []
        for cfg in self.scene["objects"]:
            if cfg["type"] == "ball":
                body, _ = _add_ball(self.space, cfg["position"][0], cfg["position"][1],
                                    cfg["radius"], cfg["mass"])
            else:
                body, _ = _add_box(self.space, cfg["position"][0], cfg["position"][1],
                                   cfg["size"], cfg["mass"])
            body.angle = cfg["angle"]
            body.velocity = cfg["velocity"]
            body.angular_velocity = cfg["angular_velocity"]
            self.bodies.append((body, cfg))

        obstacle_shapes = []
        for oc in self.scene["obstacles"]:
            if oc["type"] == "segment":
                obstacle_shapes.append(_add_segment_obstacle(self.space, oc["a"], oc["b"], oc["radius"]))
            elif oc["type"] == "polygon":
                obstacle_shapes.append(_add_polygon_obstacle(self.space, oc["vertices"]))
            elif oc["type"] == "ramp":
                obstacle_shapes.append(_add_ramp_obstacle(self.space, oc["x"], oc["y"],
                                                          oc["length"], oc["angle_deg"]))
            elif oc["type"] == "circle_pin":
                obstacle_shapes.append(_add_circle_pin_obstacle(self.space, oc["x"], oc["y"], oc["radius"]))
        self.obstacle_mat = build_obstacle_matrix(self.scene, obstacle_shapes).flatten()
        self.body_index = {id(b): i for i, (b, _) in enumerate(self.bodies)}

        self.target = (float(rng.uniform(100, self.space_w - 100)),
                       float(rng.uniform(80, self.space_h - 80)))
        self.contact = np.zeros(CONTACT_DIMS, dtype=np.float32)
        self.step_i = 0
        return self._obs()

    def _obs(self):
        obj = np.zeros(OBJ_DIMS, dtype=np.float32)
        for i, (body, cfg) in enumerate(self.bodies):
            b = i * PER_OBJ
            obj[b + LF["presence"]] = 1.0
            obj[b + LF["x"]] = body.position.x
            obj[b + LF["y"]] = body.position.y
            obj[b + LF["sin_t"]] = math.sin(body.angle)
            obj[b + LF["cos_t"]] = math.cos(body.angle)
            obj[b + LF["vx"]] = body.velocity.x
            obj[b + LF["vy"]] = body.velocity.y
            obj[b + LF["w"]] = body.angular_velocity
            obj[b + LF["type"]] = TYPE_BALL if cfg["type"] == "ball" else TYPE_BOX
            obj[b + LF["size"]] = cfg["radius"] if cfg["type"] == "ball" else cfg["size"]
            obj[b + LF["mass"]] = cfg["mass"]
        return obj, self.obstacle_mat.copy(), self.contact.copy()

    def step(self, action_norm):
        """action_norm: (3,2) 归一化净冲量。恒力 F = a·500N 作用 τ（5 个物理子步）。"""
        a = np.clip(np.asarray(action_norm, dtype=np.float32).reshape(3, 2), -1, 1)
        F = a * self.force_max
        self.contact[:] = 0
        for _ in range(SUBSTEPS):
            for i, (body, _) in enumerate(self.bodies):
                body.apply_force_at_world_point((F[i, 0], F[i, 1]), body.position)
            self.space.step(PHYSICS_DT)
            for body in _iter_touching_bodies(self.space):
                idx = self.body_index.get(id(body))
                if idx is not None:
                    self.contact[idx] = 1.0
        self.step_i += 1
        obj, obs_mat, ct = self._obs()
        d = self.target_dist(obj)
        success = d <= self.target_radius
        done = self.step_i >= self.max_steps or success
        return (obj, obs_mat, ct), d, success, done

    def target_dist(self, obj):
        return float(math.hypot(obj[LF["x"]] - self.target[0],
                                obj[LF["y"]] - self.target[1]))
