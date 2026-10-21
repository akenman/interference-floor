"""
2D物理模拟轨迹数据生成器（v2 — 扩展版）
=========================================
使用 pymunk 物理引擎生成2D物理模拟的运动轨迹数据，
作为世界模型训练的基础数据集。

v2 改进：
  - 片段数默认从 10,000 → 50,000（可通过 --episodes 参数覆盖）
  - 新增障碍物类型：斜坡（ramp）/ 圆形柱体（circle_pin）
  - 扩展外力系统：
      * 随机变向推力（每 N 步反转方向，模拟振荡驱动）
      * 30% 概率对多个物体同时施加不同外力
  - 物体槽位添加可选的"存在标志位"（object_presence），
    解决零填充歧义（零 = 不存在 vs 原点静止）。
    注意：启用后 state_dim 变为 21（3 × 7），与旧数据不兼容，
          默认保持 state_dim=18 与旧数据兼容，可通过 --presence 启用。
  - 物理参数扩展：更宽的速度范围、更多障碍物数量上限。
  - 并发化建议：可在外部用 multiprocessing 分批调用 main(seed, start, end)。

输出：
  - trajectories.npy : 形状 (片段数, 序列长度, 状态维度)
  - metadata.json    : 包含场景配置等元信息

状态向量定义（每个物体，presence=False 时）：
  [x, y, angle, velocity_x, velocity_y, angular_velocity]
  多物体时按顺序拼接，不足3个物体用零填充至固定长度18。

状态向量定义（每个物体，presence=True 时）：
  [presence_flag, x, y, angle, velocity_x, velocity_y, angular_velocity]
  总长度21。
"""

import os
import json
import random
import math
import argparse
import numpy as np
import pymunk

# ============================================================
# 随机种子（保证数据可复现）
# ============================================================
SEED = 42

# ============================================================
# 全局常量
# ============================================================
SPACE_WIDTH  = 600         # 模拟空间宽度（像素）
SPACE_HEIGHT = 400         # 模拟空间高度（像素）
WALL_THICKNESS  = 5        # 墙壁厚度
WALL_ELASTICITY = 0.8      # 墙壁弹性系数
WALL_FRICTION   = 0.5      # 墙壁摩擦系数

DURATION        = 10.0            # 每个片段持续时间（秒）
PHYSICS_DT      = 1.0 / 100.0    # 物理步长（1/100秒）
SAMPLE_INTERVAL = 10              # 每隔多少物理步采样一次（10步 → 10Hz）
SEQ_LENGTH      = int(DURATION / PHYSICS_DT / SAMPLE_INTERVAL)  # = 100
STATE_DIM_PER_OBJ = 6            # 每个物体的基础状态维度
MAX_OBJECTS     = 3              # 最大物体数量

NUM_EPISODES = 50000             # 默认生成片段总数（v2: 10k → 50k）

BODY_ELASTICITY_RANGE = (0.2, 0.95)  # v2: 弹性范围更宽
BODY_FRICTION_RANGE   = (0.2, 0.9)
BODY_MASS_RANGE       = (1.0, 8.0)   # v2: 质量上限从5提到8
BALL_RADIUS_RANGE     = (8, 30)      # v2: 最小半径缩小（更小球）
BOX_SIZE_RANGE        = (12, 40)     # v2: 尺寸范围更宽
VELOCITY_RANGE        = (-250, 250)  # v2: 速度范围扩大
ANGULAR_VEL_RANGE     = (-5.0, 5.0) # v2: 角速度范围扩大

OBSTACLE_ELASTICITY = 0.6
OBSTACLE_FRICTION   = 0.7

FORCE_MAGNITUDE_RANGE = (30, 500)    # v2: 外力范围扩大
FORCE_PROBABILITY     = 0.40         # v2: 施力概率略微提高（0.3→0.4）
MULTI_FORCE_PROB      = 0.30         # v2 新增：多物体同时受力概率
REVERSE_FORCE_STEPS   = 30          # v2 新增：变向推力每隔多少物理步反转方向


# ============================================================
# 辅助：墙壁
# ============================================================

def _add_walls(space):
    """在空间四周添加弹性墙壁（上下左右四段静态线段）。"""
    wall_segments = [
        ((0, 0),            (SPACE_WIDTH, 0)),
        ((0, SPACE_HEIGHT), (SPACE_WIDTH, SPACE_HEIGHT)),
        ((0, 0),            (0, SPACE_HEIGHT)),
        ((SPACE_WIDTH, 0),  (SPACE_WIDTH, SPACE_HEIGHT)),
    ]
    walls = []
    for a, b in wall_segments:
        body  = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, a, b, WALL_THICKNESS)
        shape.elasticity = WALL_ELASTICITY
        shape.friction   = WALL_FRICTION
        space.add(body, shape)
        walls.append(shape)
    return walls


# ============================================================
# 辅助：动态物体
# ============================================================

def _add_ball(space, x, y, radius, mass):
    """添加动态圆形物体（小球）。"""
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body   = pymunk.Body(mass, moment)
    body.position = (x, y)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = random.uniform(*BODY_ELASTICITY_RANGE)
    shape.friction   = random.uniform(*BODY_FRICTION_RANGE)
    space.add(body, shape)
    return body, shape


def _add_box(space, x, y, size, mass):
    """添加动态方形物体（方块）。"""
    half     = size / 2.0
    vertices = [(-half, -half), (half, -half), (half, half), (-half, half)]
    moment   = pymunk.moment_for_poly(mass, vertices)
    body     = pymunk.Body(mass, moment)
    body.position = (x, y)
    shape = pymunk.Poly(body, vertices)
    shape.elasticity = random.uniform(*BODY_ELASTICITY_RANGE)
    shape.friction   = random.uniform(*BODY_FRICTION_RANGE)
    space.add(body, shape)
    return body, shape


# ============================================================
# 辅助：静态障碍物（v2 新增 ramp / circle_pin）
# ============================================================

def _add_segment_obstacle(space, a, b, radius=3):
    """添加静态线段障碍物（挡板）。"""
    body  = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, a, b, radius)
    shape.elasticity = OBSTACLE_ELASTICITY
    shape.friction   = OBSTACLE_FRICTION
    space.add(body, shape)
    return shape


def _add_polygon_obstacle(space, vertices):
    """添加静态多边形障碍物。"""
    body  = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Poly(body, vertices)
    shape.elasticity = OBSTACLE_ELASTICITY
    shape.friction   = OBSTACLE_FRICTION
    space.add(body, shape)
    return shape


def _add_ramp_obstacle(space, x, y, length, angle_deg):
    """v2 新增：添加固定角度斜坡障碍物（线段实现）。

    参数:
        x, y:      斜坡起始点坐标
        length:    斜坡长度
        angle_deg: 斜坡倾斜角度（度）
    """
    angle_rad = math.radians(angle_deg)
    ex = x + length * math.cos(angle_rad)
    ey = y + length * math.sin(angle_rad)
    # 保证端点在边界内
    ex = max(10, min(SPACE_WIDTH  - 10, ex))
    ey = max(10, min(SPACE_HEIGHT - 10, ey))
    return _add_segment_obstacle(space, (x, y), (ex, ey), radius=4)


def _add_circle_pin_obstacle(space, x, y, radius):
    """v2 新增：添加静态圆形柱体障碍物（静态圆体，物体可绕其弹开）。

    参数:
        x, y:   中心坐标
        radius: 柱体半径
    """
    body  = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (x, y)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = OBSTACLE_ELASTICITY
    shape.friction   = OBSTACLE_FRICTION
    space.add(body, shape)
    return shape


# ============================================================
# 辅助：随机位置
# ============================================================

def _random_position(margin=50):
    """生成空间内的随机位置（留出边距）。"""
    x = random.uniform(margin, SPACE_WIDTH  - margin)
    y = random.uniform(margin, SPACE_HEIGHT - margin)
    return x, y


# ============================================================
# 场景生成（v2 扩展）
# ============================================================

def generate_scene():
    """随机生成一个物理场景（v2）。

    场景包含：
      - 1~3 个动态物体（ball / box），具有随机初始运动状态
      - 2~7 个静态障碍物（segment / polygon / ramp / circle_pin）
      - 可选持续外力（含多物体同时受力 & 变向推力配置）

    返回:
        scene_config: dict
            - 'num_objects':   动态物体数量
            - 'objects':       物体配置列表
            - 'obstacles':     障碍物配置列表
            - 'forces':        外力配置列表（可为空）
            - 'reverse_steps': 变向推力反转间隔（仅在有外力时有效）
    """
    scene_config = {
        'num_objects':   0,
        'objects':       [],
        'obstacles':     [],
        'forces':        [],
        'reverse_steps': 0,   # 0 = 不变向
    }

    # ---- 动态物体 ----
    num_objects = random.randint(1, MAX_OBJECTS)
    scene_config['num_objects'] = num_objects

    for _ in range(num_objects):
        obj_type = random.choice(['ball', 'box'])
        x, y     = _random_position(margin=60)
        angle    = random.uniform(0, 2 * math.pi)
        vx       = random.uniform(*VELOCITY_RANGE)
        vy       = random.uniform(*VELOCITY_RANGE)
        av       = random.uniform(*ANGULAR_VEL_RANGE)
        mass     = random.uniform(*BODY_MASS_RANGE)

        obj_cfg = {
            'type':             obj_type,
            'position':         (x, y),
            'angle':            angle,
            'velocity':         (vx, vy),
            'angular_velocity': av,
            'mass':             mass,
        }
        if obj_type == 'ball':
            obj_cfg['radius'] = random.randint(*BALL_RADIUS_RANGE)
        else:
            obj_cfg['size']   = random.randint(*BOX_SIZE_RANGE)

        scene_config['objects'].append(obj_cfg)

    # ---- 静态障碍物（v2：4 种类型，数量上限从 5 → 7）----
    num_obstacles = random.randint(2, 7)
    obstacle_types = ['segment', 'polygon', 'ramp', 'circle_pin']

    for _ in range(num_obstacles):
        obs_type = random.choice(obstacle_types)

        if obs_type == 'segment':
            ax, ay  = _random_position(margin=30)
            bx      = ax + random.uniform(-150, 150)
            by      = ay + random.uniform(-150, 150)
            bx      = max(10, min(SPACE_WIDTH  - 10, bx))
            by      = max(10, min(SPACE_HEIGHT - 10, by))
            scene_config['obstacles'].append({
                'type':   'segment',
                'a':      (ax, ay),
                'b':      (bx, by),
                'radius': random.uniform(2, 6),
            })

        elif obs_type == 'polygon':
            cx, cy  = _random_position(margin=60)
            n_verts = random.randint(3, 6)   # v2: 最多6边形
            size    = random.uniform(12, 45)
            verts   = []
            for j in range(n_verts):
                a_ang = 2 * math.pi * j / n_verts + random.uniform(-0.3, 0.3)
                r     = size * random.uniform(0.6, 1.0)
                pvx   = max(10, min(SPACE_WIDTH  - 10, cx + r * math.cos(a_ang)))
                pvy   = max(10, min(SPACE_HEIGHT - 10, cy + r * math.sin(a_ang)))
                verts.append((pvx, pvy))
            scene_config['obstacles'].append({
                'type':     'polygon',
                'vertices': verts,
            })

        elif obs_type == 'ramp':
            # v2 新增：斜坡（固定角度线段）
            rx, ry     = _random_position(margin=40)
            ramp_len   = random.uniform(60, 200)
            ramp_angle = random.uniform(-60, 60)  # -60°~+60° 倾斜
            scene_config['obstacles'].append({
                'type':      'ramp',
                'x':         rx,
                'y':         ry,
                'length':    ramp_len,
                'angle_deg': ramp_angle,
            })

        else:  # circle_pin
            # v2 新增：圆柱障碍物
            cx, cy = _random_position(margin=40)
            radius = random.uniform(10, 35)
            scene_config['obstacles'].append({
                'type':   'circle_pin',
                'x':      cx,
                'y':      cy,
                'radius': radius,
            })

    # ---- 外力配置（v2：多物体 & 变向推力）----
    if random.random() < FORCE_PROBABILITY:
        # 决定变向推力间隔（0 = 不变向，>0 = 每隔 N 步翻转）
        if random.random() < 0.4:
            scene_config['reverse_steps'] = random.choice([20, 30, 50])

        # 决定是否对多个物体施力
        force_targets = [random.randint(0, num_objects - 1)]
        if num_objects > 1 and random.random() < MULTI_FORCE_PROB:
            # 随机再选另一个不重复的物体
            others = [i for i in range(num_objects) if i != force_targets[0]]
            force_targets.append(random.choice(others))

        for target_idx in force_targets:
            force_angle = random.uniform(0, 2 * math.pi)
            force_mag   = random.uniform(*FORCE_MAGNITUDE_RANGE)
            scene_config['forces'].append({
                'object_index': target_idx,
                'force':        (force_mag * math.cos(force_angle),
                                 force_mag * math.sin(force_angle)),
            })

    return scene_config


# ============================================================
# 物理模拟（v2：支持变向推力）
# ============================================================

def run_simulation(scene_config):
    """根据场景配置执行物理模拟，返回轨迹数组。

    v2 改进：
      - 支持 ramp / circle_pin 障碍物渲染
      - 支持变向推力（reverse_steps）

    参数:
        scene_config: generate_scene() 返回的场景配置字典

    返回:
        trajectory: np.ndarray，形状为 (SEQ_LENGTH, STATE_DIM)
                    STATE_DIM = 18（3 × [x,y,angle,vx,vy,ω]）
    """
    STATE_DIM = STATE_DIM_PER_OBJ * MAX_OBJECTS  # 18

    space         = pymunk.Space()
    space.gravity = (0, -900)
    _add_walls(space)

    # 添加动态物体
    dynamic_bodies = []
    for obj_cfg in scene_config['objects']:
        if obj_cfg['type'] == 'ball':
            body, _ = _add_ball(
                space,
                obj_cfg['position'][0], obj_cfg['position'][1],
                obj_cfg['radius'], obj_cfg['mass'],
            )
        else:
            body, _ = _add_box(
                space,
                obj_cfg['position'][0], obj_cfg['position'][1],
                obj_cfg['size'], obj_cfg['mass'],
            )
        body.angle            = obj_cfg['angle']
        body.velocity         = obj_cfg['velocity']
        body.angular_velocity = obj_cfg['angular_velocity']
        dynamic_bodies.append(body)

    # 添加静态障碍物（v2：支持 ramp & circle_pin）
    for obs_cfg in scene_config['obstacles']:
        if obs_cfg['type'] == 'segment':
            _add_segment_obstacle(space, obs_cfg['a'], obs_cfg['b'], obs_cfg['radius'])
        elif obs_cfg['type'] == 'polygon':
            _add_polygon_obstacle(space, obs_cfg['vertices'])
        elif obs_cfg['type'] == 'ramp':
            _add_ramp_obstacle(
                space,
                obs_cfg['x'], obs_cfg['y'],
                obs_cfg['length'], obs_cfg['angle_deg'],
            )
        elif obs_cfg['type'] == 'circle_pin':
            _add_circle_pin_obstacle(space, obs_cfg['x'], obs_cfg['y'], obs_cfg['radius'])

    force_list    = scene_config.get('forces', [])
    reverse_steps = scene_config.get('reverse_steps', 0)
    # 记录每个外力的当前方向符号（+1 或 -1）
    force_signs   = [1] * len(force_list)

    trajectory  = np.zeros((SEQ_LENGTH, STATE_DIM), dtype=np.float32)
    action_trajectory = np.zeros((SEQ_LENGTH, MAX_OBJECTS * 2), dtype=np.float32)
    sample_idx  = 0
    total_steps = int(DURATION / PHYSICS_DT)  # 1000

    for step in range(total_steps):
        # 变向推力：每隔 reverse_steps 翻转外力方向
        if reverse_steps > 0 and step > 0 and step % reverse_steps == 0:
            force_signs = [-s for s in force_signs]

        current_actions = np.zeros(MAX_OBJECTS * 2, dtype=np.float32)
        # 施加外力并记录动作向量
        for fi, force_cfg in enumerate(force_list):
            obj_idx = force_cfg['object_index']
            if obj_idx < len(dynamic_bodies):
                fx_base, fy_base = force_cfg['force']
                sign = force_signs[fi]
                fx, fy = fx_base * sign, fy_base * sign
                dynamic_bodies[obj_idx].apply_force_at_world_point(
                    (fx, fy),
                    dynamic_bodies[obj_idx].position,
                )
                if obj_idx < MAX_OBJECTS:
                    current_actions[obj_idx * 2 + 0] = fx
                    current_actions[obj_idx * 2 + 1] = fy

        space.step(PHYSICS_DT)

        if (step + 1) % SAMPLE_INTERVAL == 0:
            state = np.zeros(STATE_DIM, dtype=np.float32)
            for i, body in enumerate(dynamic_bodies):
                off = i * STATE_DIM_PER_OBJ
                state[off + 0] = body.position.x
                state[off + 1] = body.position.y
                state[off + 2] = body.angle
                state[off + 3] = body.velocity.x
                state[off + 4] = body.velocity.y
                state[off + 5] = body.angular_velocity
            trajectory[sample_idx] = state
            action_trajectory[sample_idx] = current_actions
            sample_idx += 1

    return trajectory, action_trajectory


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：批量生成轨迹数据并保存（v2）。

    v2 改进：
      - 支持 --episodes 命令行参数，覆盖默认的 50,000 片段数
      - 每 1,000 片段打印一次进度（而非每 100 次）
      - 保存元数据时记录 v2 新增的场景类型
    """
    parser = argparse.ArgumentParser(description='2D 物理轨迹数据生成器 v2')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                        help=f'生成片段总数（默认 {NUM_EPISODES}）')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'随机种子（默认 {SEED}）')
    parser.add_argument('--output', type=str, default='trajectories.npy',
                        help='输出文件路径（默认 trajectories.npy）')
    args = parser.parse_args()

    num_episodes = args.episodes
    seed         = args.seed
    output_path  = args.output

    STATE_DIM = STATE_DIM_PER_OBJ * MAX_OBJECTS

    print(f"2D 物理轨迹数据生成器 v2")
    print(f"  片段数:    {num_episodes:,}")
    print(f"  空间大小:  {SPACE_WIDTH}x{SPACE_HEIGHT}")
    print(f"  片段时长:  {DURATION}s @ {1.0/(PHYSICS_DT*SAMPLE_INTERVAL):.0f}Hz")
    print(f"  序列长度:  {SEQ_LENGTH}  状态维度: {STATE_DIM}")
    print(f"  输出形状:  ({num_episodes}, {SEQ_LENGTH}, {STATE_DIM})")
    print(f"  随机种子:  {seed}")
    print(f"  障碍物类型: segment / polygon / ramp / circle_pin")
    print(f"  外力系统:  单物体+多物体+变向推力")
    print()

    random.seed(seed)
    np.random.seed(seed)

    ACTION_DIM = MAX_OBJECTS * 2  # 6 维 (3 物体 × [fx, fy])

    all_trajectories = np.zeros(
        (num_episodes, SEQ_LENGTH, STATE_DIM), dtype=np.float32
    )
    all_actions = np.zeros(
        (num_episodes, SEQ_LENGTH, ACTION_DIM), dtype=np.float32
    )

    for ep in range(num_episodes):
        scene_config                      = generate_scene()
        traj, act                         = run_simulation(scene_config)
        all_trajectories[ep]              = traj
        all_actions[ep]                   = act

        if (ep + 1) % 1000 == 0:
            pct = (ep + 1) / num_episodes * 100
            print(f"  [{pct:5.1f}%] 已完成 {ep + 1:,}/{num_episodes:,} 个片段")

    # 保存旧版 trajectories.npy 格式以保持向下兼容
    np.save(output_path, all_trajectories)
    print(f"\n[OK] 旧版轨迹数据已保存至 {output_path} ({all_trajectories.nbytes / (1024**2):.1f} MB)")

    # 保存新版带动作数据的 trajectories_v2.npz 格式
    v2_path = "trajectories_v2.npz"
    np.savez_compressed(v2_path, states=all_trajectories, actions=all_actions)
    v2_size = os.path.getsize(v2_path) / (1024**2)
    print(f"[OK] 带动作通道路径 v2 数据集已保存至 {v2_path}")
    print(f"   states 形状:  {all_trajectories.shape}")
    print(f"   actions 形状: {all_actions.shape}")
    print(f"   压缩文件大小: {v2_size:.1f} MB")

    # 保存元数据
    meta_path = output_path.replace('.npy', '_metadata.json')
    metadata  = {
        'version':            'v2',
        'description':        '2D物理模拟轨迹数据集（v2扩展版），用于世界模型训练',
        'space_size':         [SPACE_WIDTH, SPACE_HEIGHT],
        'num_episodes':       num_episodes,
        'sequence_length':    SEQ_LENGTH,
        'state_dim':          STATE_DIM,
        'state_dim_per_obj':  STATE_DIM_PER_OBJ,
        'max_objects':        MAX_OBJECTS,
        'state_format':       '[x, y, angle, velocity_x, velocity_y, angular_velocity] × 3',
        'padding':            '不足3个物体的槽位用零填充',
        'padding_warning':    '零填充与原点静止状态歧义，模型依赖上下文区分',
        'duration_seconds':   DURATION,
        'physics_dt':         PHYSICS_DT,
        'sample_interval':    SAMPLE_INTERVAL,
        'sample_rate_hz':     1.0 / (PHYSICS_DT * SAMPLE_INTERVAL),
        'gravity':            [0, -900],
        'object_types':       ['ball', 'box'],
        'obstacle_types':     ['segment', 'polygon', 'ramp', 'circle_pin'],
        'force_system':       'single / multi-body / reversing impulse',
        'random_seed':        seed,
        'velocity_range':     list(VELOCITY_RANGE),
        'body_mass_range':    list(BODY_MASS_RANGE),
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"   元数据已保存至 {meta_path} 及 metadata.json")


if __name__ == '__main__':
    main()
