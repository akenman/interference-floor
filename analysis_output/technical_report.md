# 观象网络世界模型训练技术报告

## 1. 项目概述

### 1.1 目标
构建一个基于"意象涌现"（Image Emergence）的世界模型，将连续物理状态离散化为有限意象原型（prototype），并学习意象之间的流转规律，实现对物理世界的预测、重构与事件分类。

### 1.2 核心思想
- **观象网络（GuanXiangEncoder）**：将历史状态序列编码为意象分布 z_img（Gumbel-Softmax 概率向量），同时输出 L2 归一化的上下文向量 c_norm
- **意象流转预测器（YiXiangPredictor）**：根据当前意象分布预测下一时刻的意象分布
- **意象重构解码器（YiXiangDecoder）**：从意象分布重构原始物理状态序列
- **事件分类头（EventClassifier）**：从上下文向量预测物理事件类别（静止/飞行/碰撞）

### 1.3 硬件环境
- GPU: NVIDIA RTX 5060 Laptop (16GB VRAM)
- 训练框架: PyTorch + AMP (自动混合精度)
- 总训练时间: ~250 epochs

---

## 2. 模型架构

### 2.1 整体结构 (WorldModel)

```
Input (B, 4, 18)
    │
    ▼
GuanXiangEncoder
    ├── LSTM(18→128) + LayerNorm + L2 Norm → c_norm (B, 128)
    ├── Prototypes (64×128, 正交初始化, L2归一化)
    ├── Cosine Similarity: c_norm @ Prototypes.T → logits (B, 64)
    └── Gumbel-Softmax(temperature, hard) → z_img (B, 64)
    │
    ├──→ YiXiangPredictor: z_img → z_img_pred (B, 64)
    │       LayerNorm → Linear(64→64) → GELU → Dropout → Linear(64→64) + Residual → Softmax
    │
    ├──→ YiXiangDecoder: z_img → recon_seq (B, 4, 18)
    │       prototype_states (64×4×18) 查表 + MLP残差(64→256→256→72)
    │
    └──→ EventClassifier: c_norm → event_logits (B, 3)
            Linear(128→64) → ReLU → Dropout → Linear(64→3)
```

### 2.2 关键设计

| 组件 | 设计要点 | 维度 |
|------|----------|------|
| LSTM编码器 | 单层, batch_first | input=18, hidden=128 |
| 原型矩阵 | 正交初始化, L2归一化, 可学习 | (64, 128) |
| 温度参数 | register_buffer, 退火策略 | 标量 |
| Gumbel-Softmax | ST估计器(hard=True): 前向one-hot, 反向soft梯度 | (B, 64) |
| 预测器 | 残差连接 + LayerNorm + GELU + Dropout(0.1) | (B, 64) |
| 解码器 | prototype_states查表 + MLP残差(skip connection) | (B, 4, 18) |
| 事件分类 | 2层MLP, CrossEntropyLoss | (B, 3) |

### 2.3 参数量
- 总可训练参数: ~200K (encoder LSTM + prototypes + predictor + decoder + event_classifier)

---

## 3. 训练数据

### 3.1 数据来源
- 文件: `trajectories.npy`
- 形状: (10000, 100, 18) — 10000条轨迹, 每条100时间步, 每步18维状态

### 3.2 状态向量布局
3个球体, 每个球体6维: [x, y, vx, vy, angle, angular_vel]
- 位置维度(x,y): 大方差, 大数值范围
- 速度维度(vx,vy): 中等方差
- 角度/角速度: 小方差

### 3.3 数据预处理
- **Per-dimension归一化**: (x - dim_mean) / dim_std, 每个维度独立归一化到std≈1
- **滑动窗口**: input_len=4, 生成 (input_seq[0:4], target_seq[1:5]) 样本对
- **总样本数**: 10000 × (100-4) = 960000
- **训练/验证集**: 90%/10% 随机划分

### 3.4 事件标签分布
| 事件 | 数量 | 占比 |
|------|------|------|
| 静止(0) | 12534 | 1.3% |
| 飞行(1) | 717997 | 74.8% |
| 碰撞(2) | 229469 | 23.9% |

**严重类别不平衡**: 飞行占75%, 静止仅1.3%

---

## 4. 三阶段课程学习策略

### 4.1 Phase 1 (Epoch 0~50): 原型分离 + 重构引导

**目标**: 让64个原型在隐空间中充分分离，建立"原型→物理状态"的映射

| 配置项 | 值 |
|--------|-----|
| 温度 | 0.5 (固定) |
| hard | True (ST估计器) |
| 编码器学习率 | 1e-4 |
| 解码器学习率 | 1e-2 (100x加速) |
| 梯度裁剪 | encoder=5.0, decoder=1.0 |

**损失函数**:
```
total = 0.3 * contrastive_loss(proto.detach)  # 推c向proto
      + 1.0 * recon_loss                       # 主驱动: z编码物理信息
      + 0.5 * ortho_loss(max_penalty=1.0)      # 推proto互相分离
      + 0.5 * commit_loss                       # c和proto双向收敛
```

**关键机制**:
- commit_loss: `(1 - cos(c, assigned_proto))`, 无detach, 双向梯度
- ortho_loss: Frobenius距离 + max惩罚, 防止collapse
- 死原型k-means重启: 每5epoch检测, 重置到离活跃原型最远的c方向
- QR硬重正交化兜底: 当原型相似度>0.5时强制正交化

### 4.2 Phase 2 (Epoch 51~150): 意象对齐与流转学习

**目标**: 学习意象间的流转规律，同时保持原型分离

| 配置项 | 值 |
|--------|-----|
| 温度 | 1.0 → 0.3 线性衰减 |
| hard | False |
| 学习率 | 1e-3 (全部参数) |
| lambda_pred_grad | T>0.5→0.05, 0.3<T≤0.5→0.1, T≤0.3→0.2 |

**损失函数**:
```
total = lambda_pred * pred_loss
      + 0.1 * recon_loss
      + 0.02 * entropy_reg
      + 1.0 * ortho_loss(max_penalty=0.5)
      + 0.1 * event_loss
      + 0.05 * smooth_loss
      + 0.02 * trans_entropy_loss
      + 0.1 * trans_contrastive_loss
```

**新增损失**:
- 事件分类损失: CrossEntropy(event_logits, event_labels)
- 时间平滑损失: ||z(t+1) - z(t)||², 鼓励相邻帧表示平滑
- 转移熵损失: -H(predictor(z, future_steps=3)), 鼓励预测多样性
- 转移对比损失: InfoNCE(z_t, z_t1_positive, z_t1_negative)

**AMP渐进启用**: Phase 2前3个epoch使用FP32, 避免float16溢出

### 4.3 Phase 3 (Epoch 151~250): 精调

**目标**: 在低温度、hard编码下精调所有组件

| 配置项 | 值 |
|--------|-----|
| 温度 | 0.3 (固定) |
| hard | True (ST估计器) |
| 学习率 | 1e-4 (全部参数) |
| EMA | decay=0.999, 从Phase 3开始 |

**损失函数**:
```
total = 0.2 * pred_loss
      + 0.1 * recon_loss
      + 0.01 * entropy_reg
      + 0.5 * ortho_loss(max_penalty=0.5)
      + 0.1 * event_loss
      + 0.05 * smooth_loss
      + 0.02 * trans_entropy_loss
      + 0.1 * trans_contrastive_loss
```

---

## 5. 训练结果

### 5.1 阶段边界关键指标

| 指标 | Phase1开始 | Phase1结束 | Phase2中期 | Phase2结束 | Phase3中期 | Phase3结束 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| TotalLoss | 4.153 | 2.324 | 0.183 | 0.788 | 2.534 | 2.533 |
| Sample_SimMax | 0.405 | 0.240 | 0.192 | 0.257 | 0.347 | 0.339 |
| Proto_SimMax | 0.038 | 0.001 | 0.002 | 0.002 | 0.141 | 0.133 |
| Avg_Entropy | 2.708 | 2.708 | 1.725 | 0.626 | 0.000 | 0.000 |
| ReconLoss | 0.551 | 0.968 | 0.968 | 0.968 | 0.968 | 0.968 |
| Proto_PPL | 63.57 | 63.45 | 63.18 | 63.00 | 62.02 | 62.11 |
| EncGradNorm | 0.042 | 0.295 | 0.305 | 0.305 | 0.304 | 0.302 |
| AlignLoss | 0.671 | 0.878 | 0.000 | 0.000 | 0.000 | 0.000 |

### 5.2 训练曲线分析

#### 5.2.1 总损失 (Train/TotalLoss)
- Phase 1: 4.15 → 2.32, 稳步下降
- Phase 2: 2.32 → 0.18 (中期) → 0.79 (末期), 先降后升（新增损失引入）
- Phase 3: 0.79 → 2.53, 上升后稳定（hard编码+低温度导致分布尖锐，损失尺度变化）

#### 5.2.2 样本-原型对齐 (Sample_SimMax)
- Phase 1: 0.405 → 0.240, **下降**（原型分离过程中c与最近原型的余弦相似度降低）
- Phase 2: 0.240 → 0.192 → 0.257, 先降后升
- Phase 3: 0.257 → 0.339, **上升**（hard编码下对齐改善）
- **最终值0.339偏低**, 理想值应>0.5

#### 5.2.3 原型间相似度 (Proto_Similarity_Max)
- Phase 1: 0.038 → 0.001, 原型充分分离（正交化成功）
- Phase 2: 稳定在0.002, 保持分离
- Phase 3: **0.002 → 0.133, 显著上升**（正交化约束放松后原型开始聚集）
- **0.133仍可接受**（<0.5阈值），但趋势需警惕

#### 5.2.4 意象分布熵 (Avg_Entropy)
- Phase 1: ~2.708 (接近log(64)/2≈2.08, 分布较均匀)
- Phase 2: 2.708 → 0.626, 温度退火使分布逐渐尖锐
- Phase 3: 0.626 → 0.000, **完全坍缩到one-hot**

#### 5.2.5 重构损失 (Loss/reconstruction)
- Phase 1: 0.551 → 0.968, **上升**（decoder学习初期）
- Phase 2/3: 稳定在0.968, **几乎不变**
- 重构MSE在归一化空间≈0.97, 原始空间≈15282, **重构质量差**

#### 5.2.6 事件分类 (Accuracy/Event)
- Phase 2: 0.943 → 0.999, 训练集准确率极高
- **但评估集总准确率仅0.232**, 严重过拟合

#### 5.2.7 预测器梯度 (PredictorGradNorm)
- Phase 1: 0.000（预测器不参与Phase 1训练）
- Phase 2: 0.000 → 0.002, 逐步激活
- Phase 3: 0.002 → 0.083, 梯度增强但仍然很小

#### 5.2.8 梯度跳过 (GradSkipCount)
- Phase 1/2: 几乎为0
- Phase 3: epoch 170/220/230/240/249出现1次跳过, 梯度偶有爆炸

#### 5.2.9 ST估计器指标
- Gini系数: 0.009 → 0.079 (Phase 1结束时), 分布逐渐不均匀
- 原型利用率: 100% (全部64个原型均被使用)

---

## 6. 模型评估结果 (EMA模型, Epoch 249)

### 6.1 对齐质量

| 指标 | 值 | 评价 |
|------|-----|------|
| SimMax均值 | 0.232 | 偏低, c与最近原型对齐不足 |
| SimMax标准差 | 0.036 | 较稳定 |
| SimMax最小值 | 0.106 | 存在严重未对齐样本 |
| Commit均值 | 0.971 | **极高**, c与被分配原型几乎正交 |
| 对齐一致性 | 0.018 | **极低**, Gumbel-Softmax选择与最近原型严重不一致 |

**关键问题**: SimMax=0.232 但 Commit=0.971, 对齐一致性仅1.8%。这意味着：
- c向量与最近原型的余弦相似度仅0.23（随机水平约1/√128≈0.088）
- 但commit_loss计算的是c与Gumbel-Softmax分配的原型的相似度，1-0.971=0.029说明c与分配原型余弦相似度仅0.029
- 一致性1.8%说明Gumbel-Softmax的选择几乎不等于最近原型选择
- **根因**: 温度0.3下Gumbel噪声仍然较大，导致采样偏离最近原型

### 6.2 预测能力

| 指标 | 值 |
|------|-----|
| 单步预测MSE | 0.0155 |
| 自回归10步MSE | 0.0301 |
| 非自回归10步MSE | 0.0302 |

**分析**:
- 单步MSE=0.0155在归一化空间中表现尚可
- 多步预测MSE在step 2后迅速饱和到0.030, 误差不再显著增长
- 自回归vs非自回归几乎一致, 说明预测器输出分布较稳定
- 但MSE在原始空间仍较大（归一化还原后）

### 6.3 重构质量

| 指标 | 值 |
|------|-----|
| 重构MSE | 15282.2 |
| 每维度MSE范围 | 3.65 ~ 125721.1 |

**每维度MSE分解** (3球×6维):
| 球体 | x | y | vx | vy | angle | angular_vel |
|------|-----|-----|-----|-----|-------|-------------|
| Ball0 | 125721 | 2736 | 94.0 | 2593 | 9432 | 9.5 |
| Ball1 | 80534 | 2052 | 58.9 | 1623 | 5129 | 5.8 |
| Ball2 | 40214 | 910 | 31.1 | 839 | 3093 | 3.7 |

**分析**:
- 位置维度(x)MSE极大(125K/80K/40K), 是重构误差的主要来源
- 速度维度(vx,vy)MSE中等(30~2600)
- 角速度维度MSE最小(3.7~9.5)
- **根因**: 位置数值范围大, per-dim归一化后MSE应≈1, 但还原到原始空间后被放大

### 6.4 事件分类

| 指标 | 值 |
|------|-----|
| 总准确率 | 23.2% |
| 静止准确率 | 0.0% |
| 飞行准确率 | 4.7% |
| 碰撞准确率 | 82.3% |

**分析**:
- 训练集准确率99.9%, 评估集23.2% → **严重过拟合**
- 碰撞类准确率82.3%: 碰撞特征明显(速度突变), 容易学习
- 飞行类准确率4.7%: 大部分飞行被误分为碰撞
- 静止类0%: 完全无法识别, 可能因为样本太少(1.3%)
- **根因**: 类别不平衡 + 分类头过拟合 + c_norm的表示能力不足以区分飞行/静止

### 6.5 意象激活统计

| 指标 | 值 |
|------|-----|
| 最活跃原型 | #58 |
| 最不活跃原型 | #8 |
| 激活次数标准差 | 707.5 |
| 零激活原型数 | 0 |

**分析**: 64个原型全部被使用(利用率100%), 但激活分布不均匀(std=707.5)

### 6.6 转移矩阵统计

| 指标 | 值 |
|------|-----|
| 总转移数 | 950,000 |
| 非零转移数 | 4,096 (=64×64) |
| 自环占比 | 1.56% |

**分析**: 所有64×64=4096种转移都被观测到, 转移矩阵完全稠密。自环占比1.56%偏低, 说明意象在相邻帧间几乎必然变化。

---

## 7. 已知问题与根因分析

### 7.1 c-原型对齐不足 (SimMax=0.232)

**现象**: 上下文向量c与最近原型的余弦相似度仅0.232, 远低于理想值(>0.5)

**根因**:
1. **L2归一化后的维度灾难**: 128维空间中, 随机向量的余弦相似度期望≈0, 标准差≈1/√128≈0.088。SimMax=0.232仅约2.6σ, 对齐信号微弱
2. **commit_loss与ortho_loss的冲突**: commit推c→proto, ortho推proto互相分离, 两者在Phase 1形成动态平衡
3. **Phase 2/3无commit_loss**: AlignLoss在Phase 2/3为0, c-原型对齐完全依赖间接信号(重构/预测)

### 7.2 重构质量差 (MSE=15282)

**现象**: 原始空间重构MSE极大, 位置维度误差占主导

**根因**:
1. **信息瓶颈**: 64维one-hot z_img → 4步×18维=72维输出, 信息严重不足
2. **prototype_states查表**: one-hot输入只能激活1个原型的标准状态, 无法表达连续变化
3. **位置维度数值范围大**: per-dim归一化后MSE≈1, 但std差异导致还原后位置维度被放大

### 7.3 事件分类过拟合 (训练99.9%, 评估23.2%)

**现象**: 训练集近完美, 评估集仅23.2%

**根因**:
1. **类别严重不平衡**: 飞行75% vs 静止1.3%, 模型偏向预测多数类
2. **分类头过参数化**: 128→64→3的MLP在c_norm表示能力有限时容易记忆训练集
3. **c_norm表示能力不足**: SimMax=0.232说明c本身对物理语义编码不足

### 7.4 Phase 3原型聚集 (Proto_SimMax: 0.002→0.133)

**现象**: Phase 3中原型间最大相似度从0.002升至0.133

**根因**:
1. **正交化权重降低**: Phase 2/3的max_penalty=0.5 vs Phase 1的1.0
2. **hard编码+低温度**: one-hot梯度信号使原型向高数据密度区域聚集
3. **预测损失驱动**: predictor倾向于让相似状态的原型靠近, 以减少预测误差

### 7.5 预测器梯度微弱 (PredictorGradNorm≈0.003~0.083)

**现象**: 预测器梯度范数远小于编码器(0.3)和解码器(0.001)

**根因**:
1. **lambda_pred_grad缩放**: Phase 2中pred_loss被0.05~0.2缩放
2. **Softmax饱和**: 低温度下z_img接近one-hot, softmax梯度消失
3. **残差连接**: predictor的残差连接使梯度主要走skip路径

---

## 8. 历史迭代与修复记录

### 8.1 Phase 1 停滞问题 (v9.3~v9.6)
- **症状**: SimMax下降(0.30→0.13), Align上升(0.45→0.62), EncGrad衰减(0.80→0.36)
- **根因**: 全局梯度裁剪max_norm=1.0过严 + 死原型重启副作用 + c和proto互追形成动态平衡
- **修复**: 分组梯度裁剪(encoder=5.0, decoder=1.0) + commit_loss双向梯度 + k-means死原型重启

### 8.2 EMA导致原型坍缩 (v9.8~v9.9)
- **症状**: EMA shadow拉向c的均值, 导致原型坍缩
- **修复**: Phase 3才启用EMA, 进入Phase 3时重新初始化shadow

### 8.3 AMP NaN/Inf (Phase 2初期)
- **症状**: Phase 2新增损失在float16下溢出
- **修复**: Phase 2前3个epoch禁用AMP(使用FP32), 渐进式启用

### 8.4 EMA加载缺少buffer键
- **症状**: 加载EMA模型时缺少`encoder.temperature`键
- **根因**: EMAHelper只跟踪`named_parameters()`, 不包含`register_buffer`
- **修复**: EMAHelper扩展为同时跟踪buffer; analyze.py先加载model_state_dict再覆盖EMA参数

### 8.5 梯度爆炸处理
- **症状**: scaler.update()在跳过梯度更新时增加scale factor
- **修复**: 手动scaler._scale *= scaler._backoff_factor后再scaler.update()

---

## 9. 超参数汇总

| 参数 | 值 |
|------|-----|
| 总epochs | 250 |
| batch_size | 512 |
| input_len | 4 |
| state_dim | 18 |
| hidden_dim | 128 |
| num_prototypes | 64 |
| init_temp | 1.0 |
| min_temp | 0.1 |
| Phase1 temp | 0.5 |
| Phase2 temp | 1.0→0.3 |
| Phase3 temp | 0.3 |
| EMA decay | 0.999 |
| Phase1 enc_lr | 1e-4 |
| Phase1 dec_lr | 1e-2 |
| Phase2 lr | 1e-3 |
| Phase3 lr | 1e-4 |
| GradScaler初始 | 65536 |
| 验证频率 | Phase1: 每5epoch, Phase2: 每10epoch, Phase3: 每10epoch |
| 检查点保存 | 每20epoch + latest + best + ema |

---

## 10. 待改进方向

### 10.1 提升c-原型对齐
- 增大原型维度(hidden_dim 128→256), 降低维度灾难
- Phase 2/3保留commit_loss或显式对齐损失
- 考虑VQ-VAE风格的codebook更新(EMA更新原型)

### 10.2 改善重构质量
- 增大解码器容量(hidden_dim 256→512)
- 使用top-k采样替代argmax, 允许混合原型
- 增加重构损失的权重

### 10.3 解决事件分类过拟合
- 对静止类过采样或使用focal loss
- 增加Dropout(0.1→0.3)
- 使用c_norm + z_img拼接作为分类输入

### 10.4 防止Phase 3原型聚集
- 增大Phase 3正交化权重(0.5→1.0)
- 或使用余弦相似度惩罚替代Frobenius距离

### 10.5 增强预测器
- 增大lambda_pred_grad
- 使用Transformer替代MLP预测器
- 增加预测步数和多样性损失
