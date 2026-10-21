# Review Response Plan — 逐条判定与行动计划

## 逐条判定

| # | 审稿人观点 | 判定 | 需要做什么 | 工作量 |
|---|---|---|---|---|
| 1 | 标题/摘要泛化过强 | **有效，写作修正** | 收窄标题+摘要限定语+贡献列表加 scope 前缀 | 0.5h 写作 |
| 2 | 匹配积分器近乎循环论证 | **部分有效，可反驳+补** | 反驳：本文贡献是量化失配的代价（未知→已测量），非"怎么匹配"；补：aliasing law 的检测流程草案（从数据选计算类的伪算法+失败模式）；承认真机验证缺失 | 1h 写作 |
| 3 | 统计力不足（~30 FF 窗、无 CI、噪声底不普适） | **有效，需补** | 补：逐臂 bootstrap CI + 混合效应模型报告 + FF 窗口数的诚实披露；噪声底标注"来自 mid-tier 估计，外推需验证" | 2h 分析+写作 |
| 4 | 解码规则报告不平衡（摘要像净赢） | **有效，写作修正** | 摘要加"at the cost of 12–21% contact-regime degradation"；§6.5 加 mixed-window 闭环数字和最差 seed | 1h 写作 |
| 5 | 符号结构 7/7 非独立证据 | **有效，写作修正** | §5.2 加"one replicated finding across four generations, not seven independent tests"；限定语"within this architecture family" | 0.5h |
| 6 | 干扰地板机制未理论化 | **有效，最重要的科学批评** | 至少一个机制消融：冻结头初始化 + 零初始化头 + 显式 silence 正则——看噪声是否消失 → 定位噪声来源（优化噪声 vs 梯度冲突 vs 表示瓶颈） | 1 GPU·h 实验 + 2h 写作 |
| 7 | 40/36/64/27 数字不一致 | **有效，文书修正** | 统一：正文给最终快照数（40 tests, 64 provenance）；附录标"27 = 主表子集"；τ 网格变化说明"从 0.01 收到 0.05 不影响 τ* 位置（已验证）" | 0.5h |
| 8 | 图表/写作问题 | **有效，文书修正** | Figure 2/3 标题拼写；Table 1 格式；术语首次出现处正式定义 | 1h |

## 优先级排序

**投稿前必须做（阻塞项）**：
- #1 收窄标题/摘要/贡献（不改 = 桌拒风险）
- #7 统一数字（不改 = 可复现性自相矛盾）
- #8 图表拼写和格式

**强烈建议做（提升过稿率的关键）**：
- #3 补 bootstrap CI + FF 窗口数披露
- #4 解码规则摘要措辞修正
- #5 结构减分的限定语
- #6 至少一个机制消融（这是审稿人说"as a reviewer, I would like to see"的明确信号）

**可反驳（不需要新实验，写 response 即可）**：
- #2 循环论证（本文贡献是测量代价，不是匹配方法）
- #5 的"七个独立证据"部分（我们已诚实声明这不是七个独立测试）

## 修订后的标题建议

当前：*Integrator Accuracy, Not Structure: What Determines Closed-Loop Feasibility in Dynamics-Level World Models*

审稿人建议的方向：收窄为受控实验结论。

**修订选项 A（最小改动）**：
*Integrator Accuracy, Not Structure: A Controlled Study of Closed-Loop Feasibility in Simulated Dynamics World Models*

**修订选项 B（更诚实）**：
*Prior Computation-Class Fidelity Sets a Closed-Loop Floor in Simulated Dynamics World Models: A Preregistered Attribution Study*

**修订选项 C（保留 punch，加 scope）**：
*Integrator Accuracy, Not Structure: What Sets the Closed-Loop Floor of Residual World Models*

选项 C 推荐：保留 punch，"Residual World Models"比"Dynamics-Level World Models"更精确（我们研究的是残差架构，不是所有类型的世界模型），且不需要 "Simulated" 前缀（Limitations 已覆盖）。

## 核心修改：贡献列表加 scope 前缀

当前 C1：Prior computational fidelity, not architecture, sets closed-loop feasibility.

修订 C1：In the two simulated worlds studied here, prior computational fidelity — not architecture — sets the closed-loop floor: a zero-parameter prior with the engine's integrator is bit-exact in closed loop and ties a 401k-parameter learned model.

（每条贡献前加 "In the two simulated worlds studied here," 或等价限定。）
