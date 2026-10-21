# 内部文档已移出投稿包

以下文件**不随 `TMLR_submission.zip` 发布**，位于仓库根的 `internal/`：

| 文件 | 内容 |
|---|---|
| `internal/PEER_REVIEW.md` | 六轮内部评审与修复记录（含论文自相矛盾条目、三次自我推翻、一次我自己引入的结构 bug） |
| `internal/QC_LOG.md` | composer 逐章质量门记录 |
| `internal/CLAIM_AUDIT_2026-10-10.md` | 全文"声称 vs 实物"终检报告：60+ 条通过项、5 处已修问题、机检覆盖边界 |
| `internal/E5_HEAD_VALUE.md` | E5 实验记录（阶段 1 的 GO/NO-GO）：残差头只在"先验错法可学"时有用；含我自己的两次口径错误 |
| `internal/E5_M2_TRUNK_ABLATION.md` | M2 trunk 消融：增益归**头**（隐式参数辨识），trunk 不需编码失配参数 |
| `internal/E5_M2C_GATE.md` | M2c 解码门交互：门在"头有用"的世界里单调有害，故门是**有条件的对策**而非通用改进 |
| `internal/PRESUBMIT_AUDIT_2026-10-11.md` | 投稿前终检：手写交叉引用/编号/自述计数全部对账；含「重排≠换主线」的方法论教训 |
| `internal/M4_IN_PROGRESS.md` | M4 多 seed 检验（进行中）：先验阶梯 4 档×seeds 45-48 + E5 消融×seeds 45-48；含判读规则 |
| `internal/M3_FLOOR_SCOPE.md` | M3：地板是 **regime 失衡**的属性而非架构属性；推翻论文"摆世界头学会静默"的说法，并给出可判定判据 |

移出原因：这些文档含"当前版本不可直接投稿""论文自相矛盾"等**只对作者有意义**的措辞，
以及未成熟的自我批评，不适合随投稿包交给审稿人。

对外自述仍在投稿包内：`docs/README.md`、`PREREG_CRITERIA_EN.md`、`provenance/`、`floor_analysis/`。

投稿包构建脚本 `paper/build_submission.py` 已移除对这两个文件的引用；
重新运行 `python paper/build_submission.py` 不会再打包它们。
