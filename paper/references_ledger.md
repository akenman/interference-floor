# 引用台账（[OK]/[WARN] 政策）

政策（继承自项目铁律 9 与 PAPER_DRAFT v0.2）：
[OK] = 已核实存在（检索命中或摘要/全文已读）；[WARN] = 仅见摘要/片段，**进入正式 bib 前必须精读原文**；◇ = 主题占位（正式稿中禁止出现）。

| key | 条目 | 状态 | 备注 |
|---|---|---|---|
| ani2026 | Alternating Neural Integrators. Nat. Commun. 2026 | [OK] 全文扫描 | 最近邻（方法侧）；区分语在 §2A |
| pinwm2025 | PIN-WM. RSS 2025 / arXiv:2504.16693 | [OK] 摘要 | 视觉→刚体辨识方法 |
| physcore2026 | PhysCoRe. arXiv:2607.20653 | [OK] 全作者对拍 2026-10-06（Yin, Tao, Chen, Gan） | 标题近邻；§2A 区分语 |
| neuralsim2021 | NeuralSim. ICRA 2021 | [OK] 检索 | |
| tossingbot2019 | TossingBot. RSS 2019 | [OK] 检索 | |
| hgn2019 | Hamiltonian Generative Networks. arXiv:1909.13789 | [OK] 检索 | |
| chnn2020 | CHNN. arXiv:2010.13581 | [OK] 检索 | |
| se3hode2021 | SE(3)-Hamiltonian NODE. RSS 2021 | [OK] 检索 | |
| swazinna2021 | Overcoming model bias. Eng. Appl. AI 2021 | [OK] 检索 | |
| lyle2021 | Capacity loss. ICML 2021 | [OK]（前期会话核实） | |
| metabalance2022 | MetaBalance. WWW 2022（DOI 10.1145/3485447.3512093） | [OK] 检索（2026-10-07 修正场馆：ACM Web Conference，非 KDD） | |
| deepseekv3 | DeepSeek-V3 Technical Report. arXiv:2412.19437 | [OK] 检索 | |
| auxlossfree2024 | Aux-Loss-Free Load Balancing. arXiv:2408.15664 | [OK] 检索 | |
| kronos2026 | Fusion abstention / deterministic floor. Zenodo 2026 | [FAIL] 已删除（2026-10-06）：Zenodo DOI 10.5281/zenodo.22645812 返回 404，精确标题检索无结果，无法核实存在；原脚本硬编码 OK 亦已移除 | 原 §2E 区分语已由 wabersich2021/ames2019 承担 |
| nbf2025 | Neural Bayesian Filtering. arXiv:2510.03614 | [OK] 摘要（[WARN] 精读后引 future-work） | 论文 #3 竞争者 |
| pderefiner | PDE-Refiner. NeurIPS 2023 (arXiv:2308.05732), Lippe et al. | [OK] 摘要核实 2026-10-04 | 标题实为 "Accurate Long Rollouts"（非 Stable）——draft 引用语与之一致 |
| mbpo2019 | When to Trust Your Model: MBPO. NeurIPS 2019 (arXiv:1906.08253), Janner et al. | [OK] 摘要核实 | model-usage/bias 主题与引用语一致 |
| dreamerv3_2025 | Mastering diverse control tasks through world models. Nature 2025 (s41586-025-08744-2), Hafner et al. | [OK] 检索核实 | GRU-RSSM 架构先例，§8 防御位 |
| colas2018 | How Many Random Seeds?. arXiv:1806.08295, Colas, Sigaud, Oudeyer | [OK] 摘要核实 2026-10-04 | §3.4/§9 统计功效引用 |
| wm_survey2026 | World Model for Robot Learning survey. arXiv:2605.00080 | [OK] 摘要+部分全文 | §1/§2 定位语 |
| wabersich2021 | Predictive safety filter. Automatica 2021 / arXiv:1812.05506 | [OK] arXiv 页核实 2026-10-06 | §2E 区分语（safety filter 正主） |
| ames2019 | Control Barrier Functions: Theory and Applications. ECC 2019 / arXiv:1903.11199 | [OK] arXiv 页核实 2026-10-06 | §2E 区分语（CBF 正主） |
| ljung1999 | System Identification: Theory for the User, 2nd ed. Prentice Hall 1999 | [OK] 经典教材（非 arXiv，不在自动审计范围内） | §2F/§4.3 离散化偏差的经典定位 |
| pymunk2025 | Pymunk Advanced 文档（"uses standard Euler"，位置先于速度） | [OK] 2026-10-07 全文核对原文 | §3.1 N7 修正的依据 |
| spong1994 | Acrobot swing-up control | ◇ 未核实 | 精读后入 §7 或删 |

> TMLR 政策提醒：正文引用与 bib 必须一一对应；[WARN] 条目未经精读不得出现在提交版。
