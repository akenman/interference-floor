# Paper Project — TMLR Submission (v2 draft)

**Manuscript**: `latex/main.tex`（TMLR 投稿版；`main_draft.md` 为 v1 初稿存档，
`latex/main.tex.preM5` 为 M5 重排前备份，可回退）
**Outline**: `../TMLR_WRITING_FRAMEWORK.md`（composer 的 required input）
**Composer skill**: academic-paper-composer（逐章写作 + 质量门 + 跨章一致性）

## 当前口径（2026-10-16）

| 项 | 实测值 |
|---|---|
| `main.tex` | 97,117 B |
| PDF | **28 页**，0 错误 / 0 未定义引用 / 0 overfull |
| 数字溯源守卫 | **319/319 PASS** |
| 守卫测试 | **43 passed** |
| checkpoint 清单 | **289 个 / 17 个根** |
| `TMLR_submission.zip` | **231 项 / 930 KiB** |
| 预注册 | 12 份 |
| 引用 | 23 条（正文引用与 bib 一一对应，无缺无余）；`citation_audit.py` 抽核 17 条 0 错误 |
| 先验阶梯原始 / 扩展对照 | `tab_ladder_extension.tex`：原始 seed 42–44 对扩展 42–51（扩展复用原 3 seed，非独立复制） |

## Progress Tracker

章节顺序是 **M5 重排之后**的（干扰地板提为主线，先验阶梯降为前提）：
§1 Intro → §2 Related → §3 Platform → §4 先验阶梯（前提）→ **§5 干扰地板（主贡献）**
→ §6 结构/配方（次级）→ §7 Discussion → §8 Limitations → §9 Reproducibility
→ 附录 A 负结果目录 / B 溯源 / C 逐窗口静默代价 / D 预注册判据总表。

- [x] §1 Introduction
- [x] §2 Related Work（机制假设标注 hypothesis；未定性处写 not established）
- [x] §3 Experimental Platform
- [x] §4 The Prior Ladder Sets the Floor（含 §4.3 aliasing law）
- [x] §5 The Interference Floor and the Cost of Silence（含 §5.8 修正器的适用边界）
- [x] §6 Structure and Recipe Cannot Substitute
- [x] §7 Discussion
- [x] §8 Limitations
- [x] §9 Reproducibility Statement
- [x] Abstract（三段：现象 → 机制 → 边界；当前约 382 词）
- [x] 贡献重排为 C1–C6；正文 §N 自称一律走 `\S\ref{}`
- [x] Fig 1–8 英文重绘脚本（数字全部取自 JSON）
- [x] LaTeX/TMLR 模板合稿（`latex/main.tex` + `provenance.tex`）
- [x] 引用 [OK]→正式 bib（`audit_citations.py`：17 条 0 差异；台账 [WARN] 条目已处理）

## 投稿包与守卫

- `python paper/build_pdf.py [--clean]` → 编译 `paper/latex/main.pdf`
  （需 MiKTeX/TeX Live；当前 **28 页、0 错误、0 未定义引用、0 overfull**）
- `python paper/build_submission.py` → 重建 `TMLR_submission.zip`
  （稿件 + 表格 tex + 12 份预注册 + 英文判据表 + 守卫脚本与 43 个测试 + 全部 JSON 报告；
  **显式排除** `internal/` 下的内部文档）
- `python paper/verify_numbers.py` → **319/319 PASS**；`python -m pytest tests -q` → **43 passed**
- `python paper/check_latex.py` → 环境配对 / `$` 奇偶 / 引用↔bib / `\ref`↔`\label` / 转义残留
- `python paper/presubmit_audit.py` → 投稿前终检（手写 §N/Table N/C# 越界、生成物一致性、
  自述计数、旧自称残留；M5 重排最容易留下的那类错）
- `python paper/figures/make_figures.py` → 图 1–5
- `python paper/figures/make_fig_floor.py` → Fig 3 / 6 / 7
- `python paper/tables/gen_provenance.py` → 附录 B（自动计数）
- `python paper/tables/build_arms_table.py` → `tab_arms.tex`（臂清单）
- `python paper/tables/build_c3_table.py` → `tab_c3_gating.tex`（四代实验证据表）
- `python paper/tables/build_criteria_table.py` → `tab_criteria.tex`
  （47 行判据总表：23 pass / 19 fail / 1 partial / 3 report-only / 1 not stored）
- `python paper/tables/build_floor_analysis.py` → `floor_analysis.json`（§5.2 与 Fig 3）
- `python paper/tables/build_silence_hist.py` → 逐窗口误差直方图（附录 C + Fig 7；
  复用 S1f 评估路径并断言与 `S1f_report.json` 逐位一致）
- `python paper/tables/build_m4_multiseed.py` → M4 汇总（**从未成功落盘**：功能已由下面两个
  脚本取代，保留供后续复核；见"已知遗留"）
- `python paper/tables/m4_floor_partial.py` → 检验 (a) 四档地板比值（对缺档容错）→
  `analysis_output/m4_floor_partial.json`
- `python paper/tables/m4_ablation_stats.py` → 检验 (b) 冻结 trunk 多 seed 统计 →
  `analysis_output/m4_ablation_stats.json`
- `python paper/tables/build_ladder_extension_table.py` → 原始 3 seed vs 扩展 **10 seed** 阶梯对照表
  (`tab_ladder_extension.tex` + `analysis_output/ladder_extension_table.json`)；共享 seeds 42–44，
  因此扩展不是独立复制。包含 own-split head/target 描述性比值。
- `python paper/tables/build_ladder_floor_r1.py` → 对全部 **40** 个 checkpoint（10 seed × 4 档，
  跨 `checkpoints-E0` 与 `checkpoints-E0-extra`）在各自 held-out split 重算头/目标比值
  （CPU-only，不训练）→ `analysis_output/ladder_floor_r1.json`。
  替代旧的共同 s42 split 比值作为正文描述；值仍是同源轨迹的敏感性扩展，不是独立重复。
- `python paper/tables/m4_noise_floor.py` → **§8 噪声底矩阵**（换 seed × 重训 × 指标，
  同一评估协议 `e0_common.validate`）→ `analysis_output/noise_floor_matrix.json`（现为 10 seed）。
  支持 `--seeds`、`--ckpt-root`（可多个，按序查找）、`--out`；默认仍是原 7-seed 行为。
  主口径 R1 = 每个 seed 用自己的划分与统计量（= `build_e0_report`，val 不与训练集重叠）；
  R2 = 共用 s42 划分的受污染敏感性诊断（对 s≠42 有 88–91% 训练集重叠，**不识别划分因果效应**）。含留一法稳健性与逐 seed 单调性计数。
- `python run_seed_extension.py --seeds 50 51` → 批次调度扩展 seed 训练（4 路并行，
  逐个校验 `Epoch 60/60`，可恢复）→ `analysis_output/seed_extension_status.json`
- `python eval_e0_extra_seeds.py --seeds 49 50 51` → 用 E0 口径评估扩展 seed →
  `analysis_output/e0_extra_seeds.json`（含 `prior_spec` 与目录名一致性断言）
- `python compare_extension_seeds.py` → 扩展 seed 对 42–48 范围的逐档对比 →
  `analysis_output/extension_seed_comparison.json`
- `python analyze_seed_extension.py` → 10-seed 汇总、7 vs 10 seed 统计与逐 seed 单调性 →
  `analysis_output/seed_extension_summary.json`
- `python paper/tables/m3_floor_absolute.py` → 地板是 regime 失衡属性（非架构属性）
- `python paper/tables/e5_{head_value,trunk_ablation,gate}.py` → 修正器适用边界三组证据
- `python paper/repro_check.py --checkpoint_root <root> --arm <arm> --seed <s>` →
  从新训练的 checkpoint 重建主表数字并与存储报告比对（落盘 `analysis_output/repro_check_*.json`）
- `paper/PREREG_CRITERIA_EN.md` → 12 份中文预注册的英文判据/阈值汇总（供审稿人核验）

**图片文件名与 PDF 图号不一一对应**（浮动顺序所致）：`fig1_floor_ladder`=Fig 1、
`fig2_p0c_2x2`=Fig 2、`fig7_floor_ceiling`=Fig 3、`fig4_dose_response`=Fig 4、
`fig5_silence_roc`=Fig 5、`fig3_interference`=Fig 6、`fig6_silence_hist`=Fig 7、
`fig8_allarms_roc`=Fig 8（由 `tmp_s1f_allarms.py` 生成，附录）。

**机制排查（§5.2）是预注册之后的 post-hoc 工作**，正文 §5.2 与 §9 均已如此声明；
原始脚本为仓库根的 `tmp_probe_replicate.py` / `tmp_crosstalk_law{,3,4}.py` /
`tmp_silence_ceiling.py` 等，日志 `tmp_*_log.txt`，全部随投稿包发布。

## 数字溯源规则

初稿阶段每个数字旁保留 `〔source〕` 注释（JSON 路径），M6 合稿时转成
附录"provenance table"引用；抽查 ≥10 条跑复现命令（框架 §4/§17/§18.7）。

**新增纪律（2026-10-10 事故后）**：引用值必须先由生成器打印、再抄进正文。
守卫只核"引用的值是否在容差内"，而容差是围绕被引用的值设的——它防漂移，不防伪造。

## 已知遗留（不阻塞投稿）

1. `analysis_output/m4_multiseed.json` 从未生成：`build_m4_multiseed.py` 的功能已被
   `m4_floor_partial.py` + `m4_ablation_stats.py` + `m4_noise_floor.py` + `build_ladder_floor_r1.py` 覆盖，
   正文不引用它。
2. 对抗式二审要求的 bootstrap CI 与 notation table 尚未补入；§4.3 的辨识结果已收窄为
   单一 Nelder–Mead 配置下的 optimizer-dependent compensation basin，未声称多起点/全局最优。
3. 第三世界、M2-b 异质失配、以及更广的 seed 扩充尚未做；不属于当前受控扩展的复现范围。
4. 内部文档（`internal/`：PEER_REVIEW / CLAIM_AUDIT / PRESUBMIT_AUDIT / E5_* / M3 / M4）
   **不随投稿包发布**，原因见 `INTERNAL_NOTES.md`。

## §8 噪声底（2026-10-12 新增；2026-10-15 扩到 10 seed）

原 §8 的噪声底是一串单维数字（重训 0.5% / 0.2% / 15%）。现改为**二维**：
同一臂同配方的 **10 seed**（42–51，跨 `checkpoints-E0` 与 `checkpoints-E0-extra`）
跨 seed CV（单步 51%、k=2 12.6%、k=5 16.4%）对重训差
（0.5% / 0.2% / 15%）——短指标上 **seed 轴是重训轴的 69–100×**，
k=5 上二者相当（16% vs 15%）。留一法（去掉最离群 seed）后单步仍为 27%，故非单点离群。
另有一条更强陈述：**四档排序在 10/10 seed 上两列都逐 seed 单调**，
即阶梯是逐 seed 事实而非平均效应。

扩到 10 seed 后的两个实质发现（`internal/M4_IN_PROGRESS.md` 有详细记录）：
- **7 seed 低估了离散度**：seed 49 在三个档的 one-step 上低于原 7-seed 最小值。
- **中间档 k=5 有重尾**：seed 51 的 euler001x5 k=5 = 12.60，是原上限 7.50 的 1.7 倍，
  使该档 CV 从 49% 升到 64%，而它的 one-step（0.149）在该档接近最好——
  正是"单步拟合好不能预测闭环存活"的干净样本。
- 10-seed 阶梯分离：one-step 17.7×，k=5 31×（论文 Table 1 仍为预注册 3-seed 的 29×）。

## 写作纪律（去 AI 味清单）

禁用词：leverage / delve / crucial / pivotal / comprehensive / landscape /
showcase / underscore / seamless；禁用结构：Moreover 链、三联排比、
每段等长、空泛总结句。句子长短交错；数字自己说话；机制假设一律标注
"hypothesis"；机制未定性的地方明说 not established。
