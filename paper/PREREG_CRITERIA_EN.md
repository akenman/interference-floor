# Preregistered decision criteria (English translation)

These are the decision rules fixed before each experiment's first run. Thresholds are quoted verbatim from the preregistration files (units as written). The preregistrations themselves are in Chinese and are included in the supplementary archive under preregistrations/.

*Rows whose ids end in `-conv-` are preregistered statistical / analysis-convention clauses (口径) that live outside the `criteria` object in the source JSON but are part of the fixed decision rules; they are included for completeness. Literal `|` characters inside verbatim thresholds are escaped as `\|`.*

## P0-a

*Is the coarse integrator really the culprit behind the finding that "the prior is a liability"?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| P0a-1 单调性 (monotonicity) | Both one-step and closed-loop k=5 error are strictly monotone across the four prior tiers. | 4 档单步与 k=5 均严格单调；basis: 解析档位已单调（3.373>0.522>0.147>0.038） |
| P0a-2 精度达标 (accuracy target) | Accuracy target reached: one-step ≤0.10px and k=5 ≤1.0px. | one-step ≤0.10px 且 k=5 ≤1.0px；basis: 精确先验 0.038 / 0.49，留 ~2× 余量 |
| P0a-3 超越无先验基线 (beats no-prior baseline) | Beats the no-prior learning baseline: k=5 ≤ strong_mlp's 1.362px. | k=5 ≤ strong_mlp 的 1.362px；basis: E0 现状 k=5 |
| P0a-4 先验消除残差负担 (prior removes residual burden) | The exact prior removes the residual head's burden: ours_base@rk4001x5 one-step ≤ 2× its own prior's one-step error (0.076px). | ours_base@rk4001x5 单步 ≤ 2× 其先验单步（0.076px）；basis: 残差头若健康，不应把 0.038px 的基座恶化到 2 倍以上 |
| P0a-conv-pairing | Pairing convention: same-seed pairing (pairwise); no cross-seed unpaired mean comparison (E0's n=3 unpaired convention buried a 0.563px gap as p=0.195; p=0.049 once paired). | 同 seed 配对（pairwise），不做跨 seed 非配对均值比——E0 的 n=3 非配对口径曾把 0.563px 的差距埋成 p=0.195（配对后 p=0.049） |
| P0a-conv-crash | Crash handling: ITT is the primary convention (including the crashed run's best.pt, verified to be saved before the crash); also report the crash-excluded convention and the crash rate; when the two conventions disagree, label DISAGREE instead of giving a single PASS/FAIL. | ITT 为主口径（含崩溃 run 的 best.pt，已验证为崩溃前保存）；同时报剔崩溃口径与崩溃率。两口径判定不一致时标 DISAGREE，不给单一 PASS/FAIL |
| P0a-conv-mde | Minimum detectable effect documented for the paired design: paired SD≈0.18px (measured in Stage B); δ=0.15px→n=12, δ=0.10px→n=26; this experiment expects an effect of 0.5–1.0px, so n=3 already has sufficient power. | 配对 SD≈0.18px（Stage B 实测）：δ=0.15px→n=12；δ=0.10px→n=26。本实验效应量预计 0.5–1.0px，n=3 即有足够功效 |
| P0a-conv-multiplicity | Multiplicity convention: the 4 criteria are reported in preregistered order with no post-hoc selection; any directional conclusion comes with paired t + effect size + CI together. | 4 个判据按预注册顺序报告，不做事后筛选；任一判据的方向性结论以配对 t + 效应量 + CI 同时给出 |

## P0-b

*All E0/P0-a results were trained with the w_roll=0 (pure one-step) recipe yet judged by closed-loop rollout; does the earlier ranking still hold once the training recipe is switched to closed-loop?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| P0b-1 核心威胁 (core threat) | Core threat: strong_mlp's closed-loop k=5 under recipe B ≤ 0.49px (= exact-prior level); n=3, same-seed paired A vs B. | strong_mlp 在 B 下闭环 k=5 ≤ 0.49px（= 精确先验水平）；n: 3；paired: 同 seed A vs B |
| P0b-2 纯学习自救 (pure-learning self-rescue) | Pure-learning self-rescue: noprior's closed-loop k=5 under B ≤ 1.0px (A was ~54px). | noprior 在 B 下闭环 k=5 ≤ 1.0px（A 为 ~54px） |
| P0b-3 粗先验自救 (coarse-prior self-rescue) | Coarse-prior self-rescue: ours_base's closed-loop k=5 under B ≤ 2.19px (= the rk4@0.05 tier level). | ours_base 在 B 下闭环 k=5 ≤ 2.19px（= rk4@0.05 档水平） |
| P0b-4 精确档不被损害 (exact tier not harmed) | Exact tier not harmed: ours_base@rk4001x5's closed-loop k=5 under B ≤ 0.54px (prior 0.49 × 1.10). | ours_base@rk4001x5 在 B 下闭环 k=5 ≤ 0.54px（先验 0.49 × 1.10） |
| P0b-5 噪声底 (noise floor) | Noise floor: same-configuration reruns measure 5–9%, so arm differences <10% are not claimed significant. | 同配置重跑实测 5–9%（F10）⇒ 臂间差距 <10% 不声称显著 |
| P0c-1 核心威胁重测 (core threat retest, the real verdict; registered in the cell-C addendum) | If strong_mlp@C closed-loop k=5 ≤ 0.49px, then a no-prior arm can match the exact prior when w_roll is on and lr is not reduced, and C1 must be rewritten; otherwise P0-a's conclusion still holds under the clean recipe. | strong_mlp@C 闭环 k=5 ≤ 0.49px ⇒ 无先验臂在「w_roll 有、lr 不减」下能追平精确先验 ⇒ C1 须改写；否则 P0-a 结论在干净配方下依然成立。 |
| P0c-2 纯学习自救 (pure-learning self-rescue) | noprior@C closed-loop k=5 ≤ 1.0px. | noprior@C 闭环 k=5 ≤ 1.0px |
| P0c-3 w_roll 净效应 (net effect of w_roll) | Net effect of w_roll: the C-vs-A improvement in closed-loop k=5 on the three arms (paired, same seed); also test whether B's one-step disadvantage is explained by the halved lr — if C's one-step ≈ A while B is clearly worse, lr rather than w_roll is confirmed. | C 相对 A 在三臂上的闭环 k=5 改善（配对、同 seed）；并检验 B 的单步劣势是否由 lr 减半解释——若 C 的单步 ≈ A 而 B 明显更差，则确认是 lr 而非 w_roll。 |
| P0b-conv-pairing | Pairing convention: same-seed pairing (recipe A vs B; C vs A; C vs B). | 同 seed 配对（配方 A vs B；C vs A；C vs B） |
| P0b-conv-crash | Crash handling: ITT is primary (including the crashed run's best.pt); the crash rate is also reported; the two conventions are labelled DISAGREE when they conflict. | ITT 为主（含崩溃 run 的 best.pt），同时报崩溃率；两口径不一致标 DISAGREE |
| P0b-conv-report | Reporting convention: one JSON per checkpoint root (build_e0_report.py --checkpoint_root ... --skip_analytic --out ...), then paired comparisons; the E0 main table (analysis_output/E0_report.json) is not modified. | 分 root 各出一份 JSON（build_e0_report.py --checkpoint_root ... --skip_analytic --out ...），再做配对比较；E0 主表（analysis_output/E0_report.json）不被改动 |

## P0-c

*Does auxiliary supervision — the only manipulable entry point of "human semantics" — give a positive gain on state prediction?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| P0c-1 爻监督主效应 (main effect of yao supervision) | Main effect of yao (hexagram) supervision = mean(C10, C11) − mean(C00, C01) on one-step one_step_model error with same-seed pairing: >+0.23px harmful, <−0.23px beneficial, in between no effect under this recipe. | test: mean(C10, C11) − mean(C00, C01)（单步 one_step_model，同 seed 配对）；verdict: >+0.23px ⇒ 有害；<−0.23px ⇒ 有益；其间 ⇒ 无影响（该配方下） |
| P0c-2 原型家族主效应 (main effect of prototype family) | Main effect of the prototype family (self-supervised clustering) = mean(C01, C11) − mean(C00, C10) on one-step error with same-seed pairing; same thresholds as P0c-1. | test: mean(C01, C11) − mean(C00, C10)（单步，同 seed 配对）；verdict: 同 P0c-1 门槛 |
| P0c-3 交互项 (interaction) | Interaction term = (C11−C10) − (C01−C00) on one-step error, same-seed paired; report effect size and CI, no decision threshold. | test: (C11−C10) − (C01−C00)（单步，同 seed 配对）；verdict: 报告效应量与 CI，不设判定门槛 |
| P0c-threshold_calibration | Threshold calibration basis (not itself a pass/fail rule): from the measured E0 effect size 0.56px and paired SD ≈ 0.2px, n=6 gives MDE ≈ 0.23px (≈24% of the coarse-prior arm's one-step 0.97px, above the 5–9% noise floor, hence detectable). | 阈值来自 E0 实测：效应量 0.56px、配对 SD ≈ 0.2px ⇒ n=6 的 MDE ≈ 0.23px（约为粗先验臂单步 0.97px 的 24%，高于 5–9% 噪声底，可检出）。 |
| P0c-secondary-1 | Secondary: closed-loop rollout k=5 under the same convention across all four cells. | 闭环 rollout k=5（四格同口径） |
| P0c-secondary-2 | Secondary: yao_acc confirms the supervision was actually learned — C10/C11 expected ≈0.96x, C00/C01 expected ≈0.7x (baseline level); if C10/C11's yao_acc does not rise, the supervision is not in effect and the experiment is voided and rechecked. | yao_acc：确认监督确实被学到——C10/C11 预期 ≈0.96x，C00/C01 预期 ≈0.7x（基线水平）；若 C10/C11 的 yao_acc 不升，说明监督未生效，实验作废重查 |
| P0c-secondary-3 | Secondary: crash rate (ITT, including the crashed run's best.pt). | 崩溃率（ITT：含崩溃 run 的 best.pt） |
| P0c-噪声底 (noise floor) | Noise floor: same-configuration reruns measure 5–9%, so arm differences <10% must not be claimed significant; the main criteria use the 0.23px paired threshold. | 同配置重跑实测 5–9%（F10）⇒ <10% 的臂间差距不得声称显著；主判据以 0.23px 配对门槛为准 |
| P0c-conv-pairing | Same-seed pairing (the four cells share seeds 42–47). | 同 seed 配对（四格共享 seed 42–47） |
| P0c-conv-itt | ITT includes the crashed run's best.pt while also reporting the crash rate; the crash-excluded convention is a sensitivity analysis only. | 含崩溃 run 的 best.pt，同时报崩溃率；剔除口径只作敏感性分析 |
| P0c-conv-order | Report in preregistered order (P0c-1 → P0c-2 → P0c-3 → secondary conventions), no post-hoc selection. | 按预注册顺序报告（P0c-1 → P0c-2 → P0c-3 → 次口径），不做事后筛选 |
| P0c-conv-reporting | Directional conclusions must give paired t, effect size, CI and a direction-consistency count (number of concordant seeds out of 6) together. | 方向性结论必须同时给配对 t、效应量、CI、方向一致性计数（6 seed 中同向个数） |

## S0-b

*Is the subtraction caused by structural priors (GuaMoE gating, +0.272px under a coarse prior) absorbed by the floor (≈0) once the prior is exact?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| S0b-threshold_calibration | Calibration basis for the two thresholds, written in absolute px to avoid depending on the unknown sd of gua@exact: lower threshold +0.010px = ~5× the noise floor; upper threshold +0.050px ≈ 18% of the coarse-tier effect and 25× the noise floor; both levels are detectable. | 门槛以绝对 px 写死，避开对 gua@exact 未知 sd 的依赖。噪声底锚点：base@exact 3-seed sd=0.001px（P0-a）、跨配方 Δone=+0.0002（P0-b）、同配置重跑 ≤5%（F10，在 0.037px 尺度即 ≤0.002px）。下门槛 +0.010px = 噪声底的 ~5×；上门槛 +0.050px ≈ 粗档效应的 18%、噪声底的 25×——两级均可检出。 |
| S0b-1 主判据（单步吸收判决） | Primary criterion (one-step absorption verdict): Δone = one_step(ours_gua@rk4001x5) − one_step(ours_base@rk4001x5), same-seed paired (n=3); Δ ≤ +0.010px absorbed, Δ ≥ +0.050px not absorbed (the subtraction contains a floor-independent channel), in between UNDECIDED. | test: Δone = one_step(ours_gua@rk4001x5) − one_step(ours_base@rk4001x5)，同 seed 配对（n=3）；verdict: Δ ≤ +0.010px ⇒ 吸收（主轴精确化 4 成立）；Δ ≥ +0.050px ⇒ 不吸收（减法含地板无关通道）；其间 ⇒ UNDECIDED |
| S0b-2 闭环同向 | Closed-loop same direction: paired ratio of k5(ours_gua@rk4001x5) to k5(ours_base@rk4001x5); ≤ ×1.10 (≤0.54px) absorbed in the same direction, ≥ ×1.30 (≥0.64px) not absorbed, in between UNDECIDED (same-style threshold as P0b-4). | test: k5(ours_gua@rk4001x5) 对 k5(ours_base@rk4001x5) 的配对比值；verdict: ≤ ×1.10（≤0.54px）⇒ 闭环同向吸收；≥ ×1.30（≥0.64px）⇒ 不吸收同向；其间 ⇒ UNDECIDED（P0b-4 同风格阈值） |
| S0b-3 效应量比（只报告） | Effect-size ratio (report only): ratio = Δone@exact / Δone@coarse, denominator = the coarse-tier paired difference +0.272px recomputed from existing E0 data; ratio < 1/10 strong absorption, ratio ≈ 1 no absorption, no threshold set. | test: ratio = Δone@exact / Δone@coarse（分母 = 粗档配对差 +0.272px，由现有 E0 数据复算）；verdict: ratio < 1/10 ⇒ 强吸收；ratio ≈ 1 ⇒ 无吸收；报告不设门槛 |
| S0b-secondary-1 | Secondary: full rollout k=1..5 curves for both arms under the same convention. | rollout k=1..5 全程曲线（两臂同口径） |
| S0b-secondary-2 | Secondary: yao_acc — supervision is on for both arms and both are expected ≈0.96x; if gua@exact's yao_acc is abnormally low, check the wiring before discussing the verdict. | yao_acc：两臂监督同开，预期均 ≈0.96x；若 gua@exact 的 yao_acc 异常低，先查接线再谈判决 |
| S0b-secondary-3 | Secondary: gua occupancy (number of surviving groups out of 64 / minimum share) — whether the gate still collapses where the residual ≈ 0. | gua occupancy（64 组存活数/最小占比）：门控在残差≈0 处是否仍塌缩（承接 10.17/10.19.16 的审计口径） |
| S0b-secondary-4 | Secondary: crash rate (ITT). | 崩溃率（ITT） |
| S0b-噪声底 (noise floor) | Noise floor: same-configuration reruns ≤5% (F10); both thresholds are ≥5× the noise floor, so the verdict does not rest on a <10% difference. | 同配置重跑 ≤5%（F10）；两级门槛均 ≥5× 噪声底 ⇒ 判决不依赖 <10% 差距 |
| S0b-conv-pairing | Same-seed pairing, n=3; directional conclusions give paired t, effect size and a direction-consistency count (3/3 or its absence reported). | 同 seed 配对，n=3；方向性结论给配对 t、效应量、方向一致性计数（3/3 或报其缺失） |
| S0b-conv-itt | Include the crashed run's best.pt; a run without best.pt is recorded as a missing pair. | 含崩溃 run 的 best.pt；无 best.pt 的 run 记为配对缺失 |
| S0b-conv-order | Report in preregistered order (S0b-1 → S0b-2 → S0b-3 → secondary conventions), no post-hoc selection. | 按预注册顺序报告（S0b-1 → S0b-2 → S0b-3 → 次口径），不做事后筛选 |

## S1

*Does "closed-loop floor = mismatch between the prior's computation class and the data-generation process" (established by P0-a in the hinge world) hold across engines?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| S1-1 引擎对拍（兼动作对齐自校验） | Engine cross-check (which also self-checks action alignment): euler001x5's FF one-step pos_rmse ≤ 0.01px; if missed, first check action alignment / implementation (a protocol fix recorded in DEEP_DIVE, not a threshold change) and re-adjudicate after the fix. | test: euler001x5 在 FF one-step pos_rmse ≤ 0.01px；note: 不命中 ⇒ 先查动作对齐/实现（协议修复，记录于 DEEP_DIVE，非阈值更改），修复后重判 |
| S1-2 排序判据 (ordering) | Ordering criterion in FF one-step: euler001x5 strictly smallest; euler005 strictly largest (> parabola005 and > rk4005); parabola005 and rk4005 equal within tolerance on the FF-0 subset (\|Δ\| ≤ 0.02px). All met ⇒ computation-class-distance ordering holds (H1). If parabola/rk4 are not equal ⇒ check the force-displacement term implementation. | test: FF one-step：euler001x5 严格最小；euler005 严格最大（> parabola005 且 > rk4005）；parabola005 与 rk4005 在 FF-0 子集平价（\|Δ\| ≤ 0.02px）；verdict: 全 ⇒ 计算类距离排序成立（H1 [OK]）；parabola/rk4 不平价 ⇒ 检查力位移项实现 |
| S1-3 量值判据（FF-0 子集） | Magnitude criterion (FF-0 subset): parabola005 and rk4005 ∈ [0.12, 0.20]px; euler005 ∈ [0.50, 0.80]px (the float32 + finite-sample envelope of the theoretical 0.159/0.636 px; if FF-0 has <5k samples, switch to the low-impulse decile (\|I\|≤10% IMPULSE_MAX) and note it in the record). Hit ⇒ the analytic prediction holds quantitatively (H2); outside ⇒ report the deviation and check the actual distribution of a. | test: parabola005 与 rk4005 ∈ [0.12, 0.20]px；euler005 ∈ [0.50, 0.80]px；note: 理论 0.159/0.636 px 的 float32+有限样本包络；FF-0 < 5k 样本时改用低冲量十分位（\|I\|≤10% IMPULSE_MAX）并在记录注明；verdict: 命中 ⇒ 解析预测定量成立（H2 [OK]）；偏出 ⇒ 报告偏差并查 a 的实际分布 |
| S1-4 闭环同序 (closed-loop same order) | Closed-loop same order: the k=5 ordering agrees with S1-2 (euler001x5 smallest, euler005 largest). | k=5 排序与 S1-2 一致（euler001x5 最小、euler005 最大） |
| S1-mixed 只报告 (report only) | Report-only: the four tiers' numbers on contact-containing windows and the span between tiers (expected to be flattened by contact mismatch). | 含接触窗口的四档数字与档间跨度（预期被接触失配压平） |
| S1-conv-deterministic | Determinism convention: no training and no seed variance, so the verdict involves no paired t; report each tier's mean and sample count; subsampling uses a fixed reproducible seed. | 无训练、无 seed 方差——判决不涉及配对 t；报告每档均值与样本数；子采样固定 seed 可复现 |
| S1-conv-order | Report in preregistered order (S1-1 → S1-2 → S1-3 → S1-4 → mixed), no post-hoc selection. | 按预注册顺序报告（S1-1 → S1-2 → S1-3 → S1-4 → mixed），不做事后筛选 |

## S1-b

*Does the trained-arm prior ladder inherit the same ordering in the ball world, and does the contact-frame margin change the answer relative to the hinge world where the exact tier had no learnable margin?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| S1b-1 FF 继承地板 | FF inheritance of the floor: FF-clean one-step (unified evaluation protocol, same-seed paired) euler001x5 < min(parabola, rk4) < euler005, with a consistent direction across the 3 seeds. All ⇒ H1; any reversal ⇒ the trained arm does not inherit the ladder, report it and check residual-head compensation. | test: FF-clean one-step（统一评估协议，同 seed 配对）：euler001x5 < min(parabola, rk4) < euler005，3 seed 方向一致；verdict: 全 ⇒ H1 [OK]；任一反序 ⇒ 训练臂不继承阶梯，报告并查残差头补偿 |
| S1b-2 接触帧余量 | Contact-frame margin: euler001x5's mixed one-step (3-seed mean) ≤ 0.80px; met ⇒ H2 (existence of the margin is a function of contact density); >1.00px ⇒ no margin (anomalous, must be investigated). | test: euler001x5 的 mixed one-step（3 seed 均值）≤ 0.80px；verdict: [OK] ⇒ H2（余量存在性=接触密度的函数）；>1.00px ⇒ 余量不存在（反常，须查） |
| S1b-3 闭环 FF 贴地 | Closed-loop FF hugging the floor: euler001x5's FF-clean closed-loop k=3 ≤ 0.50px (analytic floor ~0; the threshold is relaxed to 0.5px to allow small perturbations of the model feedback in free flight; report the actual value). | test: euler001x5 的 FF-clean 闭环 k=3 ≤ 0.50px；note: 解析地板 ~0；门槛放宽到 0.5px 容许模型反馈在自由飞行段的小扰动；报告实际值 |
| S1b-4 精确档不劣化 | Exact tier not degraded: mixed one-step euler001x5 ≤ parabola005 × 1.10 (same-seed paired mean ratio). | mixed one-step：euler001x5 ≤ parabola005 × 1.10（同 seed 配对均值比） |
| S1b-secondary-1 | Secondary: mixed closed-loop k=1..3 curves for all tiers. | mixed 闭环 k=1..3 全档曲线 |
| S1b-secondary-2 | Secondary: yao_acc (same supervision for the four tiers, expected to be at the same level). | yao_acc（四档同监督，应同水平） |
| S1b-secondary-3 | Secondary: crash rate (ITT; any crashed run ⇒ that pair is missing and is judged UNDECIDED, with no seeds added). | 崩溃率（ITT；任何 run 崩溃 ⇒ 该配对缺失，判 UNDECIDED 不补 seed） |
| S1b-secondary-4 | Secondary: ratio of each tier's model mixed one-step to that tier's own prior mixed error (residual-head contribution spectrum). | 每档模型 mixed one-step 对其自身先验 mixed 误差的比率（残差头贡献谱） |
| S1b-噪声底 (noise floor) | Noise floor: same-configuration reruns 5–9% (F10), so the S1b-2/S1b-4 thresholds are all ≥10% in magnitude; the ordering criterion (S1b-1) is judged by direction consistency (3/3). | 同配置重跑 5–9%（F10）⇒ S1b-2/S1b-4 门槛均 ≥10% 量级；排序判据（S1b-1）以方向一致性为准（3/3） |
| S1b-conv-pairing | Same-seed pairing only (each seed's val split differs, so pairing is within seed; across seeds report means). | 同 seed 配对（每 seed 的 val 划分随 seed 而异，故只在同 seed 内配对；跨 seed 报均值） |
| S1b-conv-itt | Include the crashed run's best.pt. | 含崩溃 run 的 best.pt |
| S1b-conv-order | Report in preregistered order (S1b-1 → S1b-2 → S1b-3 → S1b-4 → secondary conventions). | 按预注册顺序报告（S1b-1 → S1b-2 → S1b-3 → S1b-4 → 次口径） |
| S1b-conv-eval_protocol | Evaluation protocol: tmp_s1b_compare.py loads all 12 best.pt files and evaluates mixed / FF-clean one-step and closed loop on a fixed held-out window set (val_windows=20000); FF-clean is defined as in SLOT-1 (contact flag + velocity relation <1px/s). | tmp_s1b_compare.py：从 12 个 best.pt 加载，固定 held-out 窗口集（val_windows=20000）评 mixed/FF-clean one-step 与闭环；FF-clean 定义同 SLOT-1（contact 标志 + 速度关系 <1px/s） |

## S1-c

*Can two preregistered mechanisms make the residual head silent in the FF regime and thereby cash in the exact prior, despite SLOT-1b's tier-independent interference noise (~0.85px)?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| R1 静默 | Silence: FF-clean one-step (same-seed paired mean) ≤ 0.50px (~58% of the 0.861 baseline, >6× the noise floor); report the actual value and the ratio to baseline. | test: FF-clean one-step（同 seed 配对均值）≤ 0.50px（基线 0.861 的 ~58%，>6× 噪声底）；report: 实际值与基线比率 |
| R2 不伤接触 | Does not harm contact: mixed one-step ≤ baseline 0.571 × 1.10 = 0.628px; >0.628 ⇒ that mechanism is judged negative (silence bought at the cost of contact accuracy is unacceptable). | test: mixed one-step ≤ 基线 0.571 × 1.10 = 0.628px；verdict: >0.628 ⇒ 该机制判负（静默以接触精度为代价不可接受） |
| R3 闭环缓解 | Closed-loop mitigation: FF-clean closed-loop k=3 ≤ 1.02px (half of the 2.040 baseline). | FF-clean 闭环 k=3 ≤ 1.02px（基线 2.040 的一半） |
| R4 探针复核（只报告） | Probe re-check (report only): the decrease factor of the FF-clean frame head-output RMS (0.0197 convention). | FF-clean 帧头输出 RMS（0.0197 口径）下降倍数 |
| 统计 (statistics) | Statistics: same-seed paired (3/3 direction consistency governs); the two mechanisms are each judged against the same baseline, with no post-hoc selection of the better mechanism. | 同 seed 配对（3/3 方向一致为准）；两机制分别对同一基线判，不做机制间优劣的事后挑选 |
| S1c-conv-order | Report in preregistered order R1 → R2 → R3 → R4; both mechanisms reported side by side. | R1 → R2 → R3 → R4；两机制并列报告 |

## S1-d

*Since the regime labels are already in the training data (contact flags), can making the gate supervised (BCE teaching the gate to open/close) instead of end-to-end learned achieve constructive silence and bring the trained arm's closed-loop FF down to the analytic prior's identity (0.00003px)?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| R1d 完全静默 | Fully silent: FF-clean one-step ≤ 0.10px; 0.10–0.25px counts as partial fulfilment. | test: FF-clean one-step ≤ 0.10px；partial: 0.10–0.25px 记部分兑现 |
| R2d 不伤接触 | Does not harm contact: mixed one-step ≤ 0.628px (baseline 0.571 × 1.10). | mixed one-step ≤ 0.628px（基线 0.571 × 1.10） |
| R3d 闭环贴地 | Closed-loop hugging the floor: FF-clean closed-loop k=3 ≤ 0.20px (analytic 0.00003 + gate-leak margin). | test: FF-clean 闭环 k=3 ≤ 0.20px；note: 解析 0.00003 + 门泄漏裕度 |
| R4d 门准确率 | Gate accuracy: report the close-on-FF-frame / open-on-contact-frame accuracy (expected ≥99%). | 报告 FF 帧关门/接触帧开门准确率（预期 ≥99%） |
| R5d 混合闭环不劣化 | Mixed closed loop not degraded: mixed closed-loop k=3 ≤ baseline ×1.10 (2.33×1.10≈2.57px). | mixed 闭环 k=3 ≤ 基线 ×1.10（2.33×1.10≈2.57px） |
| 统计 (statistics) | Statistics: same-seed paired vs the SLOT-1b baseline (3/3 direction consistent); the comparison with wff10 is a secondary report. | 同 seed 配对 vs SLOT-1b baseline（3/3 方向一致）；与 wff10 的比较为次口径报告 |
| S1d-conv-order | Report in preregistered order R1d → R2d → R3d → R4d → R5d. | R1d → R2d → R3d → R4d → R5d |

## S1-e

*Can regime reweighting be extended from the one-step state loss to the rollout loss so that silence is bought on the closed-loop step distribution, pushing closed-loop FF from 1.406 to R3 (≤0.20px)?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| R1e 单步静默 | One-step silence: FF-clean one-step ≤ 0.10px; 0.10–0.25px counts as partial fulfilment. | test: FF-clean one-step ≤ 0.10px；partial: 0.10–0.25px 记部分兑现 |
| R2e 不伤接触 | Does not harm contact: mixed one-step ≤ 0.628px (baseline 0.570 × 1.10). | mixed one-step ≤ 0.628px（baseline 0.570 × 1.10） |
| R3e 闭环兑现（主判据） | Main criterion, closed-loop fulfilment: FF-clean closed-loop k=3 ≤ 0.20px; calibration basis: the perfect-silence ceiling is 0.00009px (3 seeds), the current value is 1.406 so a 7× improvement is needed, and 0.20 = analytic identity + head residual of order 0.2px. | test: FF-clean 闭环 k=3 ≤ 0.20px；标定依据: 完美静默上限 0.00009px（3 seed）；现状 1.406 ⇒ 需 7× 改善；0.20 = 解析恒等 + 头残余 0.2px 级 |
| R4e 通道归因 | Channel attribution: W2's relative closed-loop improvement ≥ arm S (directional; corresponds to H2e). | W2 的闭环相对改善 ≥ S 臂（方向性；对应 H2e） |
| R5e mixed 闭环不劣化 | Mixed closed loop not degraded: mixed closed-loop k=3 ≤ baseline ×1.10 (2.278×1.10≈2.51px). | mixed 闭环 k=3 ≤ baseline ×1.10（2.278×1.10≈2.51px） |
| 统计 (statistics) | Statistics: same-seed paired vs each arm's own reference (3/3 direction consistent) + effect size and CI; ITT crash rate is recorded as usual. | 同 seed 配对 vs 各自参照（3/3 方向一致）+ 效应量与 CI；ITT 崩溃率照记 |
| S1e-conv-ceiling_probe | Upper-bound probe convention: a hard-silence mask (skip the residual head when the yao head predicts contact=0; deployable, no oracle required) × the checkpoints of baseline/wff10/sgate/wff10+ is reported as an "upper-bound probe" and is **not claimed as this slot's method**; the budget's zero-self-soft (0.00009px) is its idealised version. | 硬静默掩码（爻头预测接触=0 时跳过残差头，部署可得，不需 oracle）× baseline/wff10/sgate/wff10+ 的 checkpoint——作为'上限探针'报告，**不作为本槽位声称的方法**；预算里的 zero-self-soft（0.00009px）是其理想版 |
| S1e-conv-order | Report in preregistered order R1e → R2e → R3e → R4e → R5e. | R1e → R2e → R3e → R4e → R5e |

## S1-f

*Can decode-time decision-rule hard silence (using the model's own contact head with a val-calibrated threshold τ) cash closed-loop FF from 1.3–2.7px to R3 (≤0.20px) without harming contact, and is the regime indicator trainable?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| R1f 方法兑现（主判据） | Main criterion, method fulfilment: on val-B at the rule-selected τ, FF-clean closed-loop k=3 ≤ 0.20px and satisfied for 3/3 seeds. | 在 val-B、规则 τ 下：FF-clean 闭环 k=3 ≤ 0.20px，且 3/3 seed 均满足 |
| R2f 不伤接触 | Does not harm contact: contact-regime closed-loop k=3 ≤ that arm's unmasked same-seed value ×1.10. | 接触 regime 闭环 k=3 ≤ 该臂无掩码同 seed ×1.10 |
| R3f 指示器可训 | Indicator trainability: F1/F2 free-unit protection rate ≥ the same arm's reference (directional, 3/3). | F1/F2 自由单元保护率 ≥ 同臂参照（方向性 3/3） |
| R4f 曲线外推（ROC 支配） | ROC dominance: at an equal contact false-close rate (±0.5pp), the trained arms' FF RMS ≤ the reference arms'. | 训练臂在『同等接触误关率（±0.5pp）』下 FF RMS ≤ 参照臂 |
| R5f mixed 不劣化 | Mixed not degraded: mixed closed-loop k=3 ≤ that arm's unmasked value ×1.10. | mixed 闭环 k=3 ≤ 该臂无掩码 ×1.10 |
| 统计 (statistics) | Statistics: same-seed paired; ITT crash rate recorded as usual; a <10% noise floor is not claimed significant (this slot's main criterion is an absolute threshold, not a significance test). | 同 seed 配对；ITT 崩溃率照记；noise floor <10% 不声称显著（本槽位主判据为绝对门槛，非显著性） |
| S1f-tau-rule (from design.tau_rule_preregistered) | Preregistered τ selection rule: on val-A (the first 10000 windows of the val set) take the largest τ satisfying "contact-unit false-close rate ≤ 2%" (grid 0.50→0.99, step 0.01); decision metrics are computed only on val-B (the last 10000 windows), separating selection from verdict so the threshold is not tuned on the test metric. | 在 **val-A（val 窗的前半 10000 窗）** 上，取满足『接触单元误关率 ≤ 2%』的最大 τ（网格 0.50→0.99 步长 0.01）；**判决指标只在 val-B（后半 10000 窗）上计算**（选择与判决分离，避免阈值在测试指标上调优）。 |
| S1f-conv-order | Report in preregistered order R1f → R2f → R3f → R4f → R5f. | R1f → R2f → R3f → R4f → R5f |

## S1-g

*Where does the head's interference noise come from: contact-frame gradient crosstalk, insufficient FF-frame (zero-target) loss, or GRU hidden-state drift?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| S1g-primary | Primary readout: comparison of the FF-clean one-step head-output RMS against baseline (0.0186) and wff10 (0.0095). | FF-clean one-step 头输出 RMS 与 baseline (0.0186) 和 wff10 (0.0095) 的对比 |
| S1g-interpretation-1 | If ≈0.019 ⇒ gradient crosstalk confirmed: contact-frame gradients are the sole source of the noise and the FF frames' own zero target has no effect. | if ≈ 0.019: 梯度串扰 confirmed——接触帧梯度是噪声的唯一来源，FF 自身零目标不起作用 |
| S1g-interpretation-2 | If ≈0.010 ⇒ mixed: crosstalk dominates but the FF zero target has a partial suppression effect. | if ≈ 0.010: 混合——串扰为主但 FF 零目标有部分抑制效果 |
| S1g-interpretation-3 | If <0.005 ⇒ the FF frames' own loss is the noise driver: crosstalk is excluded and silence is reachable by loss manipulation. | if < 0.005: FF 帧自身损失是噪声驱动力——排除串扰，静默可通过损失操作达到 |

## S2

*Do the paper §6.7 mechanism claims (pure-learning closed-loop collapse of 16–38px, the identified sampler–physics aliasing, and time-course causality), currently single-seed measurements, remain stable under 3 seeds?*

| Criterion (id) | English statement | Threshold (verbatim) |
| --- | --- | --- |
| R2-a 崩塌复现 | Collapse reproduction: 3/3 seeds, on g125 and drag, noprior closed-loop k5 ≥ 3× hybrid_fixed k5 (originally 8.4×/9.1×). All pass ⇒ the collapse claim is seed-robust. | test: 3/3 seed：g125 与 drag 上 noprior 闭环 k5 ≥ 3× hybrid_fixed k5（原 8.4×/9.1×）；verdict: 全过 ⇒ 崩塌主张种子稳健 |
| R2-b 混叠复现 | Aliasing reproduction: 3/3 seeds, clean nm5_H5 identifies ĝ ∈ [600, 850] (compensation basin, originally 706; not the true-value neighbourhood); g125 nm5sub_H5 \|ĝ/1125−1\| ≤ 0.20 (originally −15%). | 3/3 seed：clean nm5_H5 辨识 ĝ ∈ [600, 850]（补偿盆地，原 706；非真值邻域）；g125 nm5sub_H5 \|ĝ/1125−1\| ≤ 0.20（原 −15%） |
| R2-c 时程因果复现 | Time-course causality reproduction: 3/3 seeds, ĝ(H1) < ĝ(H5) < ĝ(H10) and k5(H10) ≤ k5(H1). | 3/3 seed：ĝ(H1) < ĝ(H5) < ĝ(H10) 且 k5(H10) ≤ k5(H1) |
| 报告义务 (reporting obligation) | Reporting obligation: a full per-seed table (including deviating items); any seed violation ⇒ that claim is downgraded to "reproduced in 2/3 seeds" and labelled as such. | 逐 seed 全表（含偏离项）；任何 seed 违反 ⇒ 该主张降格为'2/3 seed 复现'并如实标注 |
| S2-conv-order | Report in the order R2-a → R2-b → R2-c; no seeds are added (iron rule 3). | R2-a → R2-b → R2-c；不做 seed 追加（铁律 3） |
