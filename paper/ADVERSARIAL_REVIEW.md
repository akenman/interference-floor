# Adversarial Review — TMLR

**Manuscript:** *Computation-Class Fidelity, Not Structure: What Sets the Closed-Loop Floor of Residual World Models*
**Material reviewed:** `paper/main.pdf` (17 pp.), `paper/TMLR_submission.zip` (12 preregistrations, 79 `analysis_output/*.json`, 9 guard test files, `verify_numbers.py`, `number_provenance_check.json`, 5 figure PDFs), plus the repository's Python sources, `tests/`, and `checkpoints_manifest.json`.
**Reviewer:** independent, single-pass, artifact-verified.
**Standing of the artifacts:** I re-ran `pytest tests -q` (40 passed) and re-derived every number quoted below from the shipped JSON. All arithmetic in the paper is essentially faithful to the JSON. The problems below are almost never transcription errors; they are *interpretation*, *selection*, and *provenance* problems.

---

## Summary of Submission

The paper studies what determines whether a residual (prior + learned head) dynamics world model survives closed-loop rollout, using two small simulated worlds — a two-link underactuated acrobot and a planar ball arena — inside a fixed architecture family with a registry of four prior-integrator tiers. Its three headline claims are (C1) the prior's *computation class* (the discrete map, not the accuracy of the underlying ODE solution) sets a closed-loop floor; (C2) the training recipe multiplies gains *within* that floor and saturates; (C3) symbolic structure (gating, supervision) consistently subtracts, and the harm vanishes at an exact floor. A fourth section argues the trained ball-world arm sits on an "interference floor" — residual-head noise on windows where the exact prior is already perfect — and offers a decode-time contact-gated hard-silence rule that the authors present as restoring analytic parity at a quoted ~2% false-silence rate. The methodological frame (12 preregistrations written before the runs, guard tests binding paper numbers to JSON, a measured noise floor, and three self-corrected conclusions) is offered as a second contribution.

---

## Strengths

1. **The ball-world analytic ladder (Table 1b / `S1_ball_ladder.json`) is a genuine, sharp, non-obvious result:** a mathematically exact closed-form parabola and RK4@0.05 both sit at 0.1591 px against an explicit-Euler@0.01×5 engine at 3.1e-5 px, and the coarse same-family tier is worst at 0.6364 px — with the values *predicted analytically before the run* (0.225/√2, 0.9/√2) and hit to the fourth decimal.
2. **The "erased ladder" negative result (§6.1, `S1b_report.json`) is reported honestly at the level of numbers:** four trained ball arms land at 0.862/0.872/0.985/0.861 px against priors at 0.159/0.636/0.159/3e-5, i.e. the training mixture destroys the analytic ordering the paper spent §4 establishing.
3. **Preregistration discipline is real and unusually fine-grained:** each slot fixes arms, seeds, recipes, thresholds with a calibration source (`key_arithmetic`, `threshold_calibration`), and stop rules; the seeds are shared across paired comparisons.
4. **The negative-result catalogue (Appendix A) and the per-seed accounting in Table 5 expose caveats against the authors' own case** — e.g. that the 7-variant gate meta-analysis is one replicated finding rather than seven independent ones, and that `z = 7.14` is anticonservative.
5. **Mechanistic separation in §5.2 is properly done:** the supervision 2×2 holds architecture and capacity fixed and validates that the supervision channel was live (indicator accuracy 0.968 vs 0.474, `P0c_compare.json → yao_supervision_check`), so the +0.269 px cost is not a null manipulation.

---

## Weaknesses (ordered by severity)

### W1 — The decode rule's headline number is an oracle probe, not the deployed rule, and the reported 2% operating point is not the 2% operating point specified (`BLOCKING`)

**Position:** §6.5–§6.6, Abstract, C4; Fig. 4.

**Evidence.** §6.5 defines the deployed rule as *"if the maximum contact-indicator probability over objects exceeds a threshold τ, the head is used; below it, the head is skipped and the analytic prior acts alone"* — i.e. a threshold on the **model's own contact-head output** — and reports *"Free-flight closed-loop error under the rule is 10⁻⁴ px, in every seed"*. But `S1e_error_budget_all_summary.json` gives the analytic/identity floor at **8.767e-5 px** for *every* arm (`prior` field, identical across baseline/wff10/sgate/W1/W2/S), and the S1e preregistration explicitly attributes that 0.00009 px figure to an **oracle hard-silence mask** ("硬静默掩码（爻头预测接触=0 时跳过残差头…）… **不作为本槽位声称的方法**"). The provenance check `S1f decode-rule FF closed (max over arms/seeds) = 0.000103` is a *max over arms*, i.e. consistent with the oracle floor, while the rule-based `S1f_report.json` figure is `7.86e-5`. The text does not distinguish these two sources.

**Second, and worse:** the preregistered τ rule is *"the largest τ satisfying **contact-unit false-silence rate ≤ 2%** on val-A"* with a grid of **0.50→0.99 step 0.01** (50 points). The shipped grid is **0.50→0.95 step 0.05** (10 points; `S1f_report.json → tau_grid`), a deviation the paper does disclose. But on the shipped grid the rule lands at **τ\* = 0.75**, where val-A false silence is **1.9945%** and val-B false silence is **2.007%** — and the free-flight floor is only reached at **τ = 0.75** (val-B `ff_rms` 7.86e-5, `ff_all_silenced = 1.0`). At the two grid points nearest the registered rule at 0.01 resolution, τ ∈ {0.76, 0.77} where the unit-level constraint (1.2–1.5% at τ = 0.7, 3.1–3.5% at τ = 0.9 in the preregistration's own calibration table) would almost certainly still be satisfied, the floor would *not* be reached. So the reported headline operates exactly at the boundary its own rule forbids, and the coarse grid is what makes the boundary reachable.

**Third:** §6.6's "price of silence" is stated as a derivative *"per decade of free-flight closed-loop error purchased"* with an operating point *"at or below 2%"*, but §6.5 simultaneously reports the **×1.12–1.21** contact damage as arising from *"the 2% of contact windows falsely silenced"*. A 2% poisoning of a 2.1 px pool cannot raise it by 21% (the pooled rise would be ≈4%), so the damage is either superlinear in the poisoned subset or the quoted price and the quoted cost are not the same experiment. `S1f_report.json` (seed 42, baseline) shows the true operating point `τ*=0.75` sits **above** the 2% bound, and the floor's arrival is a knife-edge over the 15 scored val-B free-flight windows.

**Required:** (i) state explicitly and in the main text which number is oracle and which is rule-based, and re-derive the headline from the rule-based path only; (ii) re-run the τ grid at the registered resolution (0.01) and report where the rule actually lands, with the resulting free-flight error, or preregister an amended rule; (iii) define the false-silence cost metric precisely (window-level vs unit-level) and reconcile the 2% figure with the ×1.21 damage; (iv) restore the preregistered per-seed median and all-silenced-window fraction, which the S1f preregistration lists as robustness readings and which are absent from the report.

---

### W2 — "Every number is bound to a stored JSON by a guard script" is false for at least one load-bearing number, and its cited baseline contradicts the shipped JSON (`BLOCKING`)

**Position:** §6.1 (mechanism ablation), Reproducibility Statement §9, Appendix B.

**Evidence.** §6.1 states: *"excluding free-flight windows from the gradient entirely (w_ff = 0) leaves the noise at 0.0213 against the S1c reweighting baseline 0.0186 — a shift of +14%"*. Check the artifacts:

- `analysis_output/S1c_report.json → verdicts.wff10.head_rms_ff = 0.009493` (mean over seeds). The **value 0.0186 is not the wff10 arm's head RMS**; it is the mean of the *native baseline* (`S1c_report.json → baseline.head = 0.018574`, the arm with **no reweighting**). The sentence names the reweighting arm and quotes the non-reweighted arm's number.
- **0.0213 appears in no JSON in the archive.** It is absent from `number_provenance_check.json` in any form. There is no `S1g_report.json` at all; the only S1g artifact shipped is `prereg_S1g_mechanism_ablation.json`. The corresponding run's numbers exist only in `checkpoints-S1g/` (excluded from the archive).
- The derived "+14%" is therefore not reproducible from the released material by any script.

This directly contradicts §9 ("*Every number in this paper is produced by a named script from a stored JSON*") and Appendix B ("*77/77 PASS*"). Several other load-bearing numbers are likewise outside the 77-item list: the abstract's "roughly 0.85 px of noise" for the interference floor, the "5.6× the largest free-flight target" ratio, and the 14% shift.

**Required:** ship the S1g report and comparison script; state the correct comparator arm explicitly; either bind 0.0213 to a JSON or remove the claim; and either publish the guard with full coverage or narrow the sentence to "every number listed in Appendix B".

---

### W3 — Preregistered criteria that failed are omitted from the main text; the one experiment the paper most needs is the one it does not report (`BLOCKING`)

**Position:** §5.1, §6.1, Appendix A.

**Evidence.**

- **P0-b (`prereg_P0b_rollout_recipe.json`).** Its own `results_final` block records `P0b-1 不成立`, `P0b-2 不成立`, `P0b-3 不成立` — three of four primary criteria failed. Worse, the **core threat arm failed in the direction opposite to the paper's thesis**: `strong_mlp` went from **1.653 px (A) → 1.650 px (C)**, an improvement of **0%**, despite sitting **3.4× above** the exact floor. §5.1's claim that *"the gains arrange themselves exactly along the distance to the floor"* therefore has an explicit counterexample in the authors' own preregistration — an arm with a large distance-to-floor and zero gain — and the main text reports only the three arms that fit the story (41×, 2.9×, 0%). This is the paper's single most informative negative control and it is silent.
- **S1b-1 (`prereg_S1b_ball_trained.json`).** The criterion "FF-clean one-step: euler001x5 < min(parabola, rk4) < euler005, 3 seeds direction-consistent" has verdict **false**, `per_seed_ok = [false, true, false]` (`S1b_report.json → S1b-1`). The paper reports the failure as "unordered and indistinguishable" but then leans on the 0.001 px separation between the exact tier (0.861) and the parabola tier (0.862) — a difference 100× below the paper's own self-declared 5–9% noise floor. The preregistered verdict is the more honest reading and should be stated as such.
- **S1b-3** (FF-clean closed-loop k=3 ≤ 0.50 px for the exact tier) also failed: 2.040 px.
- **S2 / eval_g4.** `S2_g4_s42.json → criteria` records `G4_A_净收益: false, G4_B_先验负债: false, G4_C_类内辨识: false, G4_D_类外扩展: false` — and in that experiment the *fixed* prior (5.10 px at k=5) beats the NM-fitted prior (5.99 px) on the perturbed world, i.e. the paper's own tool wins. Appendix A asserts "3/3" for the superseding criterion set without disclosing the four failed criteria that share the same runs.

**Required:** add a single mandatory table listing *every* preregistered criterion, its pass/fail verdict, and where it is reported; restore the P0-b failure and the `strong_mlp` counterexample; and state the S1b-1 and S1b-3 verdicts in §6.1.

---

### W4 — The central rhetorical premise ("we corrected the engine's documentation") misdescribes what happened (`BLOCKING`)

**Position:** §1, §3.1, §7; `prereg_S1_ball_ladder.json → analysis_basis`.

**Evidence.** §3.1 says the ball engine's *documentation* specifies semi-implicit Euler ("position increments of v∆t + 15a∆t²"), while the data carry an explicit-Euler signature (10a∆t²). The actual documented-in-the-artifact belief that was falsified is the authors' own: `prereg_S1_ball_ladder.json` states `"引擎位置增益 15·a·dt²（SI@0.01×5）；连续精确 12.5·a·dt²；SI@0.05 25·a·dt²"` and defines `euler005` as *"半隐式欧拉 @0.05×1"* — and the script's own `step_euler_expl` docstring and `S1_ball_ladder.json → protocol_fix` record the correction ("*引擎实测为显式欧拉形式…四档实现按此修正*"). The preregistration even licenses the move: S1-1's note says "*不命中 ⇒ 先查动作对齐/实现（协议修复…），修复后重判*". So the analytic basis, the tier implementations, and the tier ordering *all* changed after the data were seen.

This matters because the change flipped the ordering: under the preregistered basis the ordering would have been SI@0.05×1 worst; under the corrected basis the exact tier wins. The paper converts what is a preregistration violation into one of its most attractive claims ("*Documentation and code disagree at exactly the scale of interest*", §3.1; "*a reader may still object that… matching the generating integrator is trivial. The matching is easy to state and hard to know… ours self-describes as the wrong integrator*", §7). If the "documentation" is the authors' own prior belief, the claim is circular. I could find no external pymunk/chipmunk documentation in the artifacts supporting the 15a∆t² reading.

**Required:** state plainly what the original belief was, where it came from, that the correction was post hoc and licensed by S1-1, and re-label the "documentation vs code" claim accordingly. If an external document does specify semi-implicit Euler, cite it verbatim.

---

### W5 — The identification experiment is a deterministic optimizer artifact presented as a measurement of a computation class (`BLOCKING` for the §4.3/§7 "aliasing law")

**Position:** §4.3, §7; `S2_multiseed_verdict.json`, `eval_nmmulti.py`.

**Evidence.** §4.3 reports ĝ = 705.7 against a true 900, then says: *"The clean dataset is fixed by the preregistration, so all three runs return 705.7333, identical to the last digit — that is a design fact, not evidence of sample-independence."* That sentence concedes the point: the three "independent" runs are the **same deterministic computation** (Nelder–Mead from a fixed start `OFF5.g = 1200`, fixed 2048 windows at `seed=123`, on an unchanged clean dataset — `eval_nmmulti.py::make_id_tensors`, `nm_multistep`). Nothing in the artifacts shows the value is the objective's global optimum rather than a local minimum 21.6% from truth. There is no multi-start, no convergence diagnostic, no basin width. The perturbed-world spread (870.87/880.68/886.45) is also optimizer output on the same fixed window seed across three worlds, and the "refined class" values (957.3/961.4/958.5) are likewise single deterministic NM returns.

The consequence is that §7's practical prescription — *"an identification or filtering pipeline that does not model the prior's computation class is fitting the mismatch, and it will do so reproducibly"* — is justified by (a) an oracle-simulator study with known ground truth, (b) a tool that fails its own ±20% recovery criteria, and (c) a mechanism that only exists because the *model class* was simplified to the truth's functional form with one parameter freed. That is the circular-argument risk in this paper, and it is concentrated in §4.3/§7 rather than in C1.

**Required:** report multi-start statistics and an optimum-certificate argument; state explicitly in §4.3 which criteria failed (the ±20% set) and that the "passing" set reframes the same basin as the target; and scope §7's prescription accordingly.

---

### W6 — The interference floor's mechanism is asserted, not established (`IMPORTANT`)

**Position:** §6.1.

**Evidence.** The causal claim is *"The noise is gradient crosstalk from the contact-dominated majority, not an optimisation artefact."* The supporting ablation is **one run on one seed** (`prereg_S1g_mechanism_ablation.json → seed: [42], n_runs: 1`), and the reported shift (+14%) is described in the same sentence as being *"at the edge of the measured 5–9% floor rather than safely inside it"*. §6.1's own definition of the floor's magnitude ("roughly 0.85 px") does not correspond to any measured head quantity: the head's free-flight output is **0.0197 in normalised units** (`S1b_probe_cached.json`, `S1c_report.json → head_rms_ff` = 0.0197/0.0189/0.0172), while 0.83–0.86 px is the arm's **total** free-flight error (`S1e_error_budget_all_summary.json → baseline.one = 0.8558`, `baseline.head = 0.8319`). Calling a total error "head noise" is the semantic bridge that makes the claim sound measured.

**Required:** report the S1g run's per-seed head RMS with the correct comparator; express the floor in a single stated unit (normalised head output OR px total error) and use it consistently; soften "not an optimisation artefact" to a hypothesis until it has at least 3 seeds.

---

### W7 — Provenance and reproducibility claims outrun the released material (`IMPORTANT`)

**Position:** §3.4, §9, Appendix B.

**Evidence.** (i) The paper claims "*40 guard tests … current pass rate 77/77*" for the number provenance; the scanned Appendix B table prints **61** rows (PDF p. 16–17) while `provenance.tex` is headed "*82 checks*" and the shipped `number_provenance_check.json` contains **77**. Three mutually inconsistent counts across three artifacts. (ii) `verify_numbers.py` reads its "computed" values from the same `analysis_output/*.json` files that the paper's quoted values were taken from — the check is a paper↔JSON consistency check, not a data↔paper check; the §3.4 sentence "*make divergence between the paper's numbers and the artifacts a test failure*" should say exactly that. (iii) `guards/tests/*` import `e0_common`, `model_e0`, `physics_prior_spec`, `layout_h`, `train_v3` and load `trajectories_hinge.npz` / `trajectories_v3.npz` — none of which are in the archive; the two trajectory datasets and all checkpoints are declared excluded. In the archive as shipped, the guard suite and the pipeline cannot be executed. (iv) Appendix B's per-number tolerances are wide relative to the digits quoted (e.g. `S1e W2 one-step FF` tolerance 0.01 on a value quoted to three decimals; `S2 g125 H5 g (min)` tolerance 0.02 on 870.9).

**Required:** make the three counts agree; relabel the check as a document-consistency check; state precisely which scripts and data are needed to reproduce which numbers; and drop "the full training and evaluation pipeline" unless the world generators (`generate_hinge.py`, `generate_trajectories_v3.py`, `generate_g4.py`) and the measurement scripts are shipped.

---

### W8 — The cross-generation "symbolic structure subtracts" claim aggregates non-commensurable, non-independent units (`IMPORTANT`)

**Position:** §5.2, §5.3, Table 5, C3.

**Evidence.** The four "experiment generations" are: one main-table comparison (n=3, closed-loop CI **[−24.98, +79.34]** — uninformative); seven ten-seed gating variants sharing seeds and architecture (the paper itself says the count is one finding and that z is anticonservative); a capacity-decoupled block that is *"paired against the coupled gated arm V3, not against an ungated arm"* (Table 5's own note (i)); and a six-seed supervision 2×2 in a **different world** (ball) on a **coarse-tier** arm (`P0c_compare.json`, thresholds in px at the 7–14 px scale). Only 4 of the 7 pooled variants have a 95% CI excluding zero (`c3_gating_table.json`: V2a [−0.079, +0.321], V2c [−0.057, +0.356], V3 [−0.087, +0.269], V4 [−0.010, +0.502]). The sentence "*A free (unstructured) gate costs the same*" rests on Vfree = +0.068, CI [−0.069, +0.204] — i.e. a null result used as positive evidence. The capacity-decoupled block's **negative** headline point estimate (A1x64 −0.099) is present in Table 5 but absent from the abstract. §5.3's "harm vanishes at exact floor" rests on a **0.0005 px** paired ratio, i.e. below the resolution at which any of these numbers were measured; and the accompanying occupancy audit is intact in both arms (2–15 of 64 experts), so the "absorption" is not a structural change.

**Required:** define the estimand that makes a pendulum one-step k=3 Δ commensurable with a ball ten-seed paired Δ, or stop pooling them into a "generations" count; report the per-generation CI-disjoint count instead of "7/7"; and state the *capacity* confound that the preregistration itself flags (`prereg_S0b…` limitation: "*ours_gua 参数量 −3.9%*", confirmed: 386,664 vs 402,280 in `checkpoints_manifest.json`) wherever "architecture held fixed" is claimed.

---

### W9 — The two headline single numbers in the abstract are not the quantities they are named (`IMPORTANT`)

**Position:** Abstract; §6.1; C1.

**Evidence.** (i) "*a 401k-parameter learned model matches this floor but does not beat it*" is true for the **pendulum** only (`E0_report.json`: trained exact arm k=5 = 0.4822–0.5006 vs prior 0.4949; one-step 0.0364–0.0377 vs prior 0.0367–0.0378). In the **ball** world the trained exact-tier arm is **0.861 px** against its own prior's **3.1e-5 px** — a factor of ~2.8×10⁴ — which §6.1 states correctly but the abstract's "The result replicates across engines" contradicts. (ii) The pendulum's exact tier is *not* float32-exact: 0.038 px one-step, 0.49 px closed-loop, flat across seeds. The true machine-precision tier is the **ball's** (3.1e-5 px). C1's "*A zero-parameter prior with the engine's integrator is exact to float32 precision in closed loop (3 × 10⁻⁵ px per step)*" is stated as a general property of engine-matched priors but holds only where the action schedule is fully observable. The pendulum's residual floor is set by **unobservable intra-interval torque granularity** (documented only in `physics_prior_spec.py`'s module docstring: "τ 的变化不可观测…实测其贡献 ≤0.06px"), never in the paper.

**Required:** separate the two worlds explicitly in the abstract; state the pendulum's 0.038 px floor and its cause; and stop calling the pendulum arm float32-exact.

---

### W10 — Small-sample and metric-design issues that the paper's own accounting exposes (`IMPORTANT` / `MINOR`)

**Position:** §4.1, §4.2, §6.5, §6.6, §4.2 (mixed windows).

**Evidence.** (a) The decisive decode-rule verdict scored **15 free-flight windows per seed and 30 in total** (the paper says so), and §6.6's price is a slope over those. (b) The ball free-flight analysis uses a velocity-consistency filter that discards **~90%** of free-flight transfers (`S1_ball_ladder.json`: `n_ff = 64,463`, `n_ff_clean = 6,597`); on the **unfiltered** free-flight slice the exact tier is *worst* (`one_step.ff`: euler001x5 0.4355 vs euler005 **0.4647** vs parabola 0.5575) — the filter choice is load-bearing and the sensitivity is not reported. (c) "*On mixed windows that include contact, all four tiers compress to 0.9–1.2 px*" (§4.2) inverts the measurement: `mixed_report_only` = euler005 **0.9049**, euler001x5 **1.1071**, parabola/rk4005 **1.2055/1.2059** — the engine-matched tier is 22% *worse* than the coarse tier there, which the "compress" phrasing hides. (d) §4.1's "10% improvement to a 50% degradation" head-contribution range is verified (`0.0978`, `0.5034`) but is computed against each tier's *own* prior one-step error, not a common reference. (e) S1g's one-seed probe is reported in the same sentence as a 3-seed mean (0.0186), mixing n.

**Required:** report the unfiltered-free-flight numbers alongside the filtered ones; give a temperature/robustness check on the filter; correct the "compress to 0.9–1.2 px" sentence.

---

### W11 — Clarity and framing (`MINOR`)

- The paper says the archive contains "twelve preregistrations with an English summary of every criterion" and then gives that document's filename in parentheses; the pointer is unnecessary in a manuscript and reads as a cross-reference outside the paper. Describe the artifact without the filename.
- Figure 1b's axis tick labels are corrupted in rendering ("la", "Eul", "RK4@", "Eu engine)") — legibility, not content.
- Figure 3's caption says "Six arms"; Table 3 lists **seven trained arms** across five families (the paper says "five families, seven trained arms" in the table caption, contradicting Fig. 3).
- §3.1 gives the pendulum window state as "two 11-dim object segments … plus a three-channel contact block" (25), matching `layout_h.py` (`PER_OBJ = 11`, `CONTACT_DIMS = 3`), but the ball's "116 dimensions (33 object, 80 obstacle, 3 contact)" sums to 116 — verify against `layout.py` and give both layouts in one place.
- S1f's preregistration promises the full τ-curve appendix and a median/all-silenced robustness reading; neither is in the shipped report.
- Terminology: "interference floor", "the floor", "the analytic identity", and "silence" are each used for two different quantities (normalised head RMS vs px arm error; oracle mask vs rule-based gate). One definition table would resolve W1, W6 and W9 partially.

---

## Claim–Evidence Mismatches

| # | What the paper says (location) | What the material actually contains (source field) |
|---|---|---|
| 1 | "*Free-flight closed-loop error under the rule is 10⁻⁴ px, in every seed*" — presented as the output of the deployed contact-head-threshold rule (§6.5) | 0.000103 is the **max over arms/seeds** of the S1e **oracle** hard-mask probe (`number_provenance_check.json`; `S1e_error_budget_all_summary.json → prior = 8.767e-5` for all arms); the rule-based S1f value is `7.86e-5` (`S1f_report.json → R1f.baseline.ff_rms`) |
| 2 | "*the preregistered operating point—the largest τ whose false-silence rate stays at or below 2%—sits at 1.9–2.0%*" (§6.6) | τ\* = 0.75 has val-A false silence **1.9945%** and val-B **2.007%** (`S1f_report.json → grid_A/grid_B/verdict_B`); the floor is reached only at τ = 0.75, so the rule operates exactly at its own boundary, on a grid coarsened from the registered 0.01 to 0.05 step |
| 3 | "*excluding free-flight windows … leaves the noise at 0.0213 against the S1c reweighting baseline 0.0186 — a shift of +14%*" (§6.1) | 0.0213 is in **no JSON** in the archive; `S1c_report.json → verdicts.wff10.head_rms_ff = 0.009493` (the named arm), while 0.0186 is the **un-reweighted** baseline (`S1c_report.json → baseline.head = 0.018574`). No `S1g_report.json` is shipped at all |
| 4 | "*the gains arrange themselves exactly along the distance to the floor*" (§5.1) | `prereg_P0b_rollout_recipe.json → results_final`: the **strong_mlp** arm, 3.4× above the floor, gains **0%** (1.653 → 1.650); three of four primary criteria recorded 不成立 |
| 5 | "*a 401k-parameter learned model matches this floor but does not beat it … The result replicates across engines*" (Abstract, C1) | Pendulum only. Ball: trained exact-tier arm **0.861 px** vs its prior **3.1e-5 px** (`S1b_report.json → summary.euler001x5.one_ff_clean`, `S1_ball_ladder.json → one_step.euler001x5.ff_clean`) — ~2.8×10⁴× above the floor |
| 6 | "*A zero-parameter prior with the engine's integrator is exact to float32 precision in closed loop (3×10⁻⁵ px per step)*" (C1) | True for the ball; the pendulum's exact tier is **0.0378 px one-step / 0.4949 px k=5** and flat at ~0.038 across seeds — limited by unobservable torque granularity, documented only in `physics_prior_spec.py`'s docstring |
| 7 | "*the 2% of contact windows falsely silenced … contact-regime closed-loop error rises by a factor of 1.12–1.21*" (§6.5) | Internally inconsistent: a 2% poisoning of a 2.12 px pool cannot raise it 21%. `S1e_error_budget_all_summary.json` shows the head term is the whole error (`baseline.head = 0.8319` of `one = 0.8558`), and `S1f_report.json → verdict_B` shows contact RMS 2.1247 → 2.5715 at `false_silence = 0.02007` — a threshold/level mismatch of roughly an order of magnitude |
| 8 | "*it emits roughly 0.85 px of noise*" (Abstract, §6.1) | The head's free-flight output is **0.0197 normalised** (`S1b_probe_cached.json`); 0.83–0.86 px is the arm's **total** free-flight error (`S1e_error_budget_all_summary.json → baseline.one = 0.8558`, `.head = 0.8319`) |
| 9 | "*the ball engine's documentation specifies a different integrator from the one its code implements*" (§1, §3.1, §7) | The falsified belief is the authors' own preregistration: `prereg_S1_ball_ladder.json → analytic_basis` asserts 15·a·dt² (SI@0.01×5) and defines euler005 as semi-implicit; `S1_ball_ladder.json → protocol_fix` records the post hoc correction to 10·a·dt² |
| 10 | "*The engine check comes first and it is the reason to trust everything after it*" — reproducing recorded transitions to 3×10⁻⁵ px (§4.2) | The engine check is **not run in the paper's pipeline**: the 3.1e-5 px figure is that prior tier's *free-flight error against the recorded data* (`S1_ball_ladder.json → one_step.euler001x5.ff_clean`); no code path compares a prior rollout against the engine itself (the S1e budget's `ident = 3.34e-5` is a separate zero-head accounting term) |
| 11 | The 6,597-transfer scope limitation is stated ("*6,597 transfers passing a contact-flag and velocity-consistency filter*", §4.2) | On the **unfiltered** free-flight slice (`n_ff = 64,463`) the ranking flips: exact 0.4355 > euler005 0.4647 in the wrong direction vs euler005, and 0.4355 vs 0.5575 for the others — the ~90% filter is load-bearing and unreported |
| 12 | "*On mixed windows … all four tiers compress to 0.9–1.2 px*" (§4.2) | `S1_ball_ladder.json → mixed_report_only`: the engine-matched tier is **1.107**, the coarse tier **0.905** — the exact tier is 22% *worse*, so "compress" understates a reversal |
| 13 | "*All quantities are in the engine's native pixel units*"; "capacity [is] identical across cells" (§5.2, §3.2) | `checkpoints_manifest.json`: ours_gua = **386,664** vs ours_base = **402,280** params (−3.9%); the preregistration flags this explicitly, and the paper's own `provenance.tex` reports checkpoints from **339,076 to 615,158** params against the text's "401k to 598k" |
| 14 | Appendix B: "*Current status: 77/77 PASS*" (§9, Appendix B) | Three counts disagree: PDF table **61 rows**, `provenance.tex` header "**82 checks**", `number_provenance_check.json` "**77**". The check compares paper numbers to JSON that the paper's numbers were taken from — a consistency, not a provenance, test |
| 15 | "*the evidence is the perturbed worlds: on three independently regenerated worlds … returns 870.9, 880.7 and 886.4*" (§4.3) vs "*all three runs return 705.7333, identical to the last digit — that is a design fact, not evidence of sample-independence*" | Both true of the same procedure: `eval_nmmulti.py` uses a **fixed** window seed 123 and a fixed start `OFF5.g = 1200`, so the "identical" runs are a deterministic NM return, and the spread measures world regeneration, not estimator variance. No multi-start or optimality evidence exists in the artifacts |

---

## Required Revisions

### Blocking
1. **Resolve the decode-rule provenance conflict (W1).** Separate the oracle probe from the deployed rule everywhere; re-run the τ grid at the registered 0.01 resolution; report the free-flight error at the grid point the registered rule actually selects; give the window-level false-silence rate and the all-silenced fraction; define the damage metric and reconcile 2% with ×1.21. If the registered rule cannot reach parity, say so and withdraw "restores parity … at a two-percent false-silence rate" from the abstract.
2. **Fix or withdraw the 0.0213/0.0186/+14% ablation (W2).** Ship the S1g JSON and script, name the correct comparator, and either bind the number or delete the causal "gradient crosstalk" sentence. Correct the §9 claim to match the actual guard coverage.
3. **Publish all preregistered verdicts (W3).** One table: criterion, test, verdict, location. Restore the P0-b failures (including the `strong_mlp` 0% counterexample), S1b-1 (2/3 seeds fail), S1b-3, and the four failed `eval_g4` criteria.
4. **Correct the "documentation vs code" narrative (W4) and the identification claims (W5).** State the original preregistered basis, that the correction was post hoc and preregistration-licensed, and re-derive the narrative accordingly. For §4.3, report multi-start statistics and label the value as an optimizer output consistent with a compensation basin, not a sample-independent property of the class.
5. **Reconcile the abstract with §6.1 (W9):** the ball world does *not* replicate the pendulum's trained-arm floor result, and the pendulum's exact tier is *not* float32-exact.

### Important
6. Correct the "0.85 px of noise" quantity and use one unit consistently (W6); state the interference-floor mechanism as a hypothesis pending ≥3 seeds.
7. Report the unfiltered free-flight slice and the filter's sensitivity; correct the "compress to 0.9–1.2 px" sentence (W10).
8. Either make the guard suite and pipeline runnable from the archive, or narrow the reproducibility statement to what the archive actually supports, and state the data/checkpoint dependency per claim (W7).
9. State the ≤3.9% capacity offset wherever "architecture held fixed" is claimed, and give the C3 pooling rule (W8).
10. Report the pendulum's 0.038 px floor cause (intra-interval torque granularity) in the main text (W9).

### Minor
11. Fix the archive/manuscript inconsistencies: provenance counts (61/77/82), model-size range (401k–598k vs 339k–615k), Fig. 3 "six arms" vs Table 3's seven, the parenthetical filename pointer in §9, Fig. 1b tick labels.
12. Add a notation table defining "floor" (computation-class floor vs interference floor), "silence" (oracle vs rule), and the px/normalised distinction.
13. Report the S1f preregistered robustness readings (per-seed median, all-silenced fraction).

---

## Overall Assessment

The paper contains one genuinely interesting and carefully isolated empirical fact — the ball-world analytic ladder — and one genuinely honest negative result — that training erases that ladder. Those two, plus the supervision 2×2, would make a solid short paper. What surrounds them is a framing that repeatedly outruns the artifacts: an oracle probe presented as a deployment rule, a mechanism ablation whose central number is untraceable, a "documentation was wrong" narrative that is really "our preregistration's analytic basis was wrong and we fixed it after seeing the data," an identification study that is a deterministic Nelder–Mead output reframed as a law of computation classes, and a set of preregistered failures that appear only in Appendix A or not at all. The preregistration apparatus is the paper's best asset and simultaneously its sharpest critic: nearly every place where the manuscript's narrative is weakest is a place where a preregistered verdict in the shipped JSON says something different.

The claim/evidence mismatches I found are not fatal individually, and the arithmetic is largely faithful. But at least three of them (W1, W2, W3) touch the abstract's headline sentences, and my attempts to verify the paper's strongest rhetorical claim (§7's "matching the integrator is hard to *know*") led to the authors' own preregistration rather than to any external documentation. That is the pattern a TMLR reviewer is asked to catch: a paper whose stated protocol is more rigorous than the protocol the artifacts record.

**Decision: Major revision.**
**Confidence: 4/5** (high on the claim–evidence analysis and on all numbers I re-derived; I did not re-run any training, so I cannot confirm or deny that the excluded checkpoints would reproduce the reported metrics).

If the authors address blocking items 1–5 with the artifacts to back them, the core contribution (C1) survives in a narrower form: *in a contact-free simulated world whose action schedule is fully observable, matching the generating discrete map collapses the residual floor to machine precision, and the residual head can then contribute nothing* — which is true, useful, and worth publishing. The broader claims (C2's "distance to the floor" law, C4's exchange rate as quoted) do not currently survive their own preregistered data.

---

## The Single Question I Would Ask the Authors

**For every number in §6.5–§6.6 — the 10⁻⁴ px free-flight closed-loop parity, the "two-percent false-silence rate," and the 1.12–1.21× contact cost — please state, individually and unambiguously, whether it comes from the oracle hard-silence mask (`S1e_error_budget`), from the model's own contact-head threshold rule (`S1f_report`), or from the τ = 0.95 endpoint of the scan; and if the reported parity depends on the oracle, what does the *deployed* rule achieve on the windows the registered 0.01-step grid would have selected?**

---

### 中文备注（审稿人自用）

- **判据**：我把"论文声称的协议"当作审查对象，而不是把"论文自报的通过率"当作事实。凡是正文叙述与其自身预注册或 JSON 冲突的地方，一律以预注册/JSON 为准，并在上表给出字段名。
- **是否循环论证**：C1 本身**不是**循环论证（"数学精确的抛物线输给引擎同款积分器"是一个真实、可反驳的测量）；但 §4.3/§7 的"混叠律"**是**（用一个已知真值的 oracle 仿真 + 一个简化到真值函数形式、只放开一个参数的模型类，去"测量"计算类的影响），W5 即为此。
- **最严重的三条**：W1（oracle 探针冒充部署规则 + 阈值网格被粗化后恰好踩线）、W3（预注册失败项未进正文，尤其是 strong_mlp 距离地板远却零增益这一反例）、W2（0.0213 在任何 JSON 中都不存在，且所引基线臂张冠李戴）。W4（把"我们自己的预注册 analytic basis 写错了"讲成"引擎文档与代码不符"）紧随其后。
- 我没有打开任何文件名含 REVIEW / review / 评审 的文件，也没有打开论文目录下的任何说明性 .md（含 README、QC_LOG 一类）；本评审的每一条依据都来自 main.pdf / main.tex、analysis_output/*.json、experiments/prereg_*.json、tests/ 与仓库 Python 源码。全程未修改论文文件（paper/latex/** 与 paper/*.md 仅被新建的 ADVERSARIAL_REVIEW.md 之外的内容未受影响）。
