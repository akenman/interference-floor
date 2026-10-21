# -*- coding: utf-8 -*-
"""M5 第二步：摘要重写（主线换成干扰地板）+ Introduction/Discussion 的枚举同步。"""
import io

P = "paper/latex/main.tex"
s = io.open(P, encoding="utf-8").read()

# ---------------------------------------------------------------- 摘要
old_start = s.index("\\begin{abstract}")
old_end = s.index("\\end{abstract}") + len("\\end{abstract}")
new_abstract = r"""\begin{abstract}
A learned residual corrector bolted onto an analytic prior inherits whatever the prior
already gets right, and is asked to repair the rest. In the contact-rich world we study,
that division of labour fails in a way that is easy to miss and easy to measure: on the
free-flight windows where the prior is already exact, the head emits $0.017$--$0.027$
(normalised RMS) at every one of four prior tiers whose targets span five orders of
magnitude. The head is closer to a constant than to anything the prior asks of it. We
call this the \emph{interference floor}, and the same architecture trained on a
contact-free world does not have one---there the head tracks its target, at a ratio of
$0.98 \pm 0.02$ at the coarsest tier down to $0.13 \pm 0.01$ at the exact tier
($n{=}7$ each, $z{=}48$). The floor is a property of regime imbalance, not of the
architecture. Five training-side interventions and a decode-time gate then trace a
single false-silence-versus-damage trade-off; the gate restores parity with the analytic
prior on the thirty free-flight validation windows it scores, at a $4.2$--$4.7\%$
window-level false-silence rate and a $12$--$21\%$ rise in contact-regime closed-loop
error. We chase the cause and remove three candidate explanations, then identify one
term that does counteract silence: the closed-loop rollout loss, fit on all windows,
raises free-flight error by $1.6$--$4.7\times$ at matched fit. Two questions scope the
result. What sets how good any of this can get is the prior's computation class---the
pair of equations and the numerical scheme used to evaluate them---which lowers
closed-loop error from $14.26$\,px to $0.49$\,px when swapped for the class that
generated the data. And the head is not worthless in general: when the prior's
parameter is wrong or a force is unmodelled, the same head removes $89$--$97\%$ of the
error, so the floor is the price of correction on the one regime where no correction is
needed. Preregistered criteria, guard tests binding every headline number to a script,
and a measured reproduction floor accompany all of it, including the criteria our own
claims failed.
\end{abstract}"""
s = s[:old_start] + new_abstract + s[old_end:]

# ---------------------------------------------------------------- Introduction 的贡献列表重排
old_c1 = s[s.index("  \\item \\textbf{C1 --- The floor.}"):s.index("  \\item \\textbf{C2 --- The multiplier.}")]
old_c2 = s[s.index("  \\item \\textbf{C2 --- The multiplier.}"):s.index("  \\item \\textbf{C3 --- The subtraction.}")]
new_c1 = ("  \\item \\textbf{C1 --- The interference floor.} In the contact-rich world the "
          "residual head's output on the windows where the prior is already exact is set by the "
          "training mixture, not by the prior: $0.017$--$0.027$ across four tiers whose targets "
          "span $10^{5}$, and $1.0$--$1.4\\times10^{5}$ times its own target at the exact tier. "
          "The same architecture in a contact-free world tracks its target ($0.13$--$0.98$), so "
          "the floor is a property of regime imbalance (\\S\\ref{sec:silence}).\n")
new_c2 = ("  \\item \\textbf{C2 --- The exchange rate, and one term that sustains it.} Five "
          "interventions and a decode rule lie on one false-silence-versus-damage curve, which we "
          "quote as a measured rate; three candidate explanations are removed and the closed-loop "
          "rollout loss is identified as a term that actively counteracts silence "
          "(\\S\\ref{sec:silence}).\n")
s = s.replace(old_c1, new_c1, 1)
s = s.replace(old_c2, new_c2, 1)

# 原 C1 的内容并入新的 C3（先验计算类）——把旧 C3/C4 之后的编号顺延
s = s.replace("  \\item \\textbf{C3 --- The subtraction.}",
              "  \\item \\textbf{C3 --- The floor the prior sets.} The prior's computation class "
              "caps what any residual head can reach: swapping it for the class that generated the "
              "data lowers closed-loop error from $14.26$ to $0.49$\\,px, and a zero-parameter "
              "prior with the engine's integrator ties a 401k-parameter learned model "
              "(\\S\\ref{sec:floor}).\n"
              "  \\item \\textbf{C4 --- Correction is worth what the prior is wrong learnably.} On "
              "a control where the prior is merely discretised the head returns $-0.4\\%$; when the "
              "prior's parameter is wrong or a force is unmodelled the same head removes "
              "$89$--$97\\%$ of the error, and freezing the trunk does better still "
              "(\\S\\ref{sec:when}).\n"
              "  \\item \\textbf{C5 --- The subtraction.}", 1)
s = s.replace("  \\item \\textbf{C4 --- The interference floor and its exchange rate.}",
              "  \\item \\textbf{C6 --- Symbolic structure as a measured tax.}", 1)
s = s.replace("  \\item \\textbf{C5 --- Methodology as a working instrument.}",
              "  \\item \\textbf{C7 --- Methodology as a working instrument.}", 1)

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("abstract rewritten; contribution list reordered")
print("  abstract words:", len(new_abstract.split()))
