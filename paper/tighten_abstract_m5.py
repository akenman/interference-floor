# -*- coding: utf-8 -*-
"""M5 第三步：摘要压缩到 ~280 词（保留全部实质内容，删冗余）。"""
import io

P = "paper/latex/main.tex"
s = io.open(P, encoding="utf-8").read()
i = s.index("\\begin{abstract}")
j = s.index("\\end{abstract}") + len("\\end{abstract}")
new = r"""\begin{abstract}
A learned residual corrector bolted onto an analytic prior inherits whatever the prior
already gets right and is asked to repair the rest. In the contact-rich world we study,
that division of labour fails in a way that is easy to miss and easy to measure: on the
free-flight windows where the prior is already exact, the head emits $0.017$--$0.027$
(normalised RMS) at every one of four prior tiers whose targets span five orders of
magnitude. The head is closer to a constant than to anything the prior asks of it. We
call this the \emph{interference floor}, and the same architecture trained on a
contact-free world does not have one---there the head tracks its target, from
$0.98 \pm 0.02$ at the coarsest tier to $0.13 \pm 0.01$ at the exact tier ($n{=}7$ each,
$z{=}48$). The floor is a property of regime imbalance, not of the architecture. Five
training-side interventions and a decode-time gate then trace a single
false-silence-versus-damage trade-off; the gate restores parity with the analytic prior
on the thirty free-flight validation windows it scores, at a $4.2$--$4.7\%$ window-level
false-silence rate and a $12$--$21\%$ rise in contact-regime closed-loop error. We
remove three candidate explanations for the floor and identify one term that sustains
it: the closed-loop rollout loss, fit on all windows, raises free-flight error by
$1.6$--$4.7\times$ at matched fit. Two measurements scope the result. What caps how good
any of this can get is the prior's computation class---the equations and the numerical
scheme used to evaluate them---which lowers closed-loop error from $14.26$\,px to
$0.49$\,px when swapped for the class that generated the data. And the head is not
worthless in general: when the prior's parameter is wrong or a force is unmodelled, the
same head removes $89$--$97\%$ of the error. The floor is the price of correction on the
one regime where no correction is needed. Preregistered criteria and guard tests binding
every headline number to a script accompany all of it, including the criteria our own
claims failed.
\end{abstract}"""
s = s[:i] + new + s[j:]
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("abstract words:", len(new.split()))
