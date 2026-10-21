# -*- coding: utf-8 -*-
"""M5 结构重排：把 §6（干扰地板）移到 §5（结构/配方）之前，并加入 E5 证据。

新顺序：
  4 The Prior Ladder Sets the Floor        （前提：地板由计算类设定）
  5 The Interference Floor and the Cost of Silence （主贡献，原 §6 + 新增"何时值得要修正器"）
  6 Structure and Recipe Cannot Substitute （次级：两个方向的杠杆）

附录 E（When the corrector is worth having）的内容提升为 §5 的一个子节，
附录本身删除，其 M3/M4 证据并入该子节。

用法：python -X utf8 paper/restructure_m5.py   （一次性脚本，跑完即归档）
"""
import io
import re

P = "paper/latex/main.tex"
s = io.open(P, encoding="utf-8").read()

# ---------------------------------------------------------------- 1) 切出三大块
i4 = s.index("\\section{The Prior Ladder Sets the Floor}")
i5_old = s.index("\\section{Structure and Recipe Cannot Substitute}")
i6_old = s.index("\\section{The Trained Arm: Interference")
i_disc = s.index("\\section{Discussion}")

head = s[:i4]
sec_ladder = s[i4:i5_old]              # §4 保持
sec_structure = s[i5_old:i6_old]       # 原 §5 → 新 §6
sec_interf = s[i6_old:i_disc]          # 原 §6 → 新 §5
tail = s[i_disc:]

assert sec_interf.startswith("\\section{The Trained Arm: Interference")
assert sec_structure.startswith("\\section{Structure and Recipe Cannot Substitute")

# ---------------------------------------------------------------- 2) 追加 E5 子节到 §5 末尾
E5_SUBSECTION = r"""
\subsection{When the corrector is worth having}
\label{sec:when}

Everything above describes what the head costs on the windows where the prior is
already right. That raises the obvious question the paper would otherwise leave open:
is the head ever worth having? We can answer it, because the answer turns on the
\emph{kind} of mismatch rather than its size. Every prior in \S\ref{sec:floor} and
\S\ref{sec:silence} is wrong in the same way---through its discretisation, an error that
is per-window exact under a fixed substep and a constant action, and therefore leaves
nothing systematic to learn. We therefore varied the kind of mismatch while holding the
architecture, data budget, schedule and window convention fixed, and measured the head's
contribution under three priors: one that is numerically right, one whose gravity
parameter is wrong by $25\%$, and one that omits a drag and Coulomb-friction term
present in the data. The window convention, action index and metric follow
\S\ref{sec:platform}'s evaluation path and reproduce its published prior error to three
decimals ($0.0377$\,px against $0.038$).

\begin{table}[t]
\centering
\caption{The head's contribution as a function of what is wrong with the prior
(pendulum, seed 42, one run per cell, weighted position RMSE in px). ``Matched'' is the
control: the prior is numerically correct, and the head returns $-0.4\%$ in closed loop,
reproducing \S\ref{sec:silence}'s claim. Under a wrong parameter or an unmodelled force
the head removes $89$--$97\%$ of the closed-loop error it is handed. The last two rows
separate which part of the model does the work: freezing the trunk on \emph{matched}
data---so its hidden state cannot encode the mismatch---recovers almost all of the gain,
which places the contribution in the head rather than the trunk.}
\label{tab:corrector}
\small
\begin{tabular}{llccc}
\toprule
Condition & What is wrong with the prior & one-step (px) & closed $k{=}5$ (px) & $k{=}5$ gain \\
\midrule
prior & --- & 0.0377 & 0.4939 & --- \\
\quad + head & nothing (control) & 0.0428 & 0.4959 & $-0.4\%$ \\
\midrule
prior & gravity off by $25\%$ & 0.2123 & 5.0409 & --- \\
\quad + head & & 0.0516 & 0.5410 & $+89.3\%$ \\
\quad + head, frozen trunk & & 0.0443 & 0.3649 & $+92.8\%$ \\
\midrule
prior & unmodelled drag and friction & 0.2335 & 5.1126 & --- \\
\quad + head & & 0.0177 & 0.1569 & $+96.9\%$ \\
\quad + head, frozen trunk & & 0.0275 & 0.1905 & $+96.3\%$ \\
\bottomrule
\end{tabular}
\end{table}

Three things follow, and together they bound the paper's negative results. First, the
claim that the head is worthless is specific rather than general: it holds when the
prior's only error is its discretisation, and it reverses by an order of magnitude when
the prior is wrong in a way that is smooth, persistent and inferable from observations.
A reader should not conclude that residual correction is worthless; the correct
conclusion is that it is worth as much as the prior is wrong \emph{learnably}, and
nothing on the windows where the prior is already right. Second, the mechanism is
implicit parameter identification performed by the head: the trunk is not required to
encode the mismatch, since a head trained on a trunk fitted to matched data recovers the
same gain. That places this paper next to the classical online-identification literature
rather than apart from it---the head is estimating parameters implicitly, and what this
paper adds is the cost of doing it where no estimation is needed. Third, that ordering
has a practical consequence we did not expect: on the wrong-parameter prior, freezing
the trunk and adapting only the head is not merely as good as training both, it is
better, lowering closed-loop error from $0.488$ to $0.390$ over four seeds ($25.4\%$,
per-seed spread $6.2\%$, $t{=}8.2$, same direction on $4/4$). On the unmodelled-force
prior the same comparison gives $-3.5\%$ with an $18.3\%$ spread ($2/4$): no effect. So
we report it for the parameter-error case only, and read it as the trunk overfitting the
majority regime when it is allowed to keep training on data whose mismatch lives in the
minority windows.

One more measurement belongs here, because it decides how far the interference floor
generalises. On the pendulum the residual targets of four priors span $100\times$, and
the head's free-flight output tracks its target in every one of them: averaged over
seeds the ratio is $0.98 \pm 0.02$ at the coarsest tier and $0.13 \pm 0.01$ at the exact
tier ($n{=}7$ each), a separation of $z{=}48$, with the intermediate tiers between them.
In the ball world the same ratio runs from $1.0$ to $1.4\times10^{5}$. The floor is
therefore not a general property of a shared residual head. It appears when the minority
window's target is small relative to the majority's, which is the ball's situation
(contact targets near $0.015$ against free-flight targets near $1.4\times10^{-7}$) and
not the pendulum's, where all targets are the same order. That is why this section's
measurements are reported for the contact-rich world, and why we do not claim the floor
for worlds whose loss is not dominated by a regime whose windows look different.

Finally, we carried the decode rule into the matched-mismatch setting, since the
pendulum exposes an analogous contact channel through its phase head and is $88\%$ free
flight. The rule is monotonically harmful there: raising the threshold so that it
actually closes windows moves closed-loop error back toward the prior, from $0.541$
ungated to $0.586$ with every free-flight window silenced under the wrong-parameter
prior, and from $0.157$ to $0.262$ under the unmodelled force. The reason is the same
fact that makes the head useful above---when the head beats the prior in \emph{both}
regimes, a rule that silences it on one of them can only remove work. The gate is
therefore not a general improvement but a response to a specific condition, namely the
head being net harmful on the regime it silences; that condition is what this section
measures, and stating it as a condition rather than a technique is what we take to be
the transferable part.

"""

# §5 标题与导语改写（主线）
sec_interf = sec_interf.replace(
    "\\section{The Trained Arm: Interference and the Price of Silence}\\label{sec:silence}",
    "\\section{The Interference Floor and the Cost of Silence}\\label{sec:silence}",
    1)

# 在 §5 末尾（即 sec_structure 之前）插入 E5 子节
sec_interf = sec_interf.rstrip() + "\n" + E5_SUBSECTION

# ---------------------------------------------------------------- 3) 新 §6 的导语改一句（避免"上节"歧义）
sec_structure = sec_structure.replace(
    "\\subsection{The harm is absorbed by the floor}",
    "\\subsection{The harm is absorbed by the floor}", 1)

# ---------------------------------------------------------------- 4) 拼回：§4 → §5(干扰+E5) → §6(结构) → tail
s2 = head + sec_ladder + sec_interf + sec_structure + tail

# ---------------------------------------------------------------- 5) 删除附录 E（内容已并入 §5）
a_start = s2.index("\\section{When the corrector is worth having}")
a_end = s2.index("\\end{document}")
s2 = s2[:a_start].rstrip() + "\n\n\\end{document}\n"

# ---------------------------------------------------------------- 6) 全局替换指向附录 E 的引用 → §5
s2 = s2.replace("Appendix~\\ref{app:corrector}", "\\S\\ref{sec:when}")
s2 = s2.replace("(\\S\\ref{sec:when})", "(\\S\\ref{sec:when})")

io.open(P, "w", encoding="utf-8", newline="\n").write(s2)
print("restructured.")
print("  app:corrector refs remaining:", s2.count("app:corrector"))
print("  sec:when present:", s2.count("sec:when"))
print("  section order:")
for m in re.finditer(r"\\section\{([^}]*)\}", s2):
    print("    ", m.group(1)[:60])
