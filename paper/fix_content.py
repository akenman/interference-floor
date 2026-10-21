# -*- coding: utf-8 -*-
"""修复 main.tex：①转义正文中的裸 %；②加入 Fig 2；③补 nbf2025 引用。"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'paper', 'latex', 'main.tex')
s = open(p, encoding='utf-8').read()
BS = chr(92)

# --- 1) 转义正文中的裸 %（跳过已有 \% 和注释行和 preamble）---
body_start = s.find(BS + 'begin{document}')
pre, body = s[:body_start], s[body_start:]
lines = body.split('\n')
fixed = []
n_fixed = 0
for ln in lines:
    stripped = ln.lstrip()
    if stripped.startswith('%'):
        fixed.append(ln)
        continue
    out = []
    i = 0
    while i < len(ln):
        if ln[i] == '%' and (i == 0 or ln[i-1] != BS):
            # 跳过 colorlinks 等 preamble 模式（已在 pre 中）
            out.append(BS + '%')
            n_fixed += 1
        else:
            out.append(ln[i])
        i += 1
    fixed.append(''.join(out))
body = '\n'.join(fixed)
s = pre + body
print(f'1. 转义裸 %: {n_fixed} 处')

# --- 2) 在 §5.2 末尾（吸收段后）加 Fig 2 引用 ---
# 找到 §5.3 末段（absorbed 那段）的末尾，在 \section{The Trained Arm 前插入
fig2_block = """
\\begin{figure}[t]
\\centering
\\includegraphics[width=0.82\\textwidth]{figs/fig2_p0c_2x2.pdf}
\\caption{Supervision $2{\\times}2$ results (means over six seeds; dashed line is the R2 tolerance). (a)\\ One-step error. (b)\\ Closed-loop error at $k{=}5$. The supervised cells (C10, C11) are consistently worse than the unsupervised ones (C00, C01).}
\\label{fig:p0c}
\\end{figure}
"""
anchor = 'The structure still wastes capacity; the waste no longer costs anything, because the residual head has nothing to be wrong about. This is why the subtraction result and the floor result are one result: the harm of structure is a tax on the residual signal, and the residual signal is bounded by the floor.'
if anchor in s and 'fig2_p0c_2x2' not in s:
    s = s.replace(anchor, anchor + '\n' + fig2_block + '\n', 1)
    # 在 §5.2 文字中加引用
    s = s.replace(
        'the cost is paid by a head that demonstrably learned what we asked it to learn.',
        'the cost is paid by a head that demonstrably learned what we asked it to learn (Fig.~\\ref{fig:p0c}).'
    )
    print('2. Fig 2 已加入 §5.2')
elif 'fig2_p0c_2x2' in s:
    print('2. Fig 2 已存在，跳过')

# --- 3) 在 §7 Discussion 的 future-work 段补 nbf2025 引用 ---
if 'nbf2025' not in s:
    old = 'Cultural priors, in our experience, earn their place as instruments, not components.'
    new = old + ' The filtering direction---maintaining beliefs over hidden states with an explicit computation-class prior---connects to Neural Bayesian Filtering~\\citep{nbf2025}, which we defer to future work.'
    s = s.replace(old, new, 1)
    print('3. nbf2025 引用已补（Discussion future-work）')
else:
    print('3. nbf2025 已被引用')

open(p, 'w', encoding='utf-8').write(s)
print('main.tex 修复完成')
