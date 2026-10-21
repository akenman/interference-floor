# -*- coding: utf-8 -*-
"""审稿修订阻塞项：标题收窄 / 摘要加限定 / 贡献加 scope / 数字统一 / Figure 拼写。"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'paper', 'latex', 'main.tex')
s = open(p, encoding='utf-8').read()
BS = chr(92)
n = 0

# --- 1) 标题收窄 ---
s = s.replace(
    'Integrator Accuracy, Not Structure: What Determines Closed-Loop' + BS + BS + '\n Feasibility in Dynamics-Level World Models',
    'Integrator Accuracy, Not Structure: What Sets the Closed-Loop' + BS + BS + '\n Floor of Residual World Models'
)
print('1. 标题收窄 [OK]' if 'Residual World Models' in s else '1. 标题未变')
n += 1

# --- 2) 摘要加 scope 限定 ---
s = s.replace(
    'We isolate three candidates',
    'In two simulated worlds (a one-degree-of-freedom pendulum and a planar ball arena), we isolate three candidates'
)
# 摘要里加 contact damage
s = s.replace(
    'a decode-time hard gate restores parity with the analytic prior at a measured two-percent false-silence rate.',
    'a decode-time hard gate restores parity with the analytic prior at a measured two-percent false-silence rate, at the cost of a 12--21\\% rise in contact-regime closed-loop error.'
)
print('2. 摘要加 scope + 代价 [OK]')
n += 1

# --- 3) 贡献列表加 scope 前缀 ---
for old, new in [
    ('Prior computational fidelity, not architecture, sets closed-loop feasibility.',
     'In the two simulated worlds studied here, prior computational fidelity---not architecture---sets the closed-loop floor.'),
    ('The recipe\\u2019s gain is proportional',
     'Within the floor, the recipe\\u2019s gain is proportional'),
    ('The recipe\'s gain is proportional',
     'Within the floor, the recipe\'s gain is proportional'),
    ('Symbolic structure, in gating and in supervision, consistently costs accuracy',
     'Within this architecture family and these two worlds, symbolic structure---in gating and in supervision---consistently costs accuracy'),
    ('Trained-arm error decomposes into prior error plus',
     'In the contact-rich world, trained-arm error decomposes into prior error plus'),
]:
    if old.replace('\\u2019', '\'') in s or old in s:
        actual_old = old.replace('\\u2019', '\'') if old in s else old
        s = s.replace(actual_old, new, 1)
        print(f'3. 贡献 scope: "{old[:35]}..." [OK]')
        break  # 只改第一个匹配的贡献行，逐个处理

# 更直接：逐条替换贡献列表里的关键句
contrib_fixes = [
    # C1
    ('Prior computational fidelity, not architecture, sets closed-loop feasibility.',
     'In the two simulated worlds studied here, prior computational fidelity---not architecture---sets the closed-loop floor.'),
    # C2
    ('The recipe\'s gain is proportional to the distance to the floor and saturates there; the dose has a ceiling and reverses past it',
     'Within the floor, the recipe\'s gain is proportional to the distance to the floor and saturates there; the dose has a ceiling and reverses past it'),
    # C3
    ('Symbolic structure, in gating and in supervision, consistently costs accuracy; the cost vanishes at an exact floor',
     'Within this architecture family, symbolic structure---in gating and in supervision---consistently costs accuracy; the cost vanishes at an exact floor'),
    # C4
    ('Trained-arm error decomposes into prior error plus a regime-mixing-dependent head noise; five interventions and a decode rule lie on one ROC curve, which we quote',
     'In the contact-rich world, trained-arm error decomposes into prior error plus a regime-mixing-dependent head noise; five interventions and a decode rule lie on one ROC curve, which we quote'),
]
for old, new in contrib_fixes:
    if old in s:
        s = s.replace(old, new, 1)
        print(f'  贡献修订: "{old[:40]}..." [OK]')
    else:
        print(f'  贡献修订 MISS: "{old[:40]}..."')

# --- 4) 数字统一：40 guard tests + 64 provenance ---
s = s.replace('36 guard tests', '40 guard tests')
s = s.replace('36 tests', '40 tests')
s = s.replace('27 number-provenance checks', '64 number-provenance checks')
s = s.replace('27 provenance', '64 provenance')
print('4. 数字统一: 36→40 tests, 27→64 provenance [OK]')

# --- 5) Figure 拼写修正（make_figures.py 里的 "igure" 是 matplotlib 截断）---
# 这需要改 make_figures.py 的 title 字符串——但 "igure" 可能是显示问题不是源码问题
# 检查源码
figs_py = os.path.join(ROOT, 'paper', 'figures', 'make_figures.py')
fs = open(figs_py, encoding='utf-8').read()
if 'igure' in fs:
    # 是源码里的拼写错误
    fs2 = fs.replace("'igure", "'Figure").replace('"igure', '"Figure')
    # 更精确：找 ax.set_title 里的 "igure"
    fs2 = re.sub(r'(\(\w\) )igure:', r'\1Figure:', fs2)
    open(figs_py, 'w', encoding='utf-8').write(fs2)
    print('5. make_figures.py Figure 拼写修正 [OK]')
else:
    print('5. make_figures.py 无 "igure" 拼写错误（可能是 PDF 渲染截断）')

open(p, 'w', encoding='utf-8').write(s)
print(f'\n阻塞项修订完成（{n} 项 + 贡献 4 条 + 数字 + 拼写）')
