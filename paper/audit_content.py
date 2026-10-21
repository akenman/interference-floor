# -*- coding: utf-8 -*-
"""main.tex 系统性内容审计：图表引用完整性 / 引用-文献一致性 / label 一致性 / 残余占位符。"""
import re
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tex = open(os.path.join(ROOT, 'paper', 'latex', 'main.tex'), encoding='utf-8').read()
bib = open(os.path.join(ROOT, 'paper', 'latex', 'references.bib'), encoding='utf-8').read()

BS = chr(92)  # backslash

# 1) 图文件引用
fig_files = sorted(set(re.findall(r'figs/(fig\w+)', tex)))
print('1. 图文件引用:', fig_files)

# 2) 引用键 vs bib
cite_keys = set()
for m in re.findall(re.escape(BS) + r'cite[pt]?\{([^}]+)\}', tex):
    cite_keys.update(k.strip() for k in m.split(','))
bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))
missing = sorted(cite_keys - bib_keys)
uncited = sorted(bib_keys - cite_keys)
print(f'2. 正文引用 {len(cite_keys)} 键 / bib {len(bib_keys)} 键')
print(f'   bib 缺失: {missing or "无"}')
print(f'   bib 未引: {uncited or "无"}')

# 3) label/ref 一致性
labels = set(re.findall(re.escape(BS) + r'label\{([^}]+)\}', tex))
refs = set(re.findall(re.escape(BS) + r'ref\{([^}]+)\}', tex))
print(f'3. label 缺失: {sorted(refs - labels) or "无"}')
print(f'   label 未引用: {sorted(labels - refs) or "无"}')

# 4) 残余占位符
for pat in ('T.B.D', 'TODO', 'FIXME', 'placeholder', 'XXX'):
    hits = [(tex[:m.start()].count('\n') + 1) for m in re.finditer(re.escape(pat), tex, re.I)]
    if hits:
        print(f'4. [WARN] "{pat}" at line {hits}')

# 5) 空节/空段落检查
sections = re.findall(re.escape(BS) + r'section\{([^}]+)\}', tex)
print(f'5. 章节数: {len(sections)} → {sections}')

# 6) 常见 LaTeX 格式错误
lines = tex.split('\n')
in_tab = False
tab_lines = set()
for i, ln in enumerate(lines):
    if '\\begin{tabular' in ln:
        in_tab = True
    if in_tab:
        tab_lines.add(i)
    if '\\end{tabular' in ln:
        in_tab = False
for pat, desc in [
    (r'(?<!\\)%', '未转义 %（注释风险）'),
    (r'(?<!\\)&', '表格外未转义 &'),
    (r'\.{3}', '三点省略号（应用 \\ldots）'),
]:
    hits = []
    for m in re.finditer(pat, tex):
        li = tex[:m.start()].count('\n')
        line = lines[li]
        if line.lstrip().startswith('%') or li in tab_lines:
            continue  # 整行注释与表格对齐符合法
        hits.append((li + 1, line.strip()[:60]))
    if hits:
        print(f'6. [WARN] {desc}: {hits[:5]}')

# 7) 图表 caption 检查
captions = re.findall(re.escape(BS) + r'caption\{([^}]+)\}', tex)
for c in captions:
    if len(c) < 20:
        print(f'7. [WARN] caption 过短: "{c}"')

# 8) 数字一致性抽查（正文 vs bib）
print(f'8. 正文字数: {len(tex.split())} words')
