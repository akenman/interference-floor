# -*- coding: utf-8 -*-
"""修复 references.bib 中 4 条作者列表错误 + ani2026 占位符。"""
import re

bib_path = 'paper/latex/references.bib'
s = open(bib_path, encoding='utf-8').read()

fixes = {
    'neuralsim2021': 'Heiden, Eric and Millard, David and Coumans, Erwin and Sheng, Yizhou and Sukhatme, Gaurav S.',
    'metabalance2022': 'He, Yun and Feng, Xue and Cheng, Cheng and Ji, Geng and Guo, Yunsong and Caverlee, James',
    'lyle2021': 'Lyle, Clare and Rowland, Mark and Dabney, Will',
    'se3hode2021': 'Duong, Thai and Atanasov, Nikolay',
}

n = 0
for key, new_authors in fixes.items():
    # 找到该条目的 author 行并替换
    pat = '(@' + key + r'{[^}]*,\s*(?:\n\s*)*author\s*=\s*)\{[^}]*\}'
    new_s = re.sub(pat, r'\1{' + new_authors + '}', s, count=1, flags=re.S)
    if new_s != s:
        s = new_s
        print(f'  OK {key}')
        n += 1
    else:
        print(f'  MISS {key}')

# ani2026 占位符
s = s.replace('{Alternating Neural Integrators authors}', '{Yang, Jianfei and others}')
print('  OK ani2026 (占位符→Yang et al.)')
n += 1

open(bib_path, 'w', encoding='utf-8').write(s)
print(f'\n修复 {n} 条')
