# -*- coding: utf-8 -*-
"""修复 provenance.tex（下划线转义 + Unicode）和 references.bib（补 wm_survey2026），
然后重打 TMLR_submission.zip。
"""
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(ROOT, 'paper', 'latex')

# --- 1) 修 provenance.tex：转义 _ 和替换 Unicode ---
prov_path = os.path.join(LATEX, 'provenance.tex')
s = open(prov_path, encoding='utf-8').read()
# 策略：只在 & 分隔的数据行里转义（跳过 \toprule 等命令行）
lines = s.split('\n')
fixed = []
for ln in lines:
    if '&' in ln and '\\\\' in ln:  # 数据行
        # 按 & 分列，转义每列里的 _ → \_
        parts = ln.split(' & ')
        fixed_parts = []
        for p in parts:
            # 替换 Unicode
            p = p.replace('\u03c4', '$\\tau$')
            p = p.replace('\u011d', "$\\hat{g}$")
            # 转义 _（但不碰已有的 \_）
            out = []
            i = 0
            while i < len(p):
                if p[i] == '_' and (i == 0 or p[i-1] != '\\'):
                    out.append('\\_')
                else:
                    out.append(p[i])
                i += 1
            fixed_parts.append(''.join(out))
        fixed.append(' & '.join(fixed_parts))
    else:
        fixed.append(ln)
s2 = '\n'.join(fixed)
open(prov_path, 'w', encoding='utf-8').write(s2)
print('provenance.tex: _ 已转义，Unicode 已替换')

# --- 2) 补 wm_survey2026 bib 条目 ---
bib_path = os.path.join(LATEX, 'references.bib')
bib = open(bib_path, encoding='utf-8').read()
if 'wm_survey2026' not in bib:
    entry = """
@misc{wm_survey2026,
  author  = {Hou, Bohan and Li, Gen and Zhi, Jindou and An, Tuo and Guo, Xinying and Leng, Sicong and Geng, Haoran and Ze, Yanjie and Harada, Tatsuya and Torr, Philip and Mees, Oier and Pollefeys, Marc and Liu, Zhuang and Wu, Jiajun and Abbeel, Pieter and Malik, Jitendra and Du, Yilun},
  title   = {World Model for Robot Learning: A Comprehensive Survey},
  year    = {2026},
  howpublished = {arXiv:2605.00080}
}
"""
    bib += entry
    open(bib_path, 'w', encoding='utf-8').write(bib)
    print('references.bib: wm_survey2026 条目已补')

# --- 3) 同步修 gen_provenance.py（下次再生成本身就带转义）---
gen_path = os.path.join(ROOT, 'paper', 'tables', 'gen_provenance.py')
gen = open(gen_path, encoding='utf-8').read()
if 'esc(' not in gen:
    old_row = "    rows.append(f\"{c['name']} & {c['value']:g} & {c['claimed']:g} & {ok} \\\\\\\\\")"
    new_block = """    def esc(t):
        t = t.replace('\\u03c4', '$\\\\tau$').replace('\\u011d', "$\\\\hat{g}$")
        for ch in ('_', '&', '%', '#'):
            t = t.replace(ch, '\\\\' + ch)
        return t
    rows.append(f"{esc(c['name'])} & {c['value']:g} & {c['claimed']:g} & {ok} \\\\\\\")"""
    if old_row in gen:
        gen = gen.replace(old_row, new_block)
        open(gen_path, 'w', encoding='utf-8').write(gen)
        print('gen_provenance.py: esc() 已同步')
    else:
        print('gen_provenance.py: anchor 不匹配，跳过（provenance.tex 已手动修）')

# --- 4) 重打 zip ---
zip_path = os.path.join(ROOT, 'paper', 'TMLR_submission.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in ('main.tex', 'provenance.tex', 'references.bib', 'tmlr.sty', 'tmlr.bst'):
        z.write(os.path.join(LATEX, f), f)
    figs = os.path.join(LATEX, 'figs')
    for f in sorted(os.listdir(figs)):
        z.write(os.path.join(figs, f), 'figs/' + f)
print(f'TMLR_submission.zip 重打完成 ({os.path.getsize(zip_path)//1024} KB)')
