# -*- coding: utf-8 -*-
"""诊断：corrected 协议是否收敛（决定 E1a 负结果是否可信）。"""
import io
import json
import sys
import statistics

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
d = json.load(open('analysis_output/E1a_corrected_report.json', encoding='utf-8'))
runs = [r for r in d['runs']]

print('=== best epoch 分布（60 上限；靠后=仍在改善）===')
eps = [r['best_epoch'] for r in runs if r['best_epoch']]
print(f'  n={len(eps)} min={min(eps)} median={statistics.median(eps):.0f} '
      f'max={max(eps)} mean={statistics.mean(eps):.1f}')
late = sum(1 for e in eps if e >= 50)
print(f'  best_epoch>=50 的比例: {late}/{len(eps)} = {late / len(eps):.0%}')

print('\n=== final/best 比（≈1 收敛；>1.3 末尾发散/过拟合）===')
ratios = []
for r in runs:
    if r['best'] and r['final']:
        ratios.append(r['final'] / r['best'])
print(f'  median={statistics.median(ratios):.2f} '
      f'mean={statistics.mean(ratios):.2f} max={max(ratios):.2f}')
big = [r for r in runs if r['best'] and r['final'] and r['final'] / r['best'] > 1.3]
for r in big:
    print(f"    发散: {r['variant']}{'_nogua' if r['no_gua'] else ''} "
          f"s{r['seed']}: best={r['best']:.3f}(ep{r['best_epoch']}) "
          f"final={r['final']:.3f} ratio={r['final'] / r['best']:.2f}")

print('\n=== 判据的统计功效（n=3, 门槛 5%）===')
by = {}
for r in runs:
    k = r['variant'] + ('_nogua' if r['no_gua'] else '')
    by.setdefault(k, []).append(r['best'])
v3 = by['V3']
print(f"  V3: mean={statistics.mean(v3):.3f} std={statistics.stdev(v3):.3f}")
for v in ['V1', 'V2a', 'V2b', 'Vfree']:
    xs = by[v]
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0
    se = sd / (len(xs) ** 0.5)
    d = (m - statistics.mean(v3)) / statistics.mean(v3)
    # 需要多少 seed 才能让 5% 效应显著（粗略：效应/SE > 2）
    need = (2 * (sd + statistics.stdev(v3)) / (statistics.mean(v3) * 0.05)) ** 2 \
        if sd else float('nan')
    print(f'  {v:<6} mean={m:.3f} std={sd:.3f} Δ={d:+.1%} '
          f'SE≈{se:.3f} 需≈{need:.0f} seeds/臂 才能验 5%')
