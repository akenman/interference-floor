# -*- coding: utf-8 -*-
import json, io, sys, statistics, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

d = json.load(open(r'analysis_output/E1a_corrected_report.json', encoding='utf-8'))
runs = d['runs']
print('status:', d.get('status'), '| runs:', len(runs),
      '| pre_reg kept:', d.get('pre_registration') is not None)

by = collections.defaultdict(list)
for r in runs:
    key = r['variant'] + ('_nogua' if r.get('no_gua') else '')
    by[key].append(r)

print('\n崩溃明细:')
for k in sorted(by):
    col = [f"s{r['seed']}(best={r['best']}, ep{r['best_epoch']})"
           for r in by[k] if r.get('collapse')]
    if col:
        print(f'  {k}: {", ".join(col)}')

print('\n门控增益对比（增益=(nogua-常态)/常态；负=去掉门控更准）:')
for v in ['V1', 'V2a', 'V2b', 'V2c', 'V3', 'V4']:
    n_all = [r['best'] for r in by.get(v, [])]
    g_all = [r['best'] for r in by.get(v + '_nogua', [])]
    n_ok = [r['best'] for r in by.get(v, []) if not r.get('collapse')]
    g_ok = [r['best'] for r in by.get(v + '_nogua', []) if not r.get('collapse')]
    if not (n_all and g_all):
        continue
    an, ag = statistics.mean(n_all), statistics.mean(g_all)
    line = f'  {v:<5} 全部: 常态={an:.3f} nogua={ag:.3f} 增益={(ag - an) / an:+6.1%}'
    if n_ok and g_ok and (len(n_ok) != len(n_all) or len(g_ok) != len(g_all)):
        mn, mg = statistics.mean(n_ok), statistics.mean(g_ok)
        line += f' | 剔崩溃: {mn:.3f} vs {mg:.3f} {(mg - mn) / mn:+6.1%}'
    print(line)

print('\n判据（基准 V3 全部3种子）:')
v3 = statistics.mean([r['best'] for r in by['V3']])
for k, vv, thr in [('E1a-1', 'V1', 0.05), ('E1a-2', 'V2a', 0.05),
                   ('E1a-3', 'V2b', 0.05), ('E1a-4', 'Vfree', 0.03)]:
    m = statistics.mean([r['best'] for r in by[vv]])
    delta = (m - v3) / v3
    print(f'  {k} {vv:<6} {m:.3f} vs {v3:.3f} D={delta:+.3f} thr>={thr} '
          f'[{"PASS" if delta >= thr else "FAIL"}]')
