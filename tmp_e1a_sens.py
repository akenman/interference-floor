# -*- coding: utf-8 -*-
"""E1a 敏感性分析 + 把 pre_registration 补回 corrected 报告。

1. 判据敏感性：剔除崩溃 run 后 E1a-3 是否仍 PASS
2. k5（rollout 第5步）维度的同口径判据
3. pre_registration 从旧 E1a_report.json 迁移进 corrected 报告
"""
import io
import json
import sys
import statistics

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dim = json.load(open('analysis_output/E1a_dimension_analysis.json',
                     encoding='utf-8'))
runs = dim['runs']

def bests(arm, stable_only=False):
    out = []
    for r in runs.get(arm, []):
        if stable_only and (r['collapsed'] or r['early']):
            continue
        out.append(r)
    return out

def mean_key(rs, key):
    return statistics.mean([r[key] for r in rs]) if rs else float('nan')

print('=== 判据敏感性（one-step；V3 基准）===')
print(f"{'判据':<8}{'变体':<8}{'全量Δ':>10}{'剔崩溃Δ':>10}{'稳健结论':>10}")
v3_all = mean_key(runs['V3'], 'best_one')
v3_st = mean_key(bests('V3', True), 'best_one')
rows = []
for key, vv, thr in [('E1a-1', 'V1', 0.05), ('E1a-2', 'V2a', 0.05),
                     ('E1a-3', 'V2b', 0.05), ('E1a-4', 'Vfree', 0.03)]:
    d_all = (mean_key(runs[vv], 'best_one') - v3_all) / v3_all
    n_st = bests(vv, True)
    if n_st:
        d_st = (mean_key(n_st, 'best_one') - v3_st) / v3_st
    else:
        d_st = float('nan')
    p_all = 'PASS' if d_all >= thr else 'FAIL'
    p_st = 'PASS' if d_st >= thr else 'FAIL'
    verdict = p_all if p_all == p_st else f'{p_all}->{p_st}'
    print(f'{key:<8}{vv:<8}{d_all:>+9.1%}{d_st:>+9.1%}  {verdict:>10}')
    rows.append(dict(criterion=key, variant=vv, threshold=thr,
                     delta_all=d_all, delta_stable=d_st,
                     pass_all=p_all == 'PASS', pass_stable=p_st == 'PASS'))

print('\n=== k5 维度判据（同口径，参考用——预注册以 one-step 为准）===')
v3k5 = mean_key(runs['V3'], 'k5_at_best')
for key, vv, thr in [('E1a-1', 'V1', 0.05), ('E1a-2', 'V2a', 0.05),
                     ('E1a-3', 'V2b', 0.05), ('E1a-4', 'Vfree', 0.03)]:
    d = (mean_key(runs[vv], 'k5_at_best') - v3k5) / v3k5
    print(f'{key:<8}{vv:<8}{d:>+9.1%}  [{"PASS" if d >= thr else "FAIL"}]')

# 补 pre_registration 进 corrected 报告
old = json.load(open('analysis_output/E1a_report.json', encoding='utf-8'))
pre = old.get('pre_registration')
cur = json.load(open('analysis_output/E1a_corrected_report.json',
                     encoding='utf-8'))
if pre and not cur.get('pre_registration'):
    cur['pre_registration'] = pre
    cur['pre_registration_note'] = (
        '预注册于旧协议（2026-09-16），判据定义（one-step RMSE 相对 V3）'
        '在 corrected 协议下保持不变；本报告补充敏感性与 k5 维度。')
cur['criteria_sensitivity'] = rows
with open('analysis_output/E1a_corrected_report.json', 'w',
          encoding='utf-8') as fh:
    json.dump(cur, fh, ensure_ascii=False, indent=1)
print('\n[patched] E1a_corrected_report.json: +pre_registration '
      '+criteria_sensitivity')
