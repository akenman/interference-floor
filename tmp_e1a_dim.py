# -*- coding: utf-8 -*-
"""E1a corrected 协议多维分析：one-step / rollout k=5 / yao_acc。

从 logs-corrected/*.log 解析每 2 epoch 的 val 行，按臂聚合。
关注三个问题：
  1. rollout 维度上，轴数消融是否比 one-step 更显著？
  2. 门控增益在 rollout / yao_acc 上是否反转（nogua 更差）？
  3. 崩溃（best 后 val 发散）在 rollout 上的表现。
"""
import io
import os
import re
import sys
import json
import statistics
import collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, 'logs-corrected')

RE_ONE = re.compile(r'\[val\] one-step:\s*\S+\s+([0-9.]+)')
RE_ROLL = re.compile(
    r'\[val\] rollout: k=1:([0-9.]+) k=2:([0-9.]+) k=3:([0-9.]+) '
    r'k=4:([0-9.]+) k=5:([0-9.]+) \|.*?acc ([0-9.]+)')
RE_EPOCH = re.compile(r'^Epoch (\d+)/60')
RE_OK = re.compile(r'\[OK\]')
RE_EARLY = re.compile(r'\[EARLY-STOP\]')


def parse_log(path):
    """返回每条 val 记录: dict(epoch, one, k1..k5, acc)。"""
    recs = []
    epoch = None
    done = early = False
    with open(path, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            m = RE_EPOCH.match(line)
            if m:
                epoch = int(m.group(1))
            m = RE_ONE.search(line)
            if m:
                one = float(m.group(1))
                recs.append({'epoch': epoch, 'one': one})
            m = RE_ROLL.search(line)
            if m and recs and recs[-1].get('epoch') == epoch and 'k5' not in recs[-1]:
                g = [float(x) for x in m.groups()]
                recs[-1].update(k1=g[0], k2=g[1], k3=g[2], k4=g[3],
                                k5=g[4], acc=g[5])
            if RE_OK.search(line):
                done = True
            if RE_EARLY.search(line):
                early = True
    return [r for r in recs if 'k5' in r], done, early


def mean(xs):
    return statistics.mean(xs) if xs else float('nan')


def main():
    data = {}  # arm -> list of run records
    for fn in sorted(os.listdir(LOGS)):
        if not fn.endswith('.log') or fn.startswith('scheduler') \
                or fn.startswith('pilot'):
            continue
        name = fn[:-4]
        m = re.match(r'(.+)_s(\d+)(_nogua)?$', name)
        if not m:
            continue
        arm = m.group(1) + ('_nogua' if m.group(3) else '')
        recs, done, early = parse_log(os.path.join(LOGS, fn))
        if not recs:
            continue
        best_one_rec = min(recs, key=lambda r: r['one'])
        # 崩溃 = best 后 one-step 发散（>10px）；否则为耐心早停
        collapsed = bool(early and recs[-1]['one'] > 10
                         and best_one_rec['one'] < 5)
        data.setdefault(arm, []).append(dict(
            seed=int(m.group(2)), done=done, early=early,
            collapsed=collapsed,
            best_one=best_one_rec['one'], best_one_ep=best_one_rec['epoch'],
            one_final=recs[-1]['one'],
            k5_at_best=best_one_rec['k5'], k5_min=min(r['k5'] for r in recs),
            k5_final=recs[-1]['k5'],
            acc_at_best=best_one_rec['acc'],
            acc_max=max(r['acc'] for r in recs),
            roll_at_best=[best_one_rec[f'k{i}'] for i in range(1, 6)],
        ))

    order = []
    for v in ['V1', 'V2a', 'V2b', 'V2c', 'V3', 'V4', 'Vfree']:
        if v in data:
            order.append(v)
        if (v + '_nogua') in data:
            order.append(v + '_nogua')

    hdr = (f"{'臂':<12}{'n':>3}{'one-step':>10}{'k5@best':>9}"
           f"{'k5_min':>9}{'yao@best':>10}{'yao_max':>9}")
    print('=== 逐臂聚合（one-step 取 best epoch，k5/acc 同 epoch）===')
    print(hdr)
    print('-' * len(hdr))
    agg = {}
    for arm in order:
        rs = data[arm]
        row = dict(
            n=len(rs),
            one=mean([r['best_one'] for r in rs]),
            k5_at_best=mean([r['k5_at_best'] for r in rs]),
            k5_min=mean([r['k5_min'] for r in rs]),
            acc_at_best=mean([r['acc_at_best'] for r in rs]),
            acc_max=mean([r['acc_max'] for r in rs]),
            n_early=sum(1 for r in rs if r['early']),
            n_collapse=sum(1 for r in rs if r['collapsed']),
            n_patience=sum(1 for r in rs if r['early'] and not r['collapsed']),
        )
        agg[arm] = row
        print(f"{arm:<12}{row['n']:>3}{row['one']:>10.3f}"
              f"{row['k5_at_best']:>9.3f}{row['k5_min']:>9.3f}"
              f"{row['acc_at_best']:>10.4f}{row['acc_max']:>9.4f}")

    print('\n=== 问题1：rollout k5 相对 V3 的劣化（one-step 对比）===')
    v3 = agg.get('V3')
    if v3:
        print(f"{'臂':<12}{'one-step Δ':>12}{'k5 Δ':>10}{'k5 更敏感?':>12}")
        for arm in order:
            if arm.endswith('_nogua'):
                continue
            a = agg[arm]
            d1 = (a['one'] - v3['one']) / v3['one']
            d5 = (a['k5_at_best'] - v3['k5_at_best']) / v3['k5_at_best']
            tag = 'YES' if d5 > d1 + 0.02 else ('no' if d5 <= d1 else '~')
            print(f"{arm:<12}{d1:>+11.1%}{d5:>+9.1%}{tag:>12}")

    print('\n=== 问题2：门控增益（nogua-常态，负=去门控更准）===')
    print(f"{'臂':<8}{'one-step':>10}{'k5@best':>10}{'yao@best':>10}{'yao_max':>10}")
    for v in ['V1', 'V2a', 'V2b', 'V2c', 'V3', 'V4']:
        if v in agg and (v + '_nogua') in agg:
            g, n = agg[v + '_nogua'], agg[v]
            f = lambda k: (g[k] - n[k]) / n[k]
            print(f"{v:<8}{f('one'):>+9.1%}{f('k5_at_best'):>+9.1%}"
                  f"{f('acc_at_best'):>+9.1%}{f('acc_max'):>+9.1%}")

    print('\n=== 问题3：早停 run 明细（崩溃=one 发散>10px，其余=耐心早停）===')
    for arm in order:
        for r in data[arm]:
            if r['early']:
                kind = 'CRASH' if r['collapsed'] else 'patience'
                print(f"  [{kind:>8}] {arm} s{r['seed']}: "
                      f"one@best={r['best_one']:.3f}(ep{r['best_one_ep']})"
                      f" -> final one={r['one_final']:.3f}"
                      f" k5_final={r['k5_final']:.2f}")

    out = dict(arms=agg,
               runs={arm: data[arm] for arm in order},
               note='one-step 与 k5/acc 均取 best-one epoch 处的值；'
                    'k5_min/acc_max 为整个训练过程最优。')
    with open(os.path.join(ROOT, 'analysis_output',
                           'E1a_dimension_analysis.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('\n[report] analysis_output/E1a_dimension_analysis.json')


if __name__ == '__main__':
    main()
