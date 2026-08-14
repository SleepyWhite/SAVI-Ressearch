#!/usr/bin/env python
"""Summarize the [ORACLE] arm: given the correct state, does the model follow it?

Primary readout = **condition follow rate**, not accuracy.
  The fed condition entails an answer: AND→c; OR→a(ponens)/b(tollens).
  Follow rate = P(model prediction == the condition-entailed answer), regardless of
  whether the condition is right.
  Looking only at accuracy under gold conflates "following the state" with "would have
  gotten it anyway": on BM items the model already has ~90%; still 90% after feeding OR
  proves nothing.

Criteria (frozen before running numbers)
- Follow rate high on both gold and swap sides (and the swap side significantly above the
  baseline frequency of that condition's entailed answer)
    → readout works, the bottleneck is purely at the fork (together with L1's zero signal,
      this completes the boundary characterization)
- gold high but swap unmoved (≈ unconditional baseline) → the model ignores the condition;
  gold's high accuracy is an illusion
- both sides low → even deterministic readout fails; task selection must be re-examined
  (the P4 criterion of PLAN §5)
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'probe_oracle')
V1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'generative')
MODES = ('letter', 'txt_sum', 'txt_mean')
WORDINGS = ('entry', 'prose')


def implied(conn, modus):
    """The condition-entailed answer."""
    if conn == 'and':
        return 'c'
    return 'a' if modus == 'ponens' else 'b'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=D)
    ap.add_argument('--agreement', type=int, default=0)
    ap.add_argument('--drop_qmismatch', action='store_true',
                    help='剔除 γ1/γ3 后件不一致的 20 行')
    args = ap.parse_args()

    for f in sorted(glob.glob(os.path.join(args.dir, '*.jsonl'))):
        recs = [json.loads(l) for l in open(f)]
        if args.agreement:
            recs = [r for r in recs if r['agreement_lv'] == args.agreement]
        if args.drop_qmismatch:
            recs = [r for r in recs if not r['q_mismatch']]
        name = os.path.basename(f)[len('time_t1_'):-len('.jsonl')]
        print(f'\n=== {name}   n={len(recs)} ===')

        # Unconditional baseline: v1 greedy DP answer distribution (same items), as the
        # reference for "the model ignoring the condition"
        g = {}
        p1 = os.path.join(V1, f'time_t1_{name}_dp.jsonl')
        if os.path.exists(p1):
            g = {json.loads(l)['idx']: json.loads(l)['extracted'] for l in open(p1)}

        print('  条件跟随率 = P(预测 == 喂进去的条件所蕴含的答案)')
        print(f"  {'措辞':<7} {'打分':<9} {'喂gold':>8} {'喂swap':>8} {'两侧均值':>9} "
              f"{'gold-pon':>9} {'gold-tol':>9} {'swap-pon':>9} {'swap-tol':>9}")
        for w in WORDINGS:
            for mode in MODES:
                cells = {}
                for side in ('gold', 'swap'):
                    for mod in ('ponens', 'tollens', 'all'):
                        v = []
                        for r in recs:
                            if mod != 'all' and r['modus'] != mod:
                                continue
                            conn = r['gold_conn'] if side == 'gold' else (
                                'or' if r['gold_conn'] == 'and' else 'and')
                            v.append(r[f'{w}_{conn}_pred_{mode}'] == implied(conn, r['modus']))
                        cells[(side, mod)] = float(np.mean(v)) if v else float('nan')
                print(f"  {w:<7} {mode:<9} {cells[('gold','all')]:>8.3f} {cells[('swap','all')]:>8.3f} "
                      f"{(cells[('gold','all')] + cells[('swap','all')]) / 2:>9.3f} "
                      f"{cells[('gold','ponens')]:>9.3f} {cells[('gold','tollens')]:>9.3f} "
                      f"{cells[('swap','ponens')]:>9.3f} {cells[('swap','tollens')]:>9.3f}")

        # Condition sensitivity: fraction of predictions that change when the condition is
        # swapped. 0 = the condition is ignored entirely.
        print('\n  条件敏感度 = P(喂 AND 与喂 OR 的预测不同)；0 表示完全无视条件')
        print(f"  {'措辞':<7} " + ' '.join(f'{m:>10}' for m in MODES))
        for w in WORDINGS:
            row = []
            for mode in MODES:
                row.append(np.mean([r[f'{w}_and_pred_{mode}'] != r[f'{w}_or_pred_{mode}']
                                    for r in recs]))
            print(f'  {w:<7} ' + ' '.join(f'{x:>10.3f}' for x in row))

        # Reference: unconditional (v1 greedy DP) answer distribution
        if g:
            c = collections.Counter(g[r['idx']] for r in recs if r['idx'] in g)
            tot = sum(c.values())
            print(f"\n  无条件基线（v1 贪心 DP）答案分布: "
                  + '  '.join(f'{k}={v / tot:.3f}' for k, v in sorted(c.items()) if k))

        # Accuracy (for comparison, not the primary readout)
        print('\n  参照：喂 gold 条件下的准确率（非主读数——与跟随率混淆，见文件头）')
        print(f"  {'措辞':<7} " + ' '.join(f'{m:>10}' for m in MODES) + f"{'BU':>9}{'BM':>9}  (letter)")
        for w in WORDINGS:
            row = [np.mean([r[f'{w}_{r["gold_conn"]}_pred_{m}'] == r['ground_truth'] for r in recs])
                   for m in MODES]
            bu = np.mean([r[f'{w}_{r["gold_conn"]}_pred_letter'] == r['ground_truth']
                          for r in recs if r['ground_truth'] == 'c'])
            bm = np.mean([r[f'{w}_{r["gold_conn"]}_pred_letter'] == r['ground_truth']
                          for r in recs if r['ground_truth'] != 'c'])
            print(f'  {w:<7} ' + ' '.join(f'{x:>10.3f}' for x in row) + f'{bu:>9.3f}{bm:>9.3f}')


if __name__ == '__main__':
    main()
