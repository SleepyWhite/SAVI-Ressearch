#!/usr/bin/env python
"""Summarize the first-layer probe: does the model's likelihood recognize the REQ/ALT fork?

Main readout = degree of separation of Δ between BU and BM (AUC / Cohen's d), not the sign
of Δ. The template is fixed library-wide, so the item-independent polarity prior cancels
in the BU−BM difference; the sign only reflects the prior.

Criteria (frozen before the run)
- If across the three wordings the AUC is uniformly ≈0.5 (|AUC−0.5| < 0.05) → no relation
  signal in the likelihood; SAVI's scoring axis has no purchase on Belief-R; clean negative.
- If exactly one wording shows a signal while the other two do not → lexical artifact,
  does not count.
- The paired set (same base, γ1γ2 verbatim identical, gold separates by intent) is the
  strongest form: the same base scenario with only γ3 swapped; sign test.
"""
import argparse
import collections
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'probe_relation')
# Belief-R dataset CSV. The data is not distributed with this repo (see the README's data section); set env var BELIEF_R_CSV to a local copy, else falls back to <repo>/data/queries_time_t1.csv.
CSV = os.environ.get('BELIEF_R_CSV') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'queries_time_t1.csv')
TEMPLATES = ('sufficiency', 'relation', 'certainty')


def auc(x, y):
    if not x or not y:
        return float('nan')
    c = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(x, y))
    return c / (len(x) * len(y))


def cohen_d(x, y):
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    s = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return (np.mean(x) - np.mean(y)) / s if s else float('nan')


def clean_pairs():
    """Set of bases with both sides present, γ1+γ2 verbatim identical, and gold separating by design intent (strong=c, weak≠c)."""
    t1 = pd.read_csv(CSV)
    p = t1[t1.modus == 'ponens'].copy()
    p['base'] = p['dataset_id'].str.split('-').str[0]
    p['intent'] = p['dataset_id'].str.split('-').str[-1]
    w = p.pivot_table(index='base', columns='intent', values='ground_truth', aggfunc='first').dropna()
    out = set()
    for b in w.index:
        if not (w.loc[b, 'strong'] == 'c' and w.loc[b, 'weak'] != 'c'):
            continue
        if len({tuple(q.split('\n')[:2]) for q in p[p.base == b]['questions']}) == 1:
            out.add(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=D)
    ap.add_argument('--metric', default='mean', choices=['mean', 'sum'])
    ap.add_argument('--agreement', type=int, default=0)
    args = ap.parse_args()
    pairs = clean_pairs()

    for f in sorted(glob.glob(os.path.join(args.dir, '*.jsonl'))):
        recs = [json.loads(l) for l in open(f)]
        if args.agreement:
            recs = [r for r in recs if r['agreement_lv'] == args.agreement]
        name = os.path.basename(f)[len('time_t1_'):-len('.jsonl')]
        print(f'\n=== {name}   n={len(recs)}   Δ 口径={args.metric}'
              + (f'   agreement_lv={args.agreement}' if args.agreement else '') + ' ===')
        print('  主读数：AUC = P(随机一道 BU 的 Δ > 随机一道 BM 的 Δ)；0.5 = 无信号')
        print(f"  {'模板':<12} {'分层':<9} {'BU 均值':>9} {'BM 均值':>9} {'差':>8} {'d':>6} {'AUC':>7}   n")
        verdict = collections.defaultdict(list)
        for t in TEMPLATES:
            k = f'{t}_delta_{args.metric}'
            for mod in ('ponens', 'tollens', '合并'):
                sel = (lambda r: True) if mod == '合并' else (lambda r, m=mod: r['modus'] == m)
                bu = [r[k] for r in recs if sel(r) and r['ground_truth'] == 'c']
                bm = [r[k] for r in recs if sel(r) and r['ground_truth'] != 'c']
                a = auc(bu, bm)
                if mod != '合并':
                    verdict[t].append(a)
                print(f'  {t:<12} {mod:<9} {np.mean(bu):>9.4f} {np.mean(bm):>9.4f} '
                      f'{np.mean(bu) - np.mean(bm):>+8.4f} {cohen_d(bu, bm):>+6.2f} {a:>7.3f}   {len(bu)}/{len(bm)}')

        # Paired set: same base, only γ3 swapped; strong should lean more REQ than weak
        pr = [r for r in recs if r['base'] in pairs]
        by = collections.defaultdict(dict)
        for r in pr:
            by[(r['base'], r['modus'])][r['intent']] = r
        usable = [v for v in by.values() if 'strong' in v and 'weak' in v]
        if usable:
            print(f"\n  配对集（同 base，γ1γ2 逐字相同，金标按意图分开）：{len(usable)} 对"
                  f"（{len({b for b, _ in by})} base × 2 modus）")
            print(f"  {'模板':<12} {'strong>weak':>12} {'占比':>7} {'ΔΔ 均值':>10}   符号检验 p")
            from math import comb
            for t in TEMPLATES:
                k = f'{t}_delta_{args.metric}'
                dd = [v['strong'][k] - v['weak'][k] for v in usable]
                pos, n = sum(d > 0 for d in dd), len(dd)
                p2 = 2 * sum(comb(n, i) for i in range(max(pos, n - pos), n + 1)) / 2 ** n
                print(f'  {t:<12} {pos:>7}/{n:<4} {pos / n:>7.3f} {np.mean(dd):>+10.4f}   {min(p2, 1.0):.4f}')

        # The verdict must consider direction, not just the magnitude of deviation from 0.5:
        # REQ should be more preferred on BU, i.e. AUC>0.5.
        # The three wordings contradicting each other in direction = we measured wording,
        # not relation; treat as an artifact.
        flat = [a for v in verdict.values() for a in v]
        per_t = {t: np.mean(v) for t, v in verdict.items()}
        signed = [1 if a > 0.5 else -1 for a in per_t.values()]
        print(f"\n  判定：6 个 (模板×modus) 格的 AUC 范围 [{min(flat):.3f}, {max(flat):.3f}]；"
              f"各模板均值 " + '  '.join(f'{t}={v:.3f}' for t, v in per_t.items()))
        if len(set(signed)) > 1:
            print('  → 三套措辞方向不一致：测到的是措辞先验，不是关系信号（伪影）')
        elif max(abs(a - 0.5) for a in flat) < 0.05:
            print('  → 一致无信号：似然里读不出 REQ/ALT 分叉')
        elif all(abs(v - 0.5) > 0.05 for v in per_t.values()) and all(s > 0 for s in signed):
            print('  → 三套措辞一致且方向正确：似然里存在关系信号，可进第三层')
        else:
            print('  → 方向一致但幅度弱/不齐，不足以支撑进入第三层')


if __name__ == '__main__':
    main()
