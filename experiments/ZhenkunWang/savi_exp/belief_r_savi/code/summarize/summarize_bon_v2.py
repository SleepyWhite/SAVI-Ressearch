#!/usr/bin/env python
"""Summarize the v2 sampling pool: voting, best-of-N, any@N at N=16/32, plus the four-term decomposition.

Scope (PLAN §8)
- statistical unit = scenario (dataset_id); CI uses scenario-clustered bootstrap, not per row
- vote ties count as wrong; format errors (no a/b/c extractable) count as wrong
- two vote denominators: all = all N chains vote (strict paper protocol); valid = only
  legal a/b/c votes count
- two BoN scopes: sum = whole-chain log-likelihood sum; mean = per-token mean
  (length-normalized)
- N=16 takes this file's first 16 chains (nested subsampling, not independent resampling)

Four-term decomposition: 1 = I[greedy] + Δvote + Δselect + εmiss
  εmiss   = 1 − any@N        the correct answer never appears in the pool
  Δselect = any@N − vote     in the pool but voting didn't select it (a selection problem)
  Δvote   = vote − greedy    voting's increment over greedy
The decomposition uses vote_all as the vote term; Δvote/Δselect can be negative — reported
as-is, not clipped.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

V1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'generative')
V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'bon_v2')
VALID = ('a', 'b', 'c')


def vote(answers, valid_only):
    pool = [x for x in answers if x in VALID] if valid_only else list(answers)
    if not pool:
        return None
    cnt = collections.Counter(pool).most_common()
    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
        return None                      # ties count as wrong
    return cnt[0][0]


def readouts(rec, n):
    ch = rec['chains'][:n]
    ans = [c['extracted'] for c in ch]
    gt = rec['ground_truth']
    return {
        'vote_all': vote(ans, False) == gt,
        'vote_valid': vote(ans, True) == gt,
        'bon_sum': ch[max(range(len(ch)), key=lambda i: ch[i]['sum_lp'])]['extracted'] == gt,
        'bon_mean': ch[max(range(len(ch)), key=lambda i: ch[i]['mean_lp'])]['extracted'] == gt,
        'any': gt in ans,
    }


def boot_ci(per_scen, iters=2000, seed=0):
    """Scenario-clustered bootstrap. per_scen: {dataset_id: [0/1, ...]}"""
    keys = list(per_scen)
    if not keys:
        return (float('nan'), float('nan'))
    rng = np.random.RandomState(seed)
    flat = np.array([v for k in keys for v in per_scen[k]], dtype=float)
    if len(flat) == 0:
        return (float('nan'), float('nan'))
    stats = []
    for _ in range(iters):
        pick = rng.randint(0, len(keys), len(keys))
        vals = np.concatenate([per_scen[keys[i]] for i in pick])
        stats.append(vals.mean())
    return tuple(np.percentile(stats, [2.5, 97.5]))


def load_greedy(model, method):
    p = os.path.join(V1_DIR, f'time_t1_{model}_{method}.jsonl')
    if not os.path.exists(p):
        return {}
    return {json.loads(l)['idx']: json.loads(l)['correct'] for l in open(p)}


def agg(recs, sel, key, greedy=None):
    """Returns (mean, n, per_scenario dict). When key is 'greedy', uses the v1 readout."""
    per = collections.defaultdict(list)
    for r, ro in recs:
        if not sel(r):
            continue
        v = greedy.get(r['idx']) if key == 'greedy' else ro[key]
        if v is None:
            continue
        per[r['dataset_id']].append(float(v))
    vals = [v for lst in per.values() for v in lst]
    return (float(np.mean(vals)) if vals else float('nan'), len(vals), per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=V2_DIR)
    ap.add_argument('--ns', default='16,32')
    ap.add_argument('--agreement', type=int, default=0, help='>0 时只用该 agreement_lv 子集（PLAN §8 主读数=5）')
    ap.add_argument('--ci', action='store_true', help='算按场景聚类的 bootstrap CI（慢）')
    args = ap.parse_args()
    NS = [int(x) for x in args.ns.split(',')]

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    if not files:
        print(f"no v2 output in {args.dir}")
        return

    for f in files:
        stem = os.path.basename(f)[len('time_t1_'):-len('.jsonl')]
        body, ntag = stem.rsplit('_N', 1)
        model, method = body.rsplit('_', 1)
        recs = []
        for l in open(f):
            r = json.loads(l)
            if args.agreement and r['agreement_lv'] != args.agreement:
                continue
            recs.append(r)
        if not recs:
            continue
        greedy = load_greedy(model, method)
        print(f'\n=== {model}  {method}  file-N={ntag}  rows={len(recs)}  '
              f'scenarios={len({r["dataset_id"] for r in recs})}'
              + (f'  agreement_lv={args.agreement}' if args.agreement else '') + ' ===')

        for n in NS:
            if n > min(len(r['chains']) for r in recs):
                print(f'  N={n}: 跳过（文件里链数不足）')
                continue
            pairs = [(r, readouts(r, n)) for r in recs]
            isbu = lambda r: r['ground_truth'] == 'c'
            keys = ['greedy'] + [k for k in ('vote_all', 'vote_valid', 'bon_sum', 'bon_mean', 'any')]
            print(f'  --- N={n} ---')
            print(f"    {'readout':<11} {'BU':>8} {'BM':>8} {'BREU':>8}"
                  + ('   BU 95%CI(场景聚类)' if args.ci else ''))
            rows = {}
            for k in keys:
                if k == 'greedy' and not greedy:
                    continue
                bu, nbu, pbu = agg(pairs, isbu, k, greedy)
                bm, nbm, _ = agg(pairs, lambda r: not isbu(r), k, greedy)
                rows[k] = (bu, bm)
                ci = ''
                if args.ci:
                    lo, hi = boot_ci(pbu)
                    ci = f'   [{lo:.3f}, {hi:.3f}]'
                print(f'    {k:<11} {bu:>8.4f} {bm:>8.4f} {(bu + bm) / 2:>8.4f}{ci}')

            if 'greedy' in rows:
                print(f"    {'四项分解':<9} {'I[greedy]':>10} {'Δvote':>8} {'Δselect':>9} {'εmiss':>8}")
                for name, sel in (('BU-ponens', lambda r: isbu(r) and r['modus'] == 'ponens'),
                                  ('BU-tollens', lambda r: isbu(r) and r['modus'] == 'tollens'),
                                  ('BM-ponens', lambda r: not isbu(r) and r['modus'] == 'ponens'),
                                  ('BM-tollens', lambda r: not isbu(r) and r['modus'] == 'tollens')):
                    g = agg(pairs, sel, 'greedy', greedy)[0]
                    v = agg(pairs, sel, 'vote_all', greedy)[0]
                    a = agg(pairs, sel, 'any', greedy)[0]
                    print(f'    {name:<11} {g:>10.4f} {v - g:>8.4f} {a - v:>9.4f} {1 - a:>8.4f}')


if __name__ == '__main__':
    main()
