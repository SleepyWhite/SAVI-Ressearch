#!/usr/bin/env python
"""Lift the §6.1 generative baseline from **row level** to **scenario level**: bootstrap CI clustered by dataset_id.

============================================================================
Why
============================================================================
STATUS §3 locks in "statistical unit = scenario (dataset_id), not row; twins do not count
as independent evidence". §6.2b (the BoN table) already complies, but §6.1 — the main
outward-facing table — has been row-level: the 1,744 rows are really only 872 independent
scenarios; ponens/tollens are two twins of the same scenario sharing γ1/γ3, and treating
them as independent samples narrows the CI by about √2. **Until this step is added, §6.1
does not enter outward-facing tables.**

============================================================================
Protocol (pinned before running)
============================================================================
1. **Cluster unit = dataset_id**. Each scenario has exactly 2 rows (ponens + tollens);
   on resampling the two twins move in and out **as a whole**, never split.
2. **BU/BM strata**: BU = gold c (REQ), BM = gold a/b (ALT).
   Verified that these two classes **never split a scenario** (relation R does not depend
   on modus, so twins belong to the same class), so scenarios nest cleanly inside BU/BM.
   Stratified bootstrap: BU and BM each resample scenarios within their own stratum —
   the two strata's sizes are fixed by dataset design, not obtained by sampling.
3. **BREU = (BU + BM) / 2, equal weight** (STATUS §3). The CI algorithm: after each
   resample, **recompute BU and BM then average**, not averaging two independent CIs.
4. **The four modus cells**: each scenario contributes only one row to a given cell, so
   that cell's cluster bootstrap degenerates to a plain bootstrap. Labeled honestly; no
   pretense of a cluster correction.
5. **Paired comparison** (added; not required by STATUS): DP/CoT/PS run on the same batch
   of scenarios, so "CoT is ~1 point above DP" should be judged by the CI of the
   **paired difference**, not by eyeballing two overlapping intervals. Within one resample,
   take the same batch of scenarios, compute both methods' BREU, then subtract.
6. Format errors (no a/b/c extractable) count as wrong per the paper's protocol, reusing
   the `correct` field.

============================================================================
Instrument self-checks (run on every invocation, raise on failure)
============================================================================
V0 the jsonl's ground_truth matches the source CSV by idx — the premise for trusting the
   agreement_lv join
V1 each scenario has exactly 2 rows, one per modus — the clustering premise
V2 twins' ground_truth both belong to BU or both to BM — the premise that strata don't
   split a scenario
V3 twins' agreement_lv are equal — the premise for taking the lv=5 subset by scenario
V4 **scenario-level point estimate == row-level point estimate** (equal rows per scenario,
   so they should be bitwise identical). This is a frozen check of the
   "identity cell equals its anchor" kind: the CI is the new part, the point estimate
   **must not move**; if it did, the clustering logic is wrong — not a discovery.

Usage
  python scripts/summarize_generative_ci.py --paired            # full set
  python scripts/summarize_generative_ci.py --agreement 5 --paired   # lv=5 main readout set
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'generative')
# Belief-R dataset CSV. The data is not distributed with this repo (see the README's data section); point the BELIEF_R_CSV env var at a local copy, falling back to
# <repo>/data/queries_time_t1.csv when unset.
CSV = os.environ.get('BELIEF_R_CSV') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'queries_time_t1.csv')
ITERS = 5000
SEED = 0
MODI = ('ponens', 'tollens')

# The generative jsonl has no agreement_lv (it wasn't written at the time); join it back
# from the source CSV by idx. idx is the CSV row number (see run_generative.py's
# enumerate), so the join is unambiguous.
_DF = pd.read_csv(CSV)
LV = {i: int(r['agreement_lv']) for i, r in _DF.iterrows()}
GT = {i: r['ground_truth'] for i, r in _DF.iterrows()}


def load_strata(path, agreement, tag):
    """→ (bu, bm, scen_ids_bu, scen_ids_bm), where bu/bm are (n_scenarios, 2) 0/1 matrices,
    column order = (ponens, tollens). Also runs the V0–V3 self-checks."""
    per = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        i = int(r['idx'])
        if GT[i] != r['ground_truth']:                                    # V0
            raise AssertionError(f'[{tag}] 第 {i} 行金标与 CSV 不符，idx 连接不可信')
        lv = LV[i]
        if agreement and lv != agreement:
            continue
        per[r['dataset_id']][r['modus']] = (float(r['correct']), r['ground_truth'], lv)

    bu_ids, bm_ids, bu, bm = [], [], [], []
    for sid, d in sorted(per.items()):
        if set(d) != set(MODI):                                           # V1
            raise AssertionError(f'[{tag}] 场景 {sid} 的 modus 不是 ponens/tollens 各一：{set(d)}')
        gts = {d[m][1] == 'c' for m in MODI}
        if len(gts) != 1:                                                 # V2
            raise AssertionError(f'[{tag}] 场景 {sid} 的孪生跨了 BU/BM，分层前提不成立')
        if len({d[m][2] for m in MODI}) != 1:                             # V3
            raise AssertionError(f'[{tag}] 场景 {sid} 的孪生 agreement_lv 不同')
        row = [d[m][0] for m in MODI]
        (bu if gts.pop() else bm).append(row)
        (bu_ids if d['ponens'][1] == 'c' else bm_ids).append(sid)
    return np.array(bu, float).reshape(-1, 2), np.array(bm, float).reshape(-1, 2), bu_ids, bm_ids


def draw(n, iters, rng):
    return rng.randint(0, n, (iters, n))


def stats(bu, bm, iters, seed):
    """Point estimates + stratified cluster bootstrap CI. Returns (pt, ci)."""
    pt = {'BU': bu.mean(), 'BM': bm.mean(),
          'BU-pon': bu[:, 0].mean(), 'BU-tol': bu[:, 1].mean(),
          'BM-pon': bm[:, 0].mean(), 'BM-tol': bm[:, 1].mean()}
    pt['BREU'] = (pt['BU'] + pt['BM']) / 2

    rng = np.random.RandomState(seed)
    sb = bu[draw(len(bu), iters, rng)]          # (iters, n_bu, 2)
    sm = bm[draw(len(bm), iters, rng)]
    d = {'BU': sb.mean(axis=(1, 2)), 'BM': sm.mean(axis=(1, 2)),
         'BU-pon': sb[:, :, 0].mean(axis=1), 'BU-tol': sb[:, :, 1].mean(axis=1),
         'BM-pon': sm[:, :, 0].mean(axis=1), 'BM-tol': sm[:, :, 1].mean(axis=1)}
    d['BREU'] = (d['BU'] + d['BM']) / 2
    ci = {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in d.items()}
    return pt, ci


def paired(a, b, iters, seed):
    """Paired cluster bootstrap of BREU(b) − BREU(a). a/b are each (bu, bm, bu_ids, bm_ids)."""
    (abu, abm, aib, aim), (bbu, bbm, bib, bim) = a, b
    if aib != bib or aim != bim:
        raise AssertionError('两个方法的场景集合不同，无法配对')
    rng = np.random.RandomState(seed)
    ib, im = draw(len(abu), iters, rng), draw(len(abm), iters, rng)

    def breu(bu, bm, ii, jj):
        return (bu[ii].mean(axis=(1, 2)) + bm[jj].mean(axis=(1, 2))) / 2

    pt = ((bbu.mean() + bbm.mean()) - (abu.mean() + abm.mean())) / 2
    dd = breu(bbu, bbm, ib, im) - breu(abu, abm, ib, im)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    return pt, lo, hi, len(aib) + len(aim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agreement', type=int, default=0,
                    help='>0 时只用该 agreement_lv 子集（STATUS §3 主读数 = 5）')
    ap.add_argument('--paired', action='store_true', help='加 CoT−DP / PS−DP 的配对 ΔBREU')
    ap.add_argument('--iters', type=int, default=ITERS)
    args = ap.parse_args()

    tag = f'agreement_lv={args.agreement}' if args.agreement else '全集'
    print(f'# 生成式基线，场景级聚类 bootstrap（{tag}，{args.iters} 次重采样，seed={SEED}）')
    print('# 统计单元 = dataset_id；BU/BM 分层内重采样；CI = 百分位 95%\n')

    store, n_v4 = {}, 0
    hdr = (f"{'model':<30} {'m':<4} {'BU场景':>6} {'BM场景':>6} {'BU':>7} {'BU 95%CI':>17} "
           f"{'BM':>7} {'BM 95%CI':>17} {'BREU':>7} {'BREU 95%CI':>17}")
    print(hdr)
    print('-' * len(hdr))
    for f in sorted(glob.glob(os.path.join(OUT, '*.jsonl'))):
        stem = os.path.basename(f)[len('time_t1_'):-len('.jsonl')]
        model, method = stem.rsplit('_', 1)
        bu, bm, bi, mi = load_strata(f, args.agreement, stem)
        if not len(bu) or not len(bm):
            continue
        pt, ci = stats(bu, bm, args.iters, SEED)

        # V4: the row-level point estimate (no scenario aggregation at all) should be bitwise identical to the scenario-level one
        rows = [json.loads(l) for l in open(f)]
        if args.agreement:
            rows = [r for r in rows if LV[int(r['idx'])] == args.agreement]
        rbu = np.mean([r['correct'] for r in rows if r['ground_truth'] == 'c'])
        rbm = np.mean([r['correct'] for r in rows if r['ground_truth'] != 'c'])
        for name, x, y in (('BU', pt['BU'], rbu), ('BM', pt['BM'], rbm)):
            if abs(x - y) > 1e-12:
                raise AssertionError(f'[{stem}] V4 失败：{name} 场景级 {x} != 行级 {y}')
        n_v4 += 1

        store[(model, method)] = (bu, bm, bi, mi)
        print(f"{model:<30} {method:<4} {len(bu):>6} {len(bm):>6} "
              f"{pt['BU']:>7.4f} [{ci['BU'][0]:.4f},{ci['BU'][1]:.4f}] "
              f"{pt['BM']:>7.4f} [{ci['BM'][0]:.4f},{ci['BM'][1]:.4f}] "
              f"{pt['BREU']:>7.4f} [{ci['BREU'][0]:.4f},{ci['BREU'][1]:.4f}]")

    print(f'\n# V0–V4 全部通过（{n_v4} 个配置）。V4：场景级点估计与行级逐位相同——'
          f'每场景等行数，本应如此；变了就是聚类逻辑写错。')
    print('# modus 四格（每场景在一格里只有 1 行，故此处 bootstrap 无聚类修正）\n')
    hdr2 = f"{'model':<30} {'m':<4} " + ' '.join(f'{k:>22}' for k in
                                                 ('BU-pon', 'BU-tol', 'BM-pon', 'BM-tol'))
    print(hdr2)
    print('-' * len(hdr2))
    for (model, method), (bu, bm, _, _) in store.items():
        pt, ci = stats(bu, bm, args.iters, SEED)
        cells = ' '.join(f"{pt[k]:.3f}[{ci[k][0]:.3f},{ci[k][1]:.3f}]".rjust(22)
                         for k in ('BU-pon', 'BU-tol', 'BM-pon', 'BM-tol'))
        print(f'{model:<30} {method:<4} {cells}')

    if args.paired:
        print('\n# 配对 ΔBREU（同一批场景，聚类 bootstrap）。CI 含 0 = 与 DP 不可区分\n')
        print(f"{'model':<30} {'对比':<10} {'ΔBREU':>9} {'95%CI':>20} {'场景':>6}  判定")
        print('-' * 88)
        for model in sorted({m for m, _ in store}):
            if (model, 'dp') not in store:
                continue
            for other in ('cot', 'ps'):
                if (model, other) not in store:
                    continue
                d, lo, hi, n = paired(store[(model, 'dp')], store[(model, other)],
                                      args.iters, SEED)
                v = '不可区分' if lo <= 0 <= hi else ('高于 DP' if lo > 0 else '低于 DP')
                print(f'{model:<30} {other + "−dp":<10} {d:>+9.4f} '
                      f'[{lo:>+7.4f},{hi:>+7.4f}] {n:>6}  {v}')


if __name__ == '__main__':
    main()
