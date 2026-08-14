#!/usr/bin/env python
"""Clustered bootstrap CI for the three-split "mean statistic" + the same readout and paired difference vs the lexical baseline. Pure CPU, zero GPU.

`probe_foldseed_sm_all.py` produced three-model × three-split SM point
readouts, but the mean is just the arithmetic mean of three numbers with no
uncertainty; a single-split CI also cannot answer "averaged over the three
splits, how far from 0.5 / from the lexical baseline". This script changes the
statistic from "bal-acc of one split" to "mean of the three splits' bal-acc"
and recomputes it inside the same atomic scenario-family clustered bootstrap
(5000 iterations, seed 20260803): each resample computes bal-acc once for each
of the three **already-stored** sets of out-of-fold predictions, then takes
the mean. No model is retrained — the three splits' out-of-fold predictions
are first written to `outputs/probe_foldseed_preds.npz`; everything afterwards
reads from disk.

The lexical baseline (G-3's tfidf_cv) is likewise extended to the three
splits: text = questions from the 4B meta (first assert the three models' meta
dataset_id sequences are identical position by position, so the lexical
predictions are model-independent and shared by the three models); each split
reruns with groups replaced by the remapped groups. Paired difference = the
three-split mean of (probe − lexical) within the same resample.

**Exploratory, not pre-registered, touches no frozen readout; the G-5
judgement stays frozen on the original split.**

Anchors (3dp digit-for-digit, stop on failure):
  probe main/SM (incl. CI endpoints) × three models × three splits = outputs/ci/hidden_probe_foldseed_sm_all3.txt
  lexical, original split, full-set out-of-fold bal = 0.556, per-fold C = [0.1, 0.01, 0.01, 10.0, 10.0]
       (outputs/ci/hidden_probe.txt [G-3]; same value for all three models — same text, folds, labels)

Usage: python scripts/probe_foldseed_meanci_tfidf.py > outputs/ci/probe_sm_meanci_tfidf.txt
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_hidden_probe import (  # noqa: E402  single source, don't write another copy
    BOOT, CGRID, N_INNER, N_OUTER, SEED, bal_acc, boot_ci, load_reps, nested_cv, tfidf_cv)
from summarize_length_matched import build_subsets, cem_match  # noqa: E402
from probe_foldseed_variant import ANCHORS as MAIN_ANCHORS, FOLD_SEEDS, remap_groups  # noqa: E402
from probe_foldseed_sm import N_MAIN, N_SM, sm_readout  # noqa: E402

MODELS = ('Qwen/Qwen3-4B', 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct')
SPLITS = (('原划分', None), (f'seed={FOLD_SEEDS[0]}', FOLD_SEEDS[0]),
          (f'seed={FOLD_SEEDS[1]}', FOLD_SEEDS[1]))
KEY = {'原划分': 'orig', f'seed={FOLD_SEEDS[0]}': f's{FOLD_SEEDS[0]}',
       f'seed={FOLD_SEEDS[1]}': f's{FOLD_SEEDS[1]}'}   # npz keys ASCII-only
# anchors = copied line by line from hidden_probe_foldseed_sm_all3.txt: model -> split_tag -> (main, SM, lo, hi)
ALL3_ANCHORS = {
    'Qwen/Qwen3-4B': {'原划分': (0.692, 0.596, 0.522, 0.674),
                      'seed=20260811': (0.744, 0.663, 0.584, 0.743),
                      'seed=20260812': (0.682, 0.612, 0.527, 0.694)},
    'Qwen/Qwen2.5-7B-Instruct': {'原划分': (0.716, 0.635, 0.561, 0.712),
                                 'seed=20260811': (0.727, 0.646, 0.573, 0.718),
                                 'seed=20260812': (0.711, 0.624, 0.547, 0.699)},
    'meta-llama/Llama-3.1-8B-Instruct': {'原划分': (0.734, 0.646, 0.566, 0.719),
                                         'seed=20260811': (0.699, 0.584, 0.514, 0.659),
                                         'seed=20260812': (0.695, 0.573, 0.499, 0.648)},
}
TFIDF_ANCHOR_BAL = 0.556                       # hidden_probe.txt [G-3] full-set out-of-fold
TFIDF_ANCHOR_C = [0.1, 0.01, 0.01, 10.0, 10.0]  # same source; C selected in each outer fold
NPZ = os.path.join(HERE, '..', 'outputs', 'probe_foldseed_preds.npz')
N_JOBS = 12


def safe(model):
    return model.replace('/', '_').replace('-', '_')


def d3(x):
    """3dp literal. Anchor comparison is literal digit-for-digit, not tolerance-based."""
    return f'{x:.3f}'


def stop(msg):
    print(f'\n❌ 锚点不复现,停:{msg}')
    sys.stdout.flush()
    sys.exit(1)


def mean_bal_ci(y, preds, groups, sel=None):
    """Statistic = bal-acc computed per prediction set, then averaged over the three; clustered bootstrap by groups.

    preds = [out-of-fold predictions of the three splits]; sel = row subset (boolean mask) or None (all).
    """
    if sel is not None:
        y, preds, groups = y[sel], [p[sel] for p in preds], groups[sel]
    point = float(np.mean([bal_acc(y, p) for p in preds]))
    lo, hi = boot_ci(lambda i: float(np.mean([bal_acc(y[i], p[i]) for p in preds])),
                     groups, np.random.default_rng(SEED), BOOT)
    return point, lo, hi


def mean_diff_ci(y, preds_a, preds_b, groups, sel=None):
    """Statistic = mean over the three splits of (bal_a − bal_b) within the same resample."""
    if sel is not None:
        y, groups = y[sel], groups[sel]
        preds_a, preds_b = [p[sel] for p in preds_a], [p[sel] for p in preds_b]
    point = float(np.mean([bal_acc(y, a) - bal_acc(y, b) for a, b in zip(preds_a, preds_b)]))
    lo, hi = boot_ci(
        lambda i: float(np.mean([bal_acc(y[i], a[i]) - bal_acc(y[i], b[i])
                                 for a, b in zip(preds_a, preds_b)])),
        groups, np.random.default_rng(SEED), BOOT)
    return point, lo, hi


def main():
    print('# 三划分均值统计量的聚类 bootstrap CI + 词面基线配对差'
          '(probe_foldseed_meanci_tfidf.py,2026-08-11)')
    print('# 探索性,无预注册,不动冻结读数;G-5 判定仍冻结在原划分')
    print(f'# 统计量 = 三划分 bal-acc 的均值;CI = atomic 聚类 bootstrap {BOOT} 次 seed {SEED}')
    print(f'#   每次重采样对三套**已存**折外预测各算一次 bal-acc 再取均值(全程不重训)')
    print(f'# 折 seed = 原划分 + {list(FOLD_SEEDS)};SM 配平子集 n={N_SM}(全集 n={N_MAIN})')
    print(f'# 词面基线 = summarize_hidden_probe.tfidf_cv(1-2gram),同折同标签,只网格 C∈{CGRID}')
    print(f'# n_jobs={N_JOBS};逐行折外预测落盘 → outputs/probe_foldseed_preds.npz')
    sys.stdout.flush()

    f = build_subsets()
    SM = cem_match(f)

    # ---------------- 1) three models × three splits: rerun nested_cv, compare anchors, write row-level preds ----------------
    store, model_state, anchor_ok = {}, {}, True
    ref_ids_full = ref_ids_main = None
    for model in MODELS:
        last, _g3, meta = load_reps(model, smoke=False)
        layers = list(range(last.shape[1]))
        main_df = meta[(meta.agreement_lv == 5) & (meta.modus == 'ponens')]
        idx = main_df.index.values
        Xm, Gm = last[idx], main_df['atomic_idx'].values
        ym = (main_df['gold'] == 'c').astype(int).values
        in_sm = main_df['dataset_id'].isin(SM).values
        assert len(idx) == N_MAIN and int(in_sm.sum()) == N_SM, '行数与登记不符'

        # the three models' meta dataset_id sequences are identical position by position → lexical predictions are model-independent, shareable
        ids_full = meta['dataset_id'].values
        ids_main = main_df['dataset_id'].values
        if ref_ids_full is None:
            ref_ids_full, ref_ids_main, ref_model = ids_full, ids_main, model
            main4b = main_df.copy()        # text source for the lexical baseline (avoids re-loading reps)
        else:
            assert np.array_equal(ids_full, ref_ids_full), \
                f'{model} 的 meta dataset_id 序列与 {ref_model} 不同 → 词面基线不可共用'
            assert np.array_equal(ids_main, ref_ids_main), f'{model} 主集 dataset_id 序列不同'

        print(f'\n{"=" * 78}\n{model}  (taps={last.shape[1]}, hidden={last.shape[2]})\n{"=" * 78}')
        print(f'  n={len(idx)}  组={pd.Series(Gm).nunique()}  '
              f'SM 行内 REQ/ALT = {int(ym[in_sm].sum())}/{N_SM - int(ym[in_sm].sum())}')
        sys.stdout.flush()

        preds_main, preds_sm_tag = [], []
        for tag, seed in SPLITS:
            groups_used = Gm if seed is None else remap_groups(Gm, seed)
            cv = nested_cv(Xm, ym, groups_used, layers, CGRID, N_OUTER, N_INNER, N_JOBS)
            b_main = bal_acc(ym, cv['oof_pred'])
            b, lo, hi, req, alt = sm_readout(ym, cv['oof_pred'], in_sm, Gm)
            a = ALL3_ANCHORS[model][tag]
            got, exp = (b_main, b, lo, hi), a
            ok = all(d3(x) == d3(z) for x, z in zip(got, exp))
            anchor_ok &= ok
            print(f'\n  --- {tag} ---')
            print(f'  主读数(全 {N_MAIN}) bal = {b_main:.3f}   '
                  f'SM(n={N_SM}) bal = {b:.3f} [{lo:.3f},{hi:.3f}]  REQ={req:.3f}/ALT={alt:.3f}')
            print(f'  [锚点 vs hidden_probe_foldseed_sm_all3.txt] '
                  f'主 {b_main:.3f}/{a[0]:.3f}  SM {b:.3f} [{lo:.3f},{hi:.3f}] vs '
                  f'{a[1]:.3f} [{a[2]:.3f},{a[3]:.3f}] → {"复现 ✅" if ok else "不复现 ❌"}')
            if not ok:
                stop(f'{model} / {tag}')
            store[f'{safe(model)}__{KEY[tag]}__pred'] = cv['oof_pred']
            store[f'{safe(model)}__{KEY[tag]}__score'] = cv['oof_score']
            preds_main.append(cv['oof_pred'])
            preds_sm_tag.append((tag, b))
            sys.stdout.flush()

        store[f'{safe(model)}__y'] = ym
        store[f'{safe(model)}__in_sm'] = in_sm
        store[f'{safe(model)}__groups'] = Gm
        model_state[model] = dict(y=ym, groups=Gm, in_sm=in_sm, preds=preds_main)
        print(f'\n  == {model} 三划分 SM bal = '
              f'[{", ".join(f"{b:.3f}" for _, b in preds_sm_tag)}] ==')
        sys.stdout.flush()
        del last, Xm

    # ---------------- 2) lexical baseline × three splits ----------------
    print(f'\n{"=" * 78}\n词面基线(G-3 tfidf_cv)× 三划分 —— 文本取自 {ref_model} 的 meta[questions]'
          f'\n{"=" * 78}')
    texts = main4b['questions'].values
    y_t = (main4b['gold'] == 'c').astype(int).values
    G_t = main4b['atomic_idx'].values
    in_sm_t = main4b['dataset_id'].isin(SM).values
    assert np.array_equal(y_t, model_state[ref_model]['y']), '词面标签与探针标签不一致'
    assert np.array_equal(G_t, model_state[ref_model]['groups']), '词面组与探针组不一致'
    assert np.array_equal(in_sm_t, model_state[ref_model]['in_sm']), '词面 SM 掩码不一致'
    print('  [断言] 三模型 meta 的 dataset_id 序列逐位相同 ✅ → 词面预测三模型共用')

    tf_preds, tf_rows = [], []
    for tag, seed in SPLITS:
        groups_used = G_t if seed is None else remap_groups(G_t, seed)
        tp, tc = tfidf_cv(texts, y_t, groups_used, CGRID, N_OUTER, N_INNER, N_JOBS)
        b_full = bal_acc(y_t, tp)
        b_sm = bal_acc(y_t[in_sm_t], tp[in_sm_t])
        print(f'\n  --- {tag} ---')
        print(f'  全集(n={N_MAIN}) bal = {b_full:.3f}   SM(n={N_SM}) bal = {b_sm:.3f}   '
              f'各折 C = {tc}')
        if seed is None:
            ok = (d3(b_full) == d3(TFIDF_ANCHOR_BAL)
                  and [float(c) for c in tc] == TFIDF_ANCHOR_C)
            print(f'  [锚点 vs hidden_probe.txt G-3] 全集 {b_full:.3f}/{TFIDF_ANCHOR_BAL:.3f}  '
                  f'C {tc} vs {TFIDF_ANCHOR_C} → {"复现 ✅" if ok else "不复现 ❌"}')
            anchor_ok &= ok
            if not ok:
                stop('词面基线原划分')
        store[f'tfidf__{KEY[tag]}__pred'] = tp
        tf_preds.append(tp)
        tf_rows.append((tag, b_full, b_sm))
        sys.stdout.flush()
    store['tfidf__y'] = y_t
    store['tfidf__in_sm'] = in_sm_t
    store['tfidf__groups'] = G_t

    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez(NPZ, **store)
    print(f'\n  [落盘] {len(store)} 个键 → {os.path.relpath(NPZ, os.path.join(HERE, ".."))}')
    sys.stdout.flush()

    # ---------------- 3) clustered bootstrap CI of the mean statistic ----------------
    print(f'\n{"=" * 78}\n均值统计量的聚类 bootstrap CI(不重训,只用已存预测)\n{"=" * 78}')
    res = {}
    for model in MODELS:
        s = model_state[model]
        res[model] = dict(
            main=mean_bal_ci(s['y'], s['preds'], s['groups']),
            sm=mean_bal_ci(s['y'], s['preds'], s['groups'], s['in_sm']))
        print(f'  {model}')
        print(f'    三划分主读数均值 = {res[model]["main"][0]:.3f} '
              f'[{res[model]["main"][1]:.3f},{res[model]["main"][2]:.3f}]')
        print(f'    三划分 SM 均值   = {res[model]["sm"][0]:.3f} '
              f'[{res[model]["sm"][1]:.3f},{res[model]["sm"][2]:.3f}]')
        sys.stdout.flush()
    tf_full = mean_bal_ci(y_t, tf_preds, G_t)
    tf_sm = mean_bal_ci(y_t, tf_preds, G_t, in_sm_t)
    print(f'  词面基线(tfidf)')
    print(f'    三划分全集均值 = {tf_full[0]:.3f} [{tf_full[1]:.3f},{tf_full[2]:.3f}]')
    print(f'    三划分 SM 均值 = {tf_sm[0]:.3f} [{tf_sm[1]:.3f},{tf_sm[2]:.3f}]')
    sys.stdout.flush()

    # ---------------- 4) probe − lexical paired difference ----------------
    print(f'\n{"=" * 78}\n探针 − 词面 配对差(同折同标签同行;SM 行为主,全集为对照)\n{"=" * 78}')
    diff = {}
    for model in MODELS:
        s = model_state[model]
        per_sm = [bal_acc(s['y'][s['in_sm']], p[s['in_sm']]) - bal_acc(y_t[in_sm_t], q[in_sm_t])
                  for p, q in zip(s['preds'], tf_preds)]
        per_full = [bal_acc(s['y'], p) - bal_acc(y_t, q) for p, q in zip(s['preds'], tf_preds)]
        d_sm = mean_diff_ci(s['y'], s['preds'], tf_preds, s['groups'], s['in_sm'])
        d_full = mean_diff_ci(s['y'], s['preds'], tf_preds, s['groups'])
        diff[model] = dict(sm=d_sm, full=d_full, per_sm=per_sm, per_full=per_full)
        print(f'  {model}')
        print(f'    SM 逐划分差 = [{", ".join(f"{v:+.3f}" for v in per_sm)}]  '
              f'均值差 = {d_sm[0]:+.3f} [{d_sm[1]:+.3f},{d_sm[2]:+.3f}]')
        print(f'    全集逐划分差 = [{", ".join(f"{v:+.3f}" for v in per_full)}]  '
              f'均值差 = {d_full[0]:+.3f} [{d_full[1]:+.3f},{d_full[2]:+.3f}]')
        sys.stdout.flush()

    # ---------------- summary table ----------------
    print(f'\n{"=" * 78}\n汇总:三划分均值 [95% 聚类 bootstrap CI]\n{"=" * 78}')
    print(f'  {"model":34s} {"探针主读数":>22s} {"探针 SM":>22s} '
          f'{"配对差 SM(−词面)":>24s}')
    for model in MODELS:
        m, s2, d = res[model]['main'], res[model]['sm'], diff[model]['sm']
        print(f'  {model:34s} {f"{m[0]:.3f} [{m[1]:.3f},{m[2]:.3f}]":>22s} '
              f'{f"{s2[0]:.3f} [{s2[1]:.3f},{s2[2]:.3f}]":>22s} '
              f'{f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]":>24s}')
    print(f'  {"词面基线 tfidf(三模型共用)":34s} '
          f'{f"{tf_full[0]:.3f} [{tf_full[1]:.3f},{tf_full[2]:.3f}]":>22s} '
          f'{f"{tf_sm[0]:.3f} [{tf_sm[1]:.3f},{tf_sm[2]:.3f}]":>22s} {"—":>24s}')
    print('\n  逐划分明细(SM bal-acc):')
    print(f'    {"model":34s} {"原划分":>9s} {"s=20260811":>12s} {"s=20260812":>12s} {"均值":>8s}')
    for model in MODELS:
        s = model_state[model]
        bs = [bal_acc(s['y'][s['in_sm']], p[s['in_sm']]) for p in s['preds']]
        print(f'    {model:34s} {bs[0]:9.3f} {bs[1]:12.3f} {bs[2]:12.3f} {np.mean(bs):8.3f}')
    print(f'    {"词面 tfidf":34s} {tf_rows[0][2]:9.3f} {tf_rows[1][2]:12.3f} '
          f'{tf_rows[2][2]:12.3f} {np.mean([r[2] for r in tf_rows]):8.3f}')
    print('  逐划分明细(全集 bal-acc):')
    for model in MODELS:
        s = model_state[model]
        bs = [bal_acc(s['y'], p) for p in s['preds']]
        print(f'    {model:34s} {bs[0]:9.3f} {bs[1]:12.3f} {bs[2]:12.3f} {np.mean(bs):8.3f}')
    print(f'    {"词面 tfidf":34s} {tf_rows[0][1]:9.3f} {tf_rows[1][1]:12.3f} '
          f'{tf_rows[2][1]:12.3f} {np.mean([r[1] for r in tf_rows]):8.3f}')

    # ---------------- mechanical check against criteria (check only, no interpretation) ----------------
    print(f'\n{"=" * 78}\n机械对号(只对号,不解读)\n{"=" * 78}')
    for model in MODELS:
        p, lo, hi = res[model]['sm']
        c1 = (p >= 0.60) and not (lo <= 0.5 <= hi)
        d, dlo, dhi = diff[model]['sm']
        c2 = not (dlo <= 0.0 <= dhi)
        print(f'  {model:34s} SM 均值 ≥0.60 且 CI 不含 0.5: {"是" if c1 else "否"}   '
              f'配对差 CI 不含 0: {"是" if c2 else "否"}')
    print(f'\n  [锚点总账] 探针 9 组(主+SM+CI 两端)与词面原划分(bal+各折 C):'
          f'{"全部逐位复现 ✅" if anchor_ok else "有不复现 ❌"}')
    print('\n本脚本只出数不判定。')


if __name__ == '__main__':
    main()
