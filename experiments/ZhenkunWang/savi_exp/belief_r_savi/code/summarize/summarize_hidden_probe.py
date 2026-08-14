#!/usr/bin/env python
"""Hidden-layer linear probe: probe training + all gates + printing the frozen criteria. CPU only.

Criteria, gates, and instrument self-checks are frozen in `PREREG_hidden_probe.md`
(2026-08-03). This script only prints "falls into tier X per the criteria" against that
set; the final judgment is left to a human — no thresholds are invented here.

Main pipeline (PREREG §3): lv=5 ponens rows (n=391, 152 atomic_idx groups).
Outer GroupKFold 5 folds (group=atomic_idx); inner GroupKFold 4 folds within each training
fold, jointly selecting (layer × C ∈ {0.01,0.1,1,10}), selection objective = inner balanced
accuracy; the layer is never picked on the outer test set.
Standardization is fit within the training fold; L2 logistic regression
class_weight='balanced'; features cast to float32.
Main readout = outer out-of-fold balanced accuracy; CI = cluster bootstrap by atomic_idx,
5000 resamples, seed 20260803.

Gates (PREREG §4): G-1 modus positive control / G-2 full-permutation label shuffle
(§6b revision 1) / G-3 TF-IDF surface baseline / G-4 length-rule registration /
G-5 SM length-matched fold (+ robust variant, registered not judged) / G-6 cross-model.
Instrument self-checks: V1 length-rule anchors (full set 0.800±0.02, SM 0.537±0.03;
raise outright if not reproduced), V3 grouped no-leak assertions. The SM subset comes from
the imported build_subsets/cem_match single source.

Usage:
  python scripts/summarize_hidden_probe.py > outputs/ci/hidden_probe.txt
  python scripts/summarize_hidden_probe.py --smoke   # only verifies the data path, no readouts
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_length_matched import build_subsets, cem_match, auc  # noqa: E402  single source

# Belief-R dataset CSV. The data is not distributed with this repo (see the README's data section); point the BELIEF_R_CSV env var at a local copy, falling back to
# <repo>/data/queries_time_t1.csv when unset.
CSV = os.environ.get('BELIEF_R_CSV') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'queries_time_t1.csv')
REP_ROOT = os.path.join(HERE, '..', 'outputs', 'hidden_probe')
MODELS = ('Qwen/Qwen3-4B', 'Qwen/Qwen2.5-7B-Instruct')
CGRID = (0.01, 0.1, 1.0, 10.0)
N_OUTER, N_INNER = 5, 4
BOOT, SEED = 5000, 20260803
# V1 anchors (PREREG §5). Failing to reproduce them is a pipeline error, not a new finding.
ANCHOR_FULL, TOL_FULL = 0.800, 0.02
ANCHOR_SM, TOL_SM = 0.537, 0.03


def bal_acc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float('nan')
    return ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean()) / 2


def boot_ci(stat, groups, rng, n_boot):
    """Cluster bootstrap: resample whole groups, stat(idx)→scalar. Returns (lo, hi)."""
    by = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    pools = [np.asarray(v) for v in by.values()]
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([pools[k] for k in rng.integers(0, len(pools), len(pools))])
        v = stat(idx)
        if not np.isnan(v):
            out.append(v)
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)


def _fit(Xtr, ytr, C):
    clf = LogisticRegression(C=C, class_weight='balanced', max_iter=5000)
    clf.fit(Xtr, ytr)
    return clf

def _inner_task(X, y, tr, va, layer, cgrid):
    """One (inner fold × layer): fit the scaler once, sweep C. Returns [n_C] balanced accuracies."""
    Xtr = X[tr, layer].astype(np.float32)
    Xva = X[va, layer].astype(np.float32)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    return [bal_acc(y[va], _fit(Xtr, y[tr], C).predict(Xva)) for C in cgrid]

def _final_task(X, y, tr, te, layer, C):
    """Outer final fit: scaler+logreg on the training fold, predict the test fold. Returns (pred, score, sc, clf)."""
    Xtr = X[tr, layer].astype(np.float32)
    Xte = X[te, layer].astype(np.float32)
    sc = StandardScaler().fit(Xtr)
    clf = _fit(sc.transform(Xtr), y[tr], C)
    return clf.predict(sc.transform(Xte)), clf.decision_function(sc.transform(Xte)), sc, clf


def nested_cv(X, y, groups, layers, cgrid, n_outer, n_inner, n_jobs):
    """Nested CV of PREREG §3. Returns oof_pred/oof_score/chosen/models/folds/inner_mats.

    On ties argmax takes the first occurrence → smaller layer first, smaller C first
    (deterministic).
    """
    y, groups = np.asarray(y), np.asarray(groups)
    dummy = np.zeros(len(y))
    folds = list(GroupKFold(n_outer).split(dummy, y, groups))
    tasks = []
    for k, (tr, te) in enumerate(folds):
        assert not set(groups[tr]) & set(groups[te]), 'V3 失败：外层 atomic_idx 跨折'
        for itr, iva in GroupKFold(n_inner).split(dummy[tr], y[tr], groups[tr]):
            assert not set(groups[tr][itr]) & set(groups[tr][iva]), 'V3 失败：内层跨折'
            tasks.append((k, tr[itr], tr[iva]))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_inner_task)(X, y, tr, va, L, cgrid)
        for (k, tr, va) in tasks for L in layers)
    res = iter(np.asarray(res).reshape(len(tasks), len(layers), len(cgrid)))
    per_fold = collections.defaultdict(list)
    for (k, _, _), mat in zip(tasks, res):
        per_fold[k].append(mat)

    oof_pred = np.full(len(y), -1)
    oof_score = np.zeros(len(y))
    chosen, models, inner_mats = [], [], []
    for k, (tr, te) in enumerate(folds):
        arr = np.mean(per_fold[k], axis=0)          # [layer, C] inner balanced accuracy
        li, ci = np.unravel_index(arr.argmax(), arr.shape)
        L, C = layers[li], cgrid[ci]
        pred, score, sc, clf = _final_task(X, y, tr, te, L, C)
        oof_pred[te], oof_score[te] = pred, score
        chosen.append((int(L), C, float(arr[li, ci])))
        models.append((int(L), C, sc, clf))
        inner_mats.append(arr)
    assert (oof_pred != -1).all(), 'out-of-fold 覆盖不完整'
    return dict(oof_pred=oof_pred, oof_score=oof_score, chosen=chosen,
                models=models, folds=folds, inner_mats=inner_mats)


def layer_curve(X, y, groups, cv, layers, cgrid, n_jobs):
    """Extra description: fixed-layer out-of-fold curve per layer. C reuses the best C for that layer from the main pipeline's inner grid."""
    tasks = []
    for k, (tr, te) in enumerate(cv['folds']):
        for li, L in enumerate(layers):
            tasks.append((tr, te, L, cgrid[int(cv['inner_mats'][k][li].argmax())]))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_final_task)(X, y, tr, te, L, C) for tr, te, L, C in tasks)
    # backfill out-of-fold predictions in (fold, layer) order
    it = iter(res)
    preds_by_layer = {L: np.full(len(y), -1) for L in layers}
    for k, (tr, te) in enumerate(cv['folds']):
        for L in layers:
            p, _, _, _ = next(it)
            preds_by_layer[L][te] = p
    return [(int(L), bal_acc(y, preds_by_layer[L])) for L in layers]


def _tfidf_inner(texts, y, tr, va, cgrid):
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(texts[tr])
    Xtr, Xva = vec.transform(texts[tr]), vec.transform(texts[va])
    return [bal_acc(y[va], _fit(Xtr, y[tr], C).predict(Xva)) for C in cgrid]

def tfidf_cv(texts, y, groups, cgrid, n_outer, n_inner, n_jobs):
    """G-3 surface baseline: same folds, same labels, same protocol; only C is grid-searched. Returns (oof_pred, chosen)."""
    texts, y, groups = np.asarray(texts, object), np.asarray(y), np.asarray(groups)
    dummy = np.zeros(len(y))
    folds = list(GroupKFold(n_outer).split(dummy, y, groups))
    tasks = [(k, tr[itr], tr[iva])
             for k, (tr, te) in enumerate(folds)
             for itr, iva in GroupKFold(n_inner).split(dummy[tr], y[tr], groups[tr])]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_tfidf_inner)(texts, y, tr, va, cgrid) for k, tr, va in tasks)
    per_fold = collections.defaultdict(list)
    for (k, _, _), r in zip(tasks, res):
        per_fold[k].append(r)
    oof_pred = np.full(len(y), -1)
    chosen = []
    for k, (tr, te) in enumerate(folds):
        C = cgrid[int(np.mean(per_fold[k], axis=0).argmax())]
        vec = TfidfVectorizer(ngram_range=(1, 2)).fit(texts[tr])
        clf = _fit(vec.transform(texts[tr]), y[tr], C)
        oof_pred[te] = clf.predict(vec.transform(texts[te]))
        chosen.append(C)
    assert (oof_pred != -1).all()
    return oof_pred, chosen


def sm_robust(X, y, groups, sm_mask, layers, cgrid, n_inner, n_jobs):
    """G-5 robust variant: training fully excludes SM rows; (layer×C) chosen by inner CV on the excluded training set; single prediction on SM."""
    y, groups, sm_mask = np.asarray(y), np.asarray(groups), np.asarray(sm_mask)
    tr, te = np.where(~sm_mask)[0], np.where(sm_mask)[0]
    overlap = len(set(groups[tr]) & set(groups[te]))
    dummy = np.zeros(len(tr))
    tasks = [(tr[itr], tr[iva])
             for itr, iva in GroupKFold(n_inner).split(dummy, y[tr], groups[tr])]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_inner_task)(X, y, a, b, L, cgrid) for a, b in tasks for L in layers)
    arr = np.asarray(res).reshape(len(tasks), len(layers), len(cgrid)).mean(axis=0)
    li, ci = np.unravel_index(arr.argmax(), arr.shape)
    L, C = layers[li], cgrid[ci]
    pred, _, _, _ = _final_task(X, y, tr, te, L, C)
    return te, pred, (int(L), C), overlap


def load_reps(model_name, smoke):
    safe = model_name.replace('/', '_').replace('-', '_')
    d = os.path.join(REP_ROOT, '_smoke', safe) if smoke else os.path.join(REP_ROOT, safe)
    z = np.load(os.path.join(d, 'reps.npz'))
    meta = pd.read_json(os.path.join(d, 'meta.jsonl'), lines=True)
    last, g3 = z['last_tok'], z['gamma3_tok']
    assert last.shape == g3.shape and len(meta) == last.shape[0], '形状与 meta 不符'
    assert np.isfinite(last.astype(np.float32)).all(), 'last_tok 有 NaN/Inf'
    assert np.isfinite(g3.astype(np.float32)).all(), 'gamma3_tok 有 NaN/Inf'
    # reconcile meta against the source CSV by idx (in the spirit of summarize_generative_ci's V0)
    src = pd.read_csv(CSV)
    for _, m in meta.iterrows():
        r = src.iloc[m['idx']]
        assert (r['ground_truth'] == m['gold'] and r['modus'] == m['modus']
                and int(r['atomic_idx']) == m['atomic_idx']), f"meta idx {m['idx']} 与 CSV 不符"
    meta = meta.join(src['questions'], on='idx')
    return last, g3, meta


def shuffle_labels(y, seed):
    """G-2: full-permutation shuffle of the main-set labels (PREREG §6b revision 1).

    The original "within-group shuffle" was ruled an instrument defect: 128 of the main
    set's 152 groups are label-pure, making it a near-identity transform. This arm's job =
    catching pipeline leaks of the label-into-features / cache-serialization kind;
    group-integrity leakage is covered by the V3 hard assertions."""
    return np.random.default_rng(seed).permutation(np.asarray(y))


def run_model(model_name, f_scen, SM, n_jobs, smoke):
    last, g3, meta = load_reps(model_name, smoke)
    taps = last.shape[1]
    layers = list(range(taps))
    print(f'\n{"=" * 78}\n{model_name}  (N={len(meta)}, taps={taps}, hidden={last.shape[2]})\n{"=" * 78}')

    if smoke:
        # Only verify the data path and the folding machinery: synthetic groups/labels,
        # shrunken grid; no readouts. Groups = 4, each with 2 rows, one per class → any
        # group-wise train/val split has both classes, so the folding machinery can run
        # (real labels on 8 rows can't sustain grouped CV, and shouldn't be read in smoke).
        n = len(meta)
        assert n >= 8 and n % 2 == 0, f'smoke 需要 ≥8 的偶数行，得到 {n}'
        gsyn = np.arange(n) % (n // 2)          # 2 rows per group: i and i+n/2
        y = (np.arange(n) >= n // 2).astype(int)  # exactly one of each class per group
        lay = [0, taps // 2, taps - 1]
        cg = (0.1, 1.0)
        cv = nested_cv(last, y, gsyn, lay, cg, 2, 2, n_jobs)
        assert (cv['oof_pred'] != -1).all() and len(cv['chosen']) == 2
        curve = layer_curve(last, y, gsyn, cv, lay, cg, n_jobs)
        assert len(curve) == len(lay)
        rng = np.random.default_rng(SEED)
        boot_ci(lambda idx: bal_acc(y[idx], cv['oof_pred'][idx]), gsyn, rng, 50)
        tp, tc = tfidf_cv(meta['questions'].values, y, gsyn, cg, 2, 2, n_jobs)
        assert (tp != -1).all()
        cv3 = nested_cv(g3, y, gsyn, lay, cg, 2, 2, n_jobs)
        assert (cv3['oof_pred'] != -1).all()
        print('[smoke] 形状/OOF 覆盖/V3 断言/层曲线/聚类 bootstrap/TF-IDF/γ3 通路全部跑通。'
              '样本太小，无读数；V0/V1/闸门断言按预案跳过。SMOKE PASS')
        return None

    # ---- main set: lv=5 ponens ----
    main = meta[(meta.agreement_lv == 5) & (meta.modus == 'ponens')]
    idx = main.index.values
    assert len(idx) == 391, f'主集行数 {len(idx)} != 391'
    Xm, Gm = last[idx], main['atomic_idx'].values
    ym = (main['gold'] == 'c').astype(int).values
    assert pd.Series(Gm).nunique() == 152

    cv = nested_cv(Xm, ym, Gm, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    rng = np.random.default_rng(SEED)
    b_main = bal_acc(ym, cv['oof_pred'])
    lo, hi = boot_ci(lambda i: bal_acc(ym[i], cv['oof_pred'][i]), Gm, rng, BOOT)
    a_main = auc(cv['oof_score'], ym)
    alo, ahi = boot_ci(lambda i: auc(cv['oof_score'][i], ym[i]), Gm, rng, BOOT)
    print(f'\n[主读数] lv=5 ponens × 主位置 × 分组 CV')
    print(f'  平衡准确率 = {b_main:.3f} [{lo:.3f},{hi:.3f}]  （随机 0.500）')
    print(f'  AUC        = {a_main:.3f} [{alo:.3f},{ahi:.3f}]  '
          f'（对表：L1 0.611 / 长度规则 0.800 / 天花板 0.843，皆 lv=5 口径）')
    print(f'  各外层折选中 (层, C, 内层bal_acc): {cv["chosen"]}')

    # ---- G-1 modus positive control: all 1,744 rows ----
    yg1 = (meta['modus'] == 'ponens').astype(int).values
    Gall = meta['atomic_idx'].values
    cv1 = nested_cv(last, yg1, Gall, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    b_g1 = bal_acc(yg1, cv1['oof_pred'])
    g1_ok = b_g1 >= 0.95
    print(f'\n[G-1] modus 正对照（全 1,744 行）: 平衡准确率 = {b_g1:.3f}  '
          f'{"✅ ≥0.95" if g1_ok else "❌ <0.95 → 仪器坏，其余数字一律不读"}')

    # ---- G-2 label shuffle (full permutation, PREREG §6b revision 1) ----
    y_shuf = shuffle_labels(ym, SEED)
    cv2 = nested_cv(Xm, y_shuf, Gm, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    b_g2 = bal_acc(y_shuf, cv2['oof_pred'])
    lo2, hi2 = boot_ci(lambda i: bal_acc(y_shuf[i], cv2['oof_pred'][i]), Gm,
                       np.random.default_rng(SEED), BOOT)
    g2_ok = lo2 <= 0.5 <= hi2
    print(f'[G-2] 全排列打乱标签（PREREG 修订 1）: 平衡准确率 = {b_g2:.3f} [{lo2:.3f},{hi2:.3f}]  '
          f'{"✅ CI 含 0.5" if g2_ok else "❌ CI 不含 0.5 → 管线泄漏，全部作废"}')

    # ---- G-3 TF-IDF surface baseline ----
    tf_pred, tf_C = tfidf_cv(main['questions'].values, ym, Gm, CGRID, N_OUTER, N_INNER, n_jobs)
    b_g3 = bal_acc(ym, tf_pred)
    g3_ok = (b_main - b_g3) >= 0.05
    print(f'[G-3] TF-IDF(1-2gram) 词面基线: 平衡准确率 = {b_g3:.3f}（各折 C={tf_C}）  '
          f'探针−基线 = {b_main - b_g3:+.3f}  {"✅ ≥0.05" if g3_ok else "❌ <0.05"}')

    # ---- G-4 length-rule registration (V1 anchors already asserted in main()) ----
    scen = f_scen.set_index('dataset_id')
    d_main = scen.loc[main['dataset_id']].d.values
    print(f'[G-4] 长度规则 Δ=len(r)−len(p): 主集 AUC = {auc(d_main, ym):.3f}（登记，不设通过线）')

    # ---- G-5 SM length-matched fold ----
    in_sm = main['dataset_id'].isin(SM).values
    b_g5 = bal_acc(ym[in_sm], cv['oof_pred'][in_sm])
    lo5, hi5 = boot_ci(lambda i: bal_acc(ym[in_sm][i], cv['oof_pred'][in_sm][i]),
                       Gm[in_sm], np.random.default_rng(SEED), BOOT)
    g5_ok = (b_g5 >= 0.60) and not (lo5 <= 0.5 <= hi5)
    print(f'[G-5] SM 配平折（out-of-fold 限制在 SM，n={int(in_sm.sum())}）: '
          f'平衡准确率 = {b_g5:.3f} [{lo5:.3f},{hi5:.3f}]  '
          f'{"✅ ≥0.60 且 CI 不含 0.5" if g5_ok else "❌"}')
    te, pred_r, (Lr, Cr), overlap = sm_robust(Xm, ym, Gm, in_sm, layers, CGRID, N_INNER, n_jobs)
    b_g5r = bal_acc(ym[te], pred_r)
    lo5r, hi5r = boot_ci(lambda i: bal_acc(ym[te][i], pred_r[i]), Gm[te],
                         np.random.default_rng(SEED), BOOT)
    print(f'      鲁棒变体（训练排除 SM 行，单次预测；层={Lr} C={Cr}；'
          f'train/test 共享 atomic 组 {overlap} 个）: {b_g5r:.3f} [{lo5r:.3f},{hi5r:.3f}]'
          f'  —— 登记，不判定')

    # ---- extra description (not judged) ----
    curve = layer_curve(Xm, ym, Gm, cv, layers, CGRID, n_jobs)
    best = max(curve, key=lambda t: t[1])
    print(f'\n[描述] 逐层固定层 out-of-fold 平衡准确率（主位置；C=该层内层最优）：')
    print('  ' + '  '.join(f'{L}:{b:.3f}' for L, b in curve))
    print(f'  峰值层 = {best[0]}（{best[1]:.3f}）——只作图表描述，不参与判定')

    cvg = nested_cv(g3[idx], ym, Gm, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    b_gam = bal_acc(ym, cvg['oof_pred'])
    log, hig = boot_ci(lambda i: bal_acc(ym[i], cvg['oof_pred'][i]), Gm,
                       np.random.default_rng(SEED), BOOT)
    print(f'[描述] γ3 行末位置（诊断，不进判据）: 平衡准确率 = {b_gam:.3f} [{log:.3f},{hig:.3f}]')

    # tollens consistency: use each outer fold's model to predict the tollens rows of that fold's test scenarios
    tol = meta[(meta.agreement_lv == 5) & (meta.modus == 'tollens')]
    tol_by_sid = dict(zip(tol['dataset_id'], tol.index))
    agree, n_pair = 0, 0
    for k, (tr, tei) in enumerate(cv['folds']):
        L, C, sc, clf = cv['models'][k]
        sids = main.iloc[tei]['dataset_id'].values
        rows = [tol_by_sid[s] for s in sids if s in tol_by_sid]
        sids = [s for s in sids if s in tol_by_sid]
        if not rows:
            continue
        pt = clf.predict(sc.transform(last[rows][:, L].astype(np.float32)))
        pp = cv['oof_pred'][tei][[s in tol_by_sid for s in main.iloc[tei]['dataset_id']]]
        agree += (pt == pp).sum()
        n_pair += len(rows)
    print(f'[描述] tollens 行与同场景 ponens 行的预测一致率: {agree / n_pair:.3f} '
          f'（n={n_pair} 场景对）')

    return dict(b=b_main, ci=(lo, hi), auc=a_main, g1=b_g1, g1_ok=g1_ok,
                g2=(b_g2, (lo2, hi2)), g2_ok=g2_ok, g3=b_g3, g3_ok=g3_ok,
                g5=b_g5, g5_ci=(lo5, hi5), g5_ok=g5_ok)


def tier(r):
    """PREREG §6 tiers. On boundary ties, clause order applies: ≥0.70 first, then the ≤0.55 negative."""
    if not r['g1_ok']:
        return 'G-1 未过：仪器坏，其余数字一律不读'
    if not r['g2_ok']:
        return 'G-2 未过：管线泄漏，全部作废'
    b = r['b']
    if b >= 0.70:
        if r['g5_ok'] and r['g3_ok']:
            return '正结果候选（≥0.70 且 G-3 ≥0.05 且 G-5 过；还须 G-6 两模型同过）'
        if not r['g5_ok'] and 0.55 <= r['g5'] < 0.60:
            return '判据未覆盖：全集 ≥0.70 但 SM 落 0.55–0.60 灰区 → 不确定，交人裁决'
        if not r['g5_ok']:
            return '构造痕迹档：全集 ≥0.70 但 G-5 不过'
        return '判据未覆盖：≥0.70 且 G-5 过但 G-3 不过 → 不确定，交人裁决'
    if b <= 0.55:
        return '阴性档：≤0.55（只排除线性读出；写成"未发现"）'
    return '中间带：0.55–0.70，表征没比行为多给东西'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--n_jobs', type=int, default=16)
    ap.add_argument('--smoke', action='store_true',
                    help='读 _smoke/ 产物，只验证通路与折叠机制，不出读数')
    args = ap.parse_args()

    print('# 隐藏层线性探针（PREREG_hidden_probe.md，2026-08-03 冻结）')
    print(f'# 主管线：lv=5 ponens，外层 GroupKFold {N_OUTER}（组=atomic_idx），'
          f'内层 {N_INNER} 折联合选 层×C∈{CGRID}')
    print(f'# CI：atomic_idx 聚类 bootstrap {BOOT} 次，seed {SEED}')

    if not args.smoke:
        # V1 anchors (recomputed from the single source; not reproduced = pipeline error, raise)
        f = build_subsets()
        SM = cem_match(f)
        lv5 = f[f.lv == 5]
        a_full = auc(lv5.d.values, lv5.REQ.values)
        sm_rows = f[f.dataset_id.isin(SM)]
        a_sm = auc(sm_rows.d.values, sm_rows.REQ.values)
        print(f'\n[V1] 长度规则锚点: lv=5 全集 AUC = {a_full:.3f}（登记 {ANCHOR_FULL}±{TOL_FULL}）'
              f'  SM = {a_sm:.3f}（登记 {ANCHOR_SM}±{TOL_SM}）')
        assert abs(a_full - ANCHOR_FULL) <= TOL_FULL, 'V1 失败：全集锚点不复现，管线错'
        assert abs(a_sm - ANCHOR_SM) <= TOL_SM, 'V1 失败：SM 锚点不复现，管线错'
        print(f'[V1] 通过。SM 场景数 = {len(SM)}')
    else:
        f, SM = None, None

    results = {}
    for m in args.models.split(','):
        results[m] = run_model(m, f, SM, args.n_jobs, args.smoke)

    if args.smoke:
        print('\n[smoke] 全部模型通路检查完成。无读数，无判定。')
        return

    print(f'\n{"=" * 78}\n判定建议（PREREG §6；G-6 = 正结果须两模型同过）\n{"=" * 78}')
    for m, r in results.items():
        print(f'  {m}: 按判据落在 → {tier(r)}')
    tiers = [tier(r) for r in results.values()]
    if len(tiers) == 2:
        pos = ['正结果候选' in t for t in tiers]
        if all(pos):
            print('  [G-6] 两模型同过 → 按判据落在 H-P 成立档')
        elif any(pos):
            print('  [G-6] 一正一负 → 不稳健，如实报告为不确定，交人裁决')
        else:
            print('  [G-6] 两模型均非正结果 → 按各自档位读，无跨模型加成')
    print('  最终判定留给人；本脚本只对表，不定案。')


if __name__ == '__main__':
    main()
