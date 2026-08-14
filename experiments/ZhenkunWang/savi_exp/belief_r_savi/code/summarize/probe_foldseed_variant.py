#!/usr/bin/env python
"""Hidden-layer linear probe: robustness rerun with a different fold assignment. CPU only, zero GPU, modifies no old file.

The problem: the main readout in `summarize_hidden_probe.py` uses
`GroupKFold(shuffle=False)`, so fold assignment is deterministic (groups sorted by size
descending, greedily placed into the currently lightest fold); logistic regression is
convex. So the only randomizable axis in this pipeline = **the group → fold assignment**.
This script runs again with a different assignment, everything else in the protocol
untouched: layer grid = all taps, C ∈ {0.01,0.1,1,10}, inner 4 folds jointly selecting
(layer, C), bal-acc convention, main readout set = lv=5 ponens.

How the folds are changed: permute the **order** of the 152 atomic groups with
`np.random.default_rng(seed)` (equivalent to giving the groups new ids), then call the
same `GroupKFold`. Because GroupKFold first encodes with `np.unique` and then greedily
allocates by size descending, changing the group order changes the precedence among
equal-size groups (140 of this set's 152 groups fall in tied size≤5 classes) → the fold
composition changes wholesale while fold row counts stay balanced. Outer and inner share
this one permutation (the inner split consumes groups[tr], which follows automatically
after the remap).

What is reused is the original script's functions themselves
(`load_reps` / `nested_cv` / `bal_acc` / `boot_ci`), not a copy: the V3
"groups don't cross folds" assertion inside `nested_cv` still applies.

Usage:
  python scripts/probe_foldseed_variant.py > outputs/ci/hidden_probe_foldseed.txt
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_hidden_probe import (  # noqa: E402  single source, don't write another copy
    BOOT, CGRID, N_INNER, N_OUTER, SEED, bal_acc, boot_ci, load_reps, nested_cv)
from summarize_length_matched import auc  # noqa: E402

OUT = os.path.join(HERE, '..', 'outputs', 'ci', 'hidden_probe_foldseed.txt')
# Registered anchors (outputs/ci/hidden_probe.txt 2026-08-03 / hidden_probe_llama.txt):
# lv=5 ponens main-readout bal-acc, must reproduce digit-for-digit to 3dp.
ANCHORS = {'Qwen/Qwen3-4B': 0.692,
           'Qwen/Qwen2.5-7B-Instruct': 0.716,
           'meta-llama/Llama-3.1-8B-Instruct': 0.734}
MODELS = tuple(ANCHORS)
FOLD_SEEDS = (20260811, 20260812)
N_MAIN, N_GROUPS, N_REQ, N_ALT = 391, 152, 257, 134


def remap_groups(groups, seed):
    """Permute the group order with rng(seed) (give the groups new ids). A bijection; no row's membership changes."""
    u = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(u))
    m = {g: int(perm[i]) for i, g in enumerate(u)}
    out = np.array([m[g] for g in groups])
    assert pd.Series(out).nunique() == len(u), '重排后组数变了'
    return out


def fold_id(folds, n):
    v = np.full(n, -1)
    for k, (_, te) in enumerate(folds):
        v[te] = k
    assert (v != -1).all()
    return v


def cofold_agreement(a, b):
    """Agreement rate of "same fold / different fold" between two partitions (invariant to fold renaming). Expected ≈0.68 for independent partitions."""
    A = a[:, None] == a[None, :]
    B = b[:, None] == b[None, :]
    return float((A == B).mean())


def run_split(Xm, ym, Gm, layers, groups_used, n_jobs):
    """One full nested_cv + readout breakdown. groups_used only affects fold assignment."""
    t0 = time.time()
    cv = nested_cv(Xm, ym, groups_used, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    p = cv['oof_pred']
    b = bal_acc(ym, p)
    lo, hi = boot_ci(lambda i: bal_acc(ym[i], p[i]), Gm, np.random.default_rng(SEED), BOOT)
    req_rec = float((p[ym == 1] == 1).mean())          # REQ recall (gold='c')
    alt_rec = float((p[ym == 0] == 0).mean())          # ALT recall (gold≠'c')
    per_fold = []
    for k, (_, te) in enumerate(cv['folds']):
        per_fold.append(dict(k=k, n=len(te), n_req=int(ym[te].sum()),
                             bal=bal_acc(ym[te], p[te]),
                             req=float((p[te][ym[te] == 1] == 1).mean()),
                             alt=float((p[te][ym[te] == 0] == 0).mean())))
    return dict(cv=cv, bal=b, ci=(lo, hi), auc=auc(cv['oof_score'], ym),
                req_rec=req_rec, alt_rec=alt_rec, per_fold=per_fold,
                secs=time.time() - t0,
                fid=fold_id(cv['folds'], len(ym)))


def show(tag, r):
    pf = r['per_fold']
    bals = [f['bal'] for f in pf]
    print(f'\n  --- {tag} ---   ({r["secs"]:.0f}s)')
    print(f'  主读数 bal-acc = {r["bal"]:.3f} [{r["ci"][0]:.3f},{r["ci"][1]:.3f}]   '
          f'AUC = {r["auc"]:.3f}')
    print(f'  REQ 召回 = {r["req_rec"]:.3f} (n={N_REQ})   ALT 召回 = {r["alt_rec"]:.3f} (n={N_ALT})')
    print(f'  逐折 bal-acc = [{", ".join(f"{b:.3f}" for b in bals)}]   '
          f'极差 = {max(bals) - min(bals):.3f}')
    print('  逐折明细 (折: n, REQ数, bal, REQ召回, ALT召回):')
    for f in pf:
        print(f'    fold{f["k"]}: n={f["n"]:3d}  REQ={f["n_req"]:3d}  bal={f["bal"]:.3f}  '
              f'req={f["req"]:.3f}  alt={f["alt"]:.3f}')
    print('  内层选中 (层, C, 内层bal_acc): '
          + str([(L, C, round(v, 4)) for L, C, v in r['cv']['chosen']]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--seeds', default=','.join(str(s) for s in FOLD_SEEDS))
    ap.add_argument('--n_jobs', type=int, default=16)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]

    print('# 隐藏层线性探针 · 换折划分稳健性复跑（probe_foldseed_variant.py）')
    print('# 协议 = PREREG_hidden_probe.md §3 原样；唯一改动 = 组→折指派前重排组顺序')
    print(f'# 主读数集 lv=5 ponens（n={N_MAIN}，组 {N_GROUPS}，REQ {N_REQ} / ALT {N_ALT}）')
    print(f'# 外层 {N_OUTER} 折 / 内层 {N_INNER} 折；C∈{CGRID}；CI = atomic 聚类 bootstrap '
          f'{BOOT} 次 seed {SEED}（用原始组标签聚类）')
    print(f'# 折 seed = {seeds}；n_jobs={args.n_jobs}；纯 CPU')
    print(f'# 登记锚点（3dp 逐位）：{ANCHORS}')

    for model in args.models.split(','):
        last, _g3, meta = load_reps(model, smoke=False)
        layers = list(range(last.shape[1]))
        main = meta[(meta.agreement_lv == 5) & (meta.modus == 'ponens')]
        idx = main.index.values
        Xm, Gm = last[idx], main['atomic_idx'].values
        ym = (main['gold'] == 'c').astype(int).values
        assert len(idx) == N_MAIN, f'主集行数 {len(idx)} != {N_MAIN}'
        assert pd.Series(Gm).nunique() == N_GROUPS, '组数不符'
        assert int(ym.sum()) == N_REQ and int((1 - ym).sum()) == N_ALT, \
            f'REQ/ALT 行数 {int(ym.sum())}/{int((1 - ym).sum())} != {N_REQ}/{N_ALT}'
        print(f'\n{"=" * 78}\n{model}  (taps={last.shape[1]}, hidden={last.shape[2]})\n{"=" * 78}')
        print(f'  [行数断言通过] n={len(idx)}  组={pd.Series(Gm).nunique()}  '
              f'REQ={int(ym.sum())}  ALT={int((1 - ym).sum())}')

        base = run_split(Xm, ym, Gm, layers, Gm, args.n_jobs)
        ok = abs(base['bal'] - ANCHORS[model]) < 5e-4
        show(f'原始确定性划分（锚点复现，登记 {ANCHORS[model]:.3f}）', base)
        print(f'  [锚点] {base["bal"]:.3f} vs 登记 {ANCHORS[model]:.3f} → '
              f'{"复现 ✅" if ok else "不复现 ❌ —— 停，不跑变体"}')
        sys.stdout.flush()
        if not ok:
            continue

        for s in seeds:
            g = remap_groups(Gm, s)
            r = run_split(Xm, ym, Gm, layers, g, args.n_jobs)
            show(f'换折划分 seed={s}', r)
            print(f'  与原划分的同折一致率 = {cofold_agreement(base["fid"], r["fid"]):.3f}'
                  f'（独立划分期望 ≈0.68，1.0 = 同一套折）')
            print(f'  Δ(主读数) 对原划分 = {r["bal"] - base["bal"]:+.3f}；'
                  f'落在原划分 CI [{base["ci"][0]:.3f},{base["ci"][1]:.3f}] '
                  f'{"内" if base["ci"][0] <= r["bal"] <= base["ci"][1] else "外"}')
            sys.stdout.flush()

    print('\n本脚本只出数不判定：换折读数与原划分的关系交人裁决。')


if __name__ == '__main__':
    main()
