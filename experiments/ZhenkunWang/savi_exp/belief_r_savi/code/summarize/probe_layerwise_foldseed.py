#!/usr/bin/env python
"""Layerwise probe curve × three fold splits. Pure CPU, zero GPU, no old files modified.

Plotting need (figure swap for the 811 report): the old figure fig_probe_layerwise.png
hangs a clustered-bootstrap confidence band on every layer — too crowded. Switch reading:
per layer, take the mean curve over **three fold splits**, with the band = min–max range
of the three splits — same layer, same protocol, only the "group → fold" assignment
changes, showing how much the curve jitters.

Three splits (the only axis of change; the rest of the protocol is untouched):
  orig  = the original deterministic split (GroupKFold(shuffle=False), groups=atomic_idx)
  seed1 = probe_foldseed_variant.remap_groups(G, 20260811)
  seed2 = probe_foldseed_variant.remap_groups(G, 20260812)

Layerwise readout = nested_cv(X, y, groups_used, [L], CGRID, N_OUTER, N_INNER): a
single-layer grid, i.e. "layer fixed at L, C chosen by inner 4-fold CV", bal-acc computed
on the outer out-of-fold predictions. This is mechanism-isomorphic to
summarize_hidden_probe.layer_curve() (the latter's per-fold-per-layer C is just argmax
over C of the inner matrix at that layer), so under the orig split it must reproduce the
layerwise point estimates of outputs/layerwise_ci.json bit-for-bit — that is this script's
instrument gate (|Δ| ≤ 5e-4).

All mechanisms are imported, not rewritten: fold structure / inner C selection /
out-of-fold prediction come from summarize_hidden_probe; fold remapping from
probe_foldseed_variant.

Checkpoint resume: chunked by (model × split); each finished chunk atomically rewrites the
json; on rerun, completed chunks are skipped.

Usage:
  python scripts/probe_layerwise_foldseed.py            # full: three models × three splits
  python scripts/probe_layerwise_foldseed.py --models Qwen/Qwen3-4B
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_hidden_probe import (  # noqa: E402  single source, don't write another copy
    CGRID, N_INNER, N_OUTER, bal_acc, load_reps, nested_cv)
from probe_foldseed_variant import FOLD_SEEDS, remap_groups  # noqa: E402

OUT = os.path.join(HERE, '..', 'outputs')
OUT_JSON = os.path.join(OUT, 'ci', 'probe_layerwise_foldseed.json')
OUT_TXT = os.path.join(OUT, 'ci', 'probe_layerwise_foldseed.txt')
ANCHOR_JSON = os.path.join(OUT, 'layerwise_ci.json')

MODELS = ('Qwen/Qwen3-4B', 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct')
SPLITS = ('orig', 'seed1', 'seed2')          # seed1/seed2 ↔ FOLD_SEEDS
TOL = 5e-4                                    # anchor: stop if any layer's |Δ| exceeds this, no silent reconciliation
N_MAIN, N_GROUPS, N_REQ, N_ALT = 391, 152, 257, 134


def load_ckpt(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(script='scripts/probe_layerwise_foldseed.py',
                readout='nested_cv 单层网格（层固定，C 由内层 CV 选）的外层折外 bal-acc',
                main_set='lv=5 ponens，组=atomic_idx',
                n=N_MAIN, n_groups=N_GROUPS, pos=N_REQ, neg=N_ALT,
                cgrid=list(CGRID), n_outer=N_OUTER, n_inner=N_INNER,
                fold_seeds=dict(zip(SPLITS[1:], FOLD_SEEDS)),
                anchor_source='outputs/layerwise_ci.json（逐层 bal_acc 点估计）',
                anchor_tol=TOL, models={})


def save_ckpt(path, obj):
    """Atomic rewrite: write .tmp then rename; dying midway leaves no half-written json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_main(model):
    """Primary readout set = lv=5 ponens (hard assertions on row/group/class counts, same as the old script)."""
    last, _g3, meta = load_reps(model, smoke=False)
    main = meta[(meta.agreement_lv == 5) & (meta.modus == 'ponens')]
    idx = main.index.values
    X, G = last[idx], main['atomic_idx'].values
    y = (main['gold'] == 'c').astype(int).values
    assert len(idx) == N_MAIN, f'主集行数 {len(idx)} != {N_MAIN}'
    assert pd.Series(G).nunique() == N_GROUPS, '组数不符'
    assert int(y.sum()) == N_REQ and int((1 - y).sum()) == N_ALT, 'REQ/ALT 行数不符'
    return X, y, G


def curve_for_split(X, y, G, groups_used, layers, n_jobs):
    """Run single-layer-grid nested_cv layer by layer; returns the list of layerwise out-of-fold bal-acc."""
    out = []
    for L in layers:
        cv = nested_cv(X, y, groups_used, [L], CGRID, N_OUTER, N_INNER, n_jobs)
        out.append(float(bal_acc(y, cv['oof_pred'])))
        print(f'    L{L:>2}: {out[-1]:.4f}', flush=True)
    return out


def summarize(ck):
    """Human-readable summary: per model, mean-curve peak layer/value + max layerwise spread across the three splits."""
    L = []
    L.append('# 逐层探针曲线 × 三套折划分（probe_layerwise_foldseed.py）')
    L.append('# 读数 = 单层网格 nested_cv（层固定，C 由内层 4 折 CV 选）的外层折外平衡准确率')
    L.append(f'# 主读数集 lv=5 ponens（n={N_MAIN}，组 {N_GROUPS}，REQ {N_REQ} / ALT {N_ALT}）')
    L.append(f'# 外层 {N_OUTER} 折 / 内层 {N_INNER} 折；C∈{tuple(CGRID)}；纯 CPU')
    L.append(f'# 三划分：orig = 原确定性划分；seed1/seed2 = remap_groups(seed={FOLD_SEEDS})')
    L.append(f'# 锚点：orig 逐层须复现 outputs/layerwise_ci.json 的点估计，|Δ| ≤ {TOL}')
    for m in MODELS:
        r = ck['models'].get(m)
        if not r or any(s not in r['splits'] for s in SPLITS):
            L.append(f'\n{"=" * 78}\n{m}\n{"=" * 78}\n  [未完成]')
            continue
        arr = np.array([r['splits'][s] for s in SPLITS])       # [3, n_layers]
        mean = arr.mean(axis=0)
        spread = arr.max(axis=0) - arr.min(axis=0)
        pk = int(mean.argmax())
        L.append(f'\n{"=" * 78}\n{m}  (taps={arr.shape[1]})\n{"=" * 78}')
        L.append(f'  [锚点] orig vs layerwise_ci.json 逐层最大 |Δ| = '
                 f'{r["anchor_max_abs_diff"]:.2e}  '
                 f'({"复现" if r["anchor_ok"] else "不复现 —— 停"})')
        L.append(f'  均值曲线峰层 = L{pk}  峰值 = {mean[pk]:.4f}'
                 f'   (三划分该层 = ' + ', '.join(f'{v:.4f}' for v in arr[:, pk]) + ')')
        for s, c in zip(SPLITS, arr):
            L.append(f'    {s:<5} 峰层 = L{int(c.argmax()):<2}  峰值 = {c.max():.4f}'
                     f'   末层 L{arr.shape[1] - 1} = {c[-1]:.4f}')
        L.append(f'  三划分逐层最大离差 = {spread.max():.4f}（在 L{int(spread.argmax())}）；'
                 f'中位 {np.median(spread):.4f}；均值 {spread.mean():.4f}')
        L.append('  逐层（层: orig / seed1 / seed2 / 均值 / 离差）：')
        for i in range(arr.shape[1]):
            L.append(f'    L{i:>2}: ' + ' / '.join(f'{v:.4f}' for v in arr[:, i])
                     + f' / {mean[i]:.4f} / {spread[i]:.4f}')
    L.append('\n本脚本只出数不判定：曲线怎么引用交人裁决。')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--n_jobs', type=int, default=16)
    ap.add_argument('--json', default=OUT_JSON)
    ap.add_argument('--txt', default=OUT_TXT)
    args = ap.parse_args()

    with open(ANCHOR_JSON) as f:
        anchor = json.load(f)['models']

    ck = load_ckpt(args.json)
    t_all = time.time()
    for model in args.models.split(','):
        X, y, G = load_main(model)
        layers = list(range(X.shape[1]))
        ref = [e['bal_acc'] for e in anchor[model]['layers']]
        assert len(ref) == len(layers), f'锚点层数 {len(ref)} != 本次 {len(layers)}'
        rec = ck['models'].setdefault(model, dict(taps=len(layers), splits={}, secs={}))
        print(f'\n{"=" * 78}\n{model}  (taps={len(layers)}, hidden={X.shape[2]})\n'
              f'{"=" * 78}', flush=True)

        for si, split in enumerate(SPLITS):
            gu = G if split == 'orig' else remap_groups(G, FOLD_SEEDS[si - 1])
            if len(rec['splits'].get(split, [])) == len(layers):
                print(f'  [{split}] 已在断点文件里，跳过', flush=True)
            else:
                print(f'  [{split}] 开跑（{len(layers)} 层，n_jobs={args.n_jobs}）', flush=True)
                t0 = time.time()
                rec['splits'][split] = curve_for_split(X, y, G, gu, layers, args.n_jobs)
                rec['secs'][split] = round(time.time() - t0, 1)
                save_ckpt(args.json, ck)                     # chunkwise write to disk
                print(f'  [{split}] 完成 {rec["secs"][split]:.0f}s → 已写 {args.json}',
                      flush=True)

            if split == 'orig':      # anchor: verify first; if it fails, stop and emit no further numbers
                d = np.abs(np.array(rec['splits']['orig']) - np.array(ref))
                rec['anchor_max_abs_diff'] = float(d.max())
                rec['anchor_ok'] = bool(d.max() <= TOL)
                rec['anchor_worst_layer'] = int(d.argmax())
                save_ckpt(args.json, ck)
                print(f'  [锚点] 逐层最大 |Δ| = {d.max():.3e}（L{int(d.argmax())}）  '
                      f'阈 {TOL} → {"复现" if rec["anchor_ok"] else "不复现"}', flush=True)
                if not rec['anchor_ok']:
                    bad = [(int(i), rec['splits']['orig'][i], ref[i])
                           for i in np.argsort(-d)[:8]]
                    print(f'  [停机] 锚点不复现，最差 8 层 (层, 本次, 登记) = {bad}',
                          flush=True)
                    raise SystemExit('锚点不复现 → 停，原样上报，不自行调和')

    with open(args.txt, 'w') as f:
        f.write(summarize(ck))
    print(f'\n总墙钟 {time.time() - t_all:.0f}s\n产物：{args.json}\n      {args.txt}')


if __name__ == '__main__':
    main()
