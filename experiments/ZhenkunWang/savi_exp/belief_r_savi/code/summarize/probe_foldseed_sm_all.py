#!/usr/bin/env python
"""Probe SM matched readout (same protocol as G-5) for three models × three fold splits. Pure CPU, zero GPU, no old file modified.

`probe_foldseed_sm.py` yesterday ran only 4B (SM on the original split 0.596,
grey zone; suspected fold-split luck). This script pushes the identical
protocol to three models: 'Qwen/Qwen3-4B', 'Qwen/Qwen2.5-7B-Instruct',
'meta-llama/Llama-3.1-8B-Instruct'. The protocol is untouched — the readout
function `sm_readout`, the fold remapping `remap_groups`, and
`nested_cv`/`bal_acc`/`boot_ci` are all imported from the original scripts,
not re-implemented. Three splits = original split + the §6.30-style
FOLD_SEEDS (20260811, 20260812).

**Exploratory, not pre-registered, touches no frozen readout; the G-5
judgement stays frozen on the original split.**

Anchors (3dp digit-for-digit, stop on failure):
  main readout, original split  4B 0.692 / 7B 0.716 / Llama 0.734  (probe_foldseed_variant.ANCHORS)
  SM original split     4B 0.596 [0.522,0.674]  (outputs/ci/hidden_probe.txt [G-5])
                7B 0.635 [0.561,0.712]  (same file)
                Llama 0.646 [0.566,0.719]  (outputs/ci/hidden_probe_llama.txt [G-5])
  4B's two remapped splits must match outputs/ci/hidden_probe_foldseed_sm.txt
  digit for digit (deterministic pipeline; a mismatch = a bug).

Usage: python scripts/probe_foldseed_sm_all.py > outputs/ci/hidden_probe_foldseed_sm_all3.txt
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_hidden_probe import (  # noqa: E402  single source, don't write another copy
    BOOT, CGRID, N_INNER, N_OUTER, SEED, bal_acc, load_reps, nested_cv)
from summarize_length_matched import build_subsets, cem_match  # noqa: E402
from probe_foldseed_variant import ANCHORS as MAIN_ANCHORS, FOLD_SEEDS, remap_groups  # noqa: E402
from probe_foldseed_sm import N_MAIN, N_SM, sm_readout  # noqa: E402

MODELS = ('Qwen/Qwen3-4B', 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct')
# SM original-split anchors = each model's [G-5] line (bal, lo, hi)
SM_ANCHORS = {'Qwen/Qwen3-4B': (0.596, 0.522, 0.674),
              'Qwen/Qwen2.5-7B-Instruct': (0.635, 0.561, 0.712),
              'meta-llama/Llama-3.1-8B-Instruct': (0.646, 0.566, 0.719)}
# 4B remapped-split anchors = hidden_probe_foldseed_sm.txt: seed -> (main, SM bal, lo, hi, REQ, ALT)
VAR_ANCHORS_4B = {20260811: (0.744, 0.663, 0.584, 0.743, 0.708, 0.618),
                  20260812: (0.682, 0.612, 0.527, 0.694, 0.685, 0.539)}
N_JOBS = 8  # shared CPU; parallelism does not affect readouts (joblib collects in submission order, logreg deterministic)


def main():
    print('# 探针 SM 配平读数 × 三套折划分 × 三模型(probe_foldseed_sm_all.py,2026-08-11)')
    print('# 探索性,无预注册,不动冻结读数;G-5 判定仍冻结在原划分')
    print(f'# 口径=G-5 原样:oof 限制在 SM(n={N_SM});CI=atomic 聚类 bootstrap {BOOT} 次 seed {SEED}')
    print(f'# 折 seed = 原划分 + {list(FOLD_SEEDS)}(§6.30 同款);闸门参考线 = ≥0.60 且 CI 不含 0.5')
    print(f'# 协议 import 复用 probe_foldseed_sm.sm_readout / probe_foldseed_variant.remap_groups')

    f = build_subsets()
    SM = cem_match(f)
    summary = {}

    for model in MODELS:
        last, _g3, meta = load_reps(model, smoke=False)
        layers = list(range(last.shape[1]))
        main_df = meta[(meta.agreement_lv == 5) & (meta.modus == 'ponens')]
        idx = main_df.index.values
        Xm, Gm = last[idx], main_df['atomic_idx'].values
        ym = (main_df['gold'] == 'c').astype(int).values
        in_sm = main_df['dataset_id'].isin(SM).values
        assert len(idx) == N_MAIN and int(in_sm.sum()) == N_SM, '行数与登记不符'
        n_req_sm = int(ym[in_sm].sum())
        print(f'\n{"=" * 78}\n{model}  (taps={last.shape[1]}, hidden={last.shape[2]})\n{"=" * 78}')
        print(f'  n={len(idx)}  组={pd.Series(Gm).nunique()}  '
              f'SM 行内 REQ/ALT = {n_req_sm}/{int(in_sm.sum()) - n_req_sm}')
        sys.stdout.flush()

        rows = []
        for tag, seed, groups_used in [('原划分(锚点)', None, Gm)] + \
                [(f'seed={s}', s, remap_groups(Gm, s)) for s in FOLD_SEEDS]:
            cv = nested_cv(Xm, ym, groups_used, layers, CGRID, N_OUTER, N_INNER, N_JOBS)
            b_main = bal_acc(ym, cv['oof_pred'])
            b, lo, hi, req, alt = sm_readout(ym, cv['oof_pred'], in_sm, Gm)
            gate = (b >= 0.60) and not (lo <= 0.5 <= hi)
            print(f'\n  --- {tag} ---')
            print(f'  主读数(全 {N_MAIN}) bal = {b_main:.3f}')
            print(f'  SM 配平(n={N_SM}) bal = {b:.3f} [{lo:.3f},{hi:.3f}]  '
                  f'REQ={req:.3f}/ALT={alt:.3f}  '
                  f'{"过参考线(≥0.60 且 CI 不含 0.5)" if gate else "灰区/未过参考线"}')

            if seed is None:
                a_main, a_sm = MAIN_ANCHORS[model], SM_ANCHORS[model]
                ok = (abs(b_main - a_main) < 5e-4 and abs(b - a_sm[0]) < 5e-4
                      and abs(lo - a_sm[1]) < 5e-4 and abs(hi - a_sm[2]) < 5e-4)
                print(f'  [锚点] 主 {b_main:.3f}/{a_main:.3f}  SM {b:.3f} [{lo:.3f},{hi:.3f}] vs '
                      f'{a_sm[0]:.3f} [{a_sm[1]:.3f},{a_sm[2]:.3f}] → '
                      f'{"复现 ✅" if ok else "不复现 ❌ 停"}')
                if not ok:
                    sys.exit(1)
            elif model == 'Qwen/Qwen3-4B':
                a = VAR_ANCHORS_4B[seed]
                got = (b_main, b, lo, hi, req, alt)
                ok = all(abs(x - y) < 5e-4 for x, y in zip(got, a))
                print(f'  [锚点·4B 换折 vs hidden_probe_foldseed_sm.txt] '
                      f'主 {b_main:.3f}/{a[0]:.3f}  SM {b:.3f} [{lo:.3f},{hi:.3f}] vs '
                      f'{a[1]:.3f} [{a[2]:.3f},{a[3]:.3f}]  '
                      f'REQ {req:.3f}/{a[4]:.3f} ALT {alt:.3f}/{a[5]:.3f} → '
                      f'{"复现 ✅" if ok else "不复现 ❌ 停"}')
                if not ok:
                    sys.exit(1)

            rows.append((tag, b_main, b, lo, hi, req, alt))
            sys.stdout.flush()

        bs = [r[2] for r in rows]
        summary[model] = bs
        print(f'\n  == {model} 三划分 SM bal = [{", ".join(f"{b:.3f}" for b in bs)}]  '
              f'均值 = {float(np.mean(bs)):.3f}  极差 = {max(bs) - min(bs):.3f} ==')
        sys.stdout.flush()

    print(f'\n{"=" * 78}\n汇总:SM 配平 bal-acc(n={N_SM})× 三套折划分\n{"=" * 78}')
    print(f'  {"model":34s} {"原划分":>8s} {"s=20260811":>11s} {"s=20260812":>11s} '
          f'{"均值":>7s} {"极差":>7s}')
    for m, bs in summary.items():
        print(f'  {m:34s} {bs[0]:8.3f} {bs[1]:11.3f} {bs[2]:11.3f} '
              f'{float(np.mean(bs)):7.3f} {max(bs) - min(bs):7.3f}')
    print('\n本脚本只出数不判定;换折读数与原划分的关系交人裁决。')


if __name__ == '__main__':
    main()
