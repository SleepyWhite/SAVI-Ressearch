#!/usr/bin/env python
"""Robustness of the 4B probe SM matched readout (same protocol as G-5) under three fold splits. Pure CPU, zero GPU, no old file modified.

Motivation (2026-08-11, user instruction): in the original G-5, 4B's SM
readout 0.596 misses the 0.60 line by 0.004 and lands in the grey zone;
suspected fold-split luck. Same axis as §6.30 (`probe_foldseed_variant.py`):
the only randomizable axis in the pipeline = the group→fold assignment; swap
in the two §6.30-style seeded splits and re-read each split's out-of-fold
predictions restricted to the SM rows.
**Exploratory, not pre-registered, touches no frozen readout; the G-5
judgement stays frozen on the original split.**

Protocol identical to the original G-5 digit for digit:
bal_acc(ym[in_sm], oof_pred[in_sm]); CI = atomic clustered bootstrap
(clustered on the original group labels, same seed as the original script).
Anchors: original-split main readout 0.692, SM 0.596 [0.522,0.674]
(outputs/ci/hidden_probe.txt); stop immediately if not reproduced.

Usage: python scripts/probe_foldseed_sm.py > outputs/ci/hidden_probe_foldseed_sm.txt
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_hidden_probe import (  # noqa: E402  single source, don't write another copy
    BOOT, CGRID, N_INNER, N_OUTER, SEED, bal_acc, boot_ci, load_reps, nested_cv)
from summarize_length_matched import build_subsets, cem_match  # noqa: E402
from probe_foldseed_variant import FOLD_SEEDS, remap_groups  # noqa: E402

MODEL = 'Qwen/Qwen3-4B'
ANCHOR_MAIN = 0.692
ANCHOR_SM = (0.596, 0.522, 0.674)
N_MAIN, N_SM = 391, 178


def sm_readout(ym, oof_pred, in_sm, Gm):
    b = bal_acc(ym[in_sm], oof_pred[in_sm])
    lo, hi = boot_ci(lambda i: bal_acc(ym[in_sm][i], oof_pred[in_sm][i]),
                     Gm[in_sm], np.random.default_rng(SEED), BOOT)
    req = float((oof_pred[in_sm][ym[in_sm] == 1] == 1).mean())
    alt = float((oof_pred[in_sm][ym[in_sm] == 0] == 0).mean())
    return b, lo, hi, req, alt


def main():
    print('# 4B 探针 SM 配平读数 × 三套折划分(probe_foldseed_sm.py,2026-08-11)')
    print('# 探索性,无预注册,不动冻结读数;G-5 判定仍冻结在原划分(0.596 灰区)')
    print(f'# 口径=G-5 原样:oof 限制在 SM(n={N_SM});CI=atomic 聚类 bootstrap {BOOT} 次 seed {SEED}')
    print(f'# 折 seed = 原划分 + {list(FOLD_SEEDS)}(§6.30 同款);闸门参考线 = ≥0.60 且 CI 不含 0.5')

    f = build_subsets()
    SM = cem_match(f)
    last, _g3, meta = load_reps(MODEL, smoke=False)
    layers = list(range(last.shape[1]))
    main_df = meta[(meta.agreement_lv == 5) & (meta.modus == 'ponens')]
    idx = main_df.index.values
    Xm, Gm = last[idx], main_df['atomic_idx'].values
    ym = (main_df['gold'] == 'c').astype(int).values
    in_sm = main_df['dataset_id'].isin(SM).values
    assert len(idx) == N_MAIN and int(in_sm.sum()) == N_SM, '行数与登记不符'
    n_req_sm = int(ym[in_sm].sum())
    print(f'# SM 行内 REQ/ALT = {n_req_sm}/{int(in_sm.sum()) - n_req_sm}')

    rows = []
    for tag, groups_used in [('原划分(锚点)', Gm)] + \
            [(f'seed={s}', remap_groups(Gm, s)) for s in FOLD_SEEDS]:
        cv = nested_cv(Xm, ym, groups_used, layers, CGRID, N_OUTER, N_INNER, 16)
        b_main = bal_acc(ym, cv['oof_pred'])
        b, lo, hi, req, alt = sm_readout(ym, cv['oof_pred'], in_sm, Gm)
        gate = (b >= 0.60) and not (lo <= 0.5 <= hi)
        print(f'\n  --- {tag} ---')
        print(f'  主读数(全 391) bal = {b_main:.3f}')
        print(f'  SM 配平(n=178) bal = {b:.3f} [{lo:.3f},{hi:.3f}]  '
              f'REQ={req:.3f}/ALT={alt:.3f}  '
              f'{"过参考线(≥0.60 且 CI 不含 0.5)" if gate else "灰区/未过参考线"}')
        if tag.startswith('原划分'):
            ok = (abs(b_main - ANCHOR_MAIN) < 5e-4 and abs(b - ANCHOR_SM[0]) < 5e-4
                  and abs(lo - ANCHOR_SM[1]) < 5e-4 and abs(hi - ANCHOR_SM[2]) < 5e-4)
            print(f'  [锚点] 主 {b_main:.3f}/0.692  SM {b:.3f} [{lo:.3f},{hi:.3f}] vs '
                  f'0.596 [0.522,0.674] → {"复现 ✅" if ok else "不复现 ❌ 停"}')
            if not ok:
                sys.exit(1)
        rows.append((tag, b, lo, hi))
        sys.stdout.flush()

    bs = [r[1] for r in rows]
    print(f'\n  == 汇总:三划分 SM bal = [{", ".join(f"{b:.3f}" for b in bs)}]  '
          f'极差 = {max(bs) - min(bs):.3f} ==')


if __name__ == '__main__':
    main()
