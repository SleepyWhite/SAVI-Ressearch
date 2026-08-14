#!/usr/bin/env python
"""Compute the "ceiling" reference line, and make it recomputable (it was previously a hard-coded constant with no producing code).

============================================================================
Why this script exists
============================================================================
Found during the 2026-08-02 close-out check: the ceiling number `0.843` exists as a
**hard-coded constant** in three places (`probe_incontext_choice.py` docstring,
`summarize_incontext_choice.py:CEILING`, and the various VERDICTs); the comments say
"actually computed", but **no code in the whole project produces it**, and it cannot be
reproduced. The actual computation does not match it (see below). Hence this script, and
the constants are changed to be computed by it.

============================================================================
What "ceiling" actually means (defined clearly here, no more vagueness)
============================================================================
The primary readout of the relation-choice task is **balanced accuracy** =
(REQ recall + ALT recall)/2, random = 0.500.
The "ceiling" answers: **how far can an ideal discriminator get on this task?** There are
two reference lines with different meanings — don't mix them:

**A. Absolute upper bound = 1.000 (on lv=5).**
   lv=5 = all 5 annotators unanimous (definition in STATUS §5). Since the gold label is
   unanimous, in principle there exists a function that recovers it perfectly. So the
   absolute bound is 1.0, not some other number.

**B. The upper bound of the strategy family "perfectly recover the design intent".**
   The dataset carried a marker when **constructing** γ3: `strong` (written as a
   stronger/more specific condition) vs `weak` (written as an alternative path of similar
   magnitude). It was an independent variable when the text was generated, so it is
   **in principle recoverable from the text**. How well this marker scores against the
   gold label is the bound for this strategy family. This line is more conservative and
   more meaningful: it shows that "some text-related attribute" can reach this level.

The main table should cite **B**, with A noted.

============================================================================
Scope (aligned with the L5 summary script)
============================================================================
- Statistical unit = scenario (dataset_id). Twins share γ1/γ3 and the relation judgment is
  verbatim identical, so counting by row uses it twice.
  (On this quantity the scenario-level and row-level values coincide — exactly 2 rows per
  scenario — but scenario-level is used anyway, for scope consistency.)
- Primary readout set = `agreement_lv == 5` exact match (excluding the 4 lv=6 data-anomaly rows).
- CI = scenario-clustered stratified bootstrap, 5000 draws, seed=0, same method as
  `summarize_generative_ci.py`.
"""
import os

import numpy as np
import pandas as pd

# Belief-R dataset CSV. The data is not distributed with this repo (see the README's data section);
# point env var BELIEF_R_CSV at a local copy; falls back to <repo>/data/queries_time_t1.csv if unset.
CSV = os.environ.get('BELIEF_R_CSV') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'queries_time_t1.csv')
ITERS, SEED = 5000, 0


def balanced_acc(is_req_pred, is_req_gold):
    """(REQ recall + ALT recall)/2. For a binary scorer the AUC is identically this."""
    p, g = np.asarray(is_req_pred, bool), np.asarray(is_req_gold, bool)
    if not g.any() or g.all():
        return float('nan')
    return (p[g].mean() + (~p[~g]).mean()) / 2


def boot_ci(pred, gold, iters=ITERS, seed=SEED):
    """Scenario resampling stratified by REQ/ALT (stratum sizes fixed by the dataset design)."""
    rng = np.random.RandomState(seed)
    gi, ai = np.where(gold)[0], np.where(~gold)[0]
    out = []
    for _ in range(iters):
        g = gi[rng.randint(0, len(gi), len(gi))]
        a = ai[rng.randint(0, len(ai), len(ai))]
        out.append((pred[g].mean() + (~pred[a]).mean()) / 2)
    return tuple(np.percentile(out, [2.5, 97.5]))


def main():
    df = pd.read_csv(CSV)
    df['intent'] = df['dataset_id'].str.split('-').str[-1]

    for tag, sub in (('全量', df), ('**lv=5（主读数集）**', df[df['agreement_lv'] == 5])):
        sub = sub.copy()
        sub['is_req'] = sub['ground_truth'] == 'c'
        s = sub.drop_duplicates('dataset_id')
        # Twins within a scenario must agree on intent / relation class, otherwise
        # dedup silently picks an arbitrary one.
        # ⚠️ The check is on is_req, not ground_truth: ALT-scenario twins have different
        # gold letters by design (ponens=a / tollens=b); checking the raw letter would
        # falsely flag 335 ALT scenarios.
        chk = sub.groupby('dataset_id').agg(ni=('intent', 'nunique'),
                                            ng=('is_req', 'nunique'))
        bad = chk[(chk.ni > 1) | (chk.ng > 1)]
        if len(bad):
            raise AssertionError(f'{tag}: {len(bad)} 个场景的孪生 intent/关系类别不一致')

        pred = (s['intent'] == 'strong').to_numpy()      # design intent as the scorer
        gold = s['is_req'].to_numpy()                    # gold c = REQ
        ba = balanced_acc(pred, gold)
        lo, hi = boot_ci(pred, gold)
        # Names read as "prediction × gold": s_req = intent strong and gold REQ, etc.
        s_req = int((pred & gold).sum())      # strong ∧ REQ  ← correct
        w_req = int((~pred & gold).sum())     # weak   ∧ REQ  ← miss
        s_alt = int((pred & ~gold).sum())     # strong ∧ ALT  ← false alarm
        w_alt = int((~pred & ~gold).sum())    # weak   ∧ ALT  ← correct
        print(f'\n=== {tag} ===  场景数 {len(s)}（REQ {gold.sum()} / ALT {(~gold).sum()}）')
        print(f'  混淆：strong∧REQ {s_req}  strong∧ALT {s_alt}  '
              f'weak∧REQ {w_req}  weak∧ALT {w_alt}')
        print(f'  REQ 召回 {s_req / (s_req + w_req):.4f}   '
              f'ALT 召回 {w_alt / (w_alt + s_alt):.4f}')
        print(f'  **平衡准确率 = {ba:.4f}**  95%CI [{lo:.4f}, {hi:.4f}]')

    print('\n' + '=' * 72)
    print('参照线（主表引用 B，并注明 A）：')
    print('  A 绝对上限（lv=5 金标全票一致，原则上可完全还原） = 1.000')
    print('  B "完美还原设计意图"这一族策略的上限            = 见上 lv=5 那行')
    print('对照：实测最好成绩 L1 0.588 / L4 0.560 / L5 0.545，随机 0.500')


if __name__ == '__main__':
    main()
