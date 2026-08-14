#!/usr/bin/env python
"""Arm B4L: relaxed-budget last-layer readout LoRA + lv=4 dose collection.

Spec = `PREREG_b4long.md` (this arm) + `PREREG_lv4_dose.md` §3 (lv=4 combination rule).

The **only design difference vs B4 (`run_b4_lora.py`) = training budget**:
    B4   MAX_EPOCHS 5   PATIENCE 1
    B4L  MAX_EPOCHS 20  PATIENCE 3        (constants local to this file; run_lora_sft's untouched)
Everything else (LoRA on lm_head r16 α32 dropout 0.05, lr 1e-4, micro 4×accum 4, five folds,
supervision target, dev early-stop metric = relation bal-acc / 32 tokens, three-step inference
chain, fallback, on-disk fields) is copied verbatim from B4; B4's instrument assertions
G-B4-1 / G-B4-2 / G-B4-2b / G-B4-3 are imported from `run_b4_lora` and executed as-is.
This file modifies no existing file.

Because the budget is relaxed, B4L is no longer iso with B3 (PREREG_b4long §1 registers it
as a budget variant, not to be used for the B3 comparison).

============================================================================
Three things added on top of B4
============================================================================
1. **Convergence logging** (PREREG_b4long §2): each fold's output adds `early_stopped` /
   `n_epochs_run` (plus `dev_curve`) and prints one `[conv]` line. The convergence judgment
   itself ("early stop triggered, OR total dev gain over the last 3 epochs <0.005";
   ≥2 folds running the full 20 epochs without converging → log only, don't archive)
   belongs to the summarizer; this script does not judge.
   Warning: early-stop semantics **copy the B3/B4 loop verbatim**: break only when
   `bad > PATIENCE`, i.e. PATIENCE=3 means tolerating 3 non-improving epochs and stopping
   on the 4th. This inequality was not rewritten ("copy verbatim" takes precedence).
2. **lv=4 dose collection**: after each fold trains and loads its best adapter, run step1
   relation prediction with that fold's fine-tuned model on **all 958 lv=4 rows**, writing
   `B4L_lv4_step1_fold{k}.jsonl`. No criteria; feeds the lv4_dose summary.
3. **`--lv4_exec` mode**: run once separately after all five folds are done — read the five
   step1 outputs, decide the relation via the combination rule of PREREG_lv4_dose §3
   (revision 1), then inject and execute with the **original model without adapter**,
   writing `B4L_lv4_exec.jsonl`.

============================================================================
lv=4 combination rule (PREREG_lv4_dose §3 revision 1, frozen 2026-08-05)
============================================================================
The original text said "no fold has ever seen an lv=4 row (assert atomics disjoint)".
**This atomic-level assertion does not hold on the data**: lv=4 has 170 atomic_idx values,
lv=5 has 152, intersection 118; 746 of the 958 lv=4 rows have their atomic appearing in
lv=5, and per fold 416–486 rows have their atomic inside that fold's **training** group.
The reason is that atomic_idx is a very coarse "scenario family" label (atomic 0 alone is
a group of 44 rows spanning a dozen-plus distinct dataset_id values and different problem
texts), not a scenario identifier. After reporting, §3 was rewritten into a rule immune to
family leakage:

- **Main rule = OOF assignment** (`rule='assigned'`, 746 rows): rows whose atomic is inside
  the lv=5 fold structure are predicted by their **assigned fold**
  (`run_probe_decode.fold_of_atomic`, i.e. the fold where that family is test in lv=5) —
  that fold's training set does not contain the family, isomorphic to arm A's assignment
  discipline for lv=4. If the assigned fold fails to parse → fall back to DP, fallback=1
  (however the other four folds vote, they never decide the relation; they are only
  recorded in `votes` as a side report).
- **Backstop = five-fold majority vote** (`rule='majority'`, 212 rows): rows whose atomic
  is not in lv=5 are clean for all five folds. Valid votes = successfully parsed '1'/'2';
  parse failure = abstain; majority of valid votes decides the relation;
  **valid-vote tie (including all-abstain) → fall back to DP, fallback=1**.
- Rows under both rules still record the `votes` five-fold vote string + `rule` /
  `assigned_fold` (null for majority rows).
- The all-958-row majority-vote version **does not run execution**; it is demoted to a
  CPU-only side report on the summary side (computed in the main session).

The no-leak premise is therefore split into three layers, all hard-asserted:
- [G-B4L-4] row indices disjoint / dataset_id disjoint / **step1 prompts verbatim disjoint
  (0/958, measured)**;
- [G-B4L-4b] atomic-level overlap is measured and logged only (per-fold numbers printed to
  stdout; each row records `atomic_in_lv5` / `atomic_in_fold_train`, so the summary can
  shrink to the 212-row clean subset at any time);
- [G-B4L-7] every assigned row's atomic is **not in its assigned fold's training group**
  (per-row assertion); every majority row's atomic is in **no** fold's training group.

============================================================================
Usage
============================================================================
  python scripts/run_b4long_lora.py --selfcheck                       # CPU only
  CUDA_VISIBLE_DEVICES=1 python scripts/run_b4long_lora.py --fold 0 --smoke
  CUDA_VISIBLE_DEVICES=1 python scripts/run_b4long_lora.py --fold 0
  CUDA_VISIBLE_DEVICES=1 python scripts/run_b4long_lora.py --lv4_exec --smoke
  CUDA_VISIBLE_DEVICES=1 python scripts/run_b4long_lora.py --lv4_exec
"""
import argparse
import gc
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
import run_twostep  # noqa: E402  single source of the step1 prompt (build_step1 / STEP1_SHA256)
from run_generative import build_prompt  # noqa: E402  single source of the fallback DP prompt
from run_lora_sft import (ACCUM, LR, MICRO_BS, MODEL, OUT_DIR,  # noqa: E402
                          SEED, TARGETS, breu, check_prompt_identity, collate,
                          encode, lv5_split)
from run_lora_sft import MAX_EPOCHS as B4_MAX_EPOCHS  # noqa: E402  only for printing the comparison
from run_lora_sft import PATIENCE as B4_PATIENCE  # noqa: E402   only for printing the comparison
# B3 is the skeleton source; importing it also runs the module-level G-B3-0
# (both template shas == registered values).
from run_b3_lora import (DEV_MAX_NEW, STEP1_MAX_NEW, STEP2_MAX_NEW,  # noqa: E402
                         STEP1_SHA_REG, TEMPLATE_SHA_REG,
                         check_gold_state_vs_C, check_label_rule, gen_batch,
                         gold_state, rel_bal_acc, state_of_relation,
                         step1_records, target_line, twin_step1_report)
# B4 is this arm's direct predecessor: target_modules and the three instrument
# assertions are all imported from it, not copied.
from run_b4_lora import (B4_TARGETS, EMB_PROBE_TEXT, LORA_LAYERS,  # noqa: E402
                         N_TRAINABLE_EXPECT, assert_trainable_is_lm_head,
                         emb_forward)
from run_probe_decode import (CSV, TEMPLATE_SHA256, build_replace_prompt,  # noqa: E402
                              fold_of_atomic, implied_answer, main_folds)
from src.prompts.utils import get_final_answer  # noqa: E402

# The only experimental-design difference vs B4 is the line below (PREREG_b4long §2).
B4L_MAX_EPOCHS, B4L_PATIENCE = 20, 3
LV4_LV, LV4_N = 4, 958          # row count of the full lv=4 set, asserted value
SMOKE_EPOCHS, SMOKE_LV4_N = 2, 32   # smoke runs 2 epochs (exercises the multi-epoch path) + 32 lv4 rows
N_FOLD = 5
CONV_TAIL_EPS = 0.005           # convergence threshold of PREREG_b4long §2 (script only logs, does not judge)


# ------------------------------------------------------------------ lv=4 row set and no-leak checks
def lv4_index(df):
    """Row indices of the full lv=4 set (CSV natural order). Row count hard-coded assert = 958."""
    idx = df.index[df.agreement_lv == LV4_LV].values
    assert len(idx) == LV4_N, f'lv=4 行数 {len(idx)} != {LV4_N} —— 数据换了'
    return idx


def smoke_lv4_index(df, idx):
    """The 32 smoke rows: stratified by modus, 16 each.

    The CSV has the ponens block first, so taking the first 32 rows would be all ponens;
    step2's implied answer depends on modus (under tollens ALT→b), so that path would get
    no smoke coverage. The selection depends only on the global lv4 order, independent of
    fold — the five folds' smoke step1 outputs therefore align row by row, which is what
    lets `--lv4_exec --smoke` vote.
    """
    p = [int(i) for i in idx if df.loc[i, 'modus'] == 'ponens'][:SMOKE_LV4_N // 2]
    t = [int(i) for i in idx if df.loc[i, 'modus'] == 'tollens'][:SMOKE_LV4_N // 2]
    return np.array(p + t)


def check_lv4_no_leak(df, lv4_idx, fold_train_atomics=None, verbose=True):
    """[G-B4L-4] hard-assert the real no-leak premise; [G-B4L-4b] only measure atomic-level overlap.

    `fold_train_atomics`: {fold: set(atomic_idx)}. If given, report the atomic-overlap row
    count per fold. See "the one conflict with the PREREG" in the module docstring.
    """
    lv5 = df[df.agreement_lv == 5]
    lv4 = df.loc[lv4_idx]

    # ---- hard assert 1: row indices disjoint (train/dev/test are all drawn from lv=5;
    # not a single lv=4 row ever entered)
    assert not (set(int(i) for i in lv4_idx) & set(int(i) for i in lv5.index)), \
        'G-B4L-4 失败：lv=4 与 lv=5 行号相交 —— 训练集里混进了 lv=4 行'
    # ---- hard assert 2: dataset_id disjoint (the same problem never appears under two lv values)
    ds4, ds5 = set(lv4.dataset_id), set(lv5.dataset_id)
    assert not (ds4 & ds5), \
        f'G-B4L-4 失败：lv=4/lv=5 的 dataset_id 相交 {len(ds4 & ds5)} 个'
    # ---- hard assert 3: step1 prompts (= the exact training-input bytes) verbatim disjoint
    s1_lv5 = {run_twostep.build_step1(df.loc[i]) for i in lv5.index}
    dup = [int(i) for i in lv4_idx if run_twostep.build_step1(df.loc[i]) in s1_lv5]
    assert not dup, \
        f'G-B4L-4 失败：{len(dup)} 行 lv=4 的 step1 提示逐字出现在 lv=5 池里，例如 {dup[:3]}'

    # ---- measure only: atomic-level overlap (the PREREG's atomic assertion does not hold on the data)
    a4, a5 = set(int(a) for a in lv4.atomic_idx), set(int(a) for a in lv5.atomic_idx)
    n_row_in_a5 = int(lv4.atomic_idx.isin(a5).sum())
    if verbose:
        print(f'[G-B4L-4] lv=4 {len(lv4_idx)} 行：与 lv=5 的行号不交、dataset_id 不交'
              f'（{len(ds4)}/{len(ds5)} 个，交 0）、step1 提示逐字不交（0/{len(lv4_idx)}'
              f' 行命中 lv=5 池）  ✅')
        print(f'⚠️ [G-B4L-4b] atomic 级**不满足** PREREG_lv4_dose §3 写的"atomic 不交"：'
              f'lv4 atomic {len(a4)} 个 / lv5 {len(a5)} 个，交 {len(a4 & a5)} 个；'
              f'{n_row_in_a5}/{len(lv4_idx)} 行 lv=4 的 atomic 在 lv=5 里。'
              f'（atomic_idx 是粗场景族标签，不是场景标识；只登记，判据归汇总）')
        if fold_train_atomics:
            per = {k: int(lv4.atomic_idx.isin(v).sum())
                   for k, v in sorted(fold_train_atomics.items())}
            print(f'⚠️ [G-B4L-4b] 逐折：atomic 落在该折**训练**组内的 lv=4 行数 = {per}'
                  f'（全五折都不含的行 = {len(lv4_idx) - n_row_in_a5} 行，'
                  f'即 atomic 也不交的干净子集）')
    return a5


# ------------------------------------------------------------------ majority vote (frozen in PREREG_lv4_dose §3)
def majority_vote(votes):
    """Five folds' step1 relation predictions → combined relation.

    votes: length-5 sequence, each item '1' / '2' / None (None = parse failure = abstain).
    Valid votes = '1'/'2'; majority of valid votes decides the relation; **valid-vote tie
    (including 0:0 all-abstain) → None = fall back to DP**.
    Returns (relation, votes_str, n_abstain, n1, n2); votes_str uses '.' for abstain.
    """
    vs = ''.join(v if v in ('1', '2') else '.' for v in votes)
    n1, n2, n_ab = vs.count('1'), vs.count('2'), vs.count('.')
    rel = '1' if n1 > n2 else ('2' if n2 > n1 else None)
    return rel, vs, n_ab, n1, n2


# (votes, expected relation, expected votes_str, expected n_abstain) — pin down every boundary, don't rely on argument
VOTE_CASES = [
    (('1', '1', '1', '1', '1'), '1', '11111', 0, '5:0 全票'),
    (('1', '1', '1', '1', '2'), '1', '11112', 0, '4:1'),
    (('1', '1', '1', '2', '2'), '1', '11122', 0, '3:2'),
    (('2', '2', '2', '1', '1'), '2', '22211', 0, '3:2 反向'),
    (('1', '1', '2', '2', None), None, '1122.', 1, '2:2 + 1 弃权 → 有效票平局 → 回退'),
    (('1', '1', '1', '2', None), '1', '1112.', 1, '3:1 + 1 弃权'),
    (('2', '2', '1', None, None), '2', '221..', 2, '有效票 2:1 + 2 弃权 → 多数方 2'),
    (('1', '1', None, None, None), '1', '11...', 3, '2:0 + 3 弃权'),
    (('1', '2', None, None, None), None, '12...', 3, '1:1 + 3 弃权 → 平局 → 回退'),
    (('1', None, None, None, None), '1', '1....', 4, '1:0 + 4 弃权 → 唯一有效票说了算'),
    ((None, None, None, None, None), None, '.....', 5, '全弃权 → 回退'),
    (('2', '1', '1', '2', '1'), '1', '21121', 0, '3:2 乱序'),
]


def assigned_fold_map(df):
    """atomic_idx → the fold where it is test in lv=5. Atomics outside the lv=5 fold structure are absent from the map.

    Single source = `run_probe_decode.main_folds` + `fold_of_atomic` (the same pair of
    functions used for arm A's assignment, and the same folds V-B0 asserts inside
    `lv5_split`).
    """
    folds, groups, _ = main_folds(df)
    return fold_of_atomic(folds, groups)


def combine_relation(atomic, votes, f_of):
    """Combination rule of PREREG_lv4_dose §3 (revision 1).

    Atomic inside the lv=5 fold structure → **take only the assigned fold's vote**
    (rule='assigned'; that fold never trained on this family, and the other four votes do
    not participate in deciding the relation); otherwise five-fold majority vote
    (rule='majority'). On both paths, parse failure / tie → relation=None → fall back to DP.

    Returns (relation, rule, assigned_fold, votes_str, n_abstain, n1, n2).
    """
    rel_m, vs, n_ab, n1, n2 = majority_vote(votes)
    a = int(atomic)
    if a in f_of:
        k = int(f_of[a])
        v = votes[k]
        return (v if v in ('1', '2') else None), 'assigned', k, vs, n_ab, n1, n2
    return rel_m, 'majority', None, vs, n_ab, n1, n2


# (atomic, votes, expected relation, expected rule, expected assigned_fold, note)
# Fake f_of for tests: atomic 7→fold 2, atomic 8→fold 0; atomic 99 is outside the fold structure.
UT_FOF = {7: 2, 8: 0}
ASSIGN_CASES = [
    (7, ('1', '1', '2', '1', '1'), '2', 'assigned', 2,
     'atomic 在折结构内 → 只认指派折(2)那一票 2；其余四票 4:1 投 1 也不算'),
    (7, ('2', '2', '1', '2', '2'), '1', 'assigned', 2,
     '同上反向：指派折(2)投 1，多数 2 被忽略'),
    (7, ('1', '1', None, '1', '1'), None, 'assigned', 2,
     '指派折(2)弃权 → 回退 DP，哪怕其余四票全票 1'),
    (8, (None, '2', '2', '2', '2'), None, 'assigned', 0,
     '指派折(0)弃权 → 回退（边界：指派折在首位）'),
    (8, ('2', '1', '1', '1', '1'), '2', 'assigned', 0,
     '指派折(0)投 2 → 2（边界：指派折在首位）'),
    (99, ('1', '1', '1', '2', '2'), '1', 'majority', None,
     'atomic 不在折结构内 → 五折多数票 3:2 → 1'),
    (99, ('1', '1', '2', '2', None), None, 'majority', None,
     '不在折结构内 + 有效票 2:2 平局 → 回退'),
    (99, (None, None, None, None, None), None, 'majority', None,
     '不在折结构内 + 全弃权 → 回退'),
]


def test_majority_vote(verbose=True):
    """Unit tests for majority vote + assignment rule (CPU only, no GPU dependency)."""
    for votes, want_rel, want_str, want_ab, why in VOTE_CASES:
        rel, vs, n_ab, n1, n2 = majority_vote(votes)
        assert (rel, vs, n_ab) == (want_rel, want_str, want_ab), (
            f'多数票单测失败：{votes} → ({rel!r}, {vs!r}, {n_ab}) '
            f'≠ 期望 ({want_rel!r}, {want_str!r}, {want_ab})')
        if verbose:
            print(f'  [ut-票] {vs}  有效 {n1}:{n2}  弃权 {n_ab}  → '
                  f'relation={rel!r}{" (fallback)" if rel is None else ""}   {why}')
    for atomic, votes, want_rel, want_rule, want_k, why in ASSIGN_CASES:
        rel, rule, k, vs, n_ab, _, _ = combine_relation(atomic, votes, UT_FOF)
        assert (rel, rule, k) == (want_rel, want_rule, want_k), (
            f'指派规则单测失败：atomic={atomic} {votes} → ({rel!r}, {rule}, {k}) '
            f'≠ 期望 ({want_rel!r}, {want_rule}, {want_k})')
        if verbose:
            print(f'  [ut-指派] atomic={atomic:<3} {vs}  rule={rule:<8} '
                  f'指派折={k}  → relation={rel!r}'
                  f'{" (fallback)" if rel is None else ""}   {why}')
    if verbose:
        print(f'[G-B4L-5] 单元测试通过  ✅  多数票 {len(VOTE_CASES)}/{len(VOTE_CASES)}'
              f' + 指派规则 {len(ASSIGN_CASES)}/{len(ASSIGN_CASES)}')


# ------------------------------------------------------------------ lv4_exec: read the five folds' step1
STEP1_META_KEYS = ('dataset_id', 'atomic_idx', 'modus', 'agreement_lv',
                   'ground_truth', 'gold_state', 'model', 'step1_sha')


def load_lv4_step1(out_dir, suffix):
    """Read the five B4L_lv4_step1_fold{k}.jsonl files; per idx, assert all five folds present and metadata consistent."""
    per = []
    for k in range(N_FOLD):
        p = os.path.join(out_dir, f'B4L_lv4_step1_fold{k}{suffix}.jsonl')
        assert os.path.exists(p), f'缺 {p} —— 五折没跑齐，lv4_exec 不能投票'
        d = {}
        for line in open(p):
            r = json.loads(line)
            assert r['fold'] == k, f'{p} 里出现 fold={r["fold"]} 的行'
            assert r['idx'] not in d, f'{p} 里 idx {r["idx"]} 重复'
            d[r['idx']] = r
        per.append(d)
    idxs = sorted(per[0])
    for k in range(1, N_FOLD):
        assert sorted(per[k]) == idxs, \
            f'折 {k} 的 idx 集合与折 0 不同（{len(per[k])} vs {len(idxs)}）'
    for i in idxs:
        for k in range(1, N_FOLD):
            for key in STEP1_META_KEYS:
                assert per[k][i][key] == per[0][i][key], \
                    f'idx {i} 的 {key} 在折 {k} 与折 0 不一致：' \
                    f'{per[k][i][key]!r} vs {per[0][i][key]!r}'
    print(f'[G-B4L-6] 五折 step1 产物齐：各 {len(idxs)} 行，逐 idx 五折都有预测；'
          f'行元数据 {list(STEP1_META_KEYS)} 逐折一致  ✅')
    return idxs, per


def lv4_exec(args):
    """--lv4_exec: OOF assignment / majority vote decides the relation → inject and execute with the original model (PREREG_lv4_dose §3 revision 1)."""
    import pandas as pd
    df = pd.read_csv(CSV)
    suffix = '_smoke' if args.smoke else ''
    print('# 臂 B4L lv=4 剂量执行（PREREG_lv4_dose.md §3 修订 1 冻结的组合规则）')
    print('# 组合规则：atomic 在 lv=5 折结构内 → 取指派折那一票（rule=assigned）；'
          '否则五折多数票（rule=majority）；两条路解析失败/平局 → 回退 DP')
    print(f'# model={args.model_name}（**不带 adapter 的原模型**，与 A/C 同执行器同模板）')
    print(f'# STEP1_SHA256    = {run_twostep.STEP1_SHA256}')
    print(f'# TEMPLATE_SHA256 = {TEMPLATE_SHA256}')
    assert run_twostep.STEP1_SHA256 == STEP1_SHA_REG and TEMPLATE_SHA256 == TEMPLATE_SHA_REG
    print('[G-B3-0] step1 模板 sha 与注入模板 sha 均 == 登记值  ✅')
    test_majority_vote(verbose=False)
    print(f'[G-B4L-5] 单元测试通过  ✅  多数票 {len(VOTE_CASES)} 例 + 指派规则 '
          f'{len(ASSIGN_CASES)} 例')

    lv4_idx = lv4_index(df)
    a5 = check_lv4_no_leak(df, lv4_idx)
    f_of = assigned_fold_map(df)
    fold_train = {k: lv5_split(k)[5]['train'] for k in range(N_FOLD)}
    idxs, per = load_lv4_step1(args.out_dir, suffix)
    if not args.smoke:
        assert len(idxs) == LV4_N, f'lv4_exec 行数 {len(idxs)} != {LV4_N}'
        assert set(idxs) == set(int(i) for i in lv4_idx), 'step1 产物的 idx 集合 != lv=4 全集'

    combos, n_ab_tot = [], 0
    for i in idxs:
        votes = [per[k][i]['relation'] for k in range(N_FOLD)]
        a = int(df.loc[i, 'atomic_idx'])
        rel, rule, k_as, vs, n_ab, n1, n2 = combine_relation(a, votes, f_of)
        # [G-B4L-7] out-of-fold discipline: the training group of the model(s) deciding
        # the relation must not contain this row's atomic family
        if rule == 'assigned':
            assert a not in fold_train[k_as], \
                f'G-B4L-7 失败：idx {i} 的 atomic {a} 在其指派折 {k_as} 的训练组里'
        else:
            bad = [kk for kk in range(N_FOLD) if a in fold_train[kk]]
            assert not bad, \
                f'G-B4L-7 失败：idx {i} 走多数票，但 atomic {a} 在折 {bad} 的训练组里'
        n_ab_tot += n_ab
        combos.append(dict(idx=i, rel=rel, rule=rule, assigned_fold=k_as,
                           votes=vs, n_abstain=n_ab, n1=n1, n2=n2))
    n_as = sum(1 for c in combos if c['rule'] == 'assigned')
    n_mj = len(combos) - n_as
    print(f'[G-B4L-7] 折外纪律逐行断言通过：assigned {n_as} 行的 atomic 均不在其指派折的'
          f'训练组内；majority {n_mj} 行的 atomic 不在任何折的训练组内  ✅')
    if not args.smoke:
        assert (n_as, n_mj) == (746, 212), \
            f'规则分布 {n_as}/{n_mj} != 登记的 746/212 —— 折结构或数据变了'
    print(f'[lv4] 规则分布：assigned {n_as} / majority {n_mj}'
          f'{"（登记值 746/212）" if not args.smoke else "（smoke，不断言）"}')
    if n_mj == 0:
        print('[lv4] ⚠️ 注意：本次没有任何 majority 行（smoke 的 32 行恰好 atomic 全在 '
              'lv=5 折结构内），多数票路径未被这次执行覆盖 —— 它由 G-B4L-5 单测覆盖。')

    prompts = []
    for c in combos:
        row = df.loc[c['idx']]
        prompts.append(build_prompt(row['questions'], 'dp') if c['rel'] is None
                       else build_replace_prompt(row, 'and' if c['rel'] == '1' else 'or'))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16,
        device_map='cuda').eval()
    print('[step2] 已加载不带 adapter 的原模型执行注入')
    s2 = gen_batch(base, tok, prompts, STEP2_MAX_NEW, args.batch, 'B4L lv4 step2')

    recs = []
    for c, p, (g, trunc, ntok) in zip(combos, prompts, s2):
        row = df.loc[c['idx']]
        r0 = per[0][c['idx']]
        rel = c['rel']
        conn = None if rel is None else ('and' if rel == '1' else 'or')
        ext = get_final_answer(g)
        imp = None if conn is None else implied_answer(conn, row['modus'])
        recs.append({
            'arm': 'B4L-lv4', 'idx': int(c['idx']), 'dataset_id': row['dataset_id'],
            'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
            'agreement_lv': int(row['agreement_lv']), 'ground_truth': row['ground_truth'],
            'model': args.model_name, 'step1_sha': run_twostep.STEP1_SHA256,
            'template_sha': TEMPLATE_SHA256,
            'relation': rel, 'pred_state': state_of_relation(rel), 'fed_conn': conn,
            'implied': imp, 'fallback': int(rel is None), 'gold_state': r0['gold_state'],
            'rule': c['rule'], 'assigned_fold': c['assigned_fold'],
            'votes': c['votes'], 'n_abstain': c['n_abstain'],
            'n_vote_1': c['n1'], 'n_vote_2': c['n2'],
            'n_step1_truncated': sum(int(per[k][c['idx']]['step1_truncated'])
                                     for k in range(N_FOLD)),
            'atomic_in_lv5': bool(int(row['atomic_idx']) in a5),
            'fold': None,
            'step2_prompt': p, 'raw_output': g, 'extracted': ext,
            'has_final_answer': 'final answer' in g.lower(),
            'n_new_tokens': ntok, 'truncated': trunc,
            'follows': None if imp is None else (ext == imp),
            'correct': ext == row['ground_truth'],
            'smoke': bool(args.smoke),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'B4L_lv4_exec{suffix}.jsonl')
    with open(out_path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n = len(recs)
    n_fb = sum(r['fallback'] for r in recs)
    n_rows_any_ab = sum(1 for r in recs if r['n_abstain'] > 0)
    rate_ab = n_ab_tot / (n * N_FOLD)
    rate_fb = n_fb / n
    print(f'[done] {out_path}  n={n}')
    print(f'[lv4] 弃权率（全五折票口径，含不定关系的那些票）= {rate_ab:.4f}'
          f'（{n_ab_tot}/{n * N_FOLD} 票）；含 ≥1 弃权的行 = {n_rows_any_ab}/{n}')
    print(f'[lv4] 回退率（行口径，主规则）= {rate_fb:.4f}（{n_fb}/{n}；'
          f'assigned 行=指派折弃权，majority 行=平局或全弃权）')
    for rule in ('assigned', 'majority'):
        sub = [r for r in recs if r['rule'] == rule]
        if not sub:
            continue
        print(f'[lv4] rule={rule}：n={len(sub)} 回退 {sum(r["fallback"] for r in sub)}'
              f'（{sum(r["fallback"] for r in sub) / len(sub):.4f}）'
              f' 关系 bal-acc={rel_bal_acc(sub):.4f} BREU={breu(sub):.4f}')
    if rate_fb > 0.2:
        print(f'⚠️ [GATE-lv4] 回退率 {rate_fb:.4f} > 0.2 —— 按 PREREG_lv4_dose §5 判仪器，'
              f'读数不得直接解读')
    else:
        print(f'[GATE-lv4] 回退率 {rate_fb:.4f} ≤ 0.2  ✅')
    v = [r['votes'] for r in recs]
    print(f'[lv4] 五折一致票（全同且无弃权）= '
          f'{sum(1 for s in v if len(set(s)) == 1 and "." not in s)}/{n}'
          f'（附报用；assigned 行不据此定关系）')
    print(f'[lv4] 组合关系边缘分布：REQ {sum(1 for r in recs if r["pred_state"] == "REQ")}'
          f' / ALT {sum(1 for r in recs if r["pred_state"] == "ALT")} / 回退 {n_fb}')
    print(f'[read] lv=4 组合关系 bal-acc = {rel_bal_acc(recs):.4f}；'
          f'BREU = {breu(recs):.4f}  noFA={sum(1 for r in recs if not r["has_final_answer"])}'
          f'  step2 截断={sum(1 for r in recs if r["truncated"])}')
    print('# 全 958 行多数票版为纯 CPU 附报（PREREG_lv4_dose §3 修订 1），本脚本不跑它的执行')
    print('# 剂量-响应判读全部归 summarize_lv4_dose（PREREG_lv4_dose §6），本脚本只产原始 JSONL')


# ------------------------------------------------------------------ main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--model_name', default=MODEL)
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--batch', type=int, default=16, help='推理批大小')
    ap.add_argument('--smoke', action='store_true',
                    help='train 64 / dev 32 / test 16+16 / 2 epoch / lv4 step1 32 行')
    ap.add_argument('--selfcheck', action='store_true',
                    help='纯 CPU：B4 全部断言头 + lv4 无泄漏 + 多数票单测 + 渲染')
    ap.add_argument('--lv4_exec', action='store_true',
                    help='五折齐后单独跑一次：多数票定关系 → 原模型注入执行')
    args = ap.parse_args()

    if args.lv4_exec:
        return lv4_exec(args)

    df, tr, dv, te, k_dev, A = lv5_split(args.fold)
    lv4_idx = lv4_index(df)
    print('# 臂 B4L 放宽预算的末层读出 LoRA（PREREG_b4long.md）')
    print(f'# fold={args.fold}（dev=折{k_dev}）  model={args.model_name}')
    print(f'# targets={B4_TARGETS}  r16 α32 dropout0.05  layers_to_transform={LORA_LAYERS}')
    print(f'# 与 B4 的唯一差别 = 预算：epochs {B4_MAX_EPOCHS}→{B4L_MAX_EPOCHS}，'
          f'patience {B4_PATIENCE}→{B4L_PATIENCE}（其余逐字相同；'
          f'与 B3 的 target_modules 差别照旧：{TARGETS} → {B4_TARGETS}）')
    print(f'# lr={LR} epochs≤{B4L_MAX_EPOCHS} patience={B4L_PATIENCE}'
          f'（循环用 bad>PATIENCE，即容忍 {B4L_PATIENCE} 轮未提升、第 {B4L_PATIENCE + 1} 轮停）'
          f' micro_bs={MICRO_BS}×accum{ACCUM}  dev_max_new={DEV_MAX_NEW} '
          f'step1_max_new={STEP1_MAX_NEW} step2_max_new={STEP2_MAX_NEW}')
    print(f'# STEP1_SHA256    = {run_twostep.STEP1_SHA256}')
    print(f'# TEMPLATE_SHA256 = {TEMPLATE_SHA256}')
    assert run_twostep.STEP1_SHA256 == STEP1_SHA_REG and TEMPLATE_SHA256 == TEMPLATE_SHA_REG
    print('[G-B3-0] step1 模板 sha 与注入模板 sha 均 == 登记值（与臂 C/A/B3/B4 同一串字节）  ✅')
    print('[V-B0] 782 行版折结构 == 391 行版（与 A/B1/B3/B4 同折）  ✅')
    print(f'[V-B1] train/dev/test 行数 = {len(tr)}/{len(dv)}/{len(te)}；'
          f'atomic 组数 = {len(A["train"])}/{len(A["dev"])}/{len(A["test"])}；两两不交  ✅')
    print(f'[V-B2] 回退用 DP 提示识别单元格：{check_prompt_identity(df, args.model_name)}')
    print(f'[V-B3] 标签闭环：{check_label_rule(df, list(tr) + list(dv) + list(te))}')
    print(f'[V-B4] 金标状态：{check_gold_state_vs_C(df, list(tr) + list(dv) + list(te))}')
    n_req = sum(gold_state(df.loc[i]) == 'REQ' for i in tr)
    print(f'[data] 训练集标签分布：REQ {n_req} / ALT {len(tr) - n_req}')
    print(f'[obs] {twin_step1_report(df, list(tr) + list(dv) + list(te))}')

    fold_atomics = None
    if args.selfcheck:
        fold_atomics = {k: lv5_split(k)[5]['train'] for k in range(N_FOLD)}
    a5 = check_lv4_no_leak(df, lv4_idx, fold_atomics)
    tr_atomics = A['train']
    if args.selfcheck:
        # Distribution of lv4_exec's combination rule on the real data + out-of-fold
        # discipline (checked once on pure CPU before running the execution)
        f_of = assigned_fold_map(df)
        rules = [combine_relation(int(df.loc[i, 'atomic_idx']), [None] * N_FOLD, f_of)[1:3]
                 for i in lv4_idx]
        n_as = sum(1 for r, _ in rules if r == 'assigned')
        for (rule, k_as), i in zip(rules, lv4_idx):
            a = int(df.loc[i, 'atomic_idx'])
            if rule == 'assigned':
                assert a not in fold_atomics[k_as], \
                    f'G-B4L-7 失败：idx {i} 的 atomic {a} 在其指派折 {k_as} 的训练组里'
            else:
                bad = [kk for kk in range(N_FOLD) if a in fold_atomics[kk]]
                assert not bad, f'G-B4L-7 失败：idx {i} 走多数票但 atomic {a} 在折 {bad} 训练组里'
        print(f'[G-B4L-7] lv4 组合规则分布（PREREG_lv4_dose §3 修订 1）：'
              f'assigned {n_as} / majority {len(rules) - n_as}；'
              f'逐行折外纪律断言全过（assigned 行的 atomic 不在其指派折训练组内，'
              f'majority 行的 atomic 不在任何折训练组内）  ✅')
        from collections import Counter
        print(f'[G-B4L-7] assigned 行按指派折的分布 = '
              f'{dict(sorted(Counter(k for r, k in rules if r == "assigned").items()))}')
    print(f'[data] lv=4 收集集 = 全部 {len(lv4_idx)} 行（'
          f'ponens {int((df.loc[lv4_idx, "modus"] == "ponens").sum())} / '
          f'tollens {int((df.loc[lv4_idx, "modus"] == "tollens").sum())}；'
          f'金标 REQ {int((df.loc[lv4_idx, "ground_truth"] == "c").sum())} / '
          f'ALT {int((df.loc[lv4_idx, "ground_truth"] != "c").sum())}）')

    # Existing B4 outputs serve as anchor: put the same-fold B4 readout alongside, so
    # "more budget should not be worse" can be eyeballed on the spot (V-B4L is judged by the summarizer)
    b4_path = os.path.join(args.out_dir, f'B4_fold{args.fold}.jsonl')
    if os.path.exists(b4_path):
        b4 = [json.loads(l) for l in open(b4_path)]
        print(f'[obs V-B4L] 同折 B4（epochs≤{B4_MAX_EPOCHS}/patience{B4_PATIENCE}）读数：'
              f'n={len(b4)} 关系 bal-acc={rel_bal_acc(b4):.4f} BREU={breu(b4):.4f} '
              f'best_epoch={b4[0]["best_epoch"]} dev={b4[0]["dev_balacc"]:.4f}'
              f'（pooled 版 V-B4L 锚点在汇总里判）')

    if args.selfcheck:
        print(f'\n{"=" * 78}\n组合规则单元测试（PREREG_lv4_dose §3 修订 1；'
              f"有效票='1'/'2'，'.'=弃权）\n{'=' * 78}")
        test_majority_vote()

        print(f'\n{"=" * 78}\nSFT 样本渲染（2 条；assistant 段即监督目标，'
              f'与 B3/B4 逐字相同）\n{"=" * 78}')
        for i in list(tr)[:2]:
            row = df.loc[i]
            print(f'{"#" * 26} idx={i} dataset_id={row["dataset_id"]} '
                  f'modus={row["modus"]} 金标={row["ground_truth"]} '
                  f'金标关系={gold_state(row)}')
            print('--- user（= 臂 C step1 提示逐字）---')
            print(run_twostep.build_step1(row))
            print('--- assistant（只在这一段算 loss）---')
            t = target_line(row)
            assert t in ('RELATION: 1', 'RELATION: 2'), f'监督目标写成了 {t!r}'
            print(t)
            print()

        # Inference chain: take 1 ponens + 1 tollens from lv=5 test (same as B4); plus
        # 1 ponens + 1 tollens from lv=4 — under tollens ALT implies b (not a); if that
        # cell can't be rendered, nothing was checked.
        picks = []
        for m in ('ponens', 'tollens'):
            cand = [i for i in te if df.loc[i, 'modus'] == m]
            if cand:
                picks.append(('lv5-test', cand[0]))
        for m in ('ponens', 'tollens'):
            cand = [i for i in lv4_idx if df.loc[i, 'modus'] == m]
            if len(cand):
                picks.append(('lv4-剂量', int(cand[0])))
        print(f'{"=" * 78}\n推理链渲染（lv=5 折内 test 各 1 + lv=4 各 1；'
              f'step1 全文 + rel=1/2 两种 step2 + 回退 DP）\n{"=" * 78}')
        for tag, i in picks:
            row = df.loc[i]
            print(f'{"#" * 26} [{tag}] idx={i} dataset_id={row["dataset_id"]} '
                  f'modus={row["modus"]} lv={row["agreement_lv"]} '
                  f'金标={row["ground_truth"]} 金标关系={gold_state(row)} '
                  f'atomic={row["atomic_idx"]} atomic在lv5内='
                  f'{int(row["atomic_idx"]) in a5} atomic在本折训练组内='
                  f'{int(row["atomic_idx"]) in tr_atomics}')
            print('----- STEP 1（lv5=微调模型 / lv4=五折微调模型各跑一遍，贪心 512 token）-----')
            print(run_twostep.build_step1(row))
            for rel in ('1', '2'):
                conn = 'and' if rel == '1' else 'or'
                print(f'----- STEP 2（原模型跑；解析出 RELATION: {rel} → '
                      f'{state_of_relation(rel)} → conn={conn} → '
                      f'蕴含答案 {implied_answer(conn, row["modus"])}）-----')
                print(build_replace_prompt(row, conn))
            print('----- 回退（解析失败 / lv4 多数票平局 → 原始 DP 提示，fallback=1）-----')
            print(build_prompt(row['questions'], 'dp'))
            print()

        print(f'[B4L] 与 B4 的唯一设计差别 = 预算：MAX_EPOCHS {B4_MAX_EPOCHS}→'
              f'{B4L_MAX_EPOCHS}，PATIENCE {B4_PATIENCE}→{B4L_PATIENCE}'
              f'（LoRA 位置/数据/折/监督/其余超参/早停指标/推理链/回退/字段全部相同）')
        print(f'[B4L] 预期可训练参数 = 16×(2560+151936) = {N_TRAINABLE_EXPECT:,}；'
              f'实际值由 GPU 路径的 [G-B4-1] 登记')
        print(f'[B4L] 产物：B4L_fold{{0..4}}.jsonl（lv5，字段同 B4 + early_stopped/'
              f'n_epochs_run/dev_curve）+ B4L_lv4_step1_fold{{0..4}}.jsonl（各 {LV4_N} 行）'
              f' + B4L_lv4_exec.jsonl（{LV4_N} 行，--lv4_exec 单独跑）')
        print('[selfcheck] G-B3-0 / V-B0…V-B4 / G-B4L-4 / G-B4L-5 / G-B4L-7 全部通过，'
              '未加载语言模型，无读数。')
        print('[selfcheck] B4 继承的三条断言（G-B4-1 可训练参数名 / G-B4-2、2b embedding '
              '前向不变 / G-B4-3 epoch0 loss 下降）与 G-B4L-6（五折 step1 产物齐）'
              '需要真模型或真产物，纯 CPU 自检里跑不到，只在 GPU 路径上执行。')
        print('SELFCHECK PASS')
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda')

    # The premise of [G-B4-2] must itself be measured: do lm_head and embed_tokens really share the same weight block?
    tied = (model.get_output_embeddings().weight.data_ptr()
            == model.get_input_embeddings().weight.data_ptr())
    probe_ids = tok(EMB_PROBE_TEXT, return_tensors='pt').input_ids.to(model.device)
    emb_before = emb_forward(model, probe_ids)

    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=B4_TARGETS, layers_to_transform=LORA_LAYERS))

    names, n_tr = assert_trainable_is_lm_head(model)
    print(f'[LoRA] 模型层数={model.config.num_hidden_layers}  可训练参数={n_tr:,}')
    print(f'[G-B4-1] 可训练参数 {len(names)} 个，逐一含 "lm_head"  ✅  {names}')
    print(f'[G-B4-1] 实际可训练参数量 = {n_tr:,}（PREREG 预期 {N_TRAINABLE_EXPECT:,}'
          f'{"，一致" if n_tr == N_TRAINABLE_EXPECT else "，不一致 ⚠️"}）')
    emb_after = emb_forward(model, probe_ids)
    assert torch.equal(emb_before, emb_after), \
        'G-B4-2 失败：挂 LoRA 后 embedding 前向变了 —— 增量漏进了绑定的输入侧'
    print(f'[G-B4-2] 权重绑定 = {tied}；挂 LoRA 后 embedding 前向逐位不变'
          f'（探针 {probe_ids.shape[1]} token，bf16 全等）  ✅')

    epochs = SMOKE_EPOCHS if args.smoke else B4L_MAX_EPOCHS
    tr_idx, dv_idx, te_idx = list(tr), list(dv), list(te)
    lv4_run = list(lv4_idx)
    if args.smoke:
        # Test slice stratified by modus, 16 each: the CSV has the ponens block first, so
        # taking the first 32 rows would be all ponens, while step2's implied answer
        # depends on modus (under tollens ALT→b) — that path would get no smoke coverage.
        # Only affects the --smoke branch; the full-run path is untouched. Copied from B3/B4.
        tr_idx, dv_idx = tr_idx[:64], dv_idx[:32]
        te_idx = ([i for i in te_idx if df.loc[i, 'modus'] == 'ponens'][:16]
                  + [i for i in te_idx if df.loc[i, 'modus'] == 'tollens'][:16])
        lv4_run = [int(i) for i in smoke_lv4_index(df, lv4_idx)]
    print(f'[budget] epochs≤{epochs}  train={len(tr_idx)} dev={len(dv_idx)} '
          f'test={len(te_idx)} lv4={len(lv4_run)}')

    samples = [encode(tok, run_twostep.build_step1(df.loc[i]), target_line(df.loc[i]))
               for i in tr_idx]
    print(f'[data] 训练样本 {len(samples)} 条；assistant 段 token 数 '
          f'{[sum(1 for x in s[1] if x != -100) for s in samples[:3]]}…')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    first_loss = None  # [G-B4-3] the first micro-batch loss of epoch 0
    dev_curve, loss_curve = [], []
    early_stopped, n_epochs_run = False, 0
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(SEED + ep).permutation(len(samples))
        tot, nb = 0.0, 0
        opt.zero_grad()
        for s in range(0, len(order), MICRO_BS):
            ids, lab, att = collate([samples[j] for j in order[s:s + MICRO_BS]],
                                    tok.pad_token_id, model.device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / ACCUM).backward()
            if ep == 0 and first_loss is None:
                first_loss = float(loss)
            tot += float(loss); nb += 1
            if (s // MICRO_BS + 1) % ACCUM == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        mean_loss = tot / max(nb, 1)
        if ep == 0:
            if mean_loss < first_loss:
                print(f'[G-B4-3] epoch0 平均 train loss {mean_loss:.4f} < 首个 micro-batch '
                      f'loss {first_loss:.4f}  ✅（lm_head LoRA 在学）')
            else:
                print(f'⚠️ [G-B4-3] 仪器警告：epoch0 平均 train loss {mean_loss:.4f} '
                      f'未低于首个 micro-batch loss {first_loss:.4f} —— '
                      f'lm_head LoRA 可能没在学。不中断，读数按 PREREG 先查仪器。')
        dev_recs = step1_records(model, tok, df, dv_idx, DEV_MAX_NEW, args.batch,
                                 f'dev ep{ep}')
        d = rel_bal_acc(dev_recs)
        n_fail = sum(1 for r in dev_recs if r['relation'] is None)
        dev_curve.append(float(d)); loss_curve.append(float(mean_loss))
        n_epochs_run = ep + 1
        print(f'[epoch {ep}] train loss = {mean_loss:.4f}  '
              f'dev 关系 bal-acc = {d:.4f}  (dev n={len(dev_recs)}, '
              f'解析失败={n_fail}, 预测 REQ={sum(1 for r in dev_recs if r["pred_state"] == "REQ")}'
              f'/ALT={sum(1 for r in dev_recs if r["pred_state"] == "ALT")})', flush=True)
        if d > best:
            best, best_ep, bad = d, ep, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items() if 'lora' in k}
        else:
            bad += 1
            if bad > B4L_PATIENCE:
                print(f'[early stop] dev 关系 bal-acc 连续 {bad} 轮未提升，停在 epoch {ep}')
                early_stopped = True
                break
    assert best_state is not None, \
        '没有任何一轮拿到可比较的 dev 指标（全 NaN？dev 单类？）——仪器坏了，不出读数'
    model.load_state_dict(best_state, strict=False)
    print(f'[best] dev 关系 bal-acc = {best:.4f}（epoch {best_ep}）')

    # ---- convergence logging (PREREG_b4long §2): log only, no judgment ----
    tail = (dev_curve[-1] - dev_curve[-4]) if len(dev_curve) >= 4 else None
    print(f'[conv] dev 曲线 = {[round(x, 4) for x in dev_curve]}')
    print(f'[conv] train loss 曲线 = {[round(x, 4) for x in loss_curve]}')
    print(f'[conv] n_epochs_run={n_epochs_run}/{epochs}  early_stopped={early_stopped}  '
          f'best_epoch={best_ep}  best_dev={best:.4f}  '
          f'最后3轮 dev 提升总和(=d[-1]-d[-4]) = '
          f'{"n/a(<4 轮)" if tail is None else f"{tail:+.4f}"}'
          f'（PREREG_b4long §2："早停触发 或 该和 <{CONV_TAIL_EPS}" 即已收敛；判定归汇总）')

    # [G-B4-2b] measure again after training + loading best_state: if anything leaks into
    # the tied input side, it leaks during training.
    assert torch.equal(emb_before, emb_forward(model, probe_ids)), \
        'G-B4-2b 失败：训练后 embedding 前向变了 —— 梯度漏到了绑定的输入侧'
    print('[G-B4-2b] 训练并载入 best_state 后，embedding 前向仍逐位不变  ✅')

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = '_smoke' if args.smoke else ''

    # ---- inference chain step 1: fine-tuned model predicts the relation (lv=5 out-of-fold test) ----
    s1 = step1_records(model, tok, df, te_idx, STEP1_MAX_NEW, args.batch, 'B4L step1')

    # ---- lv=4 dose collection: same best adapter, run step1 on all lv=4 rows; write to disk before moving on ----
    s1_lv4 = step1_records(model, tok, df, lv4_run, STEP1_MAX_NEW, args.batch,
                           'B4L lv4 step1')
    lv4_path = os.path.join(args.out_dir,
                            f'B4L_lv4_step1_fold{args.fold}{suffix}.jsonl')
    with open(lv4_path, 'w') as f:
        for r in s1_lv4:
            row = df.loc[r['idx']]
            f.write(json.dumps({
                'idx': int(r['idx']), 'dataset_id': row['dataset_id'],
                'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
                'agreement_lv': int(row['agreement_lv']),
                'ground_truth': row['ground_truth'], 'gold_state': r['gold_state'],
                'relation': r['relation'], 'pred_state': r['pred_state'],
                'step1_raw': r['step1_raw'], 'step1_truncated': r['step1_truncated'],
                'step1_n_new_tokens': r['step1_n_new_tokens'],
                'fold': args.fold, 'model': args.model_name,
                'step1_sha': run_twostep.STEP1_SHA256,
                'atomic_in_lv5': bool(int(row['atomic_idx']) in a5),
                'atomic_in_fold_train': bool(int(row['atomic_idx']) in tr_atomics),
                'smoke': bool(args.smoke),
            }, ensure_ascii=False) + '\n')
    n4 = len(s1_lv4)
    n4_fail = sum(1 for r in s1_lv4 if r['relation'] is None)
    print(f'[done] {lv4_path}  n={n4}')
    print(f'[lv4] step1 解析成功率 = {(n4 - n4_fail) / n4:.4f}；截断率 = '
          f'{sum(1 for r in s1_lv4 if r["step1_truncated"]) / n4:.4f}；'
          f'预测 REQ {sum(1 for r in s1_lv4 if r["pred_state"] == "REQ")}'
          f'/ALT {sum(1 for r in s1_lv4 if r["pred_state"] == "ALT")}'
          f'/失败 {n4_fail}')
    print(f'[lv4] 本折单独口径的 lv=4 关系 bal-acc = {rel_bal_acc(s1_lv4):.4f}'
          f'（登记用；剂量读数走五折多数票 --lv4_exec）')

    # ---- step 2: free the fine-tuned model. step2 must be executed by the
    # **un-fine-tuned original model** (variable isolation on the classifier) ----
    del model, opt, best_state, samples, emb_before, emb_after, probe_ids
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16,
        device_map='cuda').eval()
    print('[step2] 已释放微调模型，另行加载不带 adapter 的原模型执行注入')

    # ---- step 3: original model executes step2 (parse failure → fall back to DP, fallback=1) ----
    prompts = []
    for r in s1:
        row = df.loc[r['idx']]
        prompts.append(build_prompt(row['questions'], 'dp') if r['relation'] is None
                       else build_replace_prompt(row, 'and' if r['relation'] == '1' else 'or'))
    s2 = gen_batch(base, tok, prompts, STEP2_MAX_NEW, args.batch, 'B4L step2')

    recs = []
    for r, p, (g, trunc, ntok) in zip(s1, prompts, s2):
        row = df.loc[r['idx']]
        rel = r['relation']
        conn = None if rel is None else ('and' if rel == '1' else 'or')
        ext = get_final_answer(g)
        imp = None if conn is None else implied_answer(conn, row['modus'])
        recs.append({
            'arm': 'B4L', 'idx': int(r['idx']), 'dataset_id': row['dataset_id'],
            'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
            'agreement_lv': int(row['agreement_lv']), 'ground_truth': row['ground_truth'],
            'model': args.model_name, 'step1_sha': run_twostep.STEP1_SHA256,
            'template_sha': TEMPLATE_SHA256,
            'relation': rel, 'pred_state': r['pred_state'], 'fed_conn': conn,
            'implied': imp, 'fallback': int(rel is None), 'gold_state': r['gold_state'],
            'step1_raw': r['step1_raw'], 'step1_truncated': r['step1_truncated'],
            'step1_n_new_tokens': r['step1_n_new_tokens'],
            'step2_prompt': p, 'raw_output': g, 'extracted': ext,
            'has_final_answer': 'final answer' in g.lower(),
            'n_new_tokens': ntok, 'truncated': trunc,
            'follows': None if imp is None else (ext == imp),
            'correct': ext == row['ground_truth'],
            'fold': args.fold, 'dev_balacc': best, 'best_epoch': best_ep,
            'early_stopped': early_stopped, 'n_epochs_run': n_epochs_run,
            'max_epochs': epochs, 'patience': B4L_PATIENCE,
            'dev_curve': [round(x, 6) for x in dev_curve],
            'smoke': bool(args.smoke),
        })

    out_path = os.path.join(args.out_dir, f'B4L_fold{args.fold}{suffix}.jsonl')
    with open(out_path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n = len(recs)
    n_fb = sum(r['fallback'] for r in recs)
    n_tr1 = sum(1 for r in recs if r['step1_truncated'])
    n_req = sum(1 for r in recs if r['pred_state'] == 'REQ')
    n_alt = sum(1 for r in recs if r['pred_state'] == 'ALT')
    print(f'[done] {out_path}  n={n}')
    print(f'[G-B3-2] step1 解析成功率 = {(n - n_fb) / n:.4f}（闸门 ≥0.8）；'
          f'截断率 = {n_tr1 / n:.4f}（闸门 ≤0.05）')
    print(f'[G-B3-3] step1 预测边缘分布：REQ {n_req} / ALT {n_alt} / 解析失败 {n_fb}'
          f'（某类 ≥0.9 即塌缩形态，解读里须写明）')
    print(f'[read] 折内 test 关系 bal-acc（全 {n} 行口径）= {rel_bal_acc(recs):.4f}；'
          f'ponens 行口径 = '
          f'{rel_bal_acc([r for r in recs if r["modus"] == "ponens"]):.4f}')
    print(f'[read] 折内 test BREU = {breu(recs):.4f}  '
          f'noFA={sum(1 for r in recs if not r["has_final_answer"])}  '
          f'step2 截断={sum(1 for r in recs if r["truncated"])}')
    print('# 主读数与配对比较在汇总脚本里出，本脚本只产原始 JSONL（PREREG_b4long §4）')


if __name__ == '__main__':
    main()
