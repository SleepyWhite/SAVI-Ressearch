#!/usr/bin/env python
"""Arm B3L: deep LoRA with a relaxed budget (PREREG_b3long.md, Qwen3-4B only).

The question it asks: B3 (all-linear r16, relation supervision) started from an
"all predictions REQ" collapse in all five folds and one fold never climbed out
— is that collapse / level gap an **optimization difficulty** or a **budget
shortfall**? B4→B4L already falsified the "under-training" explanation at the
last layer; B3 has no corresponding budget group, and B3L adds only this one
direct control.

The **only design difference vs B3 (`run_b3_lora.py`) = training budget**
(mirroring the B4→B4L change):
    B3   MAX_EPOCHS 5   PATIENCE 1
    B3L  MAX_EPOCHS 20  PATIENCE 3        (constants defined in this file; run_lora_sft's are untouched)
Everything else (all-linear r16 α32 dropout 0.05, lr 1e-4, micro 4×accum 4,
5 folds, training samples, supervision target, dev early-stopping metric =
relation bal-acc / 32 tokens, three-step inference chain, fallback, assertions,
fields) is verbatim identical to B3. This file is a copy of `run_b3_lora.py`;
all changes fall inside the "change list vs B3" below; `--selfcheck` prints the
full diff of the two files and asserts line by line that every change falls
within that list.

Because the budget is relaxed, B3L is no longer iso with A/B4 (registered in
PREREG_b3long §4 as a budget variant).

============================================================================
Change list vs B3 (this is the table the `--selfcheck` diff assertion uses)
============================================================================
1. **Budget constants**: `MAX_EPOCHS/PATIENCE` are no longer taken from
   `run_lora_sft` (the renamed imports are only used to print the comparison);
   this file hardcodes `B3L_MAX_EPOCHS=20 / B3L_PATIENCE=3`; smoke goes from
   1 epoch to 2 epochs (`SMOKE_EPOCHS`, to exercise the multi-epoch path,
   copied from B4L's smoke change).
   ⚠️ Early-stopping semantics are **copied verbatim from B3's loop**: break
   only when `bad > PATIENCE`, i.e. PATIENCE=3 tolerates 3 epochs without
   improvement and stops on the 4th. That inequality was not rewritten
   ("verbatim copy" takes precedence).
2. **Artifact/arm names**: `B3_fold{k}.jsonl` → `B3L_fold{k}.jsonl`, `arm`
   field `B3` → `B3L`, `B3` in progress bars and prints → `B3L`.
3. **Convergence and collapse-shape registration** (extra reported fields
   required by PREREG_b3long §1/§2; the same batch B4L added relative to B4):
   new on-disk fields `early_stopped` / `n_epochs_run` / `max_epochs` /
   `patience` / `dev_curve`, plus the B3L-specific `dev_req_frac_curve`
   (fraction of REQ predictions at each dev point; collapse shape (i) of §1
   reads its first element); one `[conv]` line is printed. **The judgement
   itself belongs to the summary script** ("early stop triggered, or the sum
   of dev improvement over the last 3 epochs <0.005"; ≥2 folds running the
   full 20 epochs without converging → register only, no tier assignment);
   this script only registers, it does not judge.
4. **PREREG references**: `PREREG §4` in printed text → `PREREG_b3long §2`.
5. Comments and docstrings.
The unit of classification is the **logical statement** (tokenize merges
continuation lines), not the physical line — otherwise the second line of a
multi-line f-string would be judged an "unregistered change".

============================================================================
Protocol (frozen before running, PREREG_b3.md §2–§3, adopted verbatim by B3L)
============================================================================
- Training sample: user = arm C's step1 prompt **verbatim**
  (`run_twostep.build_step1`, single source, module-level assertion
  sha == STEP1_SHA256); assistant = `RELATION: 1` (gold REQ, i.e.
  ground_truth == 'c') / `RELATION: 2` (gold ALT). Loss is computed only on
  the assistant segment.
- Data and folds: all 782 lv=5 rows (`run_lora_sft.lv5_split`, the same 5
  folds as A/B1, so the V-B0/V-B1 assertions come for free). The two twin rows
  share the label and are **not deduplicated** (fixed in PREREG §2; row count
  / fold structure isomorphic to B1 position by position).
- Hyperparameters copied from B1: all-linear r16 α32 dropout 0.05, lr 1e-4,
  AdamW constant lr, micro 4 × accum 4, **epochs ≤20 patience 3 (the only
  design difference of this arm)**, seed 20260803.
- Inference chain (PREREG §3, three steps within each fold): the fine-tuned
  model runs step1 (greedy 512 tokens, parsed by
  `prompts/contract.parse_relation`) → release the fine-tuned model →
  **separately load the original model without the adapter** to run step2
  (`build_replace_prompt`, greedy 1024 tokens, extracted by
  `get_final_answer`). Parse failure → fall back to the original DP prompt and
  record `fallback=1` (same convention as arm C). Swapping in the original
  model as executor is a design requirement, not a shortcut: it isolates the
  variable to the classifier.

============================================================================
Implementation-level registration (registered in PREREG §2 + degrees of freedom this script does not hardcode)
============================================================================
- dev early-stopping metric = **relation bal-acc** on dev rows (mean of the
  per-class accuracies for REQ/ALT; parse failures count as wrong), dev
  generation budget 32 tokens. The supervision target changed, so the
  early-stopping metric changes with it, registered in PREREG §2; dev only
  selects the epoch and enters no readout.
- Gradient accumulation, per-epoch shuffle with seed 20260803+epoch: verbatim
  identical to B1.
- raise when best_state is empty (no epoch produced a comparable dev metric)
  — B1 would silently use the last epoch there; on B3 that is "the instrument
  broke but looks like it ran", so better to blow up.

Usage
  python scripts/run_b3long_lora.py --selfcheck                  # pure CPU (incl. diff assertion)
  CUDA_VISIBLE_DEVICES=6 python scripts/run_b3long_lora.py --fold 0 --smoke
  CUDA_VISIBLE_DEVICES=6 python scripts/run_b3long_lora.py --fold 0
"""
import argparse
import difflib
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
from run_lora_sft import (ACCUM, LR, MICRO_BS, MODEL,  # noqa: E402
                          OUT_DIR, SEED, TARGETS, breu,
                          check_prompt_identity, collate, encode, lv5_split)
from run_lora_sft import MAX_EPOCHS as B3_MAX_EPOCHS  # noqa: E402  only for printing the comparison
from run_lora_sft import PATIENCE as B3_PATIENCE  # noqa: E402      only for printing the comparison
from run_probe_decode import (TEMPLATE_SHA256, build_replace_prompt,  # noqa: E402
                              implied_answer)
from prompts.contract import parse_relation  # noqa: E402  single source of the RELATION regex
from src.prompts.utils import get_final_answer  # noqa: E402

# G-B3-0 input identity (PREREG §5): the shas of both templates are registered here;
# the check is an assertion, not eyeballing. Registered-value provenance: STEP1 =
# PREREG_b3.md §2; TEMPLATE = the `template_sha` written to disk by arms A/C
# (outputs/probe_decode/C_twostep.jsonl). If either template moves by one character, this blows up first.
STEP1_SHA_REG = '51e915d1f862944b3e2dc14b9d8720c11f312a3b5d4c358bb8949f7fd6b85e5c'
TEMPLATE_SHA_REG = '5a104f449cb08aba418c5d9d69fd974d27491f846baa5f3282497025366a9376'
assert run_twostep.STEP1_SHA256 == STEP1_SHA_REG, \
    f'G-B3-0 失败：step1 模板 sha {run_twostep.STEP1_SHA256} != 登记 {STEP1_SHA_REG}'
assert TEMPLATE_SHA256 == TEMPLATE_SHA_REG, \
    f'G-B3-0 失败：注入模板 sha {TEMPLATE_SHA256} != 登记 {TEMPLATE_SHA_REG}'

DEV_MAX_NEW, STEP1_MAX_NEW, STEP2_MAX_NEW = 32, 512, 1024
LORA_LAYERS = None  # all-linear, all layers = B1's variant B1; B3 has only this variant

# The only experimental design difference vs B3 is the following line (PREREG_b3long §1, mirroring B4→B4L).
B3L_MAX_EPOCHS, B3L_PATIENCE = 20, 3
SMOKE_EPOCHS = 2                # smoke runs 2 epochs (exercises the multi-epoch path), copied from B4L
CONV_TAIL_EPS = 0.005           # convergence threshold of PREREG_b3long §1 (script only registers, does not judge)
COLLAPSE_REQ_FRAC = 0.95        # threshold for collapse shape (i) of §1 (script only registers, does not judge)
SRC_B3 = os.path.join(HERE, 'run_b3_lora.py')   # baseline file for the diff assertion (read-only)


# <<<B3L-DIFFCHECK  (this whole block is a "selfcheck machinery" change; the diff assertion admits it by region)
# ------------------------------------------------------------------ diff assertion (PREREG_b3long §1)
# PREREG says "the diff must be empty apart from constants and artifact names". A literal
# "empty" is impossible — §2/§3 also require writing early_stopped / n_epochs_run / dev
# curves and collapse shape to disk, exactly what B4L added relative to B4.
# So "empty" is implemented as a **whitelist**: every change must fall into some category
# of the change list in the module docstring, otherwise the assertion fails.
# Classification relies on literal tokens and regions, not on human eyes.
CAT_TOKENS = [
    ('预算常量', ('MAX_EPOCHS', 'PATIENCE', 'SMOKE_EPOCHS', 'CONV_TAIL_EPS',
                  'COLLAPSE_REQ_FRAC', 'epochs≤', '1 epoch',
                  'from run_lora_sft import')),
    ('产物/臂名', ('B3L', "'B3'", 'B3_fold', 'B3 step1', 'B3 step2', '臂 B3')),
    ('收敛/塌缩登记', ('dev_curve', 'loss_curve', 'dev_req_frac', 'early_stopped',
                       'n_epochs_run', "'max_epochs'", "'patience'", '[conv]')),
    ('自检机具', ('difflib', 'SRC_B3', 'diff_report', 'CAT_TOKENS', 'DIFFCHECK',
                  '--selfcheck', '[selfcheck]', 'diff 断言')),
    ('PREREG 引用', ('PREREG_b3long', 'PREREG §4')),
]
NOISE_CATS = ('docstring', '注释', '空行')


def _logical_map(path):
    """Physical line number (0-based) → full text of the **logical statement** it belongs to.

    Line-level classification would judge the continuation line of a multi-line statement
    an "unregistered change" (the second line of an f-string does not contain `epochs≤`).
    So the unit of classification is the logical statement: continuation lines merged via
    tokenize's NEWLINE.
    """
    import io
    import tokenize
    src = open(path).read()
    lines = src.splitlines()
    out = {i: l for i, l in enumerate(lines)}      # default (blank / pure comment lines): itself
    start = None
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.NL,
                      tokenize.COMMENT, tokenize.ENDMARKER):
            continue
        if start is None:
            start = t.start[0] - 1
        if t.type == tokenize.NEWLINE:
            end = t.end[0] - 1
            text = '\n'.join(lines[start:end + 1])
            for i in range(start, end + 1):
                out[i] = text
            start = None
    return out


def _docstring_end(lines):
    """End line number of the module docstring (0-based, inclusive). Line 1 is the shebang; the docstring opens on line 2."""
    assert lines[1].startswith('"""'), '模块 docstring 不在第 2 行 —— 分类假设破了'
    for i in range(2, len(lines)):
        if lines[i].rstrip().endswith('"""'):
            return i
    raise AssertionError('模块 docstring 没有闭合')


def _region(lines, tag):
    """Line-number interval (half-open) between `# <<<tag` / `# >>>tag`. Empty interval if the marker is absent."""
    lo = hi = None
    for i, l in enumerate(lines):
        if l.startswith(f'# <<<{tag}'):
            lo = i
        elif l.startswith(f'# >>>{tag}'):
            hi = i + 1
    return (lo, hi) if lo is not None and hi is not None else (0, 0)


def _classify(line, stmt, i, doc_end, region):
    """line = physical line (for docstring/comment/blank checks), stmt = its logical statement (for token checks)."""
    s = line.strip()
    if i <= doc_end:
        return 'docstring'
    if region[0] <= i < region[1]:
        return '自检机具'
    if not s:
        return '空行'
    if s.startswith('#'):
        return '注释'
    for cat, toks in CAT_TOKENS:
        if any(t in stmt for t in toks):
            return cat
    return None


def diff_report(path_a=None, path_b=None, verbose=True):
    """Print the full diff run_b3_lora.py → run_b3long_lora.py and assert line by line that every change is whitelisted."""
    path_a = path_a or SRC_B3
    path_b = path_b or os.path.abspath(__file__)
    a = open(path_a).read().splitlines()
    b = open(path_b).read().splitlines()
    da, db = _docstring_end(a), _docstring_end(b)
    ra, rb = _region(a, 'B3L-DIFFCHECK'), _region(b, 'B3L-DIFFCHECK')
    la, lb = _logical_map(path_a), _logical_map(path_b)
    if verbose:
        print(f'\n{"=" * 78}\ndiff（基线 {os.path.basename(path_a)} → 本文件 '
              f'{os.path.basename(path_b)}；PREREG_b3long §1 要求打印全文）\n{"=" * 78}')
        for l in difflib.unified_diff(a, b, os.path.basename(path_a),
                                      os.path.basename(path_b), lineterm='', n=1):
            print(l)
    tally, bad = {}, []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == 'equal':
            continue
        for i in range(i1, i2):
            c = _classify(a[i], la[i], i, da, ra)
            tally[f'-{c}'] = tally.get(f'-{c}', 0) + 1
            if c is None:
                bad.append(f'  - {path_a}:{i + 1}: {a[i]!r}')
        for j in range(j1, j2):
            c = _classify(b[j], lb[j], j, db, rb)
            tally[f'+{c}'] = tally.get(f'+{c}', 0) + 1
            if c is None:
                bad.append(f'  + {path_b}:{j + 1}: {b[j]!r}')
    if verbose:
        print(f'\n{"=" * 78}\ndiff 分类（白名单 = 模块 docstring 的改动清单）\n{"=" * 78}')
        for k in sorted(tally):
            print(f'  {k:<22} {tally[k]:>4} 行')
    assert not bad, ('DIFF 断言失败：以下改动不在白名单内（改动清单没登记它们）：\n'
                     + '\n'.join(bad))
    n_code = sum(v for k, v in tally.items() if k[1:] not in NOISE_CATS)
    if verbose:
        print(f'  [DIFF] 全部改动落在白名单内（非注释/非 docstring 的改动 {n_code} 行）  ✅')
    return tally
# >>>B3L-DIFFCHECK


# ------------------------------------------------------------------ labels (= arm C's gold_state rule)
def gold_state(row):
    """Gold relation. Verbatim the same expression as the `gold_state` written to disk by `run_twostep.main`."""
    return 'REQ' if row['ground_truth'] == 'c' else 'ALT'


def target_line(row):
    """Assistant segment (the only line loss is computed on). REQ→1 / ALT→2, same mechanical mapping as A/C."""
    return 'RELATION: 1' if gold_state(row) == 'REQ' else 'RELATION: 2'


def state_of_relation(rel):
    """Parsed relation → state. Verbatim the same expression as arm C's `pred_state`; None on parse failure."""
    return None if rel is None else ('REQ' if rel == '1' else 'ALT')


def rel_bal_acc(records):
    """Relation bal-acc = (accuracy on REQ rows + accuracy on ALT rows)/2; parse failures count as wrong (pred_state=None)."""
    req = [r['pred_state'] == 'REQ' for r in records if r['gold_state'] == 'REQ']
    alt = [r['pred_state'] == 'ALT' for r in records if r['gold_state'] == 'ALT']
    if not req or not alt:
        return float('nan')
    return (float(np.mean(req)) + float(np.mean(alt))) / 2


# ------------------------------------------------------------------ frozen checks
def check_label_rule(df, idx):
    """V-B3 label round-trip: target line → parse_relation → arm C's state mapping → must return to gold_state.

    Merely asserting "the label rule and the readout rule are inverses" is meaningless
    (an identity); what is asserted here is that **the literal training-target string**,
    passed through the regex and mapping actually used at inference time, comes back to
    the gold label — a swapped 1/2, a target line in a shape beyond what the regex
    tolerates (it does tolerate `RELATION:1`), or a reversed mapping direction all blow
    up in this cell first.
    """
    for i in idx:
        row = df.loc[i]
        t = target_line(row)
        rel = parse_relation(t)
        assert rel is not None, f'idx {i} 的训练目标 {t!r} 过不了 parse_relation'
        assert state_of_relation(rel) == gold_state(row), \
            f'V-B3 失败：idx {i} 目标行 {t!r} 解析回 {state_of_relation(rel)} != 金标 {gold_state(row)}'
    return f'{len(idx)} 行目标行→正则→映射闭环回到金标  ✅'


def check_gold_state_vs_C(df, idx):
    """V-B4 identity cell: this script's gold_state equals arm C's on-disk `gold_state` row by row.

    The two rules are two copies of the same expression; comparing them by eye does not
    count — use C's artifact as the external reference.
    """
    p = os.path.join(OUT_DIR, 'C_twostep.jsonl')
    if not os.path.exists(p):
        return f'跳过（无 {os.path.basename(p)}）'
    want = {int(i) for i in idx}
    n = 0
    for line in open(p):
        r = json.loads(line)
        if r['idx'] not in want:
            continue
        assert r['gold_state'] == gold_state(df.loc[r['idx']]), \
            f'V-B4 失败：idx {r["idx"]} 的 gold_state 与臂 C 不同'
        n += 1
    return f'{n} 行与 C_twostep.jsonl 的 gold_state 逐行相同  ✅'


def twin_step1_report(df, idx):
    """Are the two twin rows' step1 prompts actually identical — count it, don't assume PREREG's wording.

    PREREG §2/§4 say "the twin rows' step1 prompts are verbatim identical". The step1
    problem text takes the first three lines of questions, including γ2, and γ2 is exactly
    where ponens/tollens differ — so that sentence needs to be measured, not quoted.
    This function changes no behavior (PREREG fixes "no dedup", which is followed); it
    only puts the measured numbers on stdout.
    """
    sub = df.loc[list(idx)]
    same = diff = 0
    for _, g in sub.groupby('dataset_id'):
        if len(g) < 2:
            continue
        if len({run_twostep.build_step1(r) for _, r in g.iterrows()}) == 1:
            same += 1
        else:
            diff += 1
    lab = sum(len({gold_state(r) for _, r in g.iterrows()}) > 1
              for _, g in sub.groupby('dataset_id'))
    return f'孪生场景 {same + diff}：step1 提示逐字相同 {same} / 不同 {diff}；标签不一致 {lab}'


# ------------------------------------------------------------------ generation
def gen_batch(model, tok, prompts, max_new, batch, desc):
    """Greedy batched generation → [(text, truncated, n_new_tokens)]. Same protocol as `run_twostep.main.gen`."""
    import torch
    from tqdm import tqdm
    outs = []
    model.eval()
    for st in tqdm(range(0, len(prompts), batch), desc=desc, leave=False):
        texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
                 for p in prompts[st:st + batch]]
        enc = tok(texts, return_tensors='pt', padding=True).to(model.device)
        in_len = enc['input_ids'].shape[1]
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for j, g in enumerate(tok.batch_decode(o[:, in_len:], skip_special_tokens=True)):
            ids = o[j, in_len:].tolist()
            outs.append((g, bool(len(ids) >= max_new and tok.eos_token_id not in ids),
                         (ids.index(tok.eos_token_id) + 1
                          if tok.eos_token_id in ids else len(ids))))
    return outs


def step1_records(model, tok, df, idx, max_new, batch, desc):
    """Run step1 on the idx rows → per-row relation / pred_state / truncation / token count."""
    outs = gen_batch(model, tok, [run_twostep.build_step1(df.loc[i]) for i in idx],
                     max_new, batch, desc)
    recs = []
    for i, (g, trunc, ntok) in zip(idx, outs):
        rel = parse_relation(g)
        recs.append({'idx': int(i), 'relation': rel, 'pred_state': state_of_relation(rel),
                     'gold_state': gold_state(df.loc[i]), 'step1_raw': g,
                     'step1_truncated': trunc, 'step1_n_new_tokens': ntok})
    return recs


# ------------------------------------------------------------------ main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--model_name', default=MODEL)
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--batch', type=int, default=16, help='推理批大小')
    ap.add_argument('--smoke', action='store_true',
                    help='train 64 行 / dev,test 各 32 行 / 2 epoch（SMOKE_EPOCHS）')
    ap.add_argument('--selfcheck', action='store_true',
                    help='纯 CPU：diff 断言 + 折与三集不交断言 + sha + 标签闭环 + 渲染')
    args = ap.parse_args()

    df, tr, dv, te, k_dev, A = lv5_split(args.fold)
    print('# 臂 B3L 放宽预算的深层 LoRA（PREREG_b3long.md；骨架 = PREREG_b3.md §2–§3）')
    print(f'# fold={args.fold}（dev=折{k_dev}）  model={args.model_name}')
    print(f'# targets={TARGETS}  r16 α32 dropout0.05  layers_to_transform={LORA_LAYERS}')
    print(f'# 与 B3 的唯一设计差别 = 预算：epochs {B3_MAX_EPOCHS}→{B3L_MAX_EPOCHS}，'
          f'patience {B3_PATIENCE}→{B3L_PATIENCE}（其余逐字相同，见 diff 断言）')
    print(f'# lr={LR} epochs≤{B3L_MAX_EPOCHS} patience={B3L_PATIENCE}'
          f'（循环用 bad>PATIENCE，即容忍 {B3L_PATIENCE} 轮未提升、第 {B3L_PATIENCE + 1} 轮停）'
          f' micro_bs={MICRO_BS}×accum{ACCUM}  dev_max_new={DEV_MAX_NEW} '
          f'step1_max_new={STEP1_MAX_NEW} step2_max_new={STEP2_MAX_NEW}')
    print(f'# STEP1_SHA256    = {run_twostep.STEP1_SHA256}')
    print(f'# TEMPLATE_SHA256 = {TEMPLATE_SHA256}')
    print('[G-B3-0] step1 模板 sha 与注入模板 sha 均 == 登记值（与臂 C/A 同一串字节）  ✅')
    print('[V-B0] 782 行版折结构 == 391 行版（与 A/B1 同折）  ✅')
    print(f'[V-B1] train/dev/test 行数 = {len(tr)}/{len(dv)}/{len(te)}；'
          f'atomic 组数 = {len(A["train"])}/{len(A["dev"])}/{len(A["test"])}；两两不交  ✅')
    print(f'[V-B2] 回退用 DP 提示识别单元格：{check_prompt_identity(df, args.model_name)}')
    print(f'[V-B3] 标签闭环：{check_label_rule(df, list(tr) + list(dv) + list(te))}')
    print(f'[V-B4] 金标状态：{check_gold_state_vs_C(df, list(tr) + list(dv) + list(te))}')
    n_req = sum(gold_state(df.loc[i]) == 'REQ' for i in tr)
    print(f'[data] 训练集标签分布：REQ {n_req} / ALT {len(tr) - n_req}')
    print(f'[obs] {twin_step1_report(df, list(tr) + list(dv) + list(te))}')

    if args.selfcheck:
        diff_report()
        print(f'\n{"=" * 78}\nSFT 样本渲染（2 条；assistant 段即监督目标，'
              f'与 B1 的唯一差别在这一段）\n{"=" * 78}')
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

        # Inference chain: 1 ponens + 1 tollens. Render both step2 parse outcomes and the
        # fallback — under tollens ALT implies b (not a); if this cell can't be rendered,
        # nothing was checked.
        picks = []
        for m in ('ponens', 'tollens'):
            cand = [i for i in te if df.loc[i, 'modus'] == m]
            if cand:
                picks.append(cand[0])
        print(f'{"=" * 78}\n推理链渲染（1 ponens + 1 tollens，取自本折 test；'
              f'step1 全文 + rel=1/2 两种 step2 + 回退 DP）\n{"=" * 78}')
        for i in picks:
            row = df.loc[i]
            print(f'{"#" * 26} idx={i} dataset_id={row["dataset_id"]} modus={row["modus"]} '
                  f'lv={row["agreement_lv"]} 金标={row["ground_truth"]} '
                  f'金标关系={gold_state(row)}')
            print('----- STEP 1（微调模型跑，贪心 512 token）-----')
            print(run_twostep.build_step1(row))
            for rel in ('1', '2'):
                conn = 'and' if rel == '1' else 'or'
                print(f'----- STEP 2（原模型跑；step1 解析出 RELATION: {rel} → '
                      f'{state_of_relation(rel)} → conn={conn} → '
                      f'蕴含答案 {implied_answer(conn, row["modus"])}）-----')
                print(build_replace_prompt(row, conn))
            print('----- 回退（step1 解析失败 → 原始 DP 提示，fallback=1）-----')
            print(build_prompt(row['questions'], 'dp'))
            print()
        print(f'[B3L] 与 B3 的唯一设计差别 = 预算：MAX_EPOCHS {B3_MAX_EPOCHS}→'
              f'{B3L_MAX_EPOCHS}，PATIENCE {B3_PATIENCE}→{B3L_PATIENCE}'
              f'（LoRA 位置/数据/折/监督/其余超参/早停指标/推理链/回退/字段全部相同）')
        print(f'[B3L] 产物：B3L_fold{{0..4}}.jsonl（字段同 B3 + early_stopped/n_epochs_run'
              f'/max_epochs/patience/dev_curve/dev_req_frac_curve）')
        print('[selfcheck] DIFF 白名单断言 / G-B3-0 / V-B0…V-B4 全部通过，'
              '未加载语言模型，无读数。')
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
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=TARGETS, layers_to_transform=LORA_LAYERS))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[LoRA] 模型层数={model.config.num_hidden_layers}  可训练参数={n_tr:,}')

    epochs = SMOKE_EPOCHS if args.smoke else B3L_MAX_EPOCHS
    tr_idx, dv_idx, te_idx = list(tr), list(dv), list(te)
    if args.smoke:
        # Test slice takes 16 per modus stratum: the CSV has the ponens block first, so
        # taking the first 32 rows would be all ponens, while step2's implied answer
        # depends on modus (ALT→b under tollens) — that path would never get smoked.
        # Affects only the --smoke branch; the full path is untouched.
        tr_idx, dv_idx = tr_idx[:64], dv_idx[:32]
        te_idx = ([i for i in te_idx if df.loc[i, 'modus'] == 'ponens'][:16]
                  + [i for i in te_idx if df.loc[i, 'modus'] == 'tollens'][:16])

    samples = [encode(tok, run_twostep.build_step1(df.loc[i]), target_line(df.loc[i]))
               for i in tr_idx]
    print(f'[data] 训练样本 {len(samples)} 条；assistant 段 token 数 '
          f'{[sum(1 for x in s[1] if x != -100) for s in samples[:3]]}…')

    print(f'[budget] epochs≤{epochs}  patience={B3L_PATIENCE}  '
          f'train={len(tr_idx)} dev={len(dv_idx)} test={len(te_idx)}')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    dev_curve, loss_curve, dev_req_frac = [], [], []
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
            tot += float(loss); nb += 1
            if (s // MICRO_BS + 1) % ACCUM == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        dev_recs = step1_records(model, tok, df, dv_idx, DEV_MAX_NEW, args.batch,
                                 f'dev ep{ep}')
        d = rel_bal_acc(dev_recs)
        n_fail = sum(1 for r in dev_recs if r['relation'] is None)
        dev_curve.append(float(d))
        loss_curve.append(float(tot / max(nb, 1)))
        dev_req_frac.append(sum(1 for r in dev_recs if r['pred_state'] == 'REQ')
                            / max(len(dev_recs), 1))
        n_epochs_run = ep + 1
        print(f'[epoch {ep}] train loss = {tot / max(nb, 1):.4f}  '
              f'dev 关系 bal-acc = {d:.4f}  (dev n={len(dev_recs)}, '
              f'解析失败={n_fail}, 预测 REQ={sum(1 for r in dev_recs if r["pred_state"] == "REQ")}'
              f'/ALT={sum(1 for r in dev_recs if r["pred_state"] == "ALT")})', flush=True)
        if d > best:
            best, best_ep, bad = d, ep, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items() if 'lora' in k}
        else:
            bad += 1
            if bad > B3L_PATIENCE:
                print(f'[early stop] dev 关系 bal-acc 连续 {bad} 轮未提升，停在 epoch {ep}')
                early_stopped = True
                break
    assert best_state is not None, \
        '没有任何一轮拿到可比较的 dev 指标（全 NaN？dev 单类？）——仪器坏了，不出读数'
    model.load_state_dict(best_state, strict=False)
    print(f'[best] dev 关系 bal-acc = {best:.4f}（epoch {best_ep}）')

    # ---- convergence and collapse-shape registration (PREREG_b3long §1/§2): register only, no judgement ----
    tail = (dev_curve[-1] - dev_curve[-4]) if len(dev_curve) >= 4 else None
    print(f'[conv] dev 曲线 = {[round(x, 4) for x in dev_curve]}')
    print(f'[conv] dev REQ 预测占比曲线 = {[round(x, 4) for x in dev_req_frac]}')
    print(f'[conv] train loss 曲线 = {[round(x, 4) for x in loss_curve]}')
    print(f'[conv] n_epochs_run={n_epochs_run}/{epochs}  early_stopped={early_stopped}  '
          f'best_epoch={best_ep}  best_dev={best:.4f}  '
          f'最后3轮 dev 提升总和(=d[-1]-d[-4]) = '
          f'{"n/a(<4 轮)" if tail is None else f"{tail:+.4f}"}'
          f'（PREREG_b3long §1："早停触发 或 该和 <{CONV_TAIL_EPS}" 即已收敛；判定归汇总）')
    print(f'[conv] 起步塌缩登记：第 1 个 dev 点的 REQ 预测占比 = {dev_req_frac[0]:.4f}'
          f'（§1 阈 ≥{COLLAPSE_REQ_FRAC} 记为"全 REQ 塌缩起步"；判定归汇总）')

    # ---- inference chain step 1: fine-tuned model judges the relation ----
    s1 = step1_records(model, tok, df, te_idx, STEP1_MAX_NEW, args.batch, 'B3L step1')

    # ---- step 2: release the fine-tuned model. step2 must be executed by the **untuned original model** (variable isolated to the classifier) ----
    del model, opt, best_state, samples
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
    s2 = gen_batch(base, tok, prompts, STEP2_MAX_NEW, args.batch, 'B3L step2')

    recs = []
    for r, p, (g, trunc, ntok) in zip(s1, prompts, s2):
        row = df.loc[r['idx']]
        rel = r['relation']
        conn = None if rel is None else ('and' if rel == '1' else 'or')
        ext = get_final_answer(g)
        imp = None if conn is None else implied_answer(conn, row['modus'])
        recs.append({
            'arm': 'B3L', 'idx': int(r['idx']), 'dataset_id': row['dataset_id'],
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
            'max_epochs': epochs, 'patience': B3L_PATIENCE,
            'dev_curve': [round(x, 6) for x in dev_curve],
            'dev_req_frac_curve': [round(x, 6) for x in dev_req_frac],
            'smoke': bool(args.smoke),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = '_smoke' if args.smoke else ''
    out_path = os.path.join(args.out_dir, f'B3L_fold{args.fold}{suffix}.jsonl')
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
    print('# 主读数与配对比较在汇总脚本里出，本脚本只产原始 JSONL（PREREG_b3long §2）')


if __name__ == '__main__':
    main()
