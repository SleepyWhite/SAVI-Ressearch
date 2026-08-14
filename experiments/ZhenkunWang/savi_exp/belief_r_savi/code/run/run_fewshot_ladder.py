#!/usr/bin/env python
"""R3 few-shot demonstrations (by k tier): is the REQ/ALT distinction a **learnable convention**?

Pre-registration and criteria are frozen in the internal experiment log §2b; this file only
implements, it does not change criteria. Before taking over, read §2b and §6.12
(the R1/R2 lesson: any front-loaded intervention must come with a placebo arm).

============================================================================
The question is not "can the model do it" but "is what's missing world information or a labeling convention"
============================================================================
Both ends of the prompt ladder are already pinned: R0 zero-shot 0.497/0.433, R4 given-state
0.994. The middle R1/R2 (a single instruction sentence) is already judged negative — zero
after subtracting the placebo. R3 changes the **form**: no instruction, give examples.

  Effective and passes the length gate → the information is in the text, just locked
  behind a convention; SAVI regains an entry point here
  Effective but dies at the length gate → what was learned is "long γ3 → answer c"; the
  conclusion stands, and the benchmark-validity problem worsens
  Ineffective          → the strongest form of "the information is not in the context";
  the whole ladder is flat

============================================================================
Arms (6)
============================================================================
  k2 / k4 / k8 / k16       examples are **nested** (k=2 ⊂ 4 ⊂ 8 ⊂ 16), so the k curve is
                           paired
  k8_shuf / k16_shuf       **shuffled-label arms**: same examples, same question texts and
                           order; only the k answers are permuted across examples (seed=0).
                           The label marginal distribution is unchanged; only the pairing
                           of question text and answer is broken.
                           This is the **correct placebo** for few-shot — the R1/R2 round
                           showed a generic placebo is not enough (on Llama, four
                           semantically different sentences gave the same +0.05…+0.07,
                           while the only no-added-sentence loose arm was zero).
  R0 (k=0)                 not run in this script; reuses outputs/generative/*_dp.jsonl,
                           but **must be recomputed on the same 852 scenarios** (§2b.10);
                           quoting 0.4970 is not allowed.

============================================================================
Example pool (frozen in §2b.3; every run of this script recomputes it and asserts bit-for-bit)
============================================================================
Selection = lv=5 candidates, greedy in ascending order of **(how many lv=5 scenarios this
seed would cost, this seed's total scenario count, dataset_id)**; one example per seed
(atomic_idx), 8 per class.
The first sort key is the crux: protecting the primary lv=5 readout set first is what
yields the [1×12, 2×4] recorded in §2b.3 and "lv=5 loses only 16". Sorting by total
scenario count alone would pick 85-weak, one item off from the frozen pool.

Actual: BU 8 / BM 8, 16 seeds used, eval-set loss **20/872 = 2.3%** (16/391 on lv=5),
remaining **852 scenarios (BU 527 / BM 325)**; all k tiers share the same eval set
(otherwise the k curve is not paired).

**Examples give only the answer, no rationale** — giving the rationale would write the
REQ/ALT explanation straight in, degenerating into the already-negative L5 (in-context
explicit choice, 0.545). The BM pool contains two `-strong` items — exactly the 47.4%
intent≠gold portion; **examples are always labeled by the gold standard**, because the
convention to be taught is the gold one.

============================================================================
Two things §2b left unpinned, decided by this script (must pass a human before running numbers)
============================================================================
1. **The examples' modus follows the tested item** (`--example_modus match`, default).
   I.e. when testing a ponens item, all 16 examples render as ponens; when testing a
   tollens item, all render as tollens.
   Rationale: BREU weights the two modi equally, and the tollens answer mapping is
   REQ→c / ALT→**b**. With ponens-only examples the model would have to learn the
   convention and also transfer that mapping across modus — **a negative result would be
   ambiguous** ("didn't learn the convention" vs "didn't transfer to tollens" cannot be
   separated). `--example_modus ponens` keeps the alternative, but it is not the default.
2. **Example order = strict BU/BM alternation** (§2b.4 only says "alternate, avoid block
   structure"), in greedy pick order, so that the nesting k=2 ⊂ k=4 ⊂ k=8 ⊂ k=16 is
   visible to the naked eye in the prompt.

============================================================================
Readout (byte-identical to R0/R1/R2, otherwise not comparable in the same table)
============================================================================
Free generation + `Final Answer [X]` extraction, T=0 greedy, max_new_tokens=1024,
single user turn.
The tested-item segment = `question + '\n\n' + FORMATTING`, byte-identical to the dp
branch of run_generative.py; this script only **prepends** the example block and does not
touch a single character of the tested item.
`has_final_answer` is always recomputed from raw_output (R7: upstream get_final_answer
emits an illegal letter when there is no `final answer`; format_ok is a false positive).

============================================================================
Main criterion (frozen before running numbers, unchanged afterwards; §2b.7)
============================================================================
**Paired ΔBREU** of `k16` vs `k16_shuf` on lv=5:
  ≥ +0.05, passes the four gates (G-shuf/G-len/G-k/G-leak), and both models same
  direction → the convention is learnable
  < +0.02                                          → ineffective
  in-between band or any gate fails                → report honestly as uncertain
The primary readout is **BREU**, not BU (§6.12 paid for this: always answering c gets
BU=1.000).

Usage
  python scripts/run_fewshot_ladder.py --selfcheck          # no GPU: pool/nesting/leak/permutation
  python scripts/run_fewshot_ladder.py --dry_render         # no GPU: dump the full k=2 and k=16 assembly
  CUDA_VISIBLE_DEVICES=0 python scripts/run_fewshot_ladder.py \
      --model_name Qwen/Qwen2.5-7B-Instruct --k 16
  CUDA_VISIBLE_DEVICES=0 python scripts/run_fewshot_ladder.py \
      --model_name Qwen/Qwen2.5-7B-Instruct --k 16 --shuffle_labels
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generative import BELIEF_REV, FORMATTING, TRIGGERS, get_final_answer  # noqa: E402

CSV = os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv')
MODI = ('ponens', 'tollens')
KS = (2, 4, 8, 16)
SHUF_SEED = 0
SEP = '---'                      # separator line between the example block and the tested item
BATCH_BY_K = {2: 16, 4: 16, 8: 8, 16: 4}

# Example pool frozen in §2b.3 (greedy pick order). This script recomputes it and asserts
# bit-for-bit, raising on any mismatch — if the pool changes, the k curve, loss rate, and
# G-leak all have to be re-recorded.
FROZEN_BU = ['484-strong', '502-strong', '526-strong', '611-strong',
             '672-strong', '688-strong', '727-strong', '752-strong']
FROZEN_BM = ['139-weak', '230-weak', '237-weak', '412-strong',
             '102-weak', '18-weak', '196-strong', '238-weak']
FROZEN_EVAL = {'scenes': 852, 'bu': 527, 'bm': 325, 'lost': 20, 'lost_lv5': 16}


# ------------------------------------------------------------------ example pool
def select_pool(df):
    """Recompute the §2b.3 example pool → (bu_ids, bm_ids, seeds). Includes frozen assertions."""
    pon = df[df.modus == 'ponens']
    n_all = pon.groupby('atomic_idx').size()
    n_lv5 = pon[pon.agreement_lv == 5].groupby('atomic_idx').size()
    cand = sorted(pon[pon.agreement_lv == 5].itertuples(),
                  key=lambda r: (int(n_lv5.get(r.atomic_idx, 0)),
                                 int(n_all[r.atomic_idx]), r.dataset_id))
    bu, bm, seeds = [], [], set()
    for r in cand:
        if r.atomic_idx in seeds:                      # one example per seed
            continue
        tgt = bu if r.ground_truth == 'c' else bm
        if len(tgt) >= 8:
            continue
        tgt.append(r.dataset_id)
        seeds.add(int(r.atomic_idx))

    if bu != FROZEN_BU or bm != FROZEN_BM:
        raise AssertionError(f'示例池与 §2b.3 冻结的不符：\n  BU {bu}\n  BM {bm}')
    lost = pon[pon.atomic_idx.isin(seeds)]
    keep = pon[~pon.atomic_idx.isin(seeds)]
    got = {'scenes': len(keep), 'bu': int((keep.ground_truth == 'c').sum()),
           'bm': int((keep.ground_truth != 'c').sum()), 'lost': len(lost),
           'lost_lv5': int((lost.agreement_lv == 5).sum())}
    if got != FROZEN_EVAL:
        raise AssertionError(f'评测集与 §2b.3 冻结的不符：{got} != {FROZEN_EVAL}')
    return bu, bm, seeds


def example_ids(k):
    """dataset_ids of the k examples, strict BU/BM alternation; k=2 ⊂ 4 ⊂ 8 ⊂ 16."""
    if k not in KS:
        raise ValueError(f'k 只能取 {KS}')
    out = []
    for i in range(k // 2):
        out += [FROZEN_BU[i], FROZEN_BM[i]]
    return out


# ------------------------------------------------------------------ prompt assembly
def build_prefix(df, ids, modus, shuffle):
    """→ (example block text, the letter used by each example). With shuffle=True only the
    letters are permuted; question texts and order are untouched."""
    rows = df[(df.dataset_id.isin(ids)) & (df.modus == modus)].set_index('dataset_id')
    qs = [rows.loc[i, 'questions'] for i in ids]
    labels = [rows.loc[i, 'ground_truth'] for i in ids]
    if shuffle:
        perm = np.random.RandomState(SHUF_SEED).permutation(len(ids))
        shuffled = [labels[j] for j in perm]
        if shuffled == labels:
            raise AssertionError('打乱标签臂退化成恒等置换，安慰剂无效')
        labels = shuffled
    blocks = [f'Example {i + 1}\n{q}\nFinal Answer [{a}].' for i, (q, a) in enumerate(zip(qs, labels))]
    return '\n\n'.join(blocks), labels


def build_prompt(prefix, question, trigger=None):
    """The tested-item segment is byte-identical to R0's dp prompt; the example block is only prepended."""
    p = f'{prefix}\n\n{SEP}\n\n{question}\n\n{FORMATTING}'
    return p if trigger is None else f'{p}\n\n{trigger}'


def build_messages(df, ids, modus, labels, question, trigger=None):
    """chat form: one example = one user/assistant turn.

    Smoke test on 2026-08-03 caught: in inline form, Llama-3.1-8B **continues the example
    block in 32/32 cases** (median 5064 chars, outputs full of "Example 17..."), and since
    get_final_answer uses rfind for the last `final answer`, what gets read is **the example
    label the model copied back**, not its answer to the tested item. Distribution 16×'b' +
    16×empty, ponens 0/16 — the classic "doesn't crash, just wrong data".
    The chat form puts examples into history turns, leaving the model nothing to continue.

    Each example's user turn = `question + '\n\n' + FORMATTING`, same shape as R0's dp
    prompt; the assistant turn = `Final Answer [X].`. The tested item is the last user
    turn, its content still byte-identical to R0.
    """
    rows = df[(df.dataset_id.isin(ids)) & (df.modus == modus)].set_index('dataset_id')
    msgs = []
    for i, a in zip(ids, labels):
        msgs.append({'role': 'user', 'content': rows.loc[i, 'questions'] + '\n\n' + FORMATTING})
        msgs.append({'role': 'assistant', 'content': f'Final Answer [{a}].'})
    q = question + '\n\n' + FORMATTING
    msgs.append({'role': 'user', 'content': q if trigger is None else f'{q}\n\n{trigger}'})
    return msgs


def prefixes_for(df, k, shuffle, example_modus):
    """Prepare example blocks per modus (two under match; one reused under ponens)."""
    ids = example_ids(k)
    if example_modus == 'match':
        return {m: build_prefix(df, ids, m, shuffle) for m in MODI}
    p = build_prefix(df, ids, 'ponens', shuffle)
    return {m: p for m in MODI}


def prompt_sha(pref, k, shuffle, example_modus, demo_format='inline', trigger=None):
    h = json.dumps({'k': k, 'shuffle': shuffle, 'example_modus': example_modus,
                    'demo_format': demo_format, 'trigger': trigger,
                    'ids': example_ids(k), 'sep': SEP, 'formatting': FORMATTING,
                    'prefix': {m: pref[m][0] for m in MODI}}, sort_keys=True)
    return hashlib.sha256(h.encode('utf-8')).hexdigest()


# ------------------------------------------------------------------ two no-GPU entry points
def selfcheck(df):
    bu, bm, seeds = select_pool(df)
    pon = df[df.modus == 'ponens'].set_index('dataset_id')
    print(f'# 示例池（§2b.3 冻结，已逐位复现）  种子 {len(seeds)} 个')
    print(f'  {"类":<4}{"dataset_id":<14}{"atomic":>7}{"该种子场景数":>13}{"金标(pon/tol)":>16}')
    n_all = df[df.modus == 'ponens'].groupby('atomic_idx').size()
    tol = df[df.modus == 'tollens'].set_index('dataset_id')
    for lbl, lst in (('BU', bu), ('BM', bm)):
        for d in lst:
            a = int(pon.loc[d, 'atomic_idx'])
            print(f'  {lbl:<4}{d:<14}{a:>7}{int(n_all[a]):>13}'
                  f'{pon.loc[d, "ground_truth"] + " / " + tol.loc[d, "ground_truth"]:>16}')
    print(f'\n# 评测集：{FROZEN_EVAL["scenes"]} 场景（BU {FROZEN_EVAL["bu"]} / BM {FROZEN_EVAL["bm"]}），'
          f'损失 {FROZEN_EVAL["lost"]}/872 = {FROZEN_EVAL["lost"] / 872:.3f}'
          f'（lv=5 上 {FROZEN_EVAL["lost_lv5"]}/391 = {FROZEN_EVAL["lost_lv5"] / 391:.3f}）')

    print('\n# G-leak：评测集与示例的 atomic_idx 交集')
    keep = df[~df.atomic_idx.isin(seeds)]
    inter = set(keep.atomic_idx) & seeds
    print(f'  交集 {len(inter)} 个 → {"✅ 空" if not inter else "❌ 非空，泄漏"}；'
          f'评测行数 {len(keep)}（应 {FROZEN_EVAL["scenes"] * 2}）')
    assert not inter and len(keep) == FROZEN_EVAL['scenes'] * 2

    print('\n# 嵌套：k=2 ⊂ 4 ⊂ 8 ⊂ 16')
    for a, b in zip(KS, KS[1:]):
        ok = set(example_ids(a)) < set(example_ids(b))
        print(f'  k={a:<3}⊂ k={b:<3}{"✅" if ok else "❌"}   k={a} 的示例：{example_ids(a)}')
        assert ok

    print('\n# 打乱标签臂（seed=0）：标签边缘分布须不变，配对须被打断')
    for k in (8, 16):
        for m in MODI:
            _, gold = build_prefix(df, example_ids(k), m, False)
            _, shuf = build_prefix(df, example_ids(k), m, True)
            same = sum(x == y for x, y in zip(gold, shuf))
            print(f'  k={k:<3}{m:<8} 金标 {"".join(gold)} → 打乱 {"".join(shuf)}   '
                  f'边缘分布{"相同 ✅" if sorted(gold) == sorted(shuf) else "变了 ❌"}   '
                  f'位置不变 {same}/{k}')
            assert sorted(gold) == sorted(shuf)

    print('\n# 提示 sha（每臂一个，改一个字符就变）')
    for k in KS:
        for sh in (False, True):
            if sh and k not in (8, 16):
                continue
            pref = prefixes_for(df, k, sh, 'match')
            arm = f'k{k}{"_shuf" if sh else ""}'
            n = {m: len(build_prompt(pref[m][0], df.loc[0, 'questions'])) for m in MODI}
            print(f'  {arm:<10}{prompt_sha(pref, k, sh, "match")[:16]}   '
                  f'提示长度 ponens {n["ponens"]} / tollens {n["tollens"]} 字符')
    print('\n[selfcheck] 全部通过')


def dry_render(df, example_modus):
    select_pool(df)
    keep = df[~df.atomic_idx.isin(select_pool(df)[2])]
    for k in (2, 16):
        for m in MODI:
            if k == 2 or m == 'ponens':          # k=16 dumps ponens only; tollens only at k=2 to show the shape
                pref, labels = prefixes_for(df, k, False, example_modus)[m]
                row = keep[keep.modus == m].iloc[0]
                p = build_prompt(pref, row['questions'])
                print(f'{"#" * 32} k={k}  modus={m}  示例标签={"".join(labels)}  '
                      f'{len(p)} 字符 {"#" * 32}')
                print(p)
                print()
    pref, labels = prefixes_for(df, 16, True, example_modus)['ponens']
    _, gold = prefixes_for(df, 16, False, example_modus)['ponens']
    print(f'{"#" * 32} k16_shuf 的标签（题面与顺序同上，只置换答案）{"#" * 32}')
    print(f'  金标  {"".join(gold)}\n  打乱  {"".join(labels)}')


# ------------------------------------------------------------------ main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name')
    ap.add_argument('--k', type=int, default=16, choices=KS)
    ap.add_argument('--shuffle_labels', action='store_true', help='打乱标签臂（安慰剂）')
    ap.add_argument('--example_modus', default='match', choices=['match', 'ponens'])
    ap.add_argument('--demo_format', default='inline', choices=['inline', 'chat'],
                    help='chat = 一个示例一轮 user/assistant（Llama 在 inline 下会续写示例块，'
                         '见 build_messages 的注释）')
    ap.add_argument('--trigger', default='none', choices=['none', 'cot'],
                    help='cot = 被测题后加 R0 的 CoT 触发句。示例是"只给答案"，k 一大'
                         '模型就学成零推理 token（k=16 时输出中位 17 字符），而 tollens '
                         '要先做一步逆否——仪器纪律第 1 条。加触发句把推理空间要回来，'
                         '且 k=0 的锚点已在盘上（outputs/generative/*_cot.jsonl）')
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      '..', 'outputs', 'fewshot_ladder'))
    ap.add_argument('--batch_size', type=int, default=0, help='0 = 按 k 自动（16/16/8/4）')
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0,
                    help='冒烟用：从评测集里**等距抽 limit/2 个场景**（孪生整体进出，'
                         '故两个 modus、BU/BM 都覆盖到）。CSV 是 ponens 全排在前面，'
                         '直接 head 会抽成清一色 ponens，看不到 tollens 的形态。'
                         '带 --limit 时输出写 _smoke/ 子目录，不覆盖正式文件。')
    ap.add_argument('--dry_render', action='store_true')
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    if args.selfcheck:
        return selfcheck(df)
    if args.dry_render:
        return dry_render(df, args.example_modus)
    if not args.model_name:
        raise SystemExit('需要 --model_name（或用 --selfcheck / --dry_render 看设计）')

    _, _, seeds = select_pool(df)
    ev = df[~df.atomic_idx.isin(seeds)].copy()          # 852 scenarios × 2 modi = 1,704 rows
    assert len(ev) == FROZEN_EVAL['scenes'] * 2
    if args.limit:
        sids = sorted(set(ev.dataset_id))
        pick = set(sids[::max(1, len(sids) // max(1, args.limit // 2))][:args.limit // 2])
        ev = ev[ev.dataset_id.isin(pick)]
        args.out_dir = os.path.join(args.out_dir, '_smoke')

    pref = prefixes_for(df, args.k, args.shuffle_labels, args.example_modus)
    trig = TRIGGERS['cot'] if args.trigger == 'cot' else None
    sha = prompt_sha(pref, args.k, args.shuffle_labels, args.example_modus, args.demo_format, trig)
    arm = (f'k{args.k}{"_shuf" if args.shuffle_labels else ""}'
           f'{"_chat" if args.demo_format == "chat" else ""}'
           f'{"_cot" if args.trigger == "cot" else ""}')
    bs = args.batch_size or BATCH_BY_K[args.k]

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda').eval()

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    out_path = os.path.join(args.out_dir, f'time_t1_{safe}_{arm}.jsonl')
    with open(os.path.join(args.out_dir, f'prefix_{arm}.txt'), 'w') as f:   # provenance
        f.write(f'# arm={arm} sha={sha} example_modus={args.example_modus} '
                f'demo_format={args.demo_format} trigger={args.trigger}\n'
                f'# ids={example_ids(args.k)}\n\n')
        demo_q = ev.iloc[0]                       # render exactly what goes to the model, using one real tested item
        for m in MODI:
            f.write(f'{"=" * 30} {m}  标签={"".join(pref[m][1])} {"=" * 30}\n')
            if args.demo_format == 'chat':
                for msg in build_messages(df, example_ids(args.k), m, pref[m][1],
                                          demo_q['questions'], trig):
                    f.write(f'<<{msg["role"]}>>\n{msg["content"]}\n')
            else:
                f.write(build_prompt(pref[m][0], demo_q['questions'], trig) + '\n')
            f.write('\n')

    with open(out_path, 'w') as fout:
        for st in tqdm(range(0, len(ev), bs), desc=f'{safe} {arm}'):
            batch = ev.iloc[st:st + bs]
            if args.demo_format == 'chat':
                msgs = [build_messages(df, example_ids(args.k), m, pref[m][1], q, trig)
                        for q, m in zip(batch['questions'], batch['modus'])]
            else:
                msgs = [[{'role': 'user', 'content': build_prompt(pref[m][0], q, trig)}]
                        for q, m in zip(batch['questions'], batch['modus'])]
            texts = [tok.apply_chat_template(mm, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for mm in msgs]
            enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gens = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)

            for (idx, row), gen in zip(batch.iterrows(), gens):
                has_fa = 'final answer' in gen.lower()          # R7: don't trust format_ok
                ans = get_final_answer(gen) if has_fa else ''
                fout.write(json.dumps({
                    'idx': int(idx), 'dataset_id': row['dataset_id'],
                    'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
                    'agreement_lv': int(row['agreement_lv']),
                    'ground_truth': row['ground_truth'], 'is_bu': row['ground_truth'] == 'c',
                    'arm': arm, 'k': args.k, 'shuffled': args.shuffle_labels,
                    'demo_format': args.demo_format, 'trigger': args.trigger,
                    'example_modus': args.example_modus,
                    'model': args.model_name, 'prompt_sha': sha,
                    'raw_output': gen, 'extracted': ans,
                    'has_final_answer': has_fa,
                    'correct': ans == row['ground_truth'],
                }, ensure_ascii=False) + '\n')
            fout.flush()
    print(f'[done] {out_path}  n={len(ev)}  arm={arm}  sha={sha[:12]}')


if __name__ == '__main__':
    main()
