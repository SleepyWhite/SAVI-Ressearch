#!/usr/bin/env python
"""Candidate (ii): explicitly juxtapose the two readings in context and let the model pick one **with room to reason**.

============================================================================
How it differs from L1 (this is the whole point)
============================================================================
L1 (already judged negative, including the 0.5B→7B scale ladder): **offline likelihood
scoring** — each reading is appended to the same prefix and scored once for logP; the
model generates **not a single reasoning token** (scoring happens after
`<think>\\n\\n</think>`).

This probe: both readings are **written out simultaneously in context**; the model
generates freely, reasons first, then chooses, and finally writes the `RELATION:` and
`Final Answer` lines per the contract.

**Why it deserves a separate test**: L2b already hit an isomorphic trap once — the same
capability scored 0.085 under a zero-reasoning-token forced choice readout and 0.877 under
free generation, a tenfold gap. Judging "is γ3 supplementary or restrictive" is a pragmatic
inference and almost certainly needs one reasoning step. So L1's negative result strictly
only refutes the **scoring axis**, not **the model's judgment**.
`VERDICT_probe_relation` kept this item in the "not killed" list from the start.

============================================================================
Pre-registration (frozen before running, unchanged after)
============================================================================
Statistical unit = **scenario** (dataset_id). Twins share γ1/γ3, so the relation problem
text is verbatim identical; but both twins are run — **do they give the same relation
judgment** is a free self-consistency readout (see C-2).

Main readout set = **agreement_lv = 5** (locked design decision, STATUS §3).

[P-1 main test] **Balanced accuracy** of the RELATION choice = (REQ recall + ALT recall) / 2.
    Balanced accuracy rather than accuracy: BU makes up 65.7% of lv=5 scenarios, so
    "always pick REQ" gets 0.657 accuracy while carrying no information — the same reason
    BREU is an equal-weight average. Chance = 0.500.
    **Balanced accuracy is directly comparable to L1's AUC** (a binary scorer's AUC equals
    its balanced accuracy), so it can sit in the same table as the L1 ladder's 0.588
    ceiling and the 0.843 ceiling.

[P-2 position control] Run the full set once with REQ in slot (A) and once in slot (B);
    **report both orders**. Models have positional preferences over A/B; without the swap
    you cannot separate "chose correctly" from "always chooses A".
    **Validity gate**: if the two orders' balanced accuracies differ by > 0.10, or the
    "fraction choosing A" is > 0.70 under both orders (position overwhelming content),
    that model's readout is void. Same role as L2's swap control.

[C-1 self-consistency rate] Whether `Final Answer` equals the answer implied by the
    RELATION the model **itself chose**. Needs no gold label → unaffected by the 47.4%
    label noise. Distinguishes "picked the wrong state" from "picked the right state but
    reasoned wrong".

[C-2 twin agreement] Whether the ponens/tollens phrasings of the same scenario give the
    same RELATION. The relation does not depend on γ2, so disagreement is
    self-contradiction. Also needs no gold label.

[End-to-end] `Final Answer` against gold as BU/BM/BREU, comparable to the §6.1 generative
    baseline under the same protocol.

============================================================================
Criteria (frozen kill-criteria) — judged on lv=5 P-1, and must pass the P-2 gate
============================================================================
  ≥ 0.65  → **real signal**. Something the L1 scoring axis cannot do (its ceiling 0.588);
            the difference attributes to intervention form, not scale — the control is the
            six-model L1 ladder
  0.55–0.65 → same band as L1, **the form added nothing**. Record, don't build on it
  ≤ 0.55  → **no signal**. At that point the "make the model distinguish the conditions"
            route on Belief-R closes
  significantly < 0.50 → positional preference or inverted prior; investigate per the P-2
            gate, no sign-flipping allowed

Ceiling reference: with "design intent" as a perfect scorer, the AUC ceiling on lv=5 =
**0.843** (computed 2026-08-02). So the 0.55–0.65 band is not squeezed by the ceiling;
it genuinely wasn't achieved.

============================================================================
Budget
============================================================================
6 models × 2 orders × 1,744 rows, free generation max_new_tokens=512.
Reference: oracle_gen on 4B took ~33 min for 1,744 rows / 1024 tokens, so ~15–25 min per
config here. 12 configs in 3 waves on 4 GPUs, ~1–1.5 h.

Usage
  python scripts/probe_incontext_choice.py --dry_render
  python scripts/probe_incontext_choice.py --model_name Qwen/Qwen2.5-7B-Instruct --order req_first
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from run_generative import BELIEF_REV  # noqa: E402
from probe_oracle_state import parse_cond  # noqa: E402
from prompts.contract import (build, parse_relation, implied_answer,  # noqa: E402
                              CONTRACT_SHA256, VALID_ANSWERS)
from src.prompts.utils import get_final_answer  # noqa: E402


def load_rows(limit):
    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv'))
    if limit:
        df = df.head(limit)
    out = []
    for i, row in df.iterrows():
        L = row['questions'].split('\n')
        c1, c3 = parse_cond(L[0]), parse_cond(L[2])
        if not c1 or not c3:
            raise ValueError(f"{row['dataset_id']} 的 γ1/γ3 解析失败，不该发生")
        (p, q), (r, _) = c1, c3
        out.append({'idx': int(i), 'dataset_id': row['dataset_id'],
                    'base': row['dataset_id'].split('-')[0],
                    'intent': row['dataset_id'].split('-')[-1],
                    'modus': row['modus'], 'agreement_lv': int(row['agreement_lv']),
                    'ground_truth': row['ground_truth'],
                    'gold_is_req': row['ground_truth'] == 'c',
                    'premises': '\n'.join(L[:3]), 'p': p, 'r': r, 'q': q,
                    'opts': (row['a'], row['b'], row['c'])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name')
    ap.add_argument('--order', default='req_first', choices=['req_first', 'alt_first'])
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'outputs', 'incontext_choice'))
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dry_render', action='store_true')
    args = ap.parse_args()

    rows = load_rows(args.limit)
    req_first = args.order == 'req_first'

    if args.dry_render:
        print(f'CONTRACT_SHA256 = {CONTRACT_SHA256}\n')
        for of in (True, False):
            pr, letter = build(rows[0]['premises'], rows[0]['p'], rows[0]['r'], rows[0]['q'],
                               rows[0]['opts'], of)
            print(f"########## order={'req_first' if of else 'alt_first'}  "
                  f"（REQ 在 {letter}）##########")
            print(pr)
            print()
        return

    if not args.model_name:
        raise SystemExit('需要 --model_name（或用 --dry_render 只看提示）')

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
    out_path = os.path.join(args.out_dir, f'time_t1_{safe}_{args.order}.jsonl')
    done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['idx'])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [r for r in rows if r['idx'] not in done]

    with open(out_path, 'a' if done else 'w') as fout:
        for st in tqdm(range(0, len(todo), args.batch), desc=f'{safe} {args.order}'):
            batch = todo[st:st + args.batch]
            built = [build(r['premises'], r['p'], r['r'], r['q'], r['opts'], req_first)
                     for r in batch]
            texts = [tok.apply_chat_template([{'role': 'user', 'content': pr}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
                     for pr, _ in built]
            enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gens = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)

            for r, (_, req_slot), gen in zip(batch, built, gens):
                rel = parse_relation(gen)
                has_fa = 'final answer' in gen.lower()
                ans = get_final_answer(gen) if has_fa else ''
                picked_req = None if rel is None else (rel == req_slot)
                own = None if picked_req is None else implied_answer(picked_req, r['modus'])
                fout.write(json.dumps({
                    'idx': r['idx'], 'dataset_id': r['dataset_id'], 'base': r['base'],
                    'intent': r['intent'], 'modus': r['modus'],
                    'agreement_lv': r['agreement_lv'], 'ground_truth': r['ground_truth'],
                    'gold_is_req': r['gold_is_req'],
                    'order': args.order, 'req_slot': req_slot,
                    'model': args.model_name, 'contract_sha': CONTRACT_SHA256,
                    'raw_output': gen,
                    'relation_slot': rel,              # the slot the model wrote (1/2)
                    'picked_req': picked_req,          # whether it picked REQ (position already factored out)
                    'relation_correct': None if picked_req is None else picked_req == r['gold_is_req'],
                    'own_implied': own,                # the answer implied by its own reading
                    'extracted': ans,
                    'has_final_answer': has_fa,
                    # contract-violation diagnostic: the answer slot holds something other than a/b/c
                    # (during smoke we saw it write the reading letter)
                    'extracted_valid': ans in VALID_ANSWERS,
                    'coherent': None if (own is None or not ans) else ans == own,
                    'correct': ans == r['ground_truth'],
                }, ensure_ascii=False) + '\n')
            fout.flush()
    print(f'[done] {out_path}  n={len(rows)}  sha={CONTRACT_SHA256[:12]}')


if __name__ == '__main__':
    main()
