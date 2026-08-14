#!/usr/bin/env python
"""[ORACLE] arm two: hand the state to the model, then let it **freely generate** the answer (no scoring).

Relation to `probe_oracle_state.py`
- that one reads the answer in scoring space (option-text likelihood / single-token forced
  choice);
- this one has the model write its own reasoning and emit `Final Answer [X]`, extracted
  via the paper protocol.
  Together they answer: does the "condition following" observed in L2 hold only in
  scoring space, or also under free generation.

Scope
- wording uses `prose` only: `entry` (the semi-formal AND/OR entry) was already measured
  in L2 as a bad way to write the state (Qwen3-4B ponens follow rate 0.487 vs prose
  0.978), not worth further generation compute.
- two context modes × gold/swap conditions = 4 groups.
- T=0 greedy, max_new_tokens=1024, extraction function identical to v1, for tabling
  against the unconditional baseline.
- record `has_final_answer`: upstream `get_final_answer` emits an irrelevant letter rather
  than an empty string when 'final answer' is missing (STATUS R7); format_ok is
  untrustworthy and must be recomputed from this field.
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generative import BELIEF_REV, FORMATTING  # noqa: E402
from probe_oracle_state import condition_text, parse_cond  # noqa: E402
from src.prompts.utils import get_final_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'outputs', 'probe_oracle_gen'))
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda').eval()

    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv'))
    if args.limit:
        df = df.head(args.limit)

    def build(row, conn, mode):
        L = row['questions'].split('\n')
        (p, q), (r, _) = parse_cond(L[0]), parse_cond(L[2])
        cond = condition_text('prose', conn, p, r, q)
        head = (f"{chr(10).join(L[:3])}\n\nAdditional clarification of how these fit together: {cond}"
                if mode == 'keep' else f'{cond}\n{L[1]}')
        return (f'{head}\n\nWhat necessarily had to follow assuming that the above premises '
                f"were true?\n(a) {row['a']}\n(b) {row['b']}\n(c) {row['c']}\n\n{FORMATTING}")

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    for mode in ('replace', 'keep'):
        for side in ('gold', 'swap'):
            out_path = os.path.join(args.out_dir, f'time_t1_{safe}_{mode}_{side}.jsonl')
            done = set()
            if args.resume and os.path.exists(out_path):
                with open(out_path) as f:
                    for line in f:
                        try:
                            done.add(json.loads(line)['idx'])
                        except (json.JSONDecodeError, KeyError):
                            pass
            todo = df[~df.index.isin(done)]
            with open(out_path, 'a' if done else 'w') as fout:
                for st in tqdm(range(0, len(todo), args.batch), desc=f'{safe} {mode}/{side}'):
                    batch = todo.iloc[st:st + args.batch]
                    conns = [('and' if r['ground_truth'] == 'c' else 'or') for _, r in batch.iterrows()]
                    if side == 'swap':
                        conns = ['or' if c == 'and' else 'and' for c in conns]
                    prompts = [build(r, c, mode) for (_, r), c in zip(batch.iterrows(), conns)]
                    texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                                     add_generation_prompt=True,
                                                     enable_thinking=False) for p in prompts]
                    enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
                    with torch.inference_mode():
                        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                             do_sample=False, pad_token_id=tok.pad_token_id)
                    gens = tok.batch_decode(out[:, enc['input_ids'].shape[1]:],
                                            skip_special_tokens=True)
                    for (idx, row), conn, gen in zip(batch.iterrows(), conns, gens):
                        ext = get_final_answer(gen)
                        implied = 'c' if conn == 'and' else ('a' if row['modus'] == 'ponens' else 'b')
                        fout.write(json.dumps({
                            'idx': int(idx), 'dataset_id': row['dataset_id'],
                            'modus': row['modus'], 'agreement_lv': int(row['agreement_lv']),
                            'ground_truth': row['ground_truth'], 'fed_conn': conn,
                            'implied': implied, 'context_mode': mode, 'side': side,
                            'model': args.model_name, 'raw_output': gen, 'extracted': ext,
                            'has_final_answer': 'final answer' in gen.lower(),
                            'follows': ext == implied, 'correct': ext == row['ground_truth'],
                        }, ensure_ascii=False) + '\n')
                    fout.flush()
            print(f'[done] {mode}/{side} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
