#!/usr/bin/env python
"""P0 generative baseline runner: local models reproducing the paper's DP / CoT / PS prompt routes.

- Prompt concatenation is byte-for-byte aligned with the dataset's companion code
  src/prompts/one_time.py: question + '\n\n' + formatting [+ '\n\n' + trigger]
  (one_time.py cannot be imported because its import chain goes through
  api.py->apikeys.py; the constants are copied verbatim here)
- Answer extraction reuses the repo's src/prompts/utils.py:get_final_answer; extraction
  failure='' and counts as wrong per the paper's protocol
- Raw output per item written line by line to JSONL (one JSON object per line)
- Convention: T=0 greedy (the paper's api.py sets no temperature, i.e. API default 1.0;
  deterministic reproduction is used here, difference noted)
- chat template: single user turn (aligned with the API form); enable_thinking=False is
  used only by Qwen3, other templates tolerate it
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

# Originally pointed at the dataset's companion code repo to import src.prompts.utils;
# this repo has vendored that module into code/src/prompts/utils.py, so point at code/ itself.
BELIEF_REV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, BELIEF_REV)
from src.prompts.utils import get_final_answer  # noqa: E402

FORMATTING = 'Begin! Reminder to write your final answer as "Final Answer [X]." and fill [X] with either a, b, or c.'
TRIGGERS = {
    'dp': None,
    'cot': 'Let’s think step by step.',
    'ps': "Let's first understand the problem and devise a plan to solve the problem."
          + "\nThen, let's carry out the plan and solve the problem step by step.",
}


def build_prompt(question, mode):
    text = question + '\n\n' + FORMATTING
    if TRIGGERS[mode] is not None:
        text = text + '\n\n' + TRIGGERS[mode]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--methods', default='dp,cot,ps')
    ap.add_argument('--dataset', default='time_t1', choices=['time_t', 'time_t1'])
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'generative'))
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--resume', action='store_true',
                    help='跳过输出文件中已有 idx，追加续跑（被杀/中断后用）')
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda').eval()

    csv_name = 'queries_time_t1.csv' if args.dataset == 'time_t1' else 'basic_time_t.csv'
    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', csv_name))
    if args.limit:
        df = df.head(args.limit)

    ct_kwargs = {'enable_thinking': False}  # only the Qwen3 template uses this; other templates ignore it

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    for method in args.methods.split(','):
        out_path = os.path.join(args.out_dir, f'{args.dataset}_{safe}_{method}.jsonl')
        done = set()
        if args.resume and os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)['idx'])
                    except (json.JSONDecodeError, KeyError):
                        pass
        with open(out_path, 'a' if done else 'w') as fout:
            for start in tqdm(range(0, len(df), args.batch_size), desc=f'{safe} {method}'):
                batch = df.iloc[start:start + args.batch_size]
                if done:
                    batch = batch[~batch.index.isin(done)]
                    if len(batch) == 0:
                        continue
                prompts = [build_prompt(q, method) for q in batch['questions']]
                texts = [tok.apply_chat_template([{'role': 'user', 'content': p}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 **ct_kwargs) for p in prompts]
                enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                gens = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
                for (idx, row), prompt, gen in zip(batch.iterrows(), prompts, gens):
                    ext = get_final_answer(gen)
                    rec = {
                        'idx': int(idx), 'dataset_id': row['dataset_id'],
                        'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
                        'intent': row['dataset_id'].split('-')[-1],
                        'ground_truth': row['ground_truth'],
                        'model': args.model_name, 'method': method,
                        'prompt': prompt, 'raw_output': gen,
                        'extracted': ext, 'format_ok': bool(ext),
                        'correct': ext == row['ground_truth'],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
        n_tot = n_ok = n_fmt_fail = 0
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_tot += 1
                n_ok += int(r['correct'])
                n_fmt_fail += int(not r['format_ok'])
        print(f'[summary] {safe} {method}: acc={n_ok / n_tot:.4f} (n={n_tot})  format_fail={n_fmt_fail}  -> {out_path}')


if __name__ == '__main__':
    main()
