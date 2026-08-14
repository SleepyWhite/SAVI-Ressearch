#!/usr/bin/env python
"""Hidden-layer linear probe — GPU extraction side (PREREG_hidden_probe.md §2).

- Prompt = run_generative.build_prompt(question, 'dp'), verbatim single source;
  chat template invoked exactly as in run_generative.py
  (single user turn, add_generation_prompt=True, enable_thinking=False effective for
  Qwen3 only, tolerated elsewhere).
- One bf16 forward pass per row (output_hidden_states=True, no_grad, batch_size=1,
  no generation).
- Store all-layer vectors at two positions (embedding + every layer):
    main position = last token of the sequence; diagnostic position = last token of the γ3 line.
  The γ3 position is obtained via char-offset→token mapping; asserted per row (PREREG V5):
    (a) tokenizer(text) ids == apply_chat_template(tokenize=True) ids;
    (b) the γ3 line text occurs exactly once in the templated text.
- Written to outputs/hidden_probe/<model_safe>/reps.npz (fp16, last_tok / gamma3_tok
  both [N, taps, hidden]) + meta.jsonl. With --limit>0, writes under
  outputs/hidden_probe/_smoke/ so smoke artifacts never overwrite full artifacts.
- --selfcheck: pure CPU. V0 data assertions + sample 8 rows, run the γ3 mapping and print
  the span text for manual inspection (V5).
- In smoke mode (0 < --limit ≤ 8), performs PREREG V4: a second forward pass over those
  rows, asserting the main-position vectors are element-wise identical (tolerance 0) and
  free of NaN/Inf.

Usage:
  python scripts/run_hidden_extract.py --model_name Qwen/Qwen3-4B --selfcheck
  CUDA_VISIBLE_DEVICES=6 python scripts/run_hidden_extract.py --model_name Qwen/Qwen3-4B --limit 8
  CUDA_VISIBLE_DEVICES=6 python scripts/run_hidden_extract.py --model_name Qwen/Qwen3-4B
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_generative import build_prompt                      # noqa: E402  single source of the DP prompt
from summarize_length_matched import build_subsets, cem_match  # noqa: E402  single source of the subsets

# Belief-R dataset CSV. The data is not distributed with this repo (see the README's data section); set env var BELIEF_R_CSV to a local copy, else falls back to <repo>/data/queries_time_t1.csv.
CSV = os.environ.get('BELIEF_R_CSV') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'queries_time_t1.csv')
OUT_ROOT = os.path.join(HERE, '..', 'outputs', 'hidden_probe')
CT_KWARGS = {'enable_thinking': False}  # same as run_generative.py: used by the Qwen3 template only, tolerated elsewhere
SELFCHECK_SEED = 20260803


def encode_and_locate(tok, question):
    """→ (input_ids list, index of the γ3 line-final token). Assertions = the two mapping preconditions of PREREG V5."""
    prompt = build_prompt(question, 'dp')
    text = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                   tokenize=False, add_generation_prompt=True, **CT_KWARGS)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids_ct = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                     tokenize=True, add_generation_prompt=True, **CT_KWARGS)
    assert list(enc['input_ids']) == list(ids_ct), \
        'V5 失败：重编码 ids != chat template ids，offset 映射不可用'
    g3 = question.split('\n')[2]
    assert text.count(g3) == 1, f'V5 失败：γ3 行在模板文本中出现 {text.count(g3)} 次（要求恰 1）'
    end = text.index(g3) + len(g3)          # one past the last character of the γ3 line
    g3_tok = None
    for i, (a, b) in enumerate(enc['offset_mapping']):
        if a <= end - 1 < b:                # the token covering the last character of γ3
            g3_tok = i
    assert g3_tok is not None, 'V5 失败：没有 token 覆盖 γ3 行末字符'
    return enc['input_ids'], g3_tok, text, end


def selfcheck(model_name, df):
    """Pure CPU: V0 data assertions + sample 8 rows to verify the γ3 mapping and print spans (PREREG V0/V5)."""
    print(f'== selfcheck: {model_name} ==')
    # V0: row count / ponens / scenario count / SM
    assert len(df) == 1744, f'V0 失败：行数 {len(df)} != 1744'
    assert (df.modus == 'ponens').sum() == 872, 'V0 失败：ponens 行数 != 872'
    f = build_subsets()                       # carries its own 870-scenario assertion
    sm = cem_match(f)
    assert len(sm) == 178, f'V0 失败：SM 场景数 {len(sm)} != 178'
    print(f'[V0] 1,744 行 / ponens 872 / 场景 870 / SM {len(sm)}  全部通过')

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    rng = np.random.default_rng(SELFCHECK_SEED)
    picks = sorted(rng.choice(len(df), 8, replace=False))
    for i in picks:
        q = df.iloc[i]['questions']
        ids, g3_tok, text, end = encode_and_locate(tok, q)
        g3_line = q.split('\n')[2]
        span_start = max(0, end - 40)
        tok_text = tok.decode([ids[g3_tok]])
        print(f'[V5] row {i:4d}  n_tok={len(ids)}  γ3_tok_idx={g3_tok}')
        print(f'      γ3 行  : {g3_line!r}')
        print(f'      行末上下文: ...{text[span_start:end]!r}')
        print(f'      γ3 行末 token: {tok_text!r}')
    print(f'[V5] {len(picks)} 行 γ3 映射断言全部通过（ids 一致 / 唯一出现 / 行末 token 命中）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--limit', type=int, default=0, help='0=全量；>0 时写到 _smoke/ 下')
    ap.add_argument('--selfcheck', action='store_true', help='纯 CPU 自查，不加载模型不前向')
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    if args.selfcheck:
        selfcheck(args.model_name, df)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda').eval()
    n_taps = model.config.num_hidden_layers + 1     # embedding + every layer

    if args.limit:
        df = df.head(args.limit)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    out_dir = os.path.join(OUT_ROOT, '_smoke', safe) if args.limit else os.path.join(OUT_ROOT, safe)
    os.makedirs(out_dir, exist_ok=True)

    def forward_one(question):
        """→ (last_vec [taps,H] fp16, g3_vec same shape)."""
        ids, g3_tok, _, _ = encode_and_locate(tok, question)
        inp = torch.tensor([ids], device='cuda')
        with torch.inference_mode():
            out = model(inp, output_hidden_states=True)
        hs = out.hidden_states
        assert len(hs) == n_taps, f'taps {len(hs)} != num_hidden_layers+1 {n_taps}'
        last = torch.stack([h[0, -1] for h in hs]).to(torch.float16).cpu().numpy()
        g3 = torch.stack([h[0, g3_tok] for h in hs]).to(torch.float16).cpu().numpy()
        return last, g3

    from tqdm import tqdm
    last_all, g3_all, meta = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=safe):
        last, g3 = forward_one(row['questions'])
        assert np.isfinite(last.astype(np.float32)).all() and \
            np.isfinite(g3.astype(np.float32)).all(), f'row {idx}: NaN/Inf'
        last_all.append(last)
        g3_all.append(g3)
        meta.append({'idx': int(idx), 'dataset_id': row['dataset_id'], 'modus': row['modus'],
                     'agreement_lv': int(row['agreement_lv']), 'gold': row['ground_truth'],
                     'atomic_idx': int(row['atomic_idx'])})

    last_arr = np.stack(last_all)     # [N, taps, H] fp16
    g3_arr = np.stack(g3_all)

    if 0 < args.limit <= 8:           # PREREG V4: extraction determinism (smoke mode)
        for j, (idx, row) in enumerate(df.iterrows()):
            last2, _ = forward_one(row['questions'])
            assert np.array_equal(last_arr[j], last2), f'V4 失败：row {idx} 第二次前向不一致'
        print(f'[V4] {len(df)} 行第二次前向主位置向量逐元素一致（容差 0），无 NaN/Inf')

    np.savez(os.path.join(out_dir, 'reps.npz'), last_tok=last_arr, gamma3_tok=g3_arr)
    with open(os.path.join(out_dir, 'meta.jsonl'), 'w') as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(f'[done] N={len(df)}  taps={last_arr.shape[1]}  hidden={last_arr.shape[2]}  '
          f'fp16 -> {out_dir}')


if __name__ == '__main__':
    main()
