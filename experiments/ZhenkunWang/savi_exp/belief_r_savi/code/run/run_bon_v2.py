#!/usr/bin/env python
"""P0 v2: sample N chains at T=1, supporting both best-of-N (likelihood chain selection) and voting (mode) readouts.

Relation to v1 (`run_generative.py`)
- The prompt reuses v1's `build_prompt` byte for byte (direct import, not re-typed),
  keeping comparability with the T=0 greedy batch
- v1 is a single pass with do_sample=False; this script is an N-chain pool with do_sample=True
- This script only does "sample + score + write to disk" and computes no metrics; N=16/32,
  the voting tie rule, and the BoN convention are all computed in `summarize_bon_v2.py`,
  so changing a convention does not require re-running the GPU

Conventions
- Sampling: temperature/top_k/top_p all specified explicitly, never taking the model's
  generation_config defaults (Qwen3's defaults are T=0.6/top_k=20/top_p=0.8; without
  explicit overrides this would not be pure T=1 sampling)
- N=16 = a nested subsample of this file's first 16 chains, not an independent resample;
  the budget curve is more stable nested
- Chain scoring: teacher-forcing recomputation, summing only over positions in the
  generated span up to and including the first EOS (not generate's output_scores — that
  would store n_steps × B × vocab logits, tens of GB at 1024 steps).
  Both sum and per-token mean conventions are stored at write time; which to use is left
  to the analysis
- The scoring forward pass uses left padding, so position_ids are passed explicitly;
  `--selfcheck` verifies single-chain scoring agrees with in-batch padded scoring
  (getting padding/position_ids wrong is the most likely silent bug here)
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generative import BELIEF_REV, build_prompt  # noqa: E402
from src.prompts.utils import get_final_answer  # noqa: E402


def scenario_subset(df, n_scenarios):
    """Sample by scenario (PLAN §8: the statistical unit is the scenario; twins enter and leave together).

    Stratification key = intent(strong/weak) × BU/BM, taken from the ponens row;
    equidistant sampling within each stratum, no RNG — the same arguments always yield the
    same subset. Returned rows include both modus twins.
    """
    pon = df[df['modus'] == 'ponens'].copy()
    pon['intent'] = pon['dataset_id'].str.split('-').str[-1]
    pon['cell'] = pon['intent'] + '_' + (pon['ground_truth'] == 'c').map({True: 'BU', False: 'BM'})
    keep = []
    for cell, g in pon.groupby('cell'):
        g = g.sort_values('dataset_id')
        share = max(1, round(n_scenarios * len(g) / len(pon)))
        step = max(1, len(g) // share)
        keep.extend(g['dataset_id'].iloc[::step][:share].tolist())
    return df[df['dataset_id'].isin(set(keep))].copy()


@torch.inference_mode()
def score_chains(model, ids, attn, gen_start, batch, vocab_chunk=256):
    """Per-token log-likelihood (generated span only). Returns (B, L_gen) float32."""
    outs = []
    for i in range(0, ids.shape[0], batch):
        x, a = ids[i:i + batch], attn[i:i + batch]
        pos = (a.cumsum(-1) - 1).masked_fill(a == 0, 1)
        logits = model(input_ids=x, attention_mask=a, position_ids=pos).logits[:, :-1]
        tgt = x[:, 1:]
        lp = torch.empty(tgt.shape, device=x.device, dtype=torch.float32)
        for s in range(0, logits.shape[1], vocab_chunk):  # chunked to avoid an fp32 copy of (B,L,V)
            sl = logits[:, s:s + vocab_chunk].float()
            lp[:, s:s + vocab_chunk] = (sl.gather(2, tgt[:, s:s + vocab_chunk].unsqueeze(2)).squeeze(2)
                                        - sl.logsumexp(-1))
        outs.append(lp[:, gen_start - 1:].clone())
        del logits, lp
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--methods', default='dp,cot,ps')
    ap.add_argument('--dataset', default='time_t1', choices=['time_t', 'time_t1'])
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'outputs', 'bon_v2'))
    ap.add_argument('--n_samples', type=int, default=32)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=0, help='0 = 不截断')
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--prompt_batch', type=int, default=2, help='每次 generate 的题数；显存 = prompt_batch × n_samples 条序列')
    ap.add_argument('--score_batch', type=int, default=8)
    ap.add_argument('--n_scenarios', type=int, default=0, help='>0 时只跑分层核心集（PLAN §7 降级方案）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--selfcheck', action='store_true', help='只做打分一致性自检后退出')
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda').eval()

    eos = set()
    for src in (tok.eos_token_id, model.generation_config.eos_token_id):
        if src is None:
            continue
        eos.update([src] if isinstance(src, int) else src)
    eos_t = torch.tensor(sorted(eos), device='cuda')

    csv_name = 'queries_time_t1.csv' if args.dataset == 'time_t1' else 'basic_time_t.csv'
    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', csv_name))
    if args.n_scenarios:
        df = scenario_subset(df, args.n_scenarios)
    print(f'[data] rows={len(df)} scenarios={df["dataset_id"].nunique()}', flush=True)

    ct_kwargs = {'enable_thinking': False}  # used by the Qwen3 template only, ignored elsewhere (same as v1)

    def render(prompts):
        return [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                        add_generation_prompt=True, **ct_kwargs) for p in prompts]

    if args.selfcheck:
        # What is tested is the property BoN actually relies on: scoring noise must be
        # small enough not to change "which chain wins".
        # (Note: the absolute single-vs-batched scoring difference is not a good criterion —
        #  bf16 GEMM accumulates in a different order at different batch sizes; a ~380-token
        #  chain already differs by ~0.4 nat, while the between-chain gap is ~19 nat.)
        import numpy as np
        noise, gaps, flips, tot = [], [], 0, 0
        for r in range(2):
            enc = tok(render([build_prompt(df['questions'].iloc[r], 'cot')]),
                      return_tensors='pt', padding=True).to('cuda')
            n_pre = enc['input_ids'].shape[1]
            torch.manual_seed(0)
            seqs = model.generate(**enc, max_new_tokens=384, do_sample=True,
                                  temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                  num_return_sequences=8, pad_token_id=tok.pad_token_id)
            gl = seqs[:, n_pre:]
            is_eos = torch.isin(gl, eos_t)
            keep = (is_eos.cumsum(1) - is_eos.long()) == 0
            attn = torch.cat([enc['attention_mask'].repeat_interleave(8, 0), keep.long()], 1)
            ref = None
            for b in (8, 3, 1):
                v = ((score_chains(model, seqs, attn, n_pre, b) * keep).sum(1)).float().cpu().numpy()
                if ref is None:
                    ref = v
                    srt = np.sort(v)[::-1]
                    gaps.append(srt[0] - srt[1])
                else:
                    noise.extend(np.abs(v - ref).tolist())
                    flips += int(np.argmax(v) != np.argmax(ref))
                    tot += 1
        ok = np.min(gaps) > 5 * max(np.max(noise), 1e-9) and flips == 0
        print(f'[selfcheck] 打分噪声 max|Δ|={np.max(noise):.3f} nat；'
              f'链间 top1−top2 gap min={np.min(gaps):.2f} nat；重打分改变胜出链 {flips}/{tot}')
        print('[selfcheck] PASS — 噪声不影响 BoN 排序' if ok
              else '[selfcheck] FAIL — 噪声与链间差同量级，BoN 排序不可信')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    for method in args.methods.split(','):
        out_path = os.path.join(args.out_dir, f'{args.dataset}_{safe}_{method}_N{args.n_samples}.jsonl')
        done = set()
        if args.resume and os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)['idx'])
                    except (json.JSONDecodeError, KeyError):
                        pass
        todo = df[~df.index.isin(done)]
        print(f'[{method}] todo={len(todo)} (done={len(done)})', flush=True)
        from tqdm import tqdm
        with open(out_path, 'a' if done else 'w') as fout:
            for start in tqdm(range(0, len(todo), args.prompt_batch), desc=f'{safe} {method} N{args.n_samples}'):
                batch = todo.iloc[start:start + args.prompt_batch]
                prompts = [build_prompt(q, method) for q in batch['questions']]
                enc = tok(render(prompts), return_tensors='pt', padding=True).to('cuda')
                n_pre = enc['input_ids'].shape[1]
                torch.manual_seed(args.seed * 100003 + int(batch.index[0]))
                seqs = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                      do_sample=True, temperature=args.temperature,
                                      top_k=args.top_k, top_p=args.top_p,
                                      num_return_sequences=args.n_samples,
                                      pad_token_id=tok.pad_token_id)
                gl = seqs[:, n_pre:]
                is_eos = torch.isin(gl, eos_t)
                keep = (is_eos.cumsum(1) - is_eos.long()) == 0     # keep up to and including the first EOS
                attn = torch.cat([enc['attention_mask'].repeat_interleave(args.n_samples, 0),
                                  keep.long()], 1)
                lp = score_chains(model, seqs, attn, n_pre, args.score_batch)
                sum_lp = (lp * keep).sum(1)
                ntok = keep.sum(1).clamp(min=1)
                texts = tok.batch_decode(gl, skip_special_tokens=True)
                for bi, (idx, row) in enumerate(batch.iterrows()):
                    sl = slice(bi * args.n_samples, (bi + 1) * args.n_samples)
                    chains = [{'text': t, 'extracted': get_final_answer(t),
                               'sum_lp': round(float(s), 4), 'ntok': int(n),
                               'mean_lp': round(float(s) / int(n), 6)}
                              for t, s, n in zip(texts[sl], sum_lp[sl], ntok[sl])]
                    fout.write(json.dumps({
                        'idx': int(idx), 'dataset_id': row['dataset_id'],
                        'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
                        'intent': row['dataset_id'].split('-')[-1],
                        'agreement_lv': int(row['agreement_lv']),
                        'ground_truth': row['ground_truth'],
                        'model': args.model_name, 'method': method,
                        'n_samples': args.n_samples, 'temperature': args.temperature,
                        'top_k': args.top_k, 'top_p': args.top_p, 'seed': args.seed,
                        'prompt': prompts[bi], 'chains': chains,
                    }, ensure_ascii=False) + '\n')
                fout.flush()
        print(f'[done] {safe} {method} N{args.n_samples} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
