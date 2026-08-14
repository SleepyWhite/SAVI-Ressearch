#!/usr/bin/env python
"""[ORACLE] arm: given the correct state (as a conditional statement), can the model do the deterministic readout?

Positioning (do not misquote)
- The conditional statement is back-derived from the gold label; **it is a ceiling/sanity
  check and does not enter the main table** (PLAN §8).
- It does not answer "SAVI works"; it answers "does the step after the fork go through".
  L1 already refuted the fork step itself.

State representation
- Parse p, r, q out of γ1 / γ3, and **replace** the original premises with one explicit
  conditional statement + γ2 + the question.
- REQ (gold c) uses the necessity frame: q needs p and r to hold together; p alone is not
  enough.
- ALT (gold a/b) uses the sufficiency frame: p alone suffices, and r alone also suffices.
  The **asymmetry between the two frames is required**: if ALT were also written as the
  necessity q→(p∨r), then under tollens ¬p cannot be derived from ¬q, gold b becomes
  underivable, and half the ALT data would be silently mis-measured.

Readout (the key design)
- Feed both the gold condition and the **swap condition** (AND↔OR swapped); the readout is
  the **condition-following rate**: does the model's answer track the condition fed in
  (AND→c; OR→a(ponens)/b(tollens)), without asking whether that condition is right.
  Feeding only gold and reading accuracy would conflate "follows the state" with "would
  have gotten it anyway".
- Three scoring routes: option-text sum / option-text mean / single-token forced choice
  (a|b|c after `Final Answer [`). The first two are known to fail to pick c without a
  condition (0/9/7/7 times across four models); the single-token route is added so that on
  failure we can distinguish "readout failure" from "scoring-method failure".
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generative import BELIEF_REV, FORMATTING  # noqa: E402

WORDINGS = ('entry', 'prose')


def parse_cond(s):
    s = s.strip().rstrip('.')
    if not s.lower().startswith('if '):
        return None
    parts = re.split(r',\s*then\s+|\s+then\s+', s[3:], maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else None


def condition_text(wording, conn, p, r, q):
    """conn='and' → REQ (necessity frame); conn='or' → ALT (sufficiency frame)."""
    if wording == 'entry':
        if conn == 'and':
            return (f'Condition for «{q}»: «{p}» AND «{r}» '
                    f'(both are required; neither one on its own is enough).')
        return (f'Condition for «{q}»: «{p}» OR «{r}» '
                f'(either one on its own is enough).')
    if conn == 'and':
        return (f'The fact that {p} does not by itself bring it about that {q}; '
                f'that happens only if in addition {r}.')
    return (f'The fact that {p} by itself brings it about that {q}; '
            f'and the fact that {r} by itself also brings it about that {q}.')


@torch.inference_mode()
def batch_logprob(model, tok, prefixes, conts, bs):
    """(sum, mean, ntok) of the continuation span. Left padding + explicit position_ids."""
    out = []
    for i in range(0, len(prefixes), bs):
        pres, cs = prefixes[i:i + bs], conts[i:i + bs]
        pre = [tok(x, add_special_tokens=False).input_ids for x in pres]
        full = [tok(x + y, add_special_tokens=False).input_ids for x, y in zip(pres, cs)]
        for a, b in zip(pre, full):
            assert b[:len(a)] == a, 'prefix/continuation 分词边界合并'
        L = max(len(f) for f in full)
        ids = torch.full((len(full), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(full), L), dtype=torch.long)
        for j, f in enumerate(full):
            ids[j, L - len(f):] = torch.tensor(f)
            attn[j, L - len(f):] = 1
        ids, attn = ids.to('cuda'), attn.to('cuda')
        pos = (attn.cumsum(-1) - 1).masked_fill(attn == 0, 1)
        lg = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits[:, :-1].float()
        lp = lg.gather(2, ids[:, 1:].unsqueeze(2)).squeeze(2) - lg.logsumexp(-1)
        for j, (a, f) in enumerate(zip(pre, full)):
            n = len(f) - len(a)
            s = float(lp[j, L - 1 - n:].sum())
            out.append((s, s / n, n))
    return out


@torch.inference_mode()
def batch_letter(model, tok, prefixes, letters, bs):
    """One forward pass to get the logprobs of the a/b/c tokens after the prefix."""
    ids_of = [tok(l, add_special_tokens=False).input_ids[0] for l in letters]
    out = []
    for i in range(0, len(prefixes), bs):
        enc = [tok(x, add_special_tokens=False).input_ids for x in prefixes[i:i + bs]]
        L = max(len(e) for e in enc)
        ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(enc), L), dtype=torch.long)
        for j, e in enumerate(enc):
            ids[j, L - len(e):] = torch.tensor(e)
            attn[j, L - len(e):] = 1
        ids, attn = ids.to('cuda'), attn.to('cuda')
        pos = (attn.cumsum(-1) - 1).masked_fill(attn == 0, 1)
        lg = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits[:, -1].float()
        lg = lg - lg.logsumexp(-1, keepdim=True)
        for j in range(len(enc)):
            out.append([float(lg[j, t]) for t in ids_of])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'outputs', 'probe_oracle'))
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--context_mode', default='replace', choices=['replace', 'keep'],
                    help='replace=条件语句取代 γ1/γ3（首轮口径）；'
                         'keep=保留 γ1/γ2/γ3 再追加条件语句（用于分辨 tollens 塌陷是否'
                         '由删除 If-then 表面形式造成）')
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda').eval()

    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv'))
    if args.limit:
        df = df.head(args.limit)

    recs, meta = [], []
    for i, row in df.iterrows():
        L = row['questions'].split('\n')
        c1, c3 = parse_cond(L[0]), parse_cond(L[2])
        p, q = c1
        r, q3 = c3
        premises = '\n'.join(L[:3])
        gold_conn = 'and' if row['ground_truth'] == 'c' else 'or'
        recs.append({'idx': int(i), 'dataset_id': row['dataset_id'],
                     'base': row['dataset_id'].split('-')[0],
                     'intent': row['dataset_id'].split('-')[-1], 'modus': row['modus'],
                     'agreement_lv': int(row['agreement_lv']),
                     'ground_truth': row['ground_truth'], 'gold_conn': gold_conn,
                     'q_mismatch': q.lower() != q3.lower(), 'model': args.model_name})
        meta.append((p, r, q, L[1], row['a'], row['b'], row['c'], premises))

    def user_msg(cond, gamma2, opts, premises):
        if args.context_mode == 'keep':
            head = f'{premises}\n\nAdditional clarification of how these fit together: {cond}'
        else:
            head = f'{cond}\n{gamma2}'
        return (f'{head}\n\nWhat necessarily had to follow assuming that the above '
                f'premises were true?\n(a) {opts[0]}\n(b) {opts[1]}\n(c) {opts[2]}\n\n{FORMATTING}')

    def chat(u, assistant_prefix=''):
        t = tok.apply_chat_template([{'role': 'user', 'content': u}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        return t + assistant_prefix

    for wording in WORDINGS:
        for conn in ('and', 'or'):
            users = [user_msg(condition_text(wording, conn, p, r, q), g2, (oa, ob, oc), prem)
                     for (p, r, q, g2, oa, ob, oc, prem) in meta]
            # option-text scoring for the three options
            for k, oi in (('a', 4), ('b', 5), ('c', 6)):
                sc = batch_logprob(model, tok, [chat(u) for u in users],
                                   [str(m[oi]) for m in meta], args.batch)
                for rec, (s, mn, n) in zip(recs, sc):
                    rec[f'{wording}_{conn}_txt_{k}_sum'] = round(s, 4)
                    rec[f'{wording}_{conn}_txt_{k}_mean'] = round(mn, 6)
            # single-token forced choice
            lt = batch_letter(model, tok, [chat(u, 'Final Answer [') for u in users],
                              ['a', 'b', 'c'], args.batch)
            for rec, v in zip(recs, lt):
                for k, x in zip('abc', v):
                    rec[f'{wording}_{conn}_letter_{k}'] = round(x, 6)
            for rec in recs:
                for mode in ('txt_sum', 'txt_mean', 'letter'):
                    key = (lambda k: f'{wording}_{conn}_txt_{k}_sum') if mode == 'txt_sum' else (
                        (lambda k: f'{wording}_{conn}_txt_{k}_mean') if mode == 'txt_mean'
                        else (lambda k: f'{wording}_{conn}_letter_{k}'))
                    rec[f'{wording}_{conn}_pred_{mode}'] = max('abc', key=lambda k: rec[key(k)])
            print(f'  [{args.model_name}] {wording}/{conn} done', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    out_path = os.path.join(args.out_dir, f'time_t1_{safe}.jsonl' if args.context_mode == 'replace' else f'time_t1_{safe}_keep.jsonl')
    with open(out_path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] {out_path}  n={len(recs)}')


if __name__ == '__main__':
    main()
