#!/usr/bin/env python
"""First-layer probe: does the model's likelihood recognize the REQ / ALT relation fork?

What it asks
- Belief-R's decisive step is not the answer letter but the relation of γ3 to γ1:
    REQ additional necessary condition (p∧r→q) → answer c
    ALT independent alternative route (p∨r→q) → answer a(ponens) / b(tollens)
  Once the relation is fixed, the answer follows deterministically. This script measures
  only the relation step.

How it reads
- Per item, compute Δ = logP(REQ statement | premises) − logP(ALT statement | premises)
- **The main readout is the difference of Δ between BU items and BM items, not the sign
  of Δ.** The two statements are library-wide fixed templates, so any item-independent
  lexical/polarity prior cancels term by term in the BU−BM difference; reading the sign
  alone would not measure the relation, only the model's prior over "yes/no" and
  "certain/uncertain".
- Three mutually dissimilar wordings are each run once, and the conclusions must agree;
  a signal in only one wording = lexical artifact.
- The relation is modus-independent (the γ3↔γ1 relation does not depend on what γ2 says),
  so the templates are shared across both modi.

Framing choice
- Use the chat template (user provides premises+question, assistant provides the
  statement), not bare continuation. Bare continuation is exactly the confounder behind
  the 30pt gap between the logprob-classification route and the generative route on Llama.
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generative import BELIEF_REV  # noqa: E402

# Three wordings. The REQ arm corresponds to answer c, the ALT arm to a/b.
# Within a template the two arms are kept roughly equal in length.
TEMPLATES = {
    'sufficiency': (
        'Considering all three statements, is the condition in the first statement '
        'enough on its own to bring about the outcome?',
        'No, it is not enough on its own.',        # REQ
        'Yes, it is enough on its own.',           # ALT
    ),
    'relation': (
        'How does the third statement relate to the first?',
        'It states a further condition that is also needed for the outcome.',   # REQ
        'It states a separate route that on its own leads to the outcome.',     # ALT
    ),
    'certainty': (
        'Given all three statements, is what follows certain?',
        'No, what follows is uncertain.',          # REQ
        'Yes, what follows is certain.',           # ALT
    ),
}


@torch.inference_mode()
def score_continuations(model, tok, prefixes, conts, batch):
    """Returns (sum_lp, mean_lp) per item, summing only over the continuation tokens."""
    out = []
    for i in range(0, len(prefixes), batch):
        pres, cs = prefixes[i:i + batch], conts[i:i + batch]
        pre_ids = [tok(p, add_special_tokens=False).input_ids for p in pres]
        full_ids = [tok(p + c, add_special_tokens=False).input_ids for p, c in zip(pres, cs)]
        for pi, fi in zip(pre_ids, full_ids):
            assert fi[:len(pi)] == pi, 'continuation 与 prefix 的分词在边界处合并了，切片会错位'
        L = max(len(f) for f in full_ids)
        pad = tok.pad_token_id
        ids = torch.full((len(full_ids), L), pad, dtype=torch.long)
        attn = torch.zeros((len(full_ids), L), dtype=torch.long)
        for j, f in enumerate(full_ids):                      # left padding
            ids[j, L - len(f):] = torch.tensor(f)
            attn[j, L - len(f):] = 1
        ids, attn = ids.to('cuda'), attn.to('cuda')
        pos = (attn.cumsum(-1) - 1).masked_fill(attn == 0, 1)
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits[:, :-1].float()
        lp = logits.gather(2, ids[:, 1:].unsqueeze(2)).squeeze(2) - logits.logsumexp(-1)
        for j, (pi, fi) in enumerate(zip(pre_ids, full_ids)):
            n_cont = len(fi) - len(pi)
            s = float(lp[j, L - 1 - n_cont:].sum())           # take only the continuation span
            out.append((s, s / n_cont, n_cont))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'outputs', 'probe_relation'))
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
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

    def prefix_for(question, stem):
        prem = '\n'.join(question.split('\n')[:3])
        user = f'{prem}\n\n{stem}'
        return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    out_path = os.path.join(args.out_dir, f'time_t1_{safe}.jsonl')

    recs = [{'idx': int(i), 'dataset_id': r['dataset_id'], 'base': r['dataset_id'].split('-')[0],
             'intent': r['dataset_id'].split('-')[-1], 'modus': r['modus'],
             'agreement_lv': int(r['agreement_lv']), 'atomic_idx': int(r['atomic_idx']),
             'ground_truth': r['ground_truth'], 'model': args.model_name}
            for i, r in df.iterrows()]

    from tqdm import tqdm
    for name, (stem, req, alt) in TEMPLATES.items():
        prefixes = [prefix_for(q, stem) for q in df['questions']]
        for arm, cont in (('req', req), ('alt', alt)):
            sc = score_continuations(model, tok, prefixes, [cont] * len(prefixes), args.batch)
            for rec, (s, m, n) in zip(recs, sc):
                rec[f'{name}_{arm}_sum'] = round(s, 4)
                rec[f'{name}_{arm}_mean'] = round(m, 6)
                rec[f'{name}_{arm}_ntok'] = n
            print(f'  [{safe}] {name}/{arm} done', flush=True)
        for rec in recs:
            rec[f'{name}_delta_sum'] = round(rec[f'{name}_req_sum'] - rec[f'{name}_alt_sum'], 4)
            rec[f'{name}_delta_mean'] = round(rec[f'{name}_req_mean'] - rec[f'{name}_alt_mean'], 6)

    with open(out_path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] {out_path}  n={len(recs)}')


if __name__ == '__main__':
    main()
