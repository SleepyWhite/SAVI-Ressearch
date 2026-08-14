#!/usr/bin/env python
"""Arm C: two-step classification prompting (zero-training control, Qwen3-4B only). Also clears an old P0 debt.

Spec = `PREREG_probe_decode.md` §4. It replaces arm A's "probe predicts the state" with
"the model judges the state itself"; **the second step is verbatim identical** — so the
A−C difference is exactly the "representation readout vs prompted judgment" term, with
everything else aligned.

============================================================================
The two steps
============================================================================
step1  Give only the three premises and ask for the relation judgment in free generation
       (≥512 token budget). Wording register uses the project-validated `prose` tier
       ("not enough on its own" / "another way"), not the `entry` semi-formal notation
       (L2 measured prose follow rate 0.978 vs entry 0.487).
       Readout uses **digits** 1/2, non-overlapping with the answer letters a/b/c (the
       prompts/contract.py lesson: when both lines used letters the model wrote
       `Final Answer [B]` and the upstream extractor silently read `b`).
       **No placeholders**: spell out both legal forms in full so there is nothing to copy
       (same lesson: with "RELATION: N" + "fill N with 1 or 2", 47/48 copied out
       `RELATION: N2`).
       The last line is the structure line ("Then state ... by writing either ... or ..."),
       phrased to match the upstream `FORMATTING`; that phrasing has a 0 format-failure
       rate on 7B.
step2  = arm A's injection assembly **verbatim** (`run_probe_decode.build_replace_prompt`),
       with the state taken from the model's own step1 judgment: 1→REQ→conn 'and',
       2→ALT→conn 'or'.

Parse failure → fall back to the **original DP prompt** (`run_generative.build_prompt(q,'dp')`)
and record `fallback=1`. The fallback is the conservative choice: it returns that row to
the §6.1 baseline protocol and injects nothing extra.

Parsing of the RELATION line reuses `prompts/contract.parse_relation` (takes the last
occurrence, tolerates a few decorative characters, does not cross lines) — the project-wide
single source of that regex.

Usage
  python scripts/run_twostep.py --dry_render                                # CPU only
  CUDA_VISIBLE_DEVICES=7 python scripts/run_twostep.py --limit 8
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from run_generative import BELIEF_REV, build_prompt  # noqa: E402
from run_probe_decode import (CSV, TEMPLATE_SHA256, build_replace_prompt,  # noqa: E402
                              implied_answer)
from prompts.contract import parse_relation  # noqa: E402  single source of the RELATION regex
from src.prompts.utils import get_final_answer  # noqa: E402

OUT_DIR = os.path.join(HERE, '..', 'outputs', 'probe_decode')

# ------------------------------------------------------------------ step1 prompt (frozen verbatim)
STEP1_TEMPLATE = (
    '{premises}\n\n'
    'Consider how the third statement relates to the first. Either the first statement '
    'is not enough on its own to bring about the outcome, so that the third statement '
    'states a further condition that is also needed; or the first statement is enough '
    'on its own, and the third statement states another way of bringing about the same '
    'outcome.\n\n'
    'Reason briefly about which of the two it is. Then state your judgement by writing '
    'either "RELATION: 1" if the first statement is not enough on its own, or '
    '"RELATION: 2" if the first statement is enough on its own.'
)
STEP1_SHA256 = hashlib.sha256(STEP1_TEMPLATE.encode('utf-8')).hexdigest()


def build_step1(row):
    return STEP1_TEMPLATE.format(premises='\n'.join(row['questions'].split('\n')[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='Qwen/Qwen3-4B')
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--step1_max_new_tokens', type=int, default=512)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry_render', action='store_true', help='纯 CPU：打印两步渲染全文')
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    print('# 臂 C 两步分类提示（PREREG_probe_decode.md §4）')
    print(f'# model = {args.model_name}')
    print(f'# STEP1_SHA256    = {STEP1_SHA256}')
    print(f'# TEMPLATE_SHA256 = {TEMPLATE_SHA256}  （step2 注入模板，与臂 A 同一 sha）')

    if args.dry_render:
        # 2 ponens + 2 tollens: the step1 problem text is verbatim identical across the
        # two modi (the relation does not depend on γ2), but step2's γ2 and implied answer
        # differ (under tollens ALT → b), so both must be eyeballed.
        picks = ([i for i in range(len(df)) if df.iloc[i]['modus'] == 'ponens'][:2]
                 + [i for i in range(len(df)) if df.iloc[i]['modus'] == 'tollens'][:2])
        print(f'\n{"=" * 78}\n两步渲染全文（2 ponens + 2 tollens；'
              f'step2 两种解析结果与回退各渲一遍）\n{"=" * 78}')
        for i in picks:
            row = df.iloc[i]
            print(f'{"#" * 26} idx={i} dataset_id={row["dataset_id"]} modus={row["modus"]} '
                  f'lv={row["agreement_lv"]} 金标={row["ground_truth"]}')
            print('----- STEP 1 -----')
            print(build_step1(row))
            for rel, conn in (('1', 'and'), ('2', 'or')):
                print(f'----- STEP 2（若 step1 解析出 RELATION: {rel} → conn={conn} → '
                      f'蕴含答案 {implied_answer(conn, row["modus"])}）-----')
                print(build_replace_prompt(row, conn))
            print('----- 回退（step1 解析失败 → 原始 DP 提示，fallback=1）-----')
            print(build_prompt(row['questions'], 'dp'))
            print()
        print('[dry_render] 未加载语言模型，无读数。')
        return

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda').eval()

    if args.limit:
        df = df.head(args.limit)

    def gen(prompts, max_new):
        outs = []
        for st in range(0, len(prompts), args.batch):
            texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
                     for p in prompts[st:st + args.batch]]
            enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
            in_len = enc['input_ids'].shape[1]
            with torch.inference_mode():
                o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(o[:, in_len:], skip_special_tokens=True)
            for j, g in enumerate(dec):
                ids = o[j, in_len:].tolist()
                outs.append((g, bool(len(ids) >= max_new and tok.eos_token_id not in ids),
                             (ids.index(tok.eos_token_id) + 1
                              if tok.eos_token_id in ids else len(ids))))
        return outs

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'C_twostep.jsonl')
    rows = [df.iloc[i] for i in range(len(df))]

    recs = []
    for st in tqdm(range(0, len(rows), args.batch), desc='C step1'):
        sub = rows[st:st + args.batch]
        s1 = gen([build_step1(r) for r in sub], args.step1_max_new_tokens)
        for r, (g, trunc, ntok) in zip(sub, s1):
            rel = parse_relation(g)
            recs.append({'row': r, 's1_raw': g, 's1_trunc': trunc, 's1_ntok': ntok,
                         'relation': rel})

    for st in tqdm(range(0, len(recs), args.batch), desc='C step2'):
        sub = recs[st:st + args.batch]
        prompts = []
        for e in sub:
            if e['relation'] is None:
                prompts.append(build_prompt(e['row']['questions'], 'dp'))
            else:
                prompts.append(build_replace_prompt(
                    e['row'], 'and' if e['relation'] == '1' else 'or'))
        for e, p, (g, trunc, ntok) in zip(sub, prompts, gen(prompts, args.max_new_tokens)):
            e.update(s2_prompt=p, s2_raw=g, s2_trunc=trunc, s2_ntok=ntok)

    with open(out_path, 'w') as f:
        for e in recs:
            row, rel = e['row'], e['relation']
            conn = None if rel is None else ('and' if rel == '1' else 'or')
            ext = get_final_answer(e['s2_raw'])
            imp = None if conn is None else implied_answer(conn, row['modus'])
            f.write(json.dumps({
                'arm': 'C', 'idx': int(row.name), 'dataset_id': row['dataset_id'],
                'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
                'agreement_lv': int(row['agreement_lv']),
                'ground_truth': row['ground_truth'],
                'model': args.model_name, 'step1_sha': STEP1_SHA256,
                'template_sha': TEMPLATE_SHA256,
                'relation': rel, 'pred_state': None if rel is None else
                             ('REQ' if rel == '1' else 'ALT'),
                'fed_conn': conn, 'implied': imp,
                'fallback': int(rel is None),
                'gold_state': 'REQ' if row['ground_truth'] == 'c' else 'ALT',
                'step1_raw': e['s1_raw'], 'step1_truncated': e['s1_trunc'],
                'step1_n_new_tokens': e['s1_ntok'],
                'step2_prompt': e['s2_prompt'], 'raw_output': e['s2_raw'],
                'extracted': ext, 'has_final_answer': 'final answer' in e['s2_raw'].lower(),
                'n_new_tokens': e['s2_ntok'], 'truncated': e['s2_trunc'],
                'follows': None if imp is None else (ext == imp),
                'correct': ext == row['ground_truth'],
            }, ensure_ascii=False) + '\n')
    n_fb = sum(1 for e in recs if e['relation'] is None)
    print(f'[done] {out_path}  n={len(recs)}  fallback={n_fb}  '
          f'step1_sha={STEP1_SHA256[:12]}')


if __name__ == '__main__':
    main()
