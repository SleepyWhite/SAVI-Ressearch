#!/usr/bin/env python
"""R1/R2 hint ladder: can a one-sentence "license / pointer" unlock the relation judgment?

============================================================================
What it is meant to answer
============================================================================
Our conclusion is "the state is not determined by the text" (VERDICT_pairwise_2afc §2–3:
the only thing that recovers it is sentence length, a construction trace). **This
experiment is a falsification test of that conclusion**, not a confirmation:

  If a single **instruction** (giving no answer, only a license or a pointer) produces a
  large gain → the information was in the context all along, just never invoked → our
  "information absent" conclusion must be rewritten, and SAVI regains an entry point here.
  If it cannot → what is missing is not the invocation but the information itself.

Both ends of the ladder are already anchored: bottom = R0 zero-shot (BREU 0.494–0.497),
top = L2 state given (0.994). This script runs the middle rungs.

============================================================================
Arms (wording frozen verbatim, sha in HINT_SHA256; changing one character requires a new
sha plus a changelog entry)
============================================================================
  none      R0 baseline. **Used only for the identity check** — this script's --arm none
            output must be byte-identical to the existing outputs/generative/*_dp.jsonl,
            otherwise the harness has drifted and R0 cannot serve as the paired baseline.
            This is the CLAUDE.md-required "identity cell == anchor" check.
  license   R1 (user-proposed): license the defeasible reading. **Symmetric** — says only
            "may be overturned", never when.
  strict    R1's **reverse control** (the most important gate): says the conditionals are
            strict and the conclusion cannot be overturned. If BREU is unchanged while
            BU/BM swap symmetrically → this kind of hint only moves the threshold and
            transmits no information. Isomorphic to L2's swap control (which once caught
            a false positive).
  pragmatic R2 (user-proposed): points at the pragmatic layer, **without naming the two
            readings**. ⚠️ Naming them degenerates into L5 (explicit in-context choice,
            already judged negative at 0.545).
  placebo   An instruction of equal weight but no information. Guards against "any extra
            instruction changes behavior".
  loose     Stem rewrite (**added by this script's author, not user-proposed**): change
            "What necessarily had to follow" to "What would follow".
            Rationale: the BU items' gold is (c), yet from {p→q, p, r→q} q **does** follow
            necessarily — the stem's "necessarily" directly licenses the reading that makes
            the gold wrong. If swapping this one word flips the result, a fair share of
            this benchmark's difficulty comes from a single word. This is the only arm that
            touches the stem; flagged separately.

============================================================================
Preregistration (frozen before the run)
============================================================================
[Main readout] **BREU** (= (BU-Acc + BM-Acc)/2), plus the four cells by modus. **Not BU-Acc.**
[Expectation, stated up front] license's ΔBREU falls within ±0.02 (0 included), with
    **BU significantly up, BM significantly down**. Basis: in §6.1b the prompt-style
    interventions (CoT/PS) on lv=5 had all 8 CIs covering 0; L5, which wrote both readings
    explicitly into the context, got only 0.545, and this arm is weaker than L5.
    pragmatic expected ΔBREU < 0.05.
[Gates]
  G-deg  degenerate-solution detection: each arm reports the fraction of "answer c on every
         item". The 0.5B pattern of BU 0.9953 / BM 0.0015 / BREU 0.498 must be caught —
         **looking at BU alone would read it as a big success**.
  G-rev  reverse arm: if license and strict have ΔBREU of the same sign and magnitude →
         judged a placebo effect.
  G-len  length gate (mandatory if few-shot arms are added later): this round's R1/R2
         introduce no new text features, but the length stratification is still recomputed
         by scenario (full 872 has all four bins usable; lv=5 does not, see §6.11).
  G-fmt  rows without `final answer` count as wrong per the paper protocol, but are
         **reported separately** (R7).
[Statistics] statistical unit = scenario; paired bootstrap (the same batch of scenarios
    drawn simultaneously for both arms), 5000 iterations seed=0, same method as
    summarize_generative_ci.py. **Overlapping intervals ≠ difference includes 0.**

Usage
  python scripts/run_hint_ladder.py --dry_render
  CUDA_VISIBLE_DEVICES=0 python scripts/run_hint_ladder.py --model_name Qwen/Qwen2.5-7B-Instruct --arm license
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generative import BELIEF_REV, FORMATTING, get_final_answer  # noqa: E402

# ---------------------------------------------------------------- wording (frozen verbatim)
HINTS = {
    'none': None,
    'license': ('The conditional statements below are commonsense generalizations, not strict '
                'logical rules — a conclusion drawn from them may be overturned by the '
                'information that follows. Judge using common sense.'),
    'strict': ('The conditional statements below are strict logical rules, not commonsense '
               'generalizations — a conclusion drawn from them cannot be overturned by the '
               'information that follows. Judge using strict logic.'),
    'pragmatic': 'Before answering, consider why the speaker included the third statement.',
    'placebo': ('Read the statements below carefully and in full before answering, and take '
                'your time to consider each one of them in turn.'),
    'loose': None,          # no added sentence; rewrites the stem (see STEM_FROM/STEM_TO)
}
STEM_FROM = 'What necessarily had to follow'
STEM_TO = 'What would follow'

HINT_SHA256 = hashlib.sha256(
    json.dumps({k: v for k, v in HINTS.items()}, sort_keys=True).encode('utf-8')
    + (STEM_FROM + STEM_TO).encode('utf-8')).hexdigest()


def build_prompt(question, arm):
    """R0 is `question + '\\n\\n' + FORMATTING` (the dp branch of run_generative.build_prompt).

    Each arm makes exactly **one** change: sentence arms prepend one line; the loose arm
    swaps that single word in the stem. Everything else is byte-identical — otherwise R0
    cannot serve as the paired baseline.
    """
    if arm == 'loose':
        if STEM_FROM not in question:
            raise ValueError('题干不含预期字符串，loose 臂作废')
        question = question.replace(STEM_FROM, STEM_TO)
    text = question + '\n\n' + FORMATTING
    if HINTS[arm] is not None:
        text = HINTS[arm] + '\n\n' + text
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name')
    ap.add_argument('--arm', default='license', choices=list(HINTS))
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      '..', 'outputs', 'hint_ladder'))
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry_render', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv'))

    if args.dry_render:
        print(f'HINT_SHA256 = {HINT_SHA256}\n')
        for a in HINTS:
            p = build_prompt(df.loc[0, 'questions'], a)
            extra = len(p) - len(build_prompt(df.loc[0, 'questions'], 'none'))
            print(f'{"#" * 30} arm={a}   （相对 R0 多 {extra} 字符）{"#" * 30}')
            print(p)
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

    if args.limit:
        df = df.head(args.limit)
    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    out_path = os.path.join(args.out_dir, f'time_t1_{safe}_{args.arm}.jsonl')

    with open(out_path, 'w') as fout:
        for st in tqdm(range(0, len(df), args.batch_size), desc=f'{safe} {args.arm}'):
            batch = df.iloc[st:st + args.batch_size]
            prompts = [build_prompt(q, args.arm) for q in batch['questions']]
            # chat template aligned verbatim with run_generative (single user turn + enable_thinking=False)
            texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
                     for p in prompts]
            enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gens = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)

            for (idx, row), gen in zip(batch.iterrows(), gens):
                has_fa = 'final answer' in gen.lower()
                ans = get_final_answer(gen) if has_fa else ''
                fout.write(json.dumps({
                    'idx': int(idx), 'dataset_id': row['dataset_id'], 'modus': row['modus'],
                    'agreement_lv': int(row['agreement_lv']),
                    'ground_truth': row['ground_truth'],
                    'is_bu': row['ground_truth'] == 'c',
                    'arm': args.arm, 'model': args.model_name, 'hint_sha': HINT_SHA256,
                    'raw_output': gen, 'extracted': ans,
                    'has_final_answer': has_fa,
                    'correct': ans == row['ground_truth'],
                }, ensure_ascii=False) + '\n')
            fout.flush()
    print(f'[done] {out_path}  n={len(df)}  sha={HINT_SHA256[:12]}')


if __name__ == '__main__':
    main()
