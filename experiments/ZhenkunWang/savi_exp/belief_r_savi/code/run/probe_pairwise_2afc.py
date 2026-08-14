#!/usr/bin/env python
"""L6: minimal-pair pairwise comparison (2AFC) — put the two writings of γ3 side by side and ask which one is the "appended condition".

============================================================================
How it differs from L1–L5 (this is the whole point)
============================================================================
L1–L5 are all **single-item absolute judgments**: "in this one item, what is the relation
between γ3 and γ1?"
This probe is **pairwise relative ranking**: "of these two γ3s, which one means the first
condition is not enough by itself?"

The dataset's design intent is itself **pairwise-defined**: the same premises get two
third sentences, one strong (appended condition), one weak (another equivalent path).
Absolute judgment is far harder than pairwise comparison — putting the single manipulated
variable directly in opposition is the **input-form** dimension L1–L5 never covered.

**Hypothesis H-L6**: the distinction is extractable in the relative-ranking form, and only
fails to come out in the absolute-judgment form.
- Pairwise works (≥0.70) while single-item sits at 0.52 → the information is in the input
  and the model can extract it; the single-item form just cannot use it.
- Pairwise also at chance → the strongest negative evidence so far; "closed" moves from
  "how we ask" down to the **input form** level.

============================================================================
Preregistration (frozen 2026-08-02 in PLAN §3 "L6"; unchanged before the run)
============================================================================
[Main readout] pair accuracy = fraction where the model points at the strong side,
    **design-intent labels × 63 pairs × mean over the two orders**. Chance = 0.500.
    Reference lines: L5 0.545 / L1 0.588 / ceiling 0.843 / absolute ceiling 1.000.
[Secondary readout] the 39 pairs where gold serves as label (strong side gold REQ,
    weak side gold ALT).
[Criteria] ≥0.70 new finding | 0.55–0.65 same band as L1/L4/L5, the form brought nothing |
    ≤0.55 no signal, L series closed | **0.65–0.70 is not covered by the criteria; report
    honestly as uncertain, do not force a tier**.
[Four gates] order swap / lexical control arm / cross-model direction agreement / format
    compliance ≥0.8. All decision logic lives in `summarize_pairwise_2afc.py`; this script
    only produces the raw rows.

============================================================================
Two implementation-level choices (unspecified by the prereg; declared here before the run)
============================================================================
(A) **γ2 in the template varies with modus, but the relation judgment does not depend on γ2.**
    The prereg budget reads "63 pairs × 2 orders × 6 models = 756 generations", i.e. each
    pair rendered once. Hence: **main readout = ponens rendering** (γ2 = affirming the
    antecedent, verbatim identical to PLAN's example). The tollens rendering runs alongside
    as a **secondary robustness arm** (it is free: the same 63 pairs), **does not enter the
    verdict**, and is only used to check "is the relation judgment contaminated by γ2".
    A large gap between the two arms is an instrument warning.
(B) `max_new_tokens` set to **2048** (prereg requires ≥1024). In L5's first round,
    Qwen3-4B had 89.8% of rows truncated at the 512 cap — an instrument failure, not a
    capability failure. In the summary, **check truncation rate before accuracy**.

============================================================================
Budget
============================================================================
63 pairs × 2 orders × 2 modus × 6 models = 1,512 free generations. Far below L5's 20,928.

Usage
  python scripts/probe_pairwise_2afc.py --dry_render          # inspect raw prompts + pair-list audit
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_pairwise_2afc.py --model_name Qwen/Qwen3-4B
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
from prompts.pairwise import build, parse_answer, PAIRWISE_SHA256  # noqa: E402

MODELS = ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct',
          'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct',
          'Qwen/Qwen3-4B', 'meta-llama/Llama-3.1-8B-Instruct')


def load_pairs():
    """Build the true minimal pairs: strong/weak both present under the same base, and γ1 verbatim identical.

    The counts match the tally in STATUS §0 / PLAN §3 L6 item 4 digit for digit (asserted
    at the end of this function):
      strong+weak both present under same base 133 → γ1+γ2 verbatim identical 63 →
      gold separates by intent 39 → both sides lv=5 10 → both properties 6
    γ1 identical ⟺ γ1+γ2 identical: γ2 is mechanically derived from γ1's p/q (ponens uses
    p, tollens uses ¬q), so the admitted sets for the two modus are exactly the same; no
    separate filtering needed.
    """
    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv'))
    L = df['questions'].str.split('\n')
    df['g1'], df['g2'], df['g3'] = L.str[0], L.str[1], L.str[2]
    df['base'] = df['dataset_id'].str.split('-').str[0]
    df['intent'] = df['dataset_id'].str.split('-').str[-1]

    n_both = (df.groupby('base')['intent'].nunique() == 2).sum()
    pairs = []
    for base, sub in df.groupby('base'):
        if sub['intent'].nunique() != 2:
            continue
        s, w = sub[sub.intent == 'strong'], sub[sub.intent == 'weak']
        # γ1 must be unique within a scenario (shared by the twins) and verbatim identical on both sides, else not a minimal pair
        if s['g1'].nunique() != 1 or w['g1'].nunique() != 1 or s['g1'].iloc[0] != w['g1'].iloc[0]:
            continue
        for modus in ('ponens', 'tollens'):
            sm, wm = s[s.modus == modus], w[w.modus == modus]
            if not len(sm) or not len(wm):
                raise ValueError(f'base {base} 缺 {modus} 行，不该发生')
            sm, wm = sm.iloc[0], wm.iloc[0]
            # The ponens version of γ2 (= affirming the antecedent) is mechanically derived
            # from γ1, verbatim identical across the 63 pairs; but the tollens version
            # (= denying the consequent) is **hand-written upstream**, and in 3 pairs the
            # two sides differ (base 123 "because John tried" vs "because John did not try",
            # 25 "is not required to" vs "must not", 45 "Few" vs "Fewer").
            # Those 3 pairs are thus not true minimal pairs on the tollens arm —
            # **flag, don't drop**; the main readout (ponens) is unaffected, and the tollens
            # arm additionally reports a "those 3 pairs removed" version.
            g2_same = sm['g2'] == wm['g2']
            if modus == 'ponens' and not g2_same:
                raise ValueError(f'base {base} 的 ponens γ2 两侧不同，与已核事实矛盾')
            # p/q/r are parsed only for the lexical control arm (Jaccard needs q); they do not enter the prompt
            c1, c3s = parse_cond(sm['g1']), parse_cond(sm['g3'])
            if not c1 or not c3s:
                raise ValueError(f'base {base} 的 γ1/γ3 解析失败，不该发生')
            pairs.append({
                'pair_id': f'{base}-{modus}', 'base': base, 'modus': modus,
                'atomic_idx': int(sm['atomic_idx']),
                'g1': sm['g1'], 'g2': sm['g2'],
                'g3_strong': sm['g3'], 'g3_weak': wm['g3'], 'q': c1[1],
                'g2_identical': bool(g2_same),
                'lv_strong': int(sm['agreement_lv']), 'lv_weak': int(wm['agreement_lv']),
                'gold_strong': sm['ground_truth'], 'gold_weak': wm['ground_truth'],
                # gold separates by design intent = strong side REQ(c) and weak side ALT
                'gold_separates': bool(sm['ground_truth'] == 'c' and wm['ground_truth'] != 'c'),
                'lv5_both': bool(sm['agreement_lv'] == 5 and wm['agreement_lv'] == 5),
            })

    pon = [p for p in pairs if p['modus'] == 'ponens']
    assert n_both == 133, f'齐全组数 {n_both} ≠ 133，数据或口径变了'
    assert len(pon) == 63, f'真最小对 {len(pon)} ≠ 63，口径变了'
    assert sum(p['gold_separates'] for p in pon) == 39
    assert sum(p['lv5_both'] for p in pon) == 10
    assert sum(p['gold_separates'] and p['lv5_both'] for p in pon) == 6
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name')
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      '..', 'outputs', 'pairwise_2afc'))
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=2048)
    ap.add_argument('--limit', type=int, default=0, help='冒烟用：只跑前 N 组')
    ap.add_argument('--dry_render', action='store_true')
    args = ap.parse_args()

    pairs = load_pairs()

    if args.dry_render:
        pon = [p for p in pairs if p['modus'] == 'ponens']
        print(f'PAIRWISE_SHA256 = {PAIRWISE_SHA256}\n')
        print('=' * 78)
        print(f'配对清单审计：真最小对 {len(pon)} 组（主读数），'
              f'金标分开 {sum(p["gold_separates"] for p in pon)} 组，'
              f'双侧 lv=5 {sum(p["lv5_both"] for p in pon)} 组')
        # PLAN §3 L6 known limitation 4: how many atomic_idx the 63 pairs come from — must be checked before scaling up
        ai = pd.Series([p['atomic_idx'] for p in pon]).value_counts()
        print(f'atomic_idx 来源：{len(ai)} 个种子，最大一组 {ai.iloc[0]} 组，'
              f'前三 {ai.head(3).to_dict()}')
        print('=' * 78)
        for sf in (True, False):
            pr, slot = build(pon[0]['g1'], pon[0]['g2'],
                             pon[0]['g3_strong'], pon[0]['g3_weak'], sf)
            print(f"\n########## order={'strong_first' if sf else 'weak_first'} "
                  f"（strong 在 ({slot})，即正确答案 = {slot}）##########")
            print(pr)
        print('\n########## 同一组的 tollens 渲染（次要臂，只有 γ2 不同）##########')
        tol = [p for p in pairs if p['base'] == pon[0]['base'] and p['modus'] == 'tollens'][0]
        print(build(tol['g1'], tol['g2'], tol['g3_strong'], tol['g3_weak'], True)[0])
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

    # All four configs (2 orders × 2 modus) run in one process, saving 3 model loads
    items = []
    for p in pairs:
        for strong_first in (True, False):
            items.append((p, strong_first))
    if args.limit:
        keep = {p['base'] for p in pairs[:args.limit * 2]}
        items = [(p, sf) for p, sf in items if p['base'] in keep]

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    out_path = os.path.join(args.out_dir, f'pairwise_{safe}.jsonl')

    with open(out_path, 'w') as fout:
        for st in tqdm(range(0, len(items), args.batch), desc=safe):
            batch = items[st:st + args.batch]
            built = [build(p['g1'], p['g2'], p['g3_strong'], p['g3_weak'], sf)
                     for p, sf in batch]
            texts = [tok.apply_chat_template([{'role': 'user', 'content': pr}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
                     for pr, _ in built]
            enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            new = out[:, enc['input_ids'].shape[1]:]
            gens = tok.batch_decode(new, skip_special_tokens=True)
            # Truncation check: no eos appeared in this row's generated span (with left padding there is no pad on the right)
            ntok = [int((row != tok.pad_token_id).sum()) for row in new]

            for (p, sf), (_, slot), gen, nt in zip(batch, built, gens, ntok):
                picked = parse_answer(gen)
                fout.write(json.dumps({
                    'pair_id': p['pair_id'], 'base': p['base'], 'modus': p['modus'],
                    'atomic_idx': p['atomic_idx'],
                    'order': 'strong_first' if sf else 'weak_first',
                    'strong_slot': slot,
                    'gold_separates': p['gold_separates'], 'lv5_both': p['lv5_both'],
                    'g2_identical': p['g2_identical'],
                    'model': args.model_name, 'prompt_sha': PAIRWISE_SHA256,
                    'g3_strong': p['g3_strong'], 'g3_weak': p['g3_weak'], 'q': p['q'],
                    'raw_output': gen,
                    'picked_slot': picked,                       # the slot the model wrote, 1/2
                    'format_ok': picked is not None,
                    # position removed: did it point at the strong side
                    'picked_strong': None if picked is None else picked == slot,
                    'n_new_tokens': nt,
                    'truncated': nt >= args.max_new_tokens,
                }, ensure_ascii=False) + '\n')
            fout.flush()
    print(f'[done] {out_path}  n={len(items)}  sha={PAIRWISE_SHA256[:12]}')


if __name__ == '__main__':
    main()
