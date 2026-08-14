#!/usr/bin/env python
"""External-evidence probe: judge the REQ / ALT fork by the **contrast** in NLI entailment strength.

============================================================================
Why do this (background)
============================================================================
L1 judged negative: the model's own likelihood cannot recognize γ3's role
relative to γ1 (AUC 0.40–0.57, directions contradictory across three
phrasings). C2 reinforcement: on minimal pairs Qwen3-4B changes its answer due
to a γ3 change only 1.6% of the time — it basically does not read γ3. And L2
showed: **hand it the state and it will execute** (follow rate 0.877–0.995,
confirmed by the swap control). Together the three point to the only route
still standing: **don't expect the model to extract the state itself; compute
it outside the model and feed it in.**

This probe tests exactly whether that "compute it outside" step can stand. It
**involves no SAVI machinery** — no decoding, no reranking, no trellis. It only
answers a prerequisite question: does an automatically computable evidence
source outside the model's context exist for this fork.

============================================================================
What is measured (derivation chain, four steps)
============================================================================
1. AGM says revision is uniquely determined by **epistemic entrenchment** (what
   you retract first when forced to retract), and entrenchment is an **input**
   of the theory, not a product. So the question shifts from "does the model
   understand belief revision" to "where does entrenchment come from".

2. The **minimal-pair structure of this dataset forces it into a contrast
   quantity**: within a pair γ1 is verbatim identical (so the entrenchment of
   p→q is the same number), while 61.8% of pairs have separated gold labels.
   **The strength of p→q alone cannot distinguish strong/weak.** Only what γ3
   brings can.

3. Pragmatics says what to compare: the speaker has said "if p then q", then
   adds "if r then q". If r is just a comparable parallel route → supplementary
   information → ALT; if r is a clearly stronger/more specific route → the
   implicature is "p alone is not enough" → REQ. So the comparison is **how
   much stronger r is than p as a route to q**.

4. NLI entailment strength is a **proxy** for this contrast quantity (not an
   equivalent; see "bridging assumption" below):

       Δ = s(r ⊨ q) − s(p ⊨ q)          s = P(entailment)

   Prediction: large Δ → REQ (gold c); small Δ → ALT (gold a/b).

Real example (0-strong / 0-weak, γ1 verbatim identical):
    p            = John reads a book
    q            = John learns something new
    r (strong)   = Jessica explains quantum physics to John for the first time   gold c
    r (weak)     = Jessica teaches John about astronomy                          gold a
Expected s(r_strong ⊨ q) > s(r_weak ⊨ q).

============================================================================
Bridging assumption (one of the things this experiment tests, not a premise)
============================================================================
"A conditional expressing a strong, reliable entailment relation you are
unwilling to retract; one expressing a weak, defeasible relation you are always
ready to qualify." Plausible, but **an assumption**. One point in our favor:
"entailment" in NLI datasets was never strict logical entailment but "would a
normal person reading the premise infer the hypothesis" — exactly the
defeasible notion we want.

**Main risk**: NLI models are sensitive to lexical overlap / topical
similarity. If Δ is driven mainly by topical similarity, we are just re-running
a re-skinned version of the "topical relatedness" conjecture from
VERDICT_probe_relation — and that VERDICT explicitly said topical relatedness
in this dataset **may run opposite to logical role**. Hence the Jaccard control
arm (see validity item V3).

============================================================================
Pre-registration (frozen before running, unchanged afterwards)
============================================================================
Statistical unit = **scenario** (dataset_id). Ponens/tollens twins share the
same (p, r, q) (only γ2 differs), so Δ is verbatim identical — computing per
row would use each number twice. n = 872 scenarios.

Main readout set = **agreement_lv = 5** (locked design decision, STATUS §3);
the full set is secondary.

[Main test P-1] AUC(Δ; BU vs BM), lv=5 subset.
    AUC = P(Δ of a random BU scenario > Δ of a random BM scenario).
    0.5 = no signal. Same metric as L1, so the two compare directly.

[Main test P-2] minimal-pair paired sign test. Within a pair p and q are the
    same, so Δ_strong − Δ_weak = s(r_strong⊨q) − s(r_weak⊨q); the p terms
    cancel completely — this is the cleanest form.
    **Actual n verified (2026-08-01, before running)**: content pairs 63 /
    gold separated **39** / both sides lv=5 only **6**.

    Power must be stated up front, so it can't become a post-hoc lifeline:
      - n=39: p<0.05 needs 27/39 in the same direction (computed p=0.0237).
        Usable, but only sensitive to large effects.
      - n=6:  **even 6/6 gives only p=0.031** — that is not a test, it is a
        display. The lv=5 pair tier is **qualitative reference only and enters
        no judgement**.
    So P-2 is a confirmation item, not a main item; the main judgement is P-1.
    No subset selection.

[Consistency gate] P-1 and P-2 **must agree in direction**, and **≥2 NLI
    models must agree in direction**. This is L1's direct lesson: a single
    instrument produces phrasing/model artifacts; contradictory directions
    across three phrasings were exactly the signature. Direction mismatch →
    judged negative; picking one to report is not allowed.

[Diagnostic D-1] AUC(Δ; intent=strong vs weak), i.e. computed once against the
    **design intent** rather than the gold label. Human annotators separate
    only 47.4% of pairs according to intent, so failure against gold may be
    label noise rather than missing signal. If D-1 is clearly above P-1, the
    conclusion is "signal present, labels dirty" — a different thing from "no
    signal"; the two must be reported separately.

============================================================================
Criteria (frozen kill-criteria)
============================================================================
Judged on P-1 over the lv=5 subset, and the consistency gate must pass:

  AUC ≥ 0.65 and beats the Jaccard control by ≥ 0.05 → **actionable**: this
                                              external evidence source stands;
                                              only then discuss wiring it into SAVI
  0.55–0.65                                → **signal but unusable**: BU baseline
                                              0.015; this strength cannot drive
                                              decoding. Record, don't build
  0.45–0.55                                → **no signal**, close the case
  < 0.45 (significantly reversed)          → **topical-similarity signature**.
                                              Report the negative result;
                                              **no sign flipping** — a post-hoc
                                              sign flip is p-hacking, and this
                                              line is written here to block it

============================================================================
Validity items (instrument self-checks, all run in summarize)
============================================================================
V1 twin identity: ponens/tollens must parse to verbatim identical (p, r, q).
        Asserted inside this script.
V2 smoke: a set of hand-written known NLI items (strong entailment / unrelated
        / contradiction); scores must rank in order. It catches hard faults
        like a reversed label mapping or a wrongly loaded model.
V3 Jaccard control: Δ_J = J(r,q) − J(p,q), pure lexical overlap, free on CPU.
        If AUC(Δ) ≈ AUC(Δ_J), NLI provides nothing beyond surface overlap → negative.
V4 length control: correlation of Δ with len(r) − len(p). r is usually longer
        and more specific; the trivial explanation "NLI prefers long premises"
        must be ruled out.

============================================================================
Budget
============================================================================
872 scenarios × 2 premises = 1,744 NLI forwards, very short sequences
(~30 tokens), model ~400M. Seconds at batch=64 on GPU; **total time is
dominated by the weight download** (bart-large-mnli ~1.6 GB). The local HF
cache has **no** MNLI weights (only roberta-base, not the NLI fine-tune), so a
download is needed first.

============================================================================
Expectation written up front (registered 2026-08-01, before running; no post-hoc edits)
============================================================================
Most likely lands in the **"no signal" or "signal but unusable"** tier
(P-1 AUC 0.52–0.62). Reasons:
  - Only 39/63 = 61.9% of content pairs have gold labels separated according
    to design intent — **the labels themselves only weakly reflect the
    designed contrast**, so AUC against gold is naturally capped by that noise
    layer.
  - Both γ3s are given in the form "If r then q", so r⊨q is high on both
    sides and the contrast may be squeezed near the ceiling.
  - L1 already showed likelihood-type probes chase topical relatedness, which
    in this dataset may run opposite to logical role.
Expected: **D-1 (against design intent) clearly above P-1 (against gold)**. If
that holds, the conclusion is "signal present, labels dirty" — a finding about
the **benchmark**, not about the method; the two must be written separately.
This is also the most likely valuable output of this probe.

**Even a high P-1 only says "an evidence source outside the model exists"; it
does not say SAVI works.** Decoding must still be built and tested separately.
This sentence is written here to prevent the probe result being reported as a
method result after the fact.

Usage
  python scripts/probe_nli_entrenchment.py --smoke_only          # V2 smoke only
  python scripts/probe_nli_entrenchment.py                       # bart-large-mnli
  python scripts/probe_nli_entrenchment.py --model_name roberta-large-mnli
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_generative import BELIEF_REV  # noqa: E402
from probe_oracle_state import parse_cond  # noqa: E402

# V2 smoke items: (premise, hypothesis, expected tier). Tiers only require **relative
# order**, no absolute thresholds — calibration varies a lot across NLI models, and
# absolute thresholds would produce false kills.
SMOKE = [
    ('John reads a book', 'John reads something', 'high'),
    ('Jessica explains quantum physics to John', 'John learns something new', 'high'),
    ('John reads a book', 'The weather is cold', 'low'),
    ('John reads a book', 'John does not read a book', 'low'),
]

STOP = {'a', 'an', 'the', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been',
        'that', 'this', 'it', 'for', 'and', 'or', 'in', 'on', 'at', 'by', 'with',
        'as', 'from', 'his', 'her', 'their', 'its', 'they', 'he', 'she'}


def toks(s):
    return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP}


def jaccard(a, b):
    ta, tb = toks(a), toks(b)
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def sentence(s):
    """Turn a clause cut out of an if-then into a full sentence."""
    s = s.strip().rstrip('.')
    return (s[0].upper() + s[1:] + '.') if s else s


@torch.inference_mode()
def nli_scores(model, tok, pairs, entail_id, batch):
    """Return the full three-class probability table [contradiction?, neutral?, entail] for each (premise, hypothesis) pair."""
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk], return_tensors='pt',
                  padding=True, truncation=True, max_length=128).to(model.device)
        probs = model(**enc).logits.float().softmax(-1).cpu()
        out.extend(probs.tolist())
    return [(row[entail_id], row) for row in out]


def load_scenarios(limit):
    """Deduplicate by scenario; also runs the V1 twin-identity assertion."""
    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv'))
    seen = {}
    for _, row in df.iterrows():
        L = row['questions'].split('\n')
        c1, c3 = parse_cond(L[0]), parse_cond(L[2])
        if not c1 or not c3:
            raise ValueError(f"{row['dataset_id']} 的 γ1/γ3 解析失败，不该发生")
        (p, q), (r, _) = c1, c3
        key = row['dataset_id']
        cur = {'dataset_id': key, 'base': key.split('-')[0], 'intent': key.split('-')[-1],
               'p': p, 'r': r, 'q': q, 'ground_truth': row['ground_truth'],
               'agreement_lv': int(row['agreement_lv'])}
        if key in seen:                                   # V1 twin identity
            old = seen[key]
            for f in ('p', 'r', 'q'):
                assert old[f] == cur[f], (
                    f'V1 失败：{key} 的孪生解析出不同的 {f}\n  {old[f]!r}\n  {cur[f]!r}')
            # gold labels derive across twins by rule (ponens-a→tollens-b), c↔c; both sides are BU or both BM
            assert (old['ground_truth'] == 'c') == (cur['ground_truth'] == 'c'), \
                f'V1 失败：{key} 的孪生 BU/BM 归属不一致'
        else:
            seen[key] = cur
    rows = list(seen.values())
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='facebook/bart-large-mnli')
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'outputs', 'probe_nli'))
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--local_files_only', action='store_true',
                    help='权重已在本地缓存时加上，避免联网')
    ap.add_argument('--smoke_only', action='store_true', help='只跑 V2 冒烟就退出')
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, local_files_only=args.local_files_only).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # read the label mapping from config, don't hardcode — id order differs across NLI models; hardcoding would silently measure the reverse
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    entail = [k for k, v in id2label.items() if 'entail' in v]
    assert len(entail) == 1, f'config.id2label 里找不到唯一的 entailment 类: {id2label}'
    entail_id = entail[0]
    print(f'[labels] {id2label}   entail_id={entail_id}', flush=True)

    # ---- V2 smoke ----
    sm = nli_scores(model, tok, [(sentence(p), sentence(h)) for p, h, _ in SMOKE],
                    entail_id, args.batch)
    print('[V2 冒烟] 期望 high 档全部高于 low 档：')
    for (p, h, lvl), (s, _) in zip(SMOKE, sm):
        print(f'   {lvl:<5} {s:.3f}   {p}  ⊨  {h}')
    hi = [s for (_, _, lvl), (s, _) in zip(SMOKE, sm) if lvl == 'high']
    lo = [s for (_, _, lvl), (s, _) in zip(SMOKE, sm) if lvl == 'low']
    ok = min(hi) > max(lo)
    print(f'[V2 冒烟] {"通过" if ok else "**失败**"}  min(high)={min(hi):.3f} '
          f'max(low)={max(lo):.3f}')
    if not ok:
        raise SystemExit('V2 冒烟未通过——标签映射或模型加载有问题，先修再跑正式的')
    if args.smoke_only:
        return

    # ---- main scoring ----
    rows = load_scenarios(args.limit)
    print(f'[V1 孪生恒等] 通过；场景数 n={len(rows)}', flush=True)

    pairs = []
    for x in rows:
        pairs.append((sentence(x['p']), sentence(x['q'])))
        pairs.append((sentence(x['r']), sentence(x['q'])))
    sc = nli_scores(model, tok, pairs, entail_id, args.batch)

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    out_path = os.path.join(args.out_dir, f'time_t1_{safe}.jsonl')
    with open(out_path, 'w') as f:
        for i, x in enumerate(rows):
            s_p, full_p = sc[2 * i]
            s_r, full_r = sc[2 * i + 1]
            f.write(json.dumps({
                **x,
                'model': args.model_name,
                's_p': round(s_p, 6), 's_r': round(s_r, 6),
                'delta': round(s_r - s_p, 6),
                'probs_p': [round(v, 6) for v in full_p],
                'probs_r': [round(v, 6) for v in full_r],
                'id2label': id2label,
                # V3 lexical-overlap control (free on CPU)
                'jac_p': round(jaccard(x['p'], x['q']), 6),
                'jac_r': round(jaccard(x['r'], x['q']), 6),
                'delta_jac': round(jaccard(x['r'], x['q']) - jaccard(x['p'], x['q']), 6),
                # V4 length control
                'len_p': len(x['p'].split()), 'len_r': len(x['r'].split()),
            }, ensure_ascii=False) + '\n')
    print(f'[done] {out_path}  n={len(rows)}')


if __name__ == '__main__':
    main()
