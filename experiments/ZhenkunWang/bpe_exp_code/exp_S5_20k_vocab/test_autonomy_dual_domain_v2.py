#!/usr/bin/env python3
"""V2: Error detection on BOTH real data AND model-generated data, plus
forced-error generation + error correction.

═══════════════════════════════════════════════════════════════════════════
PART A — Real-data detection (three domains, same as v1)
═══════════════════════════════════════════════════════════════════════════
  A1: 45M forward on forward-tok real data
  A2: 45M backward (backrule) on backrule-tok real data
  A3: 45M backward (tokflip) on forward-tok real data

  Error injection (same for all): plausible (from reference model top-20)
  or random (frequent token).

═══════════════════════════════════════════════════════════════════════════
PART B — Generated-text post-hoc error detection
═══════════════════════════════════════════════════════════════════════════
  B1: 124M forward generates text greedily → inject errors afterwards →
      45M forward detects
  B2: 124M backward (backrule) generates text greedily → inject errors →
      45M backward (backrule) detects

  Compares with A1/A2: is error detection harder/easier on generated text
  vs real text?

═══════════════════════════════════════════════════════════════════════════
PART C — Forced-error generation + detection + correction
═══════════════════════════════════════════════════════════════════════════
  C1: 124M forward generates text WITH forced errors baked into the
      autoregressive generation.  At position t, a low-ranked token is
      forced instead of the model's greedy choice; generation continues
      from the corrupted state (the model "adapts" to its own error).
      45M forward detects AND attempts to correct the error.

  C2: Same as C1 but with 124M backward (backrule) + 45M backward detection.

  "Correction" metric: at the error position, does the 45M detection model's
  argmax match the token the 124M model WOULD have generated greedily?

═══════════════════════════════════════════════════════════════════════════
Metrics per eval:
  - AUC:            P(score_err > score_clean) — detection separability
  - F0.5:           best F-beta (β=0.5, precision-weighted)
  - Recall@K:       fraction of errors ranked in top-K by surprise
  - Correction@1:   (Part C only) fraction of errors where argmax matches
                    the intended (greedy) token
  - Correction@5:   (Part C only) fraction where intended token is in top-5

Usage:
  # Full run:
  python test_autonomy_dual_domain_v2.py --device cuda:0 --n 200 --L 128 --steps 8

  # Smoke test:
  python test_autonomy_dual_domain_v2.py --device cuda:0 --n 3 --L 64 --steps 4 --smoke
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
BPE_DIR = HERE
sys.path.insert(0, str(BPE_DIR.parent))

from transformers import GPT2Config, GPT2LMHeadModel

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    def roc_auc_score(y, s):
        y = np.asarray(y); s = np.asarray(s)
        pos = s[y == 1]; neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        return float((pos[:, None] > neg[None, :]).mean())

OUT_DIR = HERE.parent / "data"
VOCAB_SIZE = 38001

# ---- 45M evaluation models ----
FWD_45M_CKPT = (
    OUT_DIR / "model_union_fwd_v1_20000_s512_st64_b32k_n10k"
    / "fwd_lm_s46_best.pt"
)
BWD_BACKRULE_45M_CKPT = (
    OUT_DIR / "model_union_bwd_backrule_v1_20000_s512_st64_b32k_n10k"
    / "bwd_lm_s46_best.pt"
)
BWD_TOKFLIP_45M_CKPT = (
    OUT_DIR / "model_union_bwd_fwdrule_v1_20000_tokflip_s512_st64_b32k_n10k"
    / "bwd_tokflip_lm_s46_best.pt"
)

# ---- 124M reference models ----
FWD_124M_CKPT = (
    OUT_DIR / "model_union_fwd_v1_20000_s512_st64_b16k_n10k_124M"
    / "fwd_lm_s46_best.pt"
)
BWD_BACKRULE_124M_CKPT = (
    OUT_DIR / "model_union_bwd_backrule_v1_20000_s512_st64_b16k_n10k_124M"
    / "bwd_lm_s46_best.pt"
)

# ---- Data ----
FWD_TOK_VAL = OUT_DIR / "tokenized_union_fwd_v1_20000" / "wiki_val.pt"
BWD_BACKRULE_TOK_VAL = OUT_DIR / "tokenized_union_bwd_backrule_v1_20000" / "wiki_val.pt"

S_OUT = OUT_DIR / "test_autonomy_dual_domain_v2_20k"
S_OUT.mkdir(parents=True, exist_ok=True)

_DEFAULT_N = 200
_DEFAULT_L = 128
_DEFAULT_STEPS = 8
ALPHA = 0.9

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(vocab_size, device, model_size="45M"):
    presets = {
        "45M":  dict(n_layer=6, n_embd=512, n_head=8),
        "124M": dict(n_layer=12, n_embd=768, n_head=12),
    }
    p = presets[model_size]
    cfg = GPT2Config(
        vocab_size=vocab_size, n_positions=1024,
        n_embd=p["n_embd"], n_layer=p["n_layer"], n_head=p["n_head"],
        activation_function="gelu_new",
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        layer_norm_epsilon=1e-5, initializer_range=0.02,
        bos_token_id=1, eos_token_id=2,
    )
    return GPT2LMHeadModel(cfg).to(device)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def lp_full(model, seq, dev, reverse=False):
    inp = torch.flip(seq.view(1, -1), [1]) if reverse else seq.view(1, -1)
    inp = inp.to(dev)
    hs = model.transformer(input_ids=inp, attention_mask=torch.ones_like(inp)).last_hidden_state
    return F.log_softmax(model.lm_head(hs[0]).float(), -1).cpu().numpy()


def f_beta_best(labels, scores, beta=0.5):
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); P_sum = int(y.sum())
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P_sum, 1)
    b2 = beta * beta
    f = (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12)
    return float(np.nanmax(f))


def recall_at_k(labels, scores, k):
    """Fraction of positives in top-k by score (descending)."""
    order = np.argsort(-scores)
    return float(labels[order[:k]].sum()) / max(labels.sum(), 1)


def compute_detection_metrics(sc_list, lab_list, n_seqs, rec1=0, rec5=0):
    """Compute detection metrics: pooled AUC/F0.5 + per-sequence Recall@K."""
    lab_all = np.concatenate(lab_list)
    sc_all = np.concatenate(sc_list)

    m = {
        "AUC": roc_auc_score(lab_all, sc_all),
        "F0.5": f_beta_best(lab_all, sc_all),
        "Correction@1": rec1 / max(n_seqs, 1),
        "Correction@5": rec5 / max(n_seqs, 1),
        "n_pos": len(lab_all),
        "n_err": int(lab_all.sum()),
    }

    for k in [1, 5]:
        per_seq = [recall_at_k(lab, sc, k) for lab, sc in zip(lab_list, sc_list)]
        m[f"Recall@{k}"] = float(np.mean(per_seq))

    return m

# ---------------------------------------------------------------------------
# Generation utilities (for Parts B & C)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_greedy_seq(model, seed_ids, total_len, dev):
    """Greedy autoregressive generation from seed.

    Returns:
      seq: [total_len] tensor (seed_ids padded/trimmed to total_len)
      greedy_tokens: list of tokens generated at each position (len=total_len)
    """
    seq = seed_ids.clone().tolist()
    while len(seq) < total_len:
        inp = torch.tensor([seq], device=dev)
        logits = model(input_ids=inp).logits[0, -1].float()
        seq.append(int(logits.argmax()))
    return torch.tensor(seq[:total_len], dtype=torch.long)


@torch.no_grad()
def generate_forced_error_seq(model, seed_ids, total_len, error_pos,
                               error_mode, dev, rng, freq_toks, topk,
                               error_rank_range=(50, 200)):
    """Generate a sequence with a forced error BAKED INTO autoregressive generation.

    Steps:
      1. Generate greedily from seed up to error_pos-1.
      2. At error_pos, FORCE a token that is NOT the model's greedy choice:
         - 'lowrank': pick from rank range [lo, hi) in model's sorted logits
         - 'random':  pick a random frequent token ≠ greedy choice
      3. Continue greedy generation from error_pos+1 onwards.
         The model ADAPTS to the error in step 3 — the context after the
         error is coherent with it (unlike post-hoc injection).

    Returns:
      seq:             [total_len] generated sequence with baked-in error
      greedy_at_error: token the model WOULD have generated at error_pos
      error_token:     the forced error token actually placed at error_pos
    """
    seq = seed_ids.clone().tolist()
    greedy_at_error = None
    error_token = None

    while len(seq) < total_len:
        pos = len(seq)
        inp = torch.tensor([seq], device=dev)
        logits = model(input_ids=inp).logits[0, -1].float()

        if pos == error_pos:
            # Record the intended (greedy) token
            greedy_at_error = int(logits.argmax())

            if error_mode == "lowrank":
                lo, hi = error_rank_range
                sorted_ids = logits.argsort(descending=True)
                hi_clamped = min(hi, len(sorted_ids))
                lo_clamped = min(lo, hi_clamped - 1)
                candidates = sorted_ids[lo_clamped:hi_clamped].cpu().tolist()
                # Remove greedy from candidates
                candidates = [c for c in candidates if c != greedy_at_error]
                if candidates:
                    error_token = int(rng.choice(candidates))
                else:
                    error_token = greedy_at_error  # fallback (should be rare)
            else:  # random
                error_token = int(rng.choice(freq_toks))
                while error_token == greedy_at_error:
                    error_token = int(rng.choice(freq_toks))

            seq.append(error_token)
        else:
            seq.append(int(logits.argmax()))

    return (torch.tensor(seq[:total_len], dtype=torch.long),
            greedy_at_error, error_token)


@torch.no_grad()
def compute_plausible_candidates_124m(model, blocks, dev, L, topk=20):
    """Pre-compute plausible-error candidates from a 124M reference model."""
    n_seqs = blocks.shape[0]
    npos = n_seqs * (L - 1)
    candidates = torch.zeros(npos, topk, dtype=torch.long)
    bs = 32
    for start in tqdm(range(0, n_seqs, bs), desc="Plausible candidates", unit="batch", leave=False):
        end = min(start + bs, n_seqs)
        bb = blocks[start:end].to(dev)
        lg = model(input_ids=bb, attention_mask=torch.ones_like(bb)).logits.float()
        for ri in range(bb.size(0)):
            si = start + ri
            for t in range(1, L):
                idx = si * (L - 1) + (t - 1)
                truth = blocks[si, t].item()
                _, top_ids = lg[ri, t - 1].topk(topk + 2)
                top_ids = top_ids.cpu()
                mask = top_ids != truth
                cand = top_ids[mask][:topk]
                if len(cand) < topk:
                    cand_list = [int(c) for c in top_ids if int(c) != truth]
                    while len(cand_list) < topk:
                        cand_list.append(0)
                    cand = torch.tensor(cand_list[:topk])
                candidates[idx] = cand
    return candidates

# ===================================================================
# PART A: Real-data error detection (three domains)
# ===================================================================

def eval_real_data_forward(seqs, fwd_45m, candidates, dev, L, rng, freq_toks):
    """A1: 45M forward model on forward-tok real data."""
    print("\n" + "-" * 50)
    print("A1: Real-data forward detection")
    print("-" * 50)

    results = {}
    for mode in ["plausible", "random"]:
        SC, LAB = [], []
        rec1, rec5 = 0, 0
        for si, s_orig in enumerate(tqdm(seqs, desc=f"  A1 {mode}", unit="seq", leave=False)):
            s = s_orig.clone()
            t = int(rng.randint(L // 4, 3 * L // 4))
            orig_token = int(s[t])

            # Error injection
            if mode == "plausible":
                idx = si * (L - 1) + (t - 1)
                cand = [int(c) for c in candidates[idx] if int(c) != int(s[t])]
                s[t] = int(rng.choice(cand)) if cand else s[t]
            else:
                c = int(rng.choice(freq_toks))
                while c == int(s[t]): c = int(rng.choice(freq_toks))
                s[t] = c

            lf = lp_full(fwd_45m, s, dev)
            P = np.arange(1, L - 1)
            xp = s.numpy()[P]; ar = np.arange(len(P))
            sf = -lf[P - 1][ar, xp]
            lab = (P == t).astype(int)

            # Correction (top-1 / top-5 at error position)
            t_idx = np.where(P == t)[0][0]
            lp_t = lf[t - 1]
            top5 = lp_t.argsort()[-5:][::-1]
            rec1 += int(top5[0] == orig_token)
            rec5 += int(orig_token in top5)

            SC.append(sf); LAB.append(lab)

        results[mode] = compute_detection_metrics(SC, LAB, len(seqs), rec1, rec5)
    return results


def eval_real_data_backward_backrule(seqs_rev, bwd_br_45m, candidates, dev, L, rng, freq_toks):
    """A2: 45M backward (backrule) on backrule-tok real data."""
    print("\n" + "-" * 50)
    print("A2: Real-data backward (backrule) detection")
    print("-" * 50)

    n_seqs = len(seqs_rev)
    results = {}
    for mode in ["plausible", "random"]:
        SC, LAB = [], []
        rec1, rec5 = 0, 0
        for si, s_orig in enumerate(tqdm(seqs_rev, desc=f"  A2 {mode}", unit="seq", leave=False)):
            s = s_orig.clone()
            t = int(rng.randint(L // 4, 3 * L // 4))
            orig_token = int(s[t])

            if mode == "plausible":
                idx = si * (L - 1) + (t - 1)
                cand = [int(c) for c in candidates[idx] if int(c) != int(s[t])]
                s[t] = int(rng.choice(cand)) if cand else s[t]
            else:
                c = int(rng.choice(freq_toks))
                while c == int(s[t]): c = int(rng.choice(freq_toks))
                s[t] = c

            lb = lp_full(bwd_br_45m, s, dev, reverse=False)
            P = np.arange(1, L - 1)
            xp = s.numpy()[P]; ar = np.arange(len(P))
            sb = -lb[P - 1][ar, xp]
            lab = (P == t).astype(int)

            t_idx = np.where(P == t)[0][0]
            lp_t = lb[t - 1]
            top5 = lp_t.argsort()[-5:][::-1]
            rec1 += int(top5[0] == orig_token)
            rec5 += int(orig_token in top5)

            SC.append(sb); LAB.append(lab)

        results[mode] = compute_detection_metrics(SC, LAB, n_seqs, rec1, rec5)
    return results


def eval_real_data_backward_tokflip(seqs, bwd_tf_45m, candidates, dev, L, rng, freq_toks):
    """A3: 45M backward (tokflip) on forward-tok real data."""
    print("\n" + "-" * 50)
    print("A3: Real-data backward (tokflip) detection")
    print("-" * 50)

    results = {}
    for mode in ["plausible", "random"]:
        SC, LAB = [], []
        rec1, rec5 = 0, 0
        for si, s_orig in enumerate(tqdm(seqs, desc=f"  A3 {mode}", unit="seq", leave=False)):
            s = s_orig.clone()
            t = int(rng.randint(L // 4, 3 * L // 4))
            orig_token = int(s[t])

            if mode == "plausible":
                idx = si * (L - 1) + (t - 1)
                cand = [int(c) for c in candidates[idx] if int(c) != int(s[t])]
                s[t] = int(rng.choice(cand)) if cand else s[t]
            else:
                c = int(rng.choice(freq_toks))
                while c == int(s[t]): c = int(rng.choice(freq_toks))
                s[t] = c

            lb = lp_full(bwd_tf_45m, s, dev, reverse=True)
            P = np.arange(1, L - 1)
            xp = s.numpy()[P]
            bwd_pos = L - 2 - P
            ar = np.arange(len(P))
            sb = -lb[bwd_pos, xp]
            lab = (P == t).astype(int)

            t_idx = np.where(P == t)[0][0]
            lp_t = lb[bwd_pos[t_idx]]
            top5 = lp_t.argsort()[-5:][::-1]
            rec1 += int(top5[0] == orig_token)
            rec5 += int(orig_token in top5)

            SC.append(sb); LAB.append(lab)

        results[mode] = compute_detection_metrics(SC, LAB, len(seqs), rec1, rec5)
    return results


# ===================================================================
# PART B: Generated-text post-hoc error detection
# ===================================================================

def eval_generated_posthoc_forward(gen_124m, fwd_45m, dev, L, n_seqs, rng, freq_toks):
    """B1: 124M forward generates greedily → inject errors → 45M forward detects.

    Compares with A1: are errors harder to detect in generated vs real text?
    """
    print("\n" + "-" * 50)
    print("B1: Generated-text post-hoc forward detection")
    print("    124M forward generates → inject errors → 45M forward detects")
    print("-" * 50)

    # Generate sequences
    print("  Generating sequences with 124M forward...")
    seed_len = L // 8
    # Use a fixed seed for all sequences
    seed_ids = torch.randint(0, VOCAB_SIZE, (seed_len,), dtype=torch.long)
    gen_seqs = []
    for i in tqdm(range(n_seqs), desc="  Generate", unit="seq", leave=False):
        seq = generate_greedy_seq(gen_124m, seed_ids, L, dev)
        gen_seqs.append(seq.cpu())

    results = {}
    for mode in ["plausible", "random"]:
        SC, LAB = [], []
        rec1, rec5 = 0, 0
        for si, s_orig in enumerate(tqdm(gen_seqs, desc=f"  B1 {mode}", unit="seq", leave=False)):
            s = s_orig.clone()
            t = int(rng.randint(L // 4, 3 * L // 4))
            # The "original" (intended) token = the greedy choice of 124M
            orig_token = int(s[t])

            # Plausible errors: use 124M's own top-20 at position t
            if mode == "plausible":
                inp = s[:t+1].view(1, -1).to(dev)
                logits_t = gen_124m(input_ids=inp).logits[0, -1].float()
                _, top_ids = logits_t.topk(22)
                top_ids = top_ids.cpu()
                cand = [int(c) for c in top_ids if int(c) != orig_token][:20]
                if cand:
                    s[t] = int(rng.choice(cand))
            else:
                c = int(rng.choice(freq_toks))
                while c == int(s[t]): c = int(rng.choice(freq_toks))
                s[t] = c

            # Detect with 45M forward
            lf = lp_full(fwd_45m, s, dev)
            P = np.arange(1, L - 1)
            xp = s.numpy()[P]; ar = np.arange(len(P))
            sf = -lf[P - 1][ar, xp]
            lab = (P == t).astype(int)

            t_idx = np.where(P == t)[0][0]
            lp_t = lf[t - 1]
            top5 = lp_t.argsort()[-5:][::-1]
            rec1 += int(top5[0] == orig_token)
            rec5 += int(orig_token in top5)

            SC.append(sf); LAB.append(lab)

        results[mode] = compute_detection_metrics(SC, LAB, n_seqs, rec1, rec5)
    return results


def eval_generated_posthoc_backward(gen_124m, bwd_br_45m, dev, L, n_seqs, rng, freq_toks):
    """B2: 124M backward (backrule) generates → inject errors → 45M backward detects.

    Generates on backrule-tok data (model is native forward-LM on backrule tokens).
    """
    print("\n" + "-" * 50)
    print("B2: Generated-text post-hoc backward (backrule) detection")
    print("    124M backward generates → inject errors → 45M backward detects")
    print("-" * 50)

    print("  Generating sequences with 124M backward (backrule)...")
    seed_len = L // 8
    seed_ids = torch.randint(0, VOCAB_SIZE, (seed_len,), dtype=torch.long)
    gen_seqs = []
    for i in tqdm(range(n_seqs), desc="  Generate", unit="seq", leave=False):
        seq = generate_greedy_seq(gen_124m, seed_ids, L, dev)
        gen_seqs.append(seq.cpu())

    results = {}
    for mode in ["plausible", "random"]:
        SC, LAB = [], []
        rec1, rec5 = 0, 0
        for si, s_orig in enumerate(tqdm(gen_seqs, desc=f"  B2 {mode}", unit="seq", leave=False)):
            s = s_orig.clone()
            t = int(rng.randint(L // 4, 3 * L // 4))
            orig_token = int(s[t])

            if mode == "plausible":
                inp = s[:t+1].view(1, -1).to(dev)
                logits_t = gen_124m(input_ids=inp).logits[0, -1].float()
                _, top_ids = logits_t.topk(22)
                top_ids = top_ids.cpu()
                cand = [int(c) for c in top_ids if int(c) != orig_token][:20]
                if cand:
                    s[t] = int(rng.choice(cand))
            else:
                c = int(rng.choice(freq_toks))
                while c == int(s[t]): c = int(rng.choice(freq_toks))
                s[t] = c

            lb = lp_full(bwd_br_45m, s, dev, reverse=False)
            P = np.arange(1, L - 1)
            xp = s.numpy()[P]; ar = np.arange(len(P))
            sb = -lb[P - 1][ar, xp]
            lab = (P == t).astype(int)

            t_idx = np.where(P == t)[0][0]
            lp_t = lb[t - 1]
            top5 = lp_t.argsort()[-5:][::-1]
            rec1 += int(top5[0] == orig_token)
            rec5 += int(orig_token in top5)

            SC.append(sb); LAB.append(lab)

        results[mode] = compute_detection_metrics(SC, LAB, n_seqs, rec1, rec5)
    return results


# ===================================================================
# PART C: Forced-error generation + detection + correction
# ===================================================================

def eval_forced_error_forward(gen_124m, fwd_45m, dev, L, n_seqs, rng, freq_toks):
    """C1: 124M forward generates WITH forced errors baked in → 45M forward detects.

    Key difference from B1: the error is forced DURING autoregressive generation,
    so the model ADAPTS to it — text after the error is coherent with the error.
    This makes detection harder (and more realistic).

    Also measures CORRECTION: does 45M forward's argmax at the error position
    match the token 124M WOULD have generated (greedy)?
    """
    print("\n" + "-" * 50)
    print("C1: Forced-error generation + forward detection + correction")
    print("    124M forward generates with forced low-rank error → 45M forward detects & corrects")
    print("-" * 50)

    seed_len = L // 8
    seed_ids = torch.randint(0, VOCAB_SIZE, (seed_len,), dtype=torch.long)

    # Generate sequences with forced errors
    print("  Generating sequences with forced errors (lowrank mode)...")
    gen_data_lowrank = []
    for i in tqdm(range(n_seqs), desc="  Gen lowrank", unit="seq", leave=False):
        error_pos = int(rng.randint(L // 4, 3 * L // 4))
        seq, greedy_at_err, err_tok = generate_forced_error_seq(
            gen_124m, seed_ids, L, error_pos, "lowrank", dev, rng, freq_toks, 20,
            error_rank_range=(50, 200))
        gen_data_lowrank.append({
            "seq": seq.cpu(), "error_pos": error_pos,
            "greedy_token": greedy_at_err, "error_token": err_tok,
        })

    print("  Generating sequences with forced errors (random mode)...")
    gen_data_random = []
    for i in tqdm(range(n_seqs), desc="  Gen random", unit="seq", leave=False):
        error_pos = int(rng.randint(L // 4, 3 * L // 4))
        seq, greedy_at_err, err_tok = generate_forced_error_seq(
            gen_124m, seed_ids, L, error_pos, "random", dev, rng, freq_toks, 20)
        gen_data_random.append({
            "seq": seq.cpu(), "error_pos": error_pos,
            "greedy_token": greedy_at_err, "error_token": err_tok,
        })

    results = {}
    for mode_label, gen_data in [("lowrank", gen_data_lowrank), ("random", gen_data_random)]:
        SC, LAB = [], []
        n_correct_1 = 0   # argmax == intended greedy token
        n_correct_5 = 0   # intended token in top-5
        for d in tqdm(gen_data, desc=f"  C1 {mode_label}", unit="seq", leave=False):
            s = d["seq"]; t = d["error_pos"]; intended = d["greedy_token"]

            lf = lp_full(fwd_45m, s, dev)
            P = np.arange(1, L - 1)
            xp = s.numpy()[P]; ar = np.arange(len(P))
            sf = -lf[P - 1][ar, xp]
            lab = (P == t).astype(int)

            t_idx = np.where(P == t)[0][0]
            lp_t = lf[t - 1]
            top5 = lp_t.argsort()[-5:][::-1]
            n_correct_1 += int(top5[0] == intended)
            n_correct_5 += int(intended in top5)

            SC.append(sf); LAB.append(lab)

        results[mode_label] = compute_detection_metrics(SC, LAB, n_seqs, n_correct_1, n_correct_5)
    return results


def eval_forced_error_backward(gen_124m, bwd_br_45m, dev, L, n_seqs, rng, freq_toks):
    """C2: 124M backward (backrule) generates with forced errors → 45M backward detects."""
    print("\n" + "-" * 50)
    print("C2: Forced-error generation + backward (backrule) detection + correction")
    print("    124M backward generates with forced error → 45M backward detects & corrects")
    print("-" * 50)

    seed_len = L // 8
    seed_ids = torch.randint(0, VOCAB_SIZE, (seed_len,), dtype=torch.long)

    gen_data_lowrank = []
    for i in tqdm(range(n_seqs), desc="  Gen lowrank", unit="seq", leave=False):
        error_pos = int(rng.randint(L // 4, 3 * L // 4))
        seq, greedy_at_err, err_tok = generate_forced_error_seq(
            gen_124m, seed_ids, L, error_pos, "lowrank", dev, rng, freq_toks, 20,
            error_rank_range=(50, 200))
        gen_data_lowrank.append({
            "seq": seq.cpu(), "error_pos": error_pos,
            "greedy_token": greedy_at_err, "error_token": err_tok,
        })

    gen_data_random = []
    for i in tqdm(range(n_seqs), desc="  Gen random", unit="seq", leave=False):
        error_pos = int(rng.randint(L // 4, 3 * L // 4))
        seq, greedy_at_err, err_tok = generate_forced_error_seq(
            gen_124m, seed_ids, L, error_pos, "random", dev, rng, freq_toks, 20)
        gen_data_random.append({
            "seq": seq.cpu(), "error_pos": error_pos,
            "greedy_token": greedy_at_err, "error_token": err_tok,
        })

    results = {}
    for mode_label, gen_data in [("lowrank", gen_data_lowrank), ("random", gen_data_random)]:
        SC, LAB = [], []
        n_correct_1, n_correct_5 = 0, 0
        for d in tqdm(gen_data, desc=f"  C2 {mode_label}", unit="seq", leave=False):
            s = d["seq"]; t = d["error_pos"]; intended = d["greedy_token"]

            lb = lp_full(bwd_br_45m, s, dev, reverse=False)
            P = np.arange(1, L - 1)
            xp = s.numpy()[P]; ar = np.arange(len(P))
            sb = -lb[P - 1][ar, xp]
            lab = (P == t).astype(int)

            t_idx = np.where(P == t)[0][0]
            lp_t = lb[t - 1]
            top5 = lp_t.argsort()[-5:][::-1]
            n_correct_1 += int(top5[0] == intended)
            n_correct_5 += int(intended in top5)

            SC.append(sb); LAB.append(lab)

        results[mode_label] = compute_detection_metrics(SC, LAB, n_seqs, n_correct_1, n_correct_5)
    return results


# ===================================================================
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser(description="V2: Generated-text error detection + correction")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=_DEFAULT_N)
    ap.add_argument("--L", type=int, default=_DEFAULT_L)
    ap.add_argument("--steps", type=int, default=_DEFAULT_STEPS,
                    help="Future window (unused in v2, kept for compatibility)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    L = args.L
    rng = np.random.RandomState(0)

    if args.smoke:
        args.n = min(args.n, 3)
        L = min(L, 64)
        print("=" * 70)
        print("SMOKE TEST MODE (V2)")
        print("=" * 70)

    print("=" * 70)
    print("V2: Error Detection on Real + Generated Data + Forced Errors + Correction")
    print(f"  device={dev}  L={L}  n_seqs={args.n}  vocab={VOCAB_SIZE:,}")
    print("=" * 70)
    print()
    print("  Part A: Real-data detection (3 domains)")
    print("  Part B: Generated-text post-hoc error detection (2 directions)")
    print("  Part C: Forced-error generation + detection + correction (2 directions)")
    print()

    # ── Load data ──
    print("[0] Load data")
    blocks_fwd = torch.load(str(FWD_TOK_VAL), map_location="cpu").to(torch.int64)
    n_full = blocks_fwd.numel() // L
    chunks_fwd = blocks_fwd[:n_full * L].view(n_full, L)
    seqs_fwd_all = [chunks_fwd[i].clone() for i in range(n_full)]
    rng.shuffle(seqs_fwd_all)
    seqs_fwd = seqs_fwd_all[:args.n]
    print(f"    fwd-tok: {len(seqs_fwd)} sequences")

    blocks_rev = torch.load(str(BWD_BACKRULE_TOK_VAL), map_location="cpu").to(torch.int64)
    n_full_rev = blocks_rev.numel() // L
    chunks_rev = blocks_rev[:n_full_rev * L].view(n_full_rev, L)
    seqs_rev_all = [chunks_rev[i].clone() for i in range(n_full_rev)]
    rng.shuffle(seqs_rev_all)
    seqs_rev = seqs_rev_all[:args.n]
    print(f"    backrule-tok: {len(seqs_rev)} sequences")

    counts = torch.bincount(blocks_fwd.reshape(-1), minlength=VOCAB_SIZE).numpy()
    freq_toks = np.argsort(counts)[-500:]

    # Backrule-tok frequent tokens (distribution-matched for backward evaluations)
    counts_bwd = torch.bincount(blocks_rev.reshape(-1), minlength=VOCAB_SIZE).numpy()
    freq_toks_bwd = np.argsort(counts_bwd)[-500:]

    fwd_set = set(freq_toks.tolist())
    bwd_set = set(freq_toks_bwd.tolist())
    print(f"    Freq-tok overlap (top-500, fwd vs bwd-backrule): {len(fwd_set & bwd_set)}/500")

    # ── Load models ──
    print("\n[1] Load 45M evaluation models")
    fwd_45m = build_model(VOCAB_SIZE, dev, "45M")
    fwd_45m.load_state_dict(torch.load(str(FWD_45M_CKPT), map_location=dev))
    fwd_45m.eval()
    print(f"    Forward 45M: {sum(p.numel() for p in fwd_45m.parameters())/1e6:.1f}M")

    bwd_br_45m = build_model(VOCAB_SIZE, dev, "45M")
    bwd_br_45m.load_state_dict(torch.load(str(BWD_BACKRULE_45M_CKPT), map_location=dev))
    bwd_br_45m.eval()
    print(f"    Backward backrule 45M: {sum(p.numel() for p in bwd_br_45m.parameters())/1e6:.1f}M")

    bwd_tf_45m = build_model(VOCAB_SIZE, dev, "45M")
    bwd_tf_45m.load_state_dict(torch.load(str(BWD_TOKFLIP_45M_CKPT), map_location=dev))
    bwd_tf_45m.eval()
    print(f"    Backward tokflip 45M: {sum(p.numel() for p in bwd_tf_45m.parameters())/1e6:.1f}M")

    # Sanity
    dummy = torch.randint(0, VOCAB_SIZE, (2, L), device=dev)
    assert fwd_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    assert bwd_br_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    assert bwd_tf_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    print("    Sanity check ✓")

    print("\n[2] Load 124M generation models")
    fwd_124m = build_model(VOCAB_SIZE, dev, "124M")
    fwd_124m.load_state_dict(torch.load(str(FWD_124M_CKPT), map_location=dev))
    fwd_124m.eval()
    print(f"    Forward 124M: {sum(p.numel() for p in fwd_124m.parameters())/1e6:.1f}M")

    bwd_br_124m = build_model(VOCAB_SIZE, dev, "124M")
    bwd_br_124m.load_state_dict(torch.load(str(BWD_BACKRULE_124M_CKPT), map_location=dev))
    bwd_br_124m.eval()
    print(f"    Backward backrule 124M: {sum(p.numel() for p in bwd_br_124m.parameters())/1e6:.1f}M")

    # ── Pre-compute plausible candidates ──
    print("\n[3] Pre-compute plausible-error candidates (124M models)")
    cand_fwd = compute_plausible_candidates_124m(fwd_124m, chunks_fwd[:args.n], dev, L)
    cand_bwd = compute_plausible_candidates_124m(bwd_br_124m, chunks_rev[:args.n], dev, L)
    print(f"    fwd candidates: {tuple(cand_fwd.shape)}")
    print(f"    bwd candidates: {tuple(cand_bwd.shape)}")

    # ══════════════════════════════════════════════════════════════════
    # RUN ALL EVALUATIONS
    # ══════════════════════════════════════════════════════════════════
    t0 = time.time()
    all_results = {}

    # ---- Part A: Real-data detection ----
    print("\n" + "=" * 70)
    print("PART A: Real-data error detection")
    print("=" * 70)

    all_results["A1_real_fwd"] = eval_real_data_forward(
        seqs_fwd, fwd_45m, cand_fwd, dev, L, rng, freq_toks)
    all_results["A2_real_bwd_backrule"] = eval_real_data_backward_backrule(
        seqs_rev, bwd_br_45m, cand_bwd, dev, L, rng, freq_toks_bwd)
    all_results["A3_real_bwd_tokflip"] = eval_real_data_backward_tokflip(
        seqs_fwd, bwd_tf_45m, cand_fwd, dev, L, rng, freq_toks)

    # ---- Part B: Generated-text post-hoc detection ----
    print("\n" + "=" * 70)
    print("PART B: Generated-text post-hoc error detection")
    print("=" * 70)

    all_results["B1_gen_posthoc_fwd"] = eval_generated_posthoc_forward(
        fwd_124m, fwd_45m, dev, L, args.n, rng, freq_toks)
    all_results["B2_gen_posthoc_bwd"] = eval_generated_posthoc_backward(
        bwd_br_124m, bwd_br_45m, dev, L, args.n, rng, freq_toks_bwd)

    # ---- Part C: Forced-error generation + detection + correction ----
    print("\n" + "=" * 70)
    print("PART C: Forced-error generation + detection + correction")
    print("=" * 70)

    all_results["C1_forced_error_fwd"] = eval_forced_error_forward(
        fwd_124m, fwd_45m, dev, L, args.n, rng, freq_toks)
    all_results["C2_forced_error_bwd"] = eval_forced_error_backward(
        bwd_br_124m, bwd_br_45m, dev, L, args.n, rng, freq_toks_bwd)

    dt = time.time() - t0

    # Free GPU memory
    del fwd_45m, bwd_br_45m, bwd_tf_45m, fwd_124m, bwd_br_124m
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"RESULTS  ({dt:.1f}s)")
    print("=" * 70)

    def print_section(title, evals):
        print(f"\n  {title}")
        hdr = (f"  {'Eval':<28} {'Mode':<10} {'AUC':>8}  {'F0.5':>8}  "
               f"{'Recall@1':>8}  {'Recall@5':>8}  {'Corr@1':>8}  {'Corr@5':>8}")
        print(hdr)
        print("  " + "-" * len(hdr))
        for name, modes in evals:
            for mode in (["plausible", "random"] if "plausible" in modes
                         else ["lowrank", "random"]):
                if mode in modes:
                    m = modes[mode]
                    print(f"  {name:<28} {mode:<10} "
                          f"{m['AUC']:>8.4f}  {m['F0.5']:>8.4f}  "
                          f"{m['Recall@1']:>8.1f}  {m['Recall@5']:>8.1f}  "
                          f"{m['Correction@1']:>8.3f}  {m['Correction@5']:>8.3f}")

    # Part A
    print_section("PART A — Real-data detection", [
        ("A1 fwd on real fwd-tok", all_results["A1_real_fwd"]),
        ("A2 bwd-backrule on rev-tok", all_results["A2_real_bwd_backrule"]),
        ("A3 bwd-tokflip on fwd-tok", all_results["A3_real_bwd_tokflip"]),
    ])

    # Part B
    print_section("PART B — Generated-text post-hoc detection", [
        ("B1 gen→inject→fwd detect", all_results["B1_gen_posthoc_fwd"]),
        ("B2 gen→inject→bwd detect", all_results["B2_gen_posthoc_bwd"]),
    ])

    # Part C
    print_section("PART C — Forced-error generation + detection + correction", [
        ("C1 forced-err→fwd detect", all_results["C1_forced_error_fwd"]),
        ("C2 forced-err→bwd detect", all_results["C2_forced_error_bwd"]),
    ])

    # ══════════════════════════════════════════════════════════════════
    # KEY COMPARISONS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    # Real vs Generated (plausible)
    a1_p = all_results["A1_real_fwd"]["plausible"]["AUC"]
    b1_p = all_results["B1_gen_posthoc_fwd"]["plausible"]["AUC"]
    print(f"\n  Forward detection: real vs generated text (plausible errors)")
    print(f"    A1 real data:      AUC = {a1_p:.4f}")
    print(f"    B1 generated data: AUC = {b1_p:.4f}")
    print(f"    Δ (gen − real) = {b1_p - a1_p:+.4f}")

    a2_p = all_results["A2_real_bwd_backrule"]["plausible"]["AUC"]
    b2_p = all_results["B2_gen_posthoc_bwd"]["plausible"]["AUC"]
    print(f"\n  Backward detection: real vs generated text (plausible errors)")
    print(f"    A2 real data:      AUC = {a2_p:.4f}")
    print(f"    B2 generated data: AUC = {b2_p:.4f}")
    print(f"    Δ (gen − real) = {b2_p - a2_p:+.4f}")

    # Post-hoc vs Forced-error (which is harder?)
    b1_plaus = all_results["B1_gen_posthoc_fwd"]["plausible"]["AUC"]
    c1_lr = all_results["C1_forced_error_fwd"]["lowrank"]["AUC"]
    print(f"\n  Forward detection: post-hoc vs forced-error (generated text)")
    print(f"    B1 post-hoc inject:  AUC = {b1_plaus:.4f}")
    print(f"    C1 forced (lowrank): AUC = {c1_lr:.4f}")
    print(f"    Δ (forced − posthoc) = {c1_lr - b1_plaus:+.4f}")
    print(f"    → Forced errors are {'HARDER' if c1_lr < b1_plaus else 'EASIER'} to detect")

    # Correction capability
    print(f"\n  Error correction (Correction@1):")
    for name, key in [("A1 real fwd", "A1_real_fwd"), ("B1 gen fwd", "B1_gen_posthoc_fwd"),
                       ("C1 forced fwd", "C1_forced_error_fwd")]:
        for mode in (["plausible", "random"] if "plausible" in all_results[key]
                     else ["lowrank", "random"]):
            c1 = all_results[key][mode]["Correction@1"]
            print(f"    {name} [{mode}]: {c1:.4f}")

    # ── Save ──
    out_path = S_OUT / f"v2_results_L{L}_n{args.n}.json"
    out_path.write_text(json.dumps({
        "config": {"vocab_size": VOCAB_SIZE, "L": L, "n_seqs": args.n},
        "results": all_results,
    }, indent=2, default=float))
    print(f"\n[saved] {out_path}")
    print("\nDONE ✓")


if __name__ == "__main__":
    main()
