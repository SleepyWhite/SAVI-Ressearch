#!/usr/bin/env python3
"""V5: Token-level error detection with extended Recall metrics (R@1–R@100).

Based on V2, Parts A & B only.  Adds R@10, R@20, R@50, R@100 to test
how far the model ranks error positions above chance.

═══════════════════════════════════════════════════════════════════════════
PART A — Real-data detection (three domains)
═══════════════════════════════════════════════════════════════════════════
  A1: 45M forward on forward-tok real data
  A2: 45M backward (backrule) on backrule-tok real data
  A3: 45M backward (tokflip) on forward-tok real data

═══════════════════════════════════════════════════════════════════════════
PART B — Generated-text post-hoc error detection
═══════════════════════════════════════════════════════════════════════════
  B1: 124M forward generates → inject errors → 45M forward detects
  B2: 124M backward (backrule) generates → inject errors → 45M backward detects

Metrics per eval:
  - AUC, F0.5, Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, Recall@100
  - Correction@1, Correction@5 (at oracle error position)

Usage:
  # Full run:
  python test_autonomy_dual_domain_v5.py --device cuda:0 --n 200 --L 128

  # Smoke test:
  python test_autonomy_dual_domain_v5.py --device cuda:0 --n 3 --L 64 --smoke
"""

from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Paths & constants (10k vocab)
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
VOCAB_SIZE = 18782   # 10k merges + base alphabet

# ---- 45M evaluation models (10k) ----
FWD_45M_CKPT = (
    OUT_DIR / "model_union_fwd_v1_10000_s512_st64_b65k_n10k"
    / "fwd_lm_s46_best.pt"
)
BWD_BACKRULE_45M_CKPT = (
    OUT_DIR / "model_union_bwd_backrule_v1_10000_s512_st64_b65k_n10k"
    / "bwd_lm_s46_best.pt"
)
BWD_TOKFLIP_45M_CKPT = (
    OUT_DIR / "model_union_bwd_fwdrule_v1_10000_tokflip_s512_st64_b65k_n10k"
    / "bwd_tokflip_lm_s46_best.pt"
)

# ---- 124M generation models (10k) ----
FWD_124M_CKPT = (
    OUT_DIR / "model_union_fwd_v1_10000_s512_st64_b32k_n10k_124M"
    / "fwd_lm_s46_best.pt"
)
BWD_BACKRULE_124M_CKPT = (
    OUT_DIR / "model_union_bwd_backrule_v1_10000_s512_st64_b32k_n10k_124M"
    / "bwd_lm_s46_best.pt"
)

# ---- Data (10k) ----
FWD_TOK_VAL = OUT_DIR / "tokenized_union_fwd_v1_10000" / "wiki_val.pt"
BWD_BACKRULE_TOK_VAL = OUT_DIR / "tokenized_union_bwd_backrule_v1_10000" / "wiki_val.pt"

S_OUT = OUT_DIR / "test_autonomy_dual_domain_v5"
S_OUT.mkdir(parents=True, exist_ok=True)

_DEFAULT_N = 200
_DEFAULT_L = 128

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
    """Compute detection metrics: pooled AUC/F0.5 + per-sequence Recall@K.

    sc_list, lab_list: lists of per-sequence arrays (not concatenated).
    Recall@K is computed per-sequence then averaged, so it measures:
      "fraction of sequences where the error position is in the top-K".
    """
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

    # Per-sequence Recall@K (NOT globally pooled)
    for k in [1, 5, 10, 20, 50, 100]:
        per_seq = [recall_at_k(lab, sc, k) for lab, sc in zip(lab_list, sc_list)]
        m[f"Recall@{k}"] = float(np.mean(per_seq))

    return m


# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_greedy_seq(model, seed_ids, total_len, dev):
    """Greedy autoregressive generation from seed."""
    seq = seed_ids.clone().tolist()
    while len(seq) < total_len:
        inp = torch.tensor([seq], device=dev)
        logits = model(input_ids=inp).logits[0, -1].float()
        seq.append(int(logits.argmax()))
    return torch.tensor(seq[:total_len], dtype=torch.long)


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
# PART D: Surprise vs Error-Token Frequency (revised)
# ===================================================================
# Instead of measuring clean-text surprise, we now:
#  1. Pick a fixed error position t ∈ [L/4, 3L/4] per sequence
#  2. Compute baseline surprise = -log p(correct token | context) at t
#  3. Replace correct token with error tokens from 40 frequency bins
#     (rank 0–2000, bin=50, 10 random samples/bin)
#  4. Compute error surprise = -log p(error token | context) at t
#  5. Average: across 10 samples per seq, then across 200 seqs
#  → Plot baseline horizontal line + error-surprise-vs-frequency curves
# ===================================================================

def compute_surprise_vs_freq(seqs, model, counts, dev, L, reverse=False, desc="surprise",
                              rng=None, max_rank=4000, bin_size=50, n_samples=10):
    """Measure error-token surprise at fixed positions vs frequency rank.

    Returns dict with keys: bin_results (list), baseline_avg, baseline_std, baseline_sem.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    # Frequency rank: 0 = most frequent
    freq_rank = np.zeros(VOCAB_SIZE, dtype=np.int32)
    order = np.argsort(-counts)
    for rank, tok in enumerate(order):
        freq_rank[tok] = rank

    # Pre-index tokens by frequency bin (rank 0 .. max_rank-1)
    bin_tokens = {}
    for tok_id in range(VOCAB_SIZE):
        r = int(freq_rank[tok_id])
        if r < max_rank:
            bin_start = (r // bin_size) * bin_size
            bin_tokens.setdefault(bin_start, []).append(tok_id)

    # Accumulators
    bin_surprises = defaultdict(list)  # bin_start -> [seq_avg, ...]
    baseline_vals = []

    # Local RNG for token sampling (don't consume caller's rng)
    sampling_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

    for s_orig in tqdm(seqs, desc=f"  {desc}", unit="seq", leave=False):
        s = s_orig.clone()
        t = int(rng.randint(L // 4, 3 * L // 4))
        orig_token = int(s[t])

        # Run model ONCE per sequence → full log-prob matrix
        lp = lp_full(model, s, dev, reverse=reverse)

        # Log-prob vector at error position t
        if not reverse:
            lp_t = lp[t - 1]          # forward: context [0, t-1] → predict t
        else:
            lp_t = lp[L - 2 - t]      # tokflip: reversed sequence

        # Baseline surprise at correct token
        baseline_vals.append(float(-lp_t[orig_token]))

        # For each frequency bin, sample error tokens & record surprise
        for bin_start, tokens_in_bin in bin_tokens.items():
            if len(tokens_in_bin) <= n_samples:
                sampled = tokens_in_bin
            else:
                sampled = list(sampling_rng.choice(tokens_in_bin, size=n_samples, replace=False))
            surps = [-float(lp_t[tok]) for tok in sampled]
            bin_surprises[bin_start].append(float(np.mean(surps)))

    # Aggregate across sequences per bin
    bin_results = []
    for bin_start in sorted(bin_surprises.keys()):
        vals = np.array(bin_surprises[bin_start])
        bin_results.append({
            "freq_rank_start": int(bin_start),
            "freq_rank_end": int(bin_start + bin_size - 1),
            "avg_error_surprise": float(np.mean(vals)),
            "std_error_surprise": float(np.std(vals)),
            "sem_error_surprise": float(np.std(vals) / np.sqrt(len(vals))),
            "n_tokens_in_bin": len(bin_tokens.get(bin_start, [])),
            "n_seqs": len(vals),
        })

    return {
        "bin_results": bin_results,
        "baseline_avg": float(np.mean(baseline_vals)),
        "baseline_std": float(np.std(baseline_vals)),
        "baseline_sem": float(np.std(baseline_vals) / np.sqrt(len(baseline_vals))),
    }


def plot_surprise_vs_freq(all_freq_data, out_dir, L, n):
    """Plot error-surprise vs error-token frequency rank, with baseline lines."""
    if not HAS_MPL:
        print("  [skip plot] matplotlib not available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.8))
    colors = {"A1": "#2196F3", "A2": "#FF9800", "A3": "#4CAF50",
              "B1": "#E91E63", "B2": "#00BCD4"}

    # Descriptive labels: model direction + tokenization direction
    label_map = {
        "A1": "A1: Forward, L→R tok",
        "A2": "A2: Backward, R→L tok",
        "A3": "A3: Backward, L→R tok",
        "B1": "B1: Forward gen, L→R tok",
        "B2": "B2: Backward gen, R→L tok",
    }

    for ax, part_label, evals in [
        (ax1, "Part A — Real Data (Wikipedia)", ["A1", "A2", "A3"]),
        (ax2, "Part B — Generated Text (greedy)", ["B1", "B2"]),
    ]:
        for key in evals:
            if key not in all_freq_data or not all_freq_data[key]:
                continue
            data = all_freq_data[key]
            br = data["bin_results"]
            if not br:
                continue

            xs = [d["freq_rank_start"] + 25 for d in br]   # bin centre
            ys = [d["avg_error_surprise"] for d in br]

            bl = data["baseline_avg"]

            # Error-surprise curve (solid line) — include baseline value in legend
            legend_label = f"{label_map[key]}  [base {bl:.1f}]"
            ax.plot(xs, ys, color=colors[key], label=legend_label,
                    linewidth=1.2, alpha=0.85)

            # ±1 SEM fill
            ys_lo = [d["avg_error_surprise"] - d["sem_error_surprise"] for d in br]
            ys_hi = [d["avg_error_surprise"] + d["sem_error_surprise"] for d in br]
            ax.fill_between(xs, ys_lo, ys_hi, color=colors[key], alpha=0.08)

            # Baseline horizontal dashed line
            ax.axhline(y=bl, color=colors[key], linestyle="--", linewidth=0.9, alpha=0.45)

        ax.set_xlabel("Error Token Frequency Rank (0 = most frequent)", fontsize=10)
        ax.set_ylabel("Mean Surprise at Error Position", fontsize=10)
        ax.set_title(part_label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85,
                  edgecolor="gray", fancybox=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-50, 4050)

    fig.suptitle(f"Error Surprise vs Error-Token Frequency (L={L}, n={n}, bin=50, 10 samples/bin)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / f"v5_surprise_vs_freq_L{L}_n{n}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [saved plot] {out_path}")

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

            # Correction at oracle error position
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
    """B1: 124M forward generates → inject errors → 45M forward detects."""
    print("\n" + "-" * 50)
    print("B1: Generated-text post-hoc forward detection")
    print("    124M forward generates → inject errors → 45M forward detects")
    print("-" * 50)

    print("  Generating sequences with 124M forward...")
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
        for si, s_orig in enumerate(tqdm(gen_seqs, desc=f"  B1 {mode}", unit="seq", leave=False)):
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
    """B2: 124M backward (backrule) generates → inject errors → 45M backward detects."""
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
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser(description="V5: Token-level detection with R@10, R@20")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=_DEFAULT_N)
    ap.add_argument("--L", type=int, default=_DEFAULT_L)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    L = args.L
    rng = np.random.RandomState(0)

    if args.smoke:
        args.n = min(args.n, 3)
        L = min(L, 64)
        print("=" * 70)
        print("SMOKE TEST MODE (V5)")
        print("=" * 70)

    print("=" * 70)
    print("V5: Token-Level Error Detection with Extended Recall (R@1–R@100)")
    print(f"  device={dev}  L={L}  n_seqs={args.n}  vocab={VOCAB_SIZE:,}")
    print("=" * 70)
    print()
    print("  Part A: Real-data detection (3 domains)")
    print("  Part B: Generated-text post-hoc error detection (2 directions)")
    print("  Part D: Surprise vs token frequency analysis (5 evals, bin=50 ranks)")
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

    counts_bwd = torch.bincount(blocks_rev.reshape(-1), minlength=VOCAB_SIZE).numpy()
    freq_toks_bwd = np.argsort(counts_bwd)[-500:]

    fwd_set = set(freq_toks.tolist())
    bwd_set = set(freq_toks_bwd.tolist())
    print(f"    Freq-tok overlap (top-500, fwd vs bwd-backrule): {len(fwd_set & bwd_set)}/500")

    # ── Load 45M detection models ──
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

    # Sanity check
    dummy = torch.randint(0, VOCAB_SIZE, (2, L), device=dev)
    assert fwd_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    assert bwd_br_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    assert bwd_tf_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    print("    Sanity check ✓")

    # ── Load 124M generation models ──
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

    # ---- Part D: Error Surprise vs Error-Token Frequency ----
    print("\n" + "=" * 70)
    print("PART D: Error Surprise vs Error-Token Frequency (revised)")
    print("  Baseline = surprise of correct token at error position")
    print("  Error surprise = surprise when replaced by token from freq bin")
    print("  Rank 0–4000, bin=50 (80 bins), 10 random samples/bin")
    print("=" * 70)

    # Re-generate B1/B2 clean text for frequency analysis
    print("  Generating clean sequences for B1/B2...")
    seed_len = L // 8
    seed_ids = torch.randint(0, VOCAB_SIZE, (seed_len,), dtype=torch.long)
    gen_fwd_clean = []
    for i in tqdm(range(args.n), desc="  B1 gen clean", unit="seq", leave=False):
        gen_fwd_clean.append(generate_greedy_seq(fwd_124m, seed_ids, L, dev).cpu())

    seed_ids_bwd = torch.randint(0, VOCAB_SIZE, (seed_len,), dtype=torch.long)
    gen_bwd_clean = []
    for i in tqdm(range(args.n), desc="  B2 gen clean", unit="seq", leave=False):
        gen_bwd_clean.append(generate_greedy_seq(bwd_br_124m, seed_ids_bwd, L, dev).cpu())

    freq_data = {}
    print("\n  Computing error surprise vs error-token frequency...")

    freq_data["A1"] = compute_surprise_vs_freq(
        seqs_fwd, fwd_45m, counts, dev, L, reverse=False, desc="A1: fwd data → fwd model", rng=rng)
    freq_data["A2"] = compute_surprise_vs_freq(
        seqs_rev, bwd_br_45m, counts_bwd, dev, L, reverse=False, desc="A2: bwd-backrule data → bwd model", rng=rng)
    freq_data["A3"] = compute_surprise_vs_freq(
        seqs_fwd, bwd_tf_45m, counts, dev, L, reverse=True, desc="A3: fwd data → bwd-tokflip model", rng=rng)
    freq_data["B1"] = compute_surprise_vs_freq(
        gen_fwd_clean, fwd_45m, counts, dev, L, reverse=False, desc="B1: gen fwd → fwd model", rng=rng)
    freq_data["B2"] = compute_surprise_vs_freq(
        gen_bwd_clean, bwd_br_45m, counts_bwd, dev, L, reverse=False, desc="B2: gen bwd → bwd model", rng=rng)

    # Print summary table (new format: baseline + error-surprise range)
    print(f"\n  {'Eval':<6} {'#bins':>6}  {'Baseline':>10}  {'Err Min':>10}  {'Err Max':>10}  {'Δ (max-baseline)':>16}")
    print("  " + "-" * 62)
    for key in ["A1", "A2", "A3", "B1", "B2"]:
        if key in freq_data and freq_data[key]:
            d = freq_data[key]
            br = d.get("bin_results", [])
            if br:
                means = [b["avg_error_surprise"] for b in br]
                print(f"  {key:<6} {len(br):>6}  {d['baseline_avg']:>10.4f}  "
                      f"{np.min(means):>10.4f}  {np.max(means):>10.4f}  "
                      f"{np.max(means) - d['baseline_avg']:>16.4f}")

    # Plot
    print("\n  Plotting...")
    plot_surprise_vs_freq(freq_data, S_OUT, L, args.n)

    dt = time.time() - t0

    # Free GPU memory
    del fwd_45m, bwd_br_45m, bwd_tf_45m, fwd_124m, bwd_br_124m
    torch.cuda.empty_cache()

    # Save freq data
    freq_out_path = S_OUT / f"v5_surprise_vs_freq_L{L}_n{args.n}.json"
    # Convert numpy types for JSON serialization
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    freq_out_path.write_text(json.dumps({
        "config": {"vocab_size": int(VOCAB_SIZE), "L": L, "n_seqs": args.n,
                   "bin_size": 50, "n_samples_per_bin": 10, "max_rank": 4000},
        "freq_data": _sanitize(freq_data),
    }, indent=2, default=float))
    print(f"[saved freq data] {freq_out_path}")

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"RESULTS  ({dt:.1f}s)")
    print("=" * 70)

    def print_section(title, evals):
        print(f"\n  {title}")
        hdr = (f"  {'Eval':<28} {'Mode':<10} {'AUC':>8}  {'F0.5':>8}  "
               f"{'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'R@20':>6}  "
               f"{'R@50':>6}  {'R@100':>6}  "
               f"{'Corr@1':>7}  {'Corr@5':>7}")
        print(hdr)
        print("  " + "-" * len(hdr))
        for name, modes in evals:
            for mode in (["plausible", "random"] if "plausible" in modes else []):
                if mode in modes:
                    m = modes[mode]
                    print(f"  {name:<28} {mode:<10} "
                          f"{m['AUC']:>8.4f}  {m['F0.5']:>8.4f}  "
                          f"{m['Recall@1']:>6.3f}  {m['Recall@5']:>6.3f}  "
                          f"{m['Recall@10']:>6.3f}  {m['Recall@20']:>6.3f}  "
                          f"{m['Recall@50']:>6.3f}  {m['Recall@100']:>6.3f}  "
                          f"{m['Correction@1']:>7.3f}  {m['Correction@5']:>7.3f}")

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

    # ══════════════════════════════════════════════════════════════════
    # KEY COMPARISONS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

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

    # ── Save ──
    out_path = S_OUT / f"v5_results_L{L}_n{args.n}.json"
    out_path.write_text(json.dumps({
        "config": {"vocab_size": VOCAB_SIZE, "L": L, "n_seqs": args.n},
        "results": all_results,
    }, indent=2, default=float))
    print(f"\n[saved] {out_path}")
    print("\nDONE ✓")


if __name__ == "__main__":
    main()
