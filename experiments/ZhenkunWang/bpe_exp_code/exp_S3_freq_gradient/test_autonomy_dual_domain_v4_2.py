#!/usr/bin/env python3
"""V4.2: V4 frequency-gradient experiment with 10-sample averaging per sequence.

Key improvement over V4:
  - V4:  1 error token per bucket per sequence  → high variance
  - V4.2: 10 error tokens per bucket per sequence, averaged
  - Efficiency: model runs ONCE per sequence (clean), then error-token surprise
    is a simple lookup from the pre-computed log-prob matrix.

Each sequence × bucket:
  1. Pick error position t
  2. Run model ONCE on clean sequence → full log-prob matrix
  3. Compute clean word-level surprises (for non-error words)
  4. For each of 10 sampled error tokens:
     - Look up token surprise at position t from log-prob matrix
     - Recompute error-word surprise
     - Rank all words, check if error word is in top-K
  5. Average R@K across 10 samples

Usage:
  python test_autonomy_dual_domain_v4_2.py --device cuda:0 --n 200 --L 128
  python test_autonomy_dual_domain_v4_2.py --device cuda:0 --n 3 --L 64 --smoke
"""

from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

# Vocab & bucket config
BUCKET_EDGES = [0, 500, 1000, 1500, 2000, 2500, 3000,
               3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
BUCKET_LABELS = [
    "0-500", "500-1000", "1000-1500", "1500-2000",
    "2000-2500", "2500-3000", "3000-3500", "3500-4000",
    "4000-4500", "4500-5000", "5000-5500", "5500-6000",
    "6000-6500", "6500-7000",
]
N_SAMPLES = 10  # error tokens to sample per sequence per bucket

S_OUT = OUT_DIR / "test_autonomy_dual_domain_v4_2"
S_OUT.mkdir(parents=True, exist_ok=True)

_DEFAULT_N = 200
_DEFAULT_L = 128


# ===================================================================
# Model builder
# ===================================================================
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


# ===================================================================
# Helpers
# ===================================================================
@torch.no_grad()
def lp_full(model, seq, dev, reverse=False):
    inp = torch.flip(seq.view(1, -1), [1]) if reverse else seq.view(1, -1)
    inp = inp.to(dev)
    hs = model.transformer(input_ids=inp, attention_mask=torch.ones_like(inp)).last_hidden_state
    return F.log_softmax(model.lm_head(hs[0]).float(), -1).cpu().numpy()


def load_id_to_str(vocab_path):
    with open(vocab_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {v: k for k, v in data.items()}
    return {}


def build_word_map(token_ids, id2s):
    word_map = []
    word_id = -1
    for tid in token_ids:
        if isinstance(tid, torch.Tensor):
            tid = tid.item()
        s = id2s.get(tid, '')
        if s in ('<s>', '</s>', '<pad>', '<unk>'):
            word_map.append(-1)
        elif s.startswith('Ġ'):
            word_id += 1
            word_map.append(word_id)
        else:
            word_map.append(max(word_id, 0))
    return word_map


# ===================================================================
# Core: one-pass word-level eval with 10-sample averaging per seq
# ===================================================================
def eval_word_freq_bucket_v2(seqs, model, dev, L, rng, freq_bucket, reverse, desc):
    """Word-level detection with 10-sample averaging per sequence per bucket.

    For each sequence:
      1. Pick error position t, build word map
      2. Run model ONCE on clean sequence → log-prob matrix
      3. Compute clean token-level & word-level surprises
      4. Identify error word w_t and its "other-position max surprise"
      5. For each of 10 sampled error tokens:
         - Look up error surprise at position t
         - Recompute w_t's word surprise
         - Rank all words, check if w_t ∈ top-1/5
      6. Average R@1/R@5 across 10 samples; collect all AUC pairs

    Returns dict with Word-AUC, Word-R@1, Word-R@5, etc.
    """
    n_seqs = len(seqs)
    all_auc_labels = []
    all_auc_scores = []
    seq_rec1 = []  # per-sequence averaged R@1
    seq_rec5 = []  # per-sequence averaged R@5
    n_skipped = 0

    sampling_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

    for si, s_clean in enumerate(tqdm(seqs, desc=desc, unit="seq", leave=False)):
        s = s_clean.clone()
        token_ids = s.tolist()
        word_map = build_word_map(token_ids, id2s)

        # Pick error position
        t = int(rng.randint(L // 8, 7 * L // 8))
        error_word = word_map[t]
        if error_word < 0:
            n_skipped += 1
            continue

        # Run model ONCE on clean sequence
        lp = lp_full(model, s, dev, reverse=reverse)

        # Token-level clean surprises & word-level aggregation
        if not reverse:
            P = np.arange(1, L - 1)
            ar = np.arange(len(P))
            clean_token_surps = -lp[P - 1][ar, [int(s.numpy()[p]) for p in P]]
        else:
            P = np.arange(1, L - 1)
            xp = s.numpy()[P]
            bwd_positions = L - 2 - P
            clean_token_surps = -lp[bwd_positions, xp]

        # Map position → token-level clean surprise
        pos_to_surp = {int(p): float(surp) for p, surp in zip(P, clean_token_surps)}

        # Build clean word-level surprises
        word_clean_surps = {}  # word_id → max surprise
        word_positions = defaultdict(list)  # word_id → [positions]
        for pos in range(1, L - 1):
            w = word_map[pos]
            if w < 0:
                continue
            word_positions[w].append(pos)
            s_val = pos_to_surp[pos]
            if w not in word_clean_surps or s_val > word_clean_surps[w]:
                word_clean_surps[w] = s_val

        # For error word w_t: max surprise excluding position t
        other_positions = [p for p in word_positions[error_word] if p != t]
        other_max = max(pos_to_surp[p] for p in other_positions) if other_positions else -float('inf')

        # Get log-prob vector at position t
        if not reverse:
            lp_t = lp[t - 1]
        else:
            lp_t = lp[L - 2 - t]

        # Sample 10 error tokens & evaluate
        n_available = len(freq_bucket)
        n_sample = min(N_SAMPLES, n_available)
        sampled_tokens = sampling_rng.choice(freq_bucket, size=n_sample, replace=False)
        # Filter out original token
        orig_token = int(s[t])
        sampled_tokens = [int(tok) for tok in sampled_tokens if int(tok) != orig_token]

        rec1_sum = 0
        rec5_sum = 0
        n_eval = 0

        for err_tok in sampled_tokens:
            err_surp = float(-lp_t[err_tok])

            # Error word's new surprise
            new_word_surp = max(err_surp, other_max)

            # Build word-level scores: copy clean, update error word
            word_scores = dict(word_clean_surps)
            word_scores[error_word] = new_word_surp

            # Rank
            sorted_words = sorted(word_scores.items(), key=lambda x: -x[1])
            ranked_word_ids = [w for w, _ in sorted_words]

            # Labels: only error_word is positive
            word_labels = {w: 1 if w == error_word else 0 for w in word_scores}

            # AUC data
            for w, score in word_scores.items():
                all_auc_labels.append(word_labels[w])
                all_auc_scores.append(score)

            # R@K
            if error_word == ranked_word_ids[0]:
                rec1_sum += 1
            if error_word in ranked_word_ids[:5]:
                rec5_sum += 1
            n_eval += 1

        if n_eval > 0:
            seq_rec1.append(rec1_sum / n_eval)
            seq_rec5.append(rec5_sum / n_eval)

    n_valid = n_seqs - n_skipped

    return {
        "Word-AUC": roc_auc_score(np.array(all_auc_labels), np.array(all_auc_scores))
                    if all_auc_labels else 0.5,
        "Word-Recall@1": float(np.mean(seq_rec1)) if seq_rec1 else 0.0,
        "Word-Recall@5": float(np.mean(seq_rec5)) if seq_rec5 else 0.0,
        "Word-Recall@1_std": float(np.std(seq_rec1)) if seq_rec1 else 0.0,
        "Word-Recall@5_std": float(np.std(seq_rec5)) if seq_rec5 else 0.0,
        "n_auc_pairs": len(all_auc_labels),
        "n_seqs": n_seqs,
        "n_valid": n_valid,
        "n_skipped": n_skipped,
        "n_samples_per_seq": N_SAMPLES,
    }


# ===================================================================
# Frequency buckets
# ===================================================================
def build_freq_buckets(counts, bucket_edges):
    sorted_ids = np.argsort(-counts)
    buckets = []
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        bucket = sorted_ids[lo:hi].copy()
        bucket = bucket[counts[bucket] > 0]
        buckets.append(bucket)
    return buckets


# ===================================================================
# Plotting
# ===================================================================
def plot_results(all_results, bucket_labels, out_path, L, n_seqs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available — skipping plot")
        return

    x = np.arange(len(bucket_labels))
    metrics = ["Word-AUC", "Word-Recall@1", "Word-Recall@5"]
    evals = ["D1_fwd", "D2_tokflip", "D3_backrule"]
    eval_labels = {
        "D1_fwd": "D1 Forward (L→R tok)",
        "D2_tokflip": "D2 Tokflip (L→R tok)",
        "D3_backrule": "D3 Backrule (R→L tok)",
    }
    colors = {"D1_fwd": "#1f77b4", "D2_tokflip": "#ff7f0e", "D3_backrule": "#2ca02c"}
    markers = {"D1_fwd": "o", "D2_tokflip": "s", "D3_backrule": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        for eval_key in evals:
            values = [all_results[b][eval_key][metric] for b in range(len(bucket_labels))]
            ax.plot(x, values, color=colors[eval_key], marker=markers[eval_key],
                    linewidth=2, markersize=8, label=eval_labels[eval_key])

            # Add error bars for R@K (std across sequences)
            if "Recall" in metric:
                std_key = metric + "_std"
                stds = [all_results[b][eval_key].get(std_key, 0) for b in range(len(bucket_labels))]
                stds = [s / np.sqrt(max(n_seqs, 1)) for s in stds]  # SEM
                ax.fill_between(x,
                                [v - s for v, s in zip(values, stds)],
                                [v + s for v, s in zip(values, stds)],
                                color=colors[eval_key], alpha=0.12)

        ax.set_xlabel("Frequency Bucket", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("V4.2: Word-Level Detection vs Error-Token Frequency (10-sample avg)\n"
                 f"(n={n_seqs}, L={L})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser(description="V4.2: Frequency-gradient with 10-sample averaging")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=_DEFAULT_N)
    ap.add_argument("--L", type=int, default=_DEFAULT_L)
    ap.add_argument("--vocab", type=int, default=10000, choices=[10000, 20000])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    L = args.L
    rng = np.random.RandomState(0)

    if args.smoke:
        args.n = min(args.n, 3)
        L = min(L, 64)
        print("=" * 70)
        print("SMOKE TEST MODE (V4.2)")
        print("=" * 70)

    # ── Resolve paths based on vocab size ──
    vk = str(args.vocab)
    if args.vocab == 10000:
        VOCAB_SIZE = 18782
        FWD_45M = OUT_DIR / "model_union_fwd_v1_10000_s512_st64_b65k_n10k" / "fwd_lm_s46_best.pt"
        BWD_TF_45M = OUT_DIR / "model_union_bwd_fwdrule_v1_10000_tokflip_s512_st64_b65k_n10k" / "bwd_tokflip_lm_s46_best.pt"
        BWD_BR_45M = OUT_DIR / "model_union_bwd_backrule_v1_10000_s512_st64_b65k_n10k" / "bwd_lm_s46_best.pt"
        FWD_TOK_VAL = OUT_DIR / "tokenized_union_fwd_v1_10000" / "wiki_val.pt"
        BWD_BR_TOK_VAL = OUT_DIR / "tokenized_union_bwd_backrule_v1_10000" / "wiki_val.pt"
        VOCAB_JSON = OUT_DIR / "vocab_union_v1_10000.json"
        # Also need train data for frequency counts
        FWD_TOK_TRAIN = OUT_DIR / "tokenized_union_fwd_v1_10000" / "wiki_train.pt"
        BWD_BR_TOK_TRAIN = OUT_DIR / "tokenized_union_bwd_backrule_v1_10000" / "wiki_train.pt"
    else:
        VOCAB_SIZE = 38001
        FWD_45M = OUT_DIR / "model_union_fwd_v1_20000_s512_st64_b32k_n10k" / "fwd_lm_s46_best.pt"
        BWD_TF_45M = OUT_DIR / "model_union_bwd_fwdrule_v1_20000_tokflip_s512_st64_b32k_n10k" / "bwd_tokflip_lm_s46_best.pt"
        BWD_BR_45M = OUT_DIR / "model_union_bwd_backrule_v1_20000_s512_st64_b32k_n10k" / "bwd_lm_s46_best.pt"
        FWD_TOK_VAL = OUT_DIR / "tokenized_union_fwd_v1_20000" / "wiki_val.pt"
        BWD_BR_TOK_VAL = OUT_DIR / "tokenized_union_bwd_backrule_v1_20000" / "wiki_val.pt"
        VOCAB_JSON = OUT_DIR / "vocab_union_v1_20000.json"
        FWD_TOK_TRAIN = OUT_DIR / "tokenized_union_fwd_v1_20000" / "wiki_train.pt"
        BWD_BR_TOK_TRAIN = OUT_DIR / "tokenized_union_bwd_backrule_v1_20000" / "wiki_train.pt"

    print("=" * 70)
    print(f"V4.2: Word-Level Detection with Frequency-Gradient (10-sample avg/seq)")
    print(f"  device={dev}  L={L}  n_seqs={args.n}  vocab={VOCAB_SIZE:,}")
    print(f"  samples_per_seq_per_bucket={N_SAMPLES}")
    print("=" * 70)
    print()
    for i, lbl in enumerate(BUCKET_LABELS):
        print(f"  bucket {i}: rank {BUCKET_EDGES[i]}–{BUCKET_EDGES[i+1]}  ({lbl})")
    print()

    # ── Load vocab mapping ──
    print("[0] Load vocab & data")
    global id2s
    id2s = load_id_to_str(str(VOCAB_JSON))
    print(f"    vocab entries: {len(id2s)}")

    # Load val data
    blocks_fwd = torch.load(str(FWD_TOK_VAL), map_location="cpu").to(torch.int64)
    n_full = blocks_fwd.numel() // L
    chunks_fwd = blocks_fwd[:n_full * L].view(n_full, L)
    seqs_fwd_all = [chunks_fwd[i].clone() for i in range(n_full)]
    rng.shuffle(seqs_fwd_all)
    seqs_fwd = seqs_fwd_all[:args.n]
    print(f"    fwd-tok val:    {len(seqs_fwd)} sequences")

    blocks_rev = torch.load(str(BWD_BR_TOK_VAL), map_location="cpu").to(torch.int64)
    n_full_rev = blocks_rev.numel() // L
    chunks_rev = blocks_rev[:n_full_rev * L].view(n_full_rev, L)
    seqs_rev_all = [chunks_rev[i].clone() for i in range(n_full_rev)]
    rng.shuffle(seqs_rev_all)
    seqs_rev = seqs_rev_all[:args.n]
    print(f"    backrule-tok val: {len(seqs_rev)} sequences")

    # ── Build frequency buckets from TRAIN+VAL data ──
    print("\n[1] Build frequency buckets (train + val counts)")
    train_fwd = torch.load(str(FWD_TOK_TRAIN), map_location="cpu").to(torch.int64).reshape(-1)
    val_fwd = torch.load(str(FWD_TOK_VAL), map_location="cpu").to(torch.int64).reshape(-1)

    if args.vocab == 10000:
        train_bwd = torch.load(str(BWD_BR_TOK_TRAIN), map_location="cpu").to(torch.int64).reshape(-1)
        val_bwd = torch.load(str(BWD_BR_TOK_VAL), map_location="cpu").to(torch.int64).reshape(-1)
    else:
        # For 20k, backrule train might not exist; fallback to val-only
        # Actually let's check
        if BWD_BR_TOK_TRAIN.exists():
            train_bwd = torch.load(str(BWD_BR_TOK_TRAIN), map_location="cpu").to(torch.int64).reshape(-1)
            val_bwd = torch.load(str(BWD_BR_TOK_VAL), map_location="cpu").to(torch.int64).reshape(-1)
        else:
            train_bwd = torch.load(str(BWD_BR_TOK_VAL), map_location="cpu").to(torch.int64).reshape(-1)
            val_bwd = train_bwd

    counts_fwd = torch.bincount(torch.cat([train_fwd, val_fwd]), minlength=VOCAB_SIZE).numpy()
    counts_bwd = torch.bincount(torch.cat([train_bwd, val_bwd]), minlength=VOCAB_SIZE).numpy()

    fwd_buckets = build_freq_buckets(counts_fwd, BUCKET_EDGES)
    bwd_buckets = build_freq_buckets(counts_bwd, BUCKET_EDGES)

    print(f"    Forward buckets:")
    for i, (b, lbl) in enumerate(zip(fwd_buckets, BUCKET_LABELS)):
        print(f"      bucket {i} ({lbl}): {len(b)} tokens, "
              f"count range [{counts_fwd[b].min():,}, {counts_fwd[b].max():,}]")
    print(f"    Backrule buckets:")
    for i, (b, lbl) in enumerate(zip(bwd_buckets, BUCKET_LABELS)):
        print(f"      bucket {i} ({lbl}): {len(b)} tokens, "
              f"count range [{counts_bwd[b].min():,}, {counts_bwd[b].max():,}]")

    # ── Load 45M models ──
    print("\n[2] Load 45M evaluation models")
    fwd_45m = build_model(VOCAB_SIZE, dev, "45M")
    fwd_45m.load_state_dict(torch.load(str(FWD_45M), map_location=dev))
    fwd_45m.eval()
    print(f"    Forward 45M:      {sum(p.numel() for p in fwd_45m.parameters())/1e6:.1f}M")

    bwd_tf_45m = build_model(VOCAB_SIZE, dev, "45M")
    bwd_tf_45m.load_state_dict(torch.load(str(BWD_TF_45M), map_location=dev))
    bwd_tf_45m.eval()
    print(f"    Tokflip 45M:      {sum(p.numel() for p in bwd_tf_45m.parameters())/1e6:.1f}M")

    bwd_br_45m = build_model(VOCAB_SIZE, dev, "45M")
    bwd_br_45m.load_state_dict(torch.load(str(BWD_BR_45M), map_location=dev))
    bwd_br_45m.eval()
    print(f"    Backrule 45M:     {sum(p.numel() for p in bwd_br_45m.parameters())/1e6:.1f}M")

    dummy = torch.randint(0, VOCAB_SIZE, (2, L), device=dev)
    assert fwd_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    assert bwd_tf_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    assert bwd_br_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    print("    Sanity check ✓")

    # ══════════════════════════════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════════════════════════════
    t0 = time.time()
    all_results = []

    for bucket_idx in range(len(BUCKET_LABELS)):
        bucket_label = BUCKET_LABELS[bucket_idx]
        print(f"\n{'='*70}")
        print(f"BUCKET {bucket_idx}: {bucket_label}")
        print(f"{'='*70}")

        br = {
            "bucket_idx": bucket_idx,
            "bucket_label": bucket_label,
            "bucket_range": [BUCKET_EDGES[bucket_idx], BUCKET_EDGES[bucket_idx + 1]],
        }

        # D1: Forward
        print(f"\n  D1: Forward 45M (L→R) — fwd-freq bucket {bucket_idx}")
        br["D1_fwd"] = eval_word_freq_bucket_v2(
            seqs_fwd, fwd_45m, dev, L, rng, fwd_buckets[bucket_idx],
            reverse=False, desc=f"  D1 fwd {bucket_label}")

        # D2: Tokflip
        print(f"\n  D2: Tokflip 45M (R→L) — fwd-freq bucket {bucket_idx}")
        br["D2_tokflip"] = eval_word_freq_bucket_v2(
            seqs_fwd, bwd_tf_45m, dev, L, rng, fwd_buckets[bucket_idx],
            reverse=True, desc=f"  D2 tokflip {bucket_label}")

        # D3: Backrule
        print(f"\n  D3: Backrule 45M (R→L) — bwd-freq bucket {bucket_idx}")
        br["D3_backrule"] = eval_word_freq_bucket_v2(
            seqs_rev, bwd_br_45m, dev, L, rng, bwd_buckets[bucket_idx],
            reverse=False, desc=f"  D3 backrule {bucket_label}")

        all_results.append(br)

    dt = time.time() - t0

    del fwd_45m, bwd_tf_45m, bwd_br_45m
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"RESULTS  ({dt:.1f}s)")
    print(f"{'='*70}")

    hdr = (f"  {'Bucket':<14}  {'D1 W-AUC':>9}  {'D1 R@1':>7}  {'D1 R@5':>7}  "
           f"{'D2 W-AUC':>9}  {'D2 R@1':>7}  {'D2 R@5':>7}  "
           f"{'D3 W-AUC':>9}  {'D3 R@1':>7}  {'D3 R@5':>7}")
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))
    for br in all_results:
        d1 = br["D1_fwd"]; d2 = br["D2_tokflip"]; d3 = br["D3_backrule"]
        print(f"  {br['bucket_label']:<14}  "
              f"{d1['Word-AUC']:>9.4f}  {d1['Word-Recall@1']:>7.3f}  {d1['Word-Recall@5']:>7.3f}  "
              f"{d2['Word-AUC']:>9.4f}  {d2['Word-Recall@1']:>7.3f}  {d2['Word-Recall@5']:>7.3f}  "
              f"{d3['Word-AUC']:>9.4f}  {d3['Word-Recall@1']:>7.3f}  {d3['Word-Recall@5']:>7.3f}")

    # ── Save ──
    out_json = S_OUT / f"v4_2_results_L{L}_n{args.n}.json"
    out_json.write_text(json.dumps({
        "config": {
            "vocab_size": VOCAB_SIZE, "L": L, "n_seqs": args.n,
            "n_samples_per_seq": N_SAMPLES,
            "bucket_edges": BUCKET_EDGES, "bucket_labels": BUCKET_LABELS,
        },
        "results": all_results,
    }, indent=2, default=float))
    print(f"\n[saved] {out_json}")

    out_plot = S_OUT / f"v4_2_freq_gradient_L{L}_n{args.n}.png"
    plot_results(all_results, BUCKET_LABELS, out_plot, L, args.n)

    print("\nDONE ✓")


if __name__ == "__main__":
    main()
