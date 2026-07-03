#!/usr/bin/env python3
"""V5.2: Compare token frequency distributions between training and validation sets.

Checks whether test-set token frequencies match training-set frequencies,
to determine if "low-frequency test token" ≈ "low-frequency training token"
(which would explain higher surprise for low-frequency tokens).

Output:
  - Frequency rank scatter: train rank vs val rank
  - Spearman / Pearson correlation
  - Per-bin frequency comparison
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data"
VOCAB_SIZE = 18782

FWD_TRAIN = OUT_DIR / "tokenized_union_fwd_v1_10000" / "wiki_train.pt"
FWD_VAL = OUT_DIR / "tokenized_union_fwd_v1_10000" / "wiki_val.pt"

S_OUT = OUT_DIR / "test_autonomy_dual_domain_v5"
S_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import torch

    print("=" * 70)
    print("V5.2: Train vs Validation Token Frequency Distribution")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    train_ids = torch.load(str(FWD_TRAIN), map_location="cpu").to(torch.int64).reshape(-1)
    val_ids = torch.load(str(FWD_VAL), map_location="cpu").to(torch.int64).reshape(-1)
    print(f"    Train tokens: {len(train_ids):,}")
    print(f"    Val tokens:   {len(val_ids):,}")

    # Compute frequency counts
    print("\n[2] Computing token frequencies...")
    train_counts = torch.bincount(train_ids, minlength=VOCAB_SIZE).numpy().astype(np.int64)
    val_counts = torch.bincount(val_ids, minlength=VOCAB_SIZE).numpy().astype(np.int64)

    # Normalize to per-million
    train_total = train_counts.sum()
    val_total = val_counts.sum()
    train_freq = train_counts / train_total * 1e6
    val_freq = val_counts / val_total * 1e6

    # Compute frequency ranks (0 = most frequent)
    train_rank = np.zeros(VOCAB_SIZE, dtype=np.int32)
    val_rank = np.zeros(VOCAB_SIZE, dtype=np.int32)
    for rank, tok in enumerate(np.argsort(-train_counts)):
        train_rank[tok] = rank
    for rank, tok in enumerate(np.argsort(-val_counts)):
        val_rank[tok] = rank

    # Only consider tokens that appear in both sets
    mask = (train_counts > 0) & (val_counts > 0)
    n_both = mask.sum()
    print(f"    Tokens in both train & val: {n_both} / {VOCAB_SIZE}")

    train_rank_both = train_rank[mask]
    val_rank_both = val_rank[mask]
    train_freq_both = train_freq[mask]
    val_freq_both = val_freq[mask]

    # Correlations
    from scipy.stats import spearmanr, pearsonr

    spearman_r, spearman_p = spearmanr(train_rank_both, val_rank_both)
    pearson_r, pearson_p = pearsonr(np.log10(train_freq_both + 1), np.log10(val_freq_both + 1))

    print(f"\n[3] Rank correlation (tokens present in both sets, n={n_both}):")
    print(f"    Spearman ρ (rank):  {spearman_r:.6f}  (p = {spearman_p:.2e})")
    print(f"    Pearson r (log10 freq): {pearson_r:.6f}  (p = {pearson_p:.2e})")

    # Check top-K overlap
    for k in [100, 500, 1000, 2000, 5000]:
        train_topk = set(np.argsort(-train_counts)[:k])
        val_topk = set(np.argsort(-val_counts)[:k])
        overlap = len(train_topk & val_topk)
        print(f"    Top-{k:>5} overlap: {overlap:>5}/{k}  ({100*overlap/k:.1f}%)")

    # Check: for tokens in val's top-2000 (our Part D range), what's their train rank?
    val_top2000 = np.argsort(-val_counts)[:2000]
    train_ranks_of_val_top2000 = train_rank[val_top2000]
    print(f"\n    Val top-2000 tokens: train rank mean={train_ranks_of_val_top2000.mean():.1f}, "
          f"median={np.median(train_ranks_of_val_top2000):.1f}, "
          f"max={train_ranks_of_val_top2000.max()}")

    # ═══════════════════════════════════════════════════════════
    # Plot
    # ═══════════════════════════════════════════════════════════
    if not HAS_MPL:
        print("\n[skip plot] matplotlib not available")
        return

    print("\n[4] Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Rank scatter ---
    ax = axes[0]
    # Sample for scatter (too many points)
    n_sample = min(5000, n_both)
    idx = np.random.RandomState(42).choice(n_both, size=n_sample, replace=False)
    ax.scatter(train_rank_both[idx], val_rank_both[idx], s=2, alpha=0.3, c="#2196F3", edgecolors="none")
    ax.plot([0, VOCAB_SIZE], [0, VOCAB_SIZE], "k--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("Train Frequency Rank (0 = most frequent)")
    ax.set_ylabel("Val Frequency Rank")
    ax.set_title(f"Rank Scatter (n={n_sample} sampled)\nSpearman ρ = {spearman_r:.4f}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Frequency scatter (log-log) ---
    ax = axes[1]
    ax.scatter(train_freq_both[idx], val_freq_both[idx], s=2, alpha=0.3, c="#FF9800", edgecolors="none")
    ax.plot([1e-1, 1e5], [1e-1, 1e5], "k--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Train Frequency (per million)")
    ax.set_ylabel("Val Frequency (per million)")
    ax.set_title(f"Frequency Scatter (log-log)\nPearson r (log10) = {pearson_r:.4f}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Rank difference distribution ---
    ax = axes[2]
    rank_diff = train_rank_both - val_rank_both
    ax.hist(rank_diff, bins=80, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Train Rank − Val Rank")
    ax.set_ylabel("Token Count")
    ax.set_title(f"Rank Difference Distribution\n"
                 f"Mean={rank_diff.mean():+.1f}, Std={rank_diff.std():.1f}, "
                 f"Median={np.median(rank_diff):+.1f}",
                 fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Train vs Validation Token Frequency Distribution (Forward BPE, 10k vocab)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = S_OUT / "v5_2_train_val_freq_dist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {out_path}")

    # ── Save data ──
    out_json = S_OUT / "v5_2_train_val_freq.json"
    out_json.write_text(json.dumps({
        "n_train_tokens": int(train_total),
        "n_val_tokens": int(val_total),
        "tokens_in_both": int(n_both),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r_log10": float(pearson_r),
        "pearson_p": float(pearson_p),
        "top_k_overlap": {
            "100": int(len(set(np.argsort(-train_counts)[:100]) & set(np.argsort(-val_counts)[:100]))),
            "500": int(len(set(np.argsort(-train_counts)[:500]) & set(np.argsort(-val_counts)[:500]))),
            "1000": int(len(set(np.argsort(-train_counts)[:1000]) & set(np.argsort(-val_counts)[:1000]))),
            "2000": int(len(set(np.argsort(-train_counts)[:2000]) & set(np.argsort(-val_counts)[:2000]))),
            "5000": int(len(set(np.argsort(-train_counts)[:5000]) & set(np.argsort(-val_counts)[:5000]))),
        },
    }, indent=2))
    print(f"    [saved] {out_json}")
    print("\nDONE ✓")


if __name__ == "__main__":
    main()
