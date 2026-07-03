#!/usr/bin/env python3
"""Train forward and reverse-corpus BPE tokenizers on wiki text (HuggingFace tokenizers backend).

Uses wiki text decoded by ``decode_wiki_to_text.py`` (wiki_train.txt / wiki_val.txt).
BPE training is backed by the HuggingFace ``tokenizers`` library (Rust) — orders of
magnitude faster than pure Python.

Trains TWO byte-level BPE tokenizers from the same raw text:
  - Forward BPE:  standard training on normal text.
  - Reverse-corpus BPE: each line reversed, BPE trained, token strings stay in
    reversed order (e.g. "ehT" for "The").  Use reverse_vocab() for forward-order
    comparison.

Pipeline:
  1. Verify wiki text files exist (run decode_wiki_to_text.py first if missing).
  2. Train forward BPE on a fixed subset of wiki_train.txt  → tokenizer_forward_{vocab_size}/
  3. Train reverse-corpus BPE on the same fixed subset       → tokenizer_bwd_rev_{vocab_size}/
  4. Report vocabulary comparison (intersection, union, Jaccard)
  5. Optionally tokenize the full train/val corpus with both tokenizers.

Only a fixed fraction (--tok_train_frac, default 0.2) of training lines is used for
BPE training; the subset is drawn with a fixed seed (42) so it is deterministic across
runs.  The validation set is NEVER used for tokenizer training.

Usage:
  python train_backward_bpe.py --phase 1 --vocab_size 10000
  python train_backward_bpe.py --phase all --vocab_size 50257
"""
from __future__ import annotations

import argparse, json, sys, time, tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE / "data"


# ===================================================================
# Utility: reverse vocabulary token strings
# ===================================================================

def reverse_vocab(vocab: dict[str, int]) -> dict[str, int]:
    """Reverse every token string in a vocabulary dict.

    Used to bring reverse-corpus BPE vocabulary back to forward order for
    comparison with the forward vocabulary.

    Args:
        vocab: {token_str: id} dict (token strings in reversed order).

    Returns:
        {reversed_token_str: id} dict (token strings in forward order).
    """
    return {k[::-1]: v for k, v in vocab.items()}


def vocab_set_operations(vocab_a: dict[str, int], vocab_b: dict[str, int]) -> dict:
    """Compute set operations between two vocabularies.

    Args:
        vocab_a: First vocabulary {token_str: id}.
        vocab_b: Second vocabulary {token_str: id}.

    Returns:
        Dict with keys: size_a, size_b, intersection, union,
        tokens_only_in_a, tokens_only_in_b, jaccard, overlap_a, overlap_b.
    """
    set_a = set(vocab_a.keys())
    set_b = set(vocab_b.keys())
    inter = set_a & set_b
    union = set_a | set_b
    return {
        "size_a": len(set_a),
        "size_b": len(set_b),
        "intersection": inter,
        "union": union,
        "tokens_only_in_a": set_a - set_b,
        "tokens_only_in_b": set_b - set_a,
        "jaccard": len(inter) / len(union) if union else 0.0,
        "overlap_a": len(inter) / len(set_a) if set_a else 0.0,
        "overlap_b": len(inter) / len(set_b) if set_b else 0.0,
    }


# ===================================================================
# Step 0: Verify wiki text files
# ===================================================================

def verify_wiki_text(output_dir: Path, n_docs: int = 0) -> dict[str, Path]:
    """Verify that wiki text files exist (produced by decode_wiki_to_text.py).

    Args:
        output_dir: Directory containing wiki_train.txt / wiki_val.txt.
        n_docs: If > 0, truncate each split to at most this many lines.

    Returns:
        {"train": train_path, "val": val_path}
    """
    train_path = output_dir / "wiki_train.txt"
    val_path = output_dir / "wiki_val.txt"

    for p, name in [(train_path, "wiki_train.txt"), (val_path, "wiki_val.txt")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run decode_wiki_to_text.py first"
            )

    n_train = sum(1 for _ in open(train_path, encoding="utf-8"))
    n_val = sum(1 for _ in open(val_path, encoding="utf-8"))
    print(f"[Init] Wiki text files found: train={n_train:,} lines, val={n_val:,} lines")

    if n_docs > 0:
        for split_name, path, max_lines in [
            ("train", train_path, min(n_docs, n_train)),
            ("val", val_path, min(max(1, n_docs // 10), n_val)),
        ]:
            if max_lines < (n_train if split_name == "train" else n_val):
                truncated_path = output_dir / f"wiki_{split_name}_truncated.txt"
                if not truncated_path.exists():
                    lines = []
                    with open(path, encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i >= max_lines:
                                break
                            lines.append(line)
                    truncated_path.write_text("".join(lines), encoding="utf-8")
                    print(f"[Init] Truncated {split_name} to {max_lines:,} lines → {truncated_path}")
                if split_name == "train":
                    train_path = truncated_path
                else:
                    val_path = truncated_path

    return {"train": train_path, "val": val_path}


# ===================================================================
# Phase 1: Tokenizer training (HuggingFace tokenizers backend)
# ===================================================================

def _build_training_subset(train_path: Path, tok_train_frac: float,
                           seed: int, output_dir: Path) -> Path:
    """Sample a fixed fraction of training lines for BPE training.

    Returns path to the subset file (one line per doc).
    """
    train_lines = [l.strip() for l in open(train_path, encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(seed)
    n_tok_train = max(1, int(len(train_lines) * tok_train_frac))
    idx = rng.permutation(len(train_lines))[:n_tok_train]
    tok_train_lines = [train_lines[i] for i in idx]
    tok_train_path = output_dir / "wiki_train_tok_subset.txt"
    tok_train_path.write_text("\n".join(tok_train_lines), encoding="utf-8")
    total_chars = sum(len(l) for l in tok_train_lines)
    print(f"[Tokenizer] Using {n_tok_train:,}/{len(train_lines):,} "
          f"({tok_train_frac:.0%}) training docs for BPE training "
          f"({total_chars/1e6:.1f}M chars, seed={seed})")
    return tok_train_path


def _train_hf_bpe(files: list[str], vocab_size: int, min_frequency: int) -> Tokenizer:
    """Train a byte-level BPE tokenizer using HuggingFace tokenizers (Rust backend).

    Args:
        files: List of corpus file paths.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum pair frequency for a merge.

    Returns:
        A trained HuggingFace Tokenizer.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True,
    )

    tokenizer.train(files, trainer)
    return tokenizer


def train_tokenizers(train_path: Path, vocab_size: int, min_frequency: int,
                     output_dir: Path, tok_train_frac: float = 0.2,
                     seed: int = 42) -> dict[str, Tokenizer]:
    """Train both forward and reverse-corpus BPE tokenizers.

    Uses only ``tok_train_frac`` of the training docs for BPE training.
    The validation set is NEVER touched.

    Returns dict with keys "forward" (HF Tokenizer) and "bwd_rev" (HF Tokenizer
    with token strings in reversed-text order).
    """
    # Build fixed training subset
    tok_train_path = _build_training_subset(train_path, tok_train_frac, seed, output_dir)

    tokenizers_out: dict[str, Tokenizer] = {}

    # --- Forward BPE ---
    fwd_dir = output_dir / f"tokenizer_forward_{vocab_size}"
    fwd_file = fwd_dir / "tokenizer.json"
    t0 = time.time()
    if fwd_file.exists():
        print(f"[Forward] Loading existing tokenizer from {fwd_dir}")
        tokenizers_out["forward"] = Tokenizer.from_file(str(fwd_file))
    else:
        print(f"[Forward] Training BPE (vocab_size={vocab_size}, "
              f"min_freq={min_frequency}) ...")
        tokenizers_out["forward"] = _train_hf_bpe(
            [str(tok_train_path)], vocab_size, min_frequency,
        )
        fwd_dir.mkdir(parents=True, exist_ok=True)
        tokenizers_out["forward"].save(str(fwd_file))
        dt = time.time() - t0
        print(f"[Forward] Done in {dt:.1f}s → {fwd_dir}")

    fwd_vocab = tokenizers_out["forward"].get_vocab()
    print(f"[Forward] vocab_size={len(fwd_vocab):,}")

    # --- Reverse-corpus BPE ---
    rev_dir = output_dir / f"tokenizer_bwd_rev_{vocab_size}"
    rev_file = rev_dir / "tokenizer.json"
    t0 = time.time()
    if rev_file.exists():
        print(f"[BwdRev] Loading existing tokenizer from {rev_dir}")
        tokenizers_out["bwd_rev"] = Tokenizer.from_file(str(rev_file))
    else:
        # Build reversed corpus in a temporary file
        reversed_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False,
        )
        with open(tok_train_path, encoding="utf-8") as fin:
            for line in fin:
                reversed_tmp.write(line.rstrip("\n\r")[::-1] + "\n")
        reversed_path = reversed_tmp.name
        reversed_tmp.close()

        try:
            print(f"[BwdRev] Training BPE on REVERSED corpus "
                  f"(vocab_size={vocab_size}, min_freq={min_frequency}) ...")
            tokenizers_out["bwd_rev"] = _train_hf_bpe(
                [reversed_path], vocab_size, min_frequency,
            )
            rev_dir.mkdir(parents=True, exist_ok=True)
            tokenizers_out["bwd_rev"].save(str(rev_file))
            dt = time.time() - t0
            print(f"[BwdRev] Done in {dt:.1f}s → {rev_dir}")
        finally:
            Path(reversed_path).unlink(missing_ok=True)

    rev_vocab = tokenizers_out["bwd_rev"].get_vocab()
    print(f"[BwdRev] vocab_size={len(rev_vocab):,}")

    # --- Vocabulary comparison ---
    rev_vocab_fwd = reverse_vocab(rev_vocab)
    stats = vocab_set_operations(fwd_vocab, rev_vocab_fwd)
    print(f"\n[Compare] |forward|={stats['size_a']:,}, |reverse|={stats['size_b']:,}, "
          f"|intersection|={len(stats['intersection']):,}, |union|={len(stats['union']):,}")
    print(f"[Compare] Jaccard={stats['jaccard']:.4f}, "
          f"forward-overlap={stats['overlap_a']:.3f}, reverse-overlap={stats['overlap_b']:.3f}")

    fwd_only = sorted(stats['tokens_only_in_a'], key=lambda t: -len(t))[:10]
    rev_only = sorted(stats['tokens_only_in_b'], key=lambda t: -len(t))[:10]
    inter = sorted(stats['intersection'], key=lambda t: -len(t))[:10]
    print(f"[Compare] Forward-only examples (longest): {fwd_only}")
    print(f"[Compare] Reverse-only examples (longest): {rev_only}")
    print(f"[Compare] Intersection examples (longest): {inter}")

    comp = {
        "size_a": stats["size_a"], "size_b": stats["size_b"],
        "n_intersection": len(stats["intersection"]),
        "n_union": len(stats["union"]),
        "n_only_forward": len(stats["tokens_only_in_a"]),
        "n_only_reverse": len(stats["tokens_only_in_b"]),
        "jaccard": stats["jaccard"],
        "overlap_forward": stats["overlap_a"],
        "overlap_reverse": stats["overlap_b"],
        "forward_only_longest": fwd_only,
        "reverse_only_longest": rev_only,
        "intersection_longest": inter,
    }
    (output_dir / "vocab_comparison.json").write_text(
        json.dumps(comp, indent=2, ensure_ascii=False),
    )

    return tokenizers_out


# ===================================================================
# Phase 2: Corpus tokenization
# ===================================================================

def tokenize_split(tok: Tokenizer, text_path: Path, output_dir: Path,
                   direction: str, reverse_text: bool = False) -> Path:
    """Encode a text file with the given tokenizer and save as flat .pt tensor.

    Args:
        tok: HuggingFace Tokenizer.
        text_path: Path to raw text file (one doc per line).
        output_dir: Output directory.
        direction: "forward" or "backward" (for logging).
        reverse_text: If True, reverse each line *before* tokenization.
            Used for bwd_rev tokenizer (trained on reversed text).

    Returns:
        Path to the saved .pt file.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = text_path.stem
    out_path = output_dir / f"{stem}.pt"

    lines = []
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    all_ids: list[int] = []
    for i, text in enumerate(lines):
        if reverse_text:
            text = text[::-1]
        try:
            ids = tok.encode(text).ids
            # HF byte-level BPE adds a leading space token; strip it for
            # concatenated flat-tensor storage (like GPT-2 tokenization).
            if ids:
                all_ids.extend(ids)
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            print(f"  [{direction}] {i+1}/{len(lines)} docs, "
                  f"{len(all_ids):,} tokens", flush=True)

    tensor = torch.tensor(all_ids, dtype=torch.int64)
    torch.save(tensor, str(out_path))

    meta = {
        "source": str(text_path),
        "tokenizer_direction": direction,
        "reverse_text": reverse_text,
        "n_docs": len(lines),
        "n_tokens": len(all_ids),
        "vocab_size": tok.get_vocab_size(),
    }
    json.dump(meta, (output_dir / f"{stem}_meta.json").open("w"), indent=2)

    print(f"  [{direction}] {len(lines)} docs → {len(all_ids):,} tokens → {out_path}",
          flush=True)
    return out_path


def run_phase2(tokenizers: dict[str, Tokenizer], splits: dict[str, Path],
               output_dir: Path,
               vocab_size: int = 10000) -> dict[str, dict[str, Path]]:
    """Tokenize train and val splits with both tokenizers.

    Forward:  standard text → forward BPE encode.
    Backward: reversed text → bwd_rev BPE encode (merge rules match reversed text).
    """
    import torch

    print(f"\n[Phase2] Tokenizing corpus with both tokenizers ...")
    data_paths: dict[str, dict[str, Path]] = {}

    # --- Forward tokenization ---
    fwd_dir = output_dir / f"tokenized_forward_{vocab_size}"
    fwd_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Phase2-Forward] Standard encode with forward BPE ...")
    data_paths["forward"] = {}
    for split_name in ("train", "val"):
        data_paths["forward"][split_name] = tokenize_split(
            tok=tokenizers["forward"],
            text_path=splits[split_name],
            output_dir=fwd_dir,
            direction=f"fwd-{split_name}",
            reverse_text=False,
        )
    train_tensor = torch.load(str(data_paths["forward"]["train"]), map_location="cpu")
    decoded = tokenizers["forward"].decode(train_tensor[:50].tolist())
    print(f"  [forward] decode check (first 50 tokens): '{decoded[:100]}...'")

    # --- Backward tokenization ---
    bwd_dir = output_dir / f"tokenized_backward_{vocab_size}"
    bwd_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Phase2-Backward] Reverse text → encode with bwd_rev BPE ...")
    data_paths["backward"] = {}
    for split_name in ("train", "val"):
        data_paths["backward"][split_name] = tokenize_split(
            tok=tokenizers["bwd_rev"],
            text_path=splits[split_name],
            output_dir=bwd_dir,
            direction=f"bwd-{split_name}",
            reverse_text=True,
        )
    train_tensor = torch.load(str(data_paths["backward"]["train"]), map_location="cpu")
    decoded = tokenizers["bwd_rev"].decode(train_tensor[:50].tolist())
    print(f"  [backward] decode check (first 50 tokens → reversed): '{decoded[:100]}...'")
    print(f"  [backward] decode check (un-reversed): '{decoded[::-1][:100]}...'")

    return data_paths


# ===================================================================
# Validation
# ===================================================================

def validate_tokenizers(tokenizers: dict[str, Tokenizer],
                        val_path: Path) -> None:
    """Quick validation: encode/decode roundtrip on held-out val set."""
    print(f"\n[Validate] Testing encode/decode on held-out val set ...")

    val_lines = []
    with open(val_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            line = line.strip()
            if line:
                val_lines.append(line)

    for name, tok in [("forward", tokenizers["forward"]),
                       ("bwd_rev", tokenizers["bwd_rev"])]:
        for text in val_lines[:3]:
            text_short = text[:80] + ("..." if len(text) > 80 else "")
            try:
                if name == "bwd_rev":
                    ids = tok.encode(text[::-1]).ids
                    decoded = tok.decode(ids)
                    # ByteLevel BPE adds a leading space; reversing moves it to
                    # the end — strip both sides for comparison.
                    match = "✓" if decoded[::-1].strip() == text else "✗"
                else:
                    ids = tok.encode(text).ids
                    decoded = tok.decode(ids)
                    # ByteLevel BPE with add_prefix_space=True adds a leading
                    # space; strip for roundtrip comparison.
                    match = "✓" if decoded.lstrip() == text else "✗"
                print(f"  [{name}] {match} '{text_short}' → {len(ids)} tokens")
                if match == "✗":
                    if name == "bwd_rev":
                        print(f"         expected: '{text[:80]}...'")
                        print(f"         un-reversed decoded: '{decoded[::-1][:80]}...'")
                    else:
                        print(f"         decoded: '{decoded[:80]}...'")
            except Exception as e:
                print(f"  [{name}] ✗ ERROR: {e}")


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Train forward + reverse-corpus BPE tokenizers on wiki text "
                    "(HuggingFace tokenizers backend — Rust, fast)")
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all",
                    help="1=tokenizer training only, 2=tokenization only, all=both")
    ap.add_argument("--n_docs", type=int, default=0,
                    help="Max training docs to use (0=all lines in wiki_train.txt)")
    ap.add_argument("--vocab_size", type=int, default=10000,
                    help="Target BPE vocabulary size (default: 10000)")
    ap.add_argument("--min_frequency", type=int, default=2,
                    help="Min pair frequency for BPE merge (default: 2)")
    ap.add_argument("--tok_train_frac", type=float, default=0.2,
                    help="Fraction of training docs used for BPE training (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="",
                    help="Override output directory")
    ap.add_argument("--skip_forward", action="store_true",
                    help="Skip forward tokenizer training")
    ap.add_argument("--skip_reverse", action="store_true",
                    help="Skip reverse-corpus tokenizer training")

    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BPE Tokenizer Training (wiki text, HuggingFace tokenizers backend)")
    print(f"  phase={args.phase}, n_docs={args.n_docs if args.n_docs else 'all'}")
    print(f"  vocab_size={args.vocab_size}, min_frequency={args.min_frequency}")
    print(f"  tok_train_frac={args.tok_train_frac}, seed={args.seed}")
    print(f"  output → {out_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 0: Verify wiki text files exist
    # ------------------------------------------------------------------
    splits = verify_wiki_text(output_dir=out_dir, n_docs=args.n_docs)

    tokenizers_out: dict[str, Tokenizer] = {}

    # ------------------------------------------------------------------
    # Phase 1: Train tokenizers
    # ------------------------------------------------------------------
    if args.phase in ("1", "all"):
        if not args.skip_forward or not args.skip_reverse:
            tokenizers_out = train_tokenizers(
                train_path=splits["train"],
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                output_dir=out_dir,
                tok_train_frac=args.tok_train_frac,
                seed=args.seed,
            )
            validate_tokenizers(tokenizers_out, splits["val"])
        if args.phase == "1":
            print(f"\n{'=' * 60}")
            print(f"Phase 1 done. Outputs in {out_dir}/")
            print(f"  tokenizer_forward_{args.vocab_size}/    — forward BPE tokenizer")
            print(f"  tokenizer_bwd_rev_{args.vocab_size}/    — reverse-corpus BPE tokenizer")
            print(f"  vocab_comparison.json — intersection/union stats")
            print(f"{'=' * 60}")
            return
    else:
        fwd_file = out_dir / f"tokenizer_forward_{args.vocab_size}" / "tokenizer.json"
        rev_file = out_dir / f"tokenizer_bwd_rev_{args.vocab_size}" / "tokenizer.json"
        if fwd_file.exists():
            tokenizers_out["forward"] = Tokenizer.from_file(str(fwd_file))
            print(f"[Init] Loaded forward tokenizer "
                  f"(vocab={tokenizers_out['forward'].get_vocab_size():,})")
        else:
            print(f"[Init] WARNING: tokenizer_forward_{args.vocab_size}/tokenizer.json "
                  f"not found — skipping")
        if rev_file.exists():
            tokenizers_out["bwd_rev"] = Tokenizer.from_file(str(rev_file))
            print(f"[Init] Loaded bwd_rev tokenizer "
                  f"(vocab={tokenizers_out['bwd_rev'].get_vocab_size():,})")
        else:
            print(f"[Init] WARNING: tokenizer_bwd_rev_{args.vocab_size}/tokenizer.json "
                  f"not found — skipping")

    # ------------------------------------------------------------------
    # Phase 2: Tokenize corpus with both tokenizers
    # ------------------------------------------------------------------
    if args.phase in ("2", "all"):
        if "forward" not in tokenizers_out or "bwd_rev" not in tokenizers_out:
            print("[Phase2] ERROR: need both tokenizers. Run --phase 1 first.")
            return

        data_paths = run_phase2(tokenizers_out, splits, out_dir, vocab_size=args.vocab_size)

        print(f"\n{'=' * 60}")
        print(f"Phase 2 done. Outputs in {out_dir}/")
        print(f"  tokenized_forward_{args.vocab_size}/  — forward BPE tokenized .pt files")
        print(f"    {data_paths['forward']['train'].name}")
        print(f"    {data_paths['forward']['val'].name}")
        print(f"  tokenized_backward_{args.vocab_size}/ — backward (reversed) tokenized .pt files")
        print(f"    {data_paths['backward']['train'].name}")
        print(f"    {data_paths['backward']['val'].name}")
        print(f"{'=' * 60}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
