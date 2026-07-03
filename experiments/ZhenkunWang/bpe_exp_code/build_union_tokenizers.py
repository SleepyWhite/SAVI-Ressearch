#!/usr/bin/env python3
"""Build union-vocabulary tokenizers from forward and reverse-corpus BPE tokenizers.

Loads the forward and reverse-corpus BPE tokenizers (HuggingFace format), computes
the union of their vocabularies, and creates THREE tokenizers that share the SAME
vocabulary but use different merge-rule priorities:

  1. union_fwd_<N>        — forward rules,  forward direction (normal text)
  2. union_bwd_backrule_<N> — backward rules, backward direction (reversed text)
  3. union_fwd_tokflip_<N>  — forward rules,  forward direction (normal text)
                              for use with token-level flip backward LM
                              (ids[::-1] per doc in train_union_lm.py)

All three share an identical token→ID mapping, making cross-direction comparisons
(surprisal deltas, logit differences) well-defined for joint forward+backward LMs.

The backward tokenizer's vocabulary is in *reversed-text order* (e.g. "ehT" for
"The"), because it operates on reversed text. The union therefore contains both
forward-order and backward-order strings; each tokenizer uses the subset relevant
to its direction.

Usage:
  python build_union_tokenizers.py
  python build_union_tokenizers.py --vocab_size 10000
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders

_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE / "data"

# Special tokens that live at fixed IDs in byte-level BPE
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]


# ===================================================================
# Helpers
# ===================================================================

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _merge_to_tuple(m):
    """Convert a merge pair (list or tuple) to a canonical tuple."""
    return (m[0], m[1]) if isinstance(m, (list, tuple)) else m


def _dedup_merges(primary: list, secondary: list) -> list:
    """Return primary merges + secondary merges with duplicates removed."""
    seen = set()
    result = []
    for m in primary:
        key = _merge_to_tuple(m)
        if key not in seen:
            seen.add(key)
            result.append(list(key) if isinstance(m, list) else key)
    for m in secondary:
        key = _merge_to_tuple(m)
        if key not in seen:
            seen.add(key)
            result.append(list(key) if isinstance(m, list) else key)
    return result


def _reverse_merge(m):
    """Reverse a merge pair for backward-text space.

    Forward merge ['Ġ', 't'] → 'Ġt' becomes backward merge ['t', 'Ġ'] → 'tĠ'.
    The reversed merge is only valid if 'tĠ' exists in the union vocab.
    """
    a, b = _merge_to_tuple(m)
    return (b[::-1], a[::-1])


def reverse_vocab(vocab: dict[str, int]) -> dict[str, int]:
    """Reverse every token string in a vocabulary dict."""
    return {k[::-1]: v for k, v in vocab.items()}


# ===================================================================
# Build union vocabulary
# ===================================================================

def build_union_vocab(fwd_vocab: dict[str, int],
                      bwd_vocab: dict[str, int]) -> tuple[dict[str, int], dict]:
    """Build the union vocabulary with contiguous IDs.

    HF BPE models assign arbitrary IDs — there is no fixed "base byte" range.
    So we collect ALL unique token strings from both tokenizers and assign new
    contiguous IDs.

    Build order: special tokens first (IDs 0-3), then forward tokens in their
    original merge order (from the forward JSON merges), then backward-only
    tokens (present in backward vocab but not in forward vocab).

    To identify forward merge order we need the forward merges list, which we
    pass via ``fwd_merges``.

    Returns:
        (union_vocab, stats_dict)
    """
    all_tokens: list[str] = []
    seen: set[str] = set()

    # Special tokens at fixed positions
    for tok in SPECIAL_TOKENS:
        all_tokens.append(tok)
        seen.add(tok)

    fwd_set = set(fwd_vocab.keys())

    # Collect all forward tokens (excluding specials), in merge-priority order
    # if available, otherwise just all forward tokens
    for tok_str in fwd_vocab:
        if tok_str not in seen:
            all_tokens.append(tok_str)
            seen.add(tok_str)

    # Backward-only tokens (not already added from forward vocab)
    bwd_only_count = 0
    for tok_str in bwd_vocab:
        if tok_str not in seen:
            all_tokens.append(tok_str)
            seen.add(tok_str)
            bwd_only_count += 1

    # Build new vocab: token_str → new_id
    union_vocab: dict[str, int] = {}
    for new_id, token_str in enumerate(all_tokens):
        union_vocab[token_str] = new_id

    n_fwd_tokens = len(fwd_vocab) - len(set(SPECIAL_TOKENS) & fwd_set)

    stats = {
        "n_forward": len(fwd_vocab),
        "n_backward": len(bwd_vocab),
        "n_union": len(union_vocab),
        "n_fwd_tokens": n_fwd_tokens,
        "n_bwd_only": bwd_only_count,
    }
    return union_vocab, stats


# ===================================================================
# Build tokenizer
# ===================================================================

def _build_hf_tokenizer(vocab: dict[str, int],
                        merges: list,
                        ) -> Tokenizer:
    """Create a byte-level BPE tokenizer with the given vocab and merges.

    Filters out merges whose merged token (a+b) is not in the provided vocab.
    This can happen when cross-direction merges (e.g. backward merges mixed into
    a forward-rules tokenizer) produce tokens that only exist in the other
    direction's vocab and weren't carried into the union.
    """
    merge_tuples = []
    dropped = 0
    for m in merges:
        a, b = _merge_to_tuple(m)
        # BPE constructor validates that both components AND the merged result
        # are all in the vocabulary
        merged = a + b
        if a in vocab and b in vocab and merged in vocab:
            merge_tuples.append((a, b))
        else:
            dropped += 1
    if dropped:
        print(f"  (dropped {dropped} merges — component(s) or result not in union vocab)")
    bpe = models.BPE(
        vocab=vocab,
        merges=merge_tuples,
        unk_token="<unk>",
        fuse_unk=False,
    )
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    return tok


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build union-vocab tokenizers from forward + backward BPE")
    ap.add_argument("--vocab_size", type=int, default=10000,
                    help="Vocab size of source tokenizers (default: 10000)")
    args = ap.parse_args()

    FWD_DIR = OUT_DIR / f"tokenizer_forward_{args.vocab_size}"
    BWD_DIR = OUT_DIR / f"tokenizer_bwd_rev_{args.vocab_size}"

    fwd_json_path = FWD_DIR / "tokenizer.json"
    bwd_json_path = BWD_DIR / "tokenizer.json"

    if not fwd_json_path.exists():
        sys.exit(f"ERROR: {fwd_json_path} not found — run train_backward_bpe.py first")
    if not bwd_json_path.exists():
        sys.exit(f"ERROR: {bwd_json_path} not found — run train_backward_bpe.py first")

    # Output directory names include vocab size
    suffix = f"v1_{args.vocab_size}"
    UNION_FWD_DIR = OUT_DIR / f"tokenizer_union_fwd_{suffix}"           # #1
    UNION_BWD_BACKRULE_DIR = OUT_DIR / f"tokenizer_union_bwd_backrule_{suffix}"  # #2
    UNION_FWD_TOKFLIP_DIR = OUT_DIR / f"tokenizer_union_fwd_tokflip_{suffix}"    # #3

    print("=" * 60)
    print(f"Building union-vocab tokenizers (vocab_size={args.vocab_size})")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load tokenizer JSONs
    # ------------------------------------------------------------------
    print(f"\n[1] Loading tokenizer JSONs ...")
    fwd_json = _load_json(fwd_json_path)
    bwd_json = _load_json(bwd_json_path)

    fwd_tokenizer = Tokenizer.from_file(str(fwd_json_path))
    bwd_tokenizer = Tokenizer.from_file(str(bwd_json_path))

    fwd_vocab = fwd_tokenizer.get_vocab()
    bwd_vocab = bwd_tokenizer.get_vocab()
    fwd_merges = fwd_json["model"]["merges"]
    bwd_merges = bwd_json["model"]["merges"]

    print(f"  Forward:  vocab={len(fwd_vocab):,}, merges={len(fwd_merges):,}")
    print(f"  Backward: vocab={len(bwd_vocab):,}, merges={len(bwd_merges):,}")

    # ------------------------------------------------------------------
    # 2. Build union vocabulary
    # ------------------------------------------------------------------
    print(f"\n[2] Building union vocabulary ...")
    union_vocab, vocab_stats = build_union_vocab(fwd_vocab, bwd_vocab)
    print(f"  Union vocab: {vocab_stats['n_union']:,} tokens "
          f"({vocab_stats['n_fwd_tokens']} fwd + {vocab_stats['n_bwd_only']} bwd-only)")

    # Save union vocab reference
    union_vocab_path = OUT_DIR / f"vocab_union_{suffix}.json"
    _save_json(union_vocab_path, union_vocab)
    print(f"  Saved → {union_vocab_path}")

    # Also save stats for later reference
    fwd_set = set(fwd_vocab.keys())
    bwd_set = set(bwd_vocab.keys())
    bwd_fwd_set = set(reverse_vocab(bwd_vocab).keys())
    inter_fwd = fwd_set & bwd_fwd_set
    jaccard = len(inter_fwd) / len(fwd_set | bwd_fwd_set) if (fwd_set | bwd_fwd_set) else 0.0
    vocab_stats["jaccard_fwd_order"] = jaccard
    vocab_stats["n_intersection_fwd_order"] = len(inter_fwd)
    _save_json(OUT_DIR / f"vocab_union_{suffix}_stats.json", vocab_stats)

    # ------------------------------------------------------------------
    # 3. Tokenizer #1: forward rules, forward direction
    # ------------------------------------------------------------------
    print(f"\n[3] Building tokenizer #1: forward rules, forward direction ...")
    # Only forward merges — backward-only tokens in the union vocab are
    # unreachable but that's intentional (shared vocab for cross-tokenizer
    # comparison, distinct rule sets for direction-specific behavior)
    merges_1 = fwd_merges
    UNION_FWD_DIR.mkdir(parents=True, exist_ok=True)
    tok1 = _build_hf_tokenizer(union_vocab, merges_1)
    tok1.save(str(UNION_FWD_DIR / "tokenizer.json"))

    _save_json(UNION_FWD_DIR / "meta.json", {
        "version": f"v1-union-{args.vocab_size}",
        "type": "union_forward_rules",
        "direction": "forward (normal text)",
        "merge_priority": "forward_rules_first",
        "n_vocab": len(union_vocab),
        "n_merges": len(merges_1),
        "source_forward": str(fwd_json_path),
        "source_backward": str(bwd_json_path),
    })
    print(f"  ✓ {UNION_FWD_DIR.name}/  vocab={len(union_vocab):,}  merges={len(merges_1):,}")

    # ------------------------------------------------------------------
    # 4. Tokenizer #2: backward rules, backward direction
    # ------------------------------------------------------------------
    print(f"\n[4] Building tokenizer #2: backward rules, backward direction ...")
    # Only backward merges
    merges_2 = bwd_merges
    UNION_BWD_BACKRULE_DIR.mkdir(parents=True, exist_ok=True)
    tok2 = _build_hf_tokenizer(union_vocab, merges_2)
    tok2.save(str(UNION_BWD_BACKRULE_DIR / "tokenizer.json"))

    _save_json(UNION_BWD_BACKRULE_DIR / "meta.json", {
        "version": f"v1-union-{args.vocab_size}",
        "type": "union_backward_rules",
        "direction": "backward (reversed text)",
        "merge_priority": "backward_rules_first",
        "n_vocab": len(union_vocab),
        "n_merges": len(merges_2),
        "source_forward": str(fwd_json_path),
        "source_backward": str(bwd_json_path),
    })
    print(f"  ✓ {UNION_BWD_BACKRULE_DIR.name}/  vocab={len(union_vocab):,}  merges={len(merges_2):,}")

    # ------------------------------------------------------------------
    # 5. Tokenizer #3: forward rules, forward direction (for token-flip)
    # ------------------------------------------------------------------
    print(f"\n[5] Building tokenizer #3: forward rules, forward direction "
          f"(for token-level flip backward LM) ...")
    # Uses the SAME forward merge rules as #1.  Token-level flip is
    # applied later by train_union_lm.py (ids[::-1] per doc), NOT by
    # the tokenizer itself.
    merges_3 = fwd_merges
    UNION_FWD_TOKFLIP_DIR.mkdir(parents=True, exist_ok=True)
    tok3 = _build_hf_tokenizer(union_vocab, merges_3)
    tok3.save(str(UNION_FWD_TOKFLIP_DIR / "tokenizer.json"))

    _save_json(UNION_FWD_TOKFLIP_DIR / "meta.json", {
        "version": f"v1-union-{args.vocab_size}",
        "type": "union_forward_rules_for_token_flip",
        "direction": "forward (normal text), then token-level flip per doc",
        "merge_priority": "forward_rules (same as #1)",
        "n_vocab": len(union_vocab),
        "n_merges": len(merges_3),
        "note": "Token-level flip (ids[::-1] per doc) is done in train_union_lm.py, not here.",
        "source_forward": str(fwd_json_path),
        "source_backward": str(bwd_json_path),
    })
    print(f"  ✓ {UNION_FWD_TOKFLIP_DIR.name}/  vocab={len(union_vocab):,}  merges={len(merges_3):,}"
          f"  (identical rules to #1)")

    # ------------------------------------------------------------------
    # 6. Verify: all three share the same vocab
    # ------------------------------------------------------------------
    print(f"\n[6] Verification ...")
    t1 = Tokenizer.from_file(str(UNION_FWD_DIR / "tokenizer.json"))
    t2 = Tokenizer.from_file(str(UNION_BWD_BACKRULE_DIR / "tokenizer.json"))
    t3 = Tokenizer.from_file(str(UNION_FWD_TOKFLIP_DIR / "tokenizer.json"))

    v1 = t1.get_vocab()
    v2 = t2.get_vocab()
    v3 = t3.get_vocab()

    assert v1 == v2 == v3, "ERROR: tokenizers have different vocabularies!"
    print(f"  ✓ All three tokenizers share identical vocab ({len(v1):,} tokens)")

    # ------------------------------------------------------------------
    # 7. Roundtrip tests
    # ------------------------------------------------------------------
    test_text = "The quick brown fox jumps over the lazy dog."
    rev_text = test_text[::-1]

    print(f"\n[7] Roundtrip tests on: '{test_text}'")

    # #1: forward rules, forward direction
    ids1 = t1.encode(test_text).ids
    dec1 = t1.decode(ids1)
    ok1 = dec1.lstrip() == test_text
    print(f"  [#1 fwd-rule fwd-dir] {len(ids1):3d} tokens  "
          f"decode✓={ok1}  '{dec1[:50]}...'")

    # #2: backward rules, backward direction
    ids2 = t2.encode(rev_text).ids
    dec2 = t2.decode(ids2)
    ok2 = dec2[::-1].strip() == test_text
    print(f"  [#2 bwd-rule bwd-dir] {len(ids2):3d} tokens  "
          f"roundtrip✓={ok2}  unreversed: '{dec2[::-1][:50]}...'")

    # #3: forward rules, forward direction (for token-flip, same as #1)
    ids3 = t3.encode(test_text).ids
    dec3 = t3.decode(ids3)
    ok3 = dec3.lstrip() == test_text
    print(f"  [#3 fwd-rule fwd-dir] {len(ids3):3d} tokens  "
          f"decode✓={ok3}  '{dec3[:50]}...'  (token-flip done in train_union_lm.py)")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Done.  Three union-vocab tokenizers ({len(union_vocab):,} tokens each):")
    print(f"  1. {UNION_FWD_DIR.name}/")
    print(f"     Forward rules, forward direction (normal text)")
    print(f"     merges: {len(merges_1):,}")
    print(f"  2. {UNION_BWD_BACKRULE_DIR.name}/")
    print(f"     Backward rules, backward direction (reversed text)")
    print(f"     merges: {len(merges_2):,}")
    print(f"  3. {UNION_FWD_TOKFLIP_DIR.name}/")
    print(f"     Forward rules, forward direction (normal text)")
    print(f"     merges: {len(merges_3):,} (same as #1, for token-flip backward LM)")
    print(f"     Token-level flip applied in train_union_lm.py, not here.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
