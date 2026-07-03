#!/usr/bin/env python3
"""V3: Word-level error detection with dual BPE tokenization.

═══════════════════════════════════════════════════════════════════════════
Core insight
═══════════════════════════════════════════════════════════════════════════
V2 showed Recall@1 = 0 at the token level: the max-surprise token is never
the actual error position.  But the max-surprise token may fall inside the
SAME WORD as the error — especially when the word is split into multiple
BPE subword tokens.

V3 tests this hypothesis by:
  1. Mapping BPE tokens to words using the 'Ġ' (space) boundary prefix
  2. Aggregating token-level surprise to word-level (max)
  3. Measuring Word-Recall@K: does the error WORD rank in top-K by surprise?

V3 also exploits the fact that forward BPE and backrule BPE split the same
text differently — giving two complementary "views" for word-level detection.

═══════════════════════════════════════════════════════════════════════════
Evaluations
═══════════════════════════════════════════════════════════════════════════
  D1: 45M forward  (L→R)  on forward-tok  — baseline
  D2: 45M tokflip  (R→L)  on forward-tok  — test: backward on forward tokens
  D3: 45M backrule (R→L)  on backrule-tok — test: backward on backrule tokens
  D4: Cross-tokenization ensemble (D2 + D3 on aligned pairs)

Each with two injection methods:
  Method 1 (independent): random token position, per-path
  Method 2 (same_word):   pick a word, inject at a subword token within it

Metrics:
  - Word-AUC:      P(score_err_word > score_clean_word) using max aggregation
  - Word-Recall@1: fraction of sequences where error word is rank-1 by surprise
  - Word-Recall@5: fraction where error word is in top-5 by surprise

Usage:
  python test_autonomy_dual_domain_v3.py --device cuda:0 --n 200 --L 128
  python test_autonomy_dual_domain_v3.py --device cuda:0 --n 3 --L 64 --smoke
"""

from __future__ import annotations

import argparse, json, sys, time
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

# Try loading tokenizers for re-encoding (needed for cross-tokenization alignment)
try:
    from tokenizers import Tokenizer as TokLib
    _HAVE_TOKENIZERS = True
except Exception:
    _HAVE_TOKENIZERS = False

OUT_DIR = HERE.parent / "data"
VOCAB_SIZE = 18782

# ---- 45M evaluation models ----
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

# ---- Data ----
FWD_TOK_VAL = OUT_DIR / "tokenized_union_fwd_v1_10000" / "wiki_val.pt"
BWD_BACKRULE_TOK_VAL = OUT_DIR / "tokenized_union_bwd_backrule_v1_10000" / "wiki_val.pt"

# ---- Tokenizers (for alignment in Method 2) ----
TOK_FWD_PATH = OUT_DIR / "tokenizer_union_fwd_v1_10000" / "tokenizer.json"
TOK_BWD_PATH = OUT_DIR / "tokenizer_union_bwd_backrule_v1_10000" / "tokenizer.json"

# ---- Vocab JSON (for Ġ-based word mapping) ----
VOCAB_PATH = OUT_DIR / "vocab_union_v1_10000.json"

S_OUT = OUT_DIR / "test_autonomy_dual_domain_v3"
S_OUT.mkdir(parents=True, exist_ok=True)

_DEFAULT_N = 200
_DEFAULT_L = 128
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
# Log-prob helpers  (same as v1/v2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def lp_full(model, seq, dev, reverse=False):
    """Full-sequence log-probs.  If reverse=True, the input is flipped so the
    model processes it R→L."""
    inp = torch.flip(seq.view(1, -1), [1]) if reverse else seq.view(1, -1)
    inp = inp.to(dev)
    hs = model.transformer(input_ids=inp, attention_mask=torch.ones_like(inp)).last_hidden_state
    return F.log_softmax(model.lm_head(hs[0]).float(), -1).cpu().numpy()


# ---------------------------------------------------------------------------
# Word mapping  (Ġ-prefix based — no tokenizer library needed)
# ---------------------------------------------------------------------------
def load_id_to_str():
    """Load vocab JSON → dict mapping token id → string."""
    with open(VOCAB_PATH) as f:
        data = json.load(f)
    if isinstance(data, dict):
        # vocab_union_v1_10000.json: {"token_str": id, ...}
        return {v: k for k, v in data.items()}
    return {}


# Global cache
_ID_TO_STR = None


def get_id_to_str():
    global _ID_TO_STR
    if _ID_TO_STR is None:
        _ID_TO_STR = load_id_to_str()
    return _ID_TO_STR


def build_word_map(token_ids):
    """Map each token position → word index using 'Ġ' prefix convention.

    A token whose string representation starts with 'Ġ' begins a new word.
    Special tokens (<s>, </s>, <pad>, <unk>) get word_id = -1.

    Returns:
      word_ids: list[int]  length = len(token_ids)
    """
    id2s = get_id_to_str()
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
            # Continuation subword — same word as previous
            word_map.append(max(word_id, 0))  # clamp: if no word yet, assign 0
    return word_map


# ---------------------------------------------------------------------------
# Word-level aggregation
# ---------------------------------------------------------------------------
def aggregate_to_words(token_surprise, word_map, offset=1):
    """Aggregate token-level surprise scores to word-level (max per word).

    Args:
      token_surprise: np.array [L-2]  surprise for positions offset .. L-2
      word_map:       list[int] [L]   word index per token position
      offset:         int             first valid position (= 1 usually)

    Returns:
      word_scores: dict  word_id → max surprise among its tokens
      word_labels: dict  word_id → 1 if any of its tokens is the error position
    """
    word_scores = {}
    word_labels = {}

    L = len(word_map)
    for i, pos in enumerate(range(offset, L - 1)):
        w = word_map[pos]
        if w < 0:
            continue
        s = float(token_surprise[i])
        if w not in word_scores or s > word_scores[w]:
            word_scores[w] = s
        # Labels are set externally; initialise to 0 here
        if w not in word_labels:
            word_labels[w] = 0

    return word_scores, word_labels


def word_arrays(word_scores, word_labels):
    """Convert dicts to aligned numpy arrays for metric computation."""
    wids = sorted(word_scores.keys())
    scores = np.array([word_scores[w] for w in wids])
    labels = np.array([word_labels[w] for w in wids])
    return labels, scores


def recall_at_k(labels, scores, k):
    order = np.argsort(-scores)
    return float(labels[order[:k]].sum()) / max(labels.sum(), 1)


# ---------------------------------------------------------------------------
# D1: Forward 45M on forward-tok  (L→R baseline, word-level)
# ---------------------------------------------------------------------------
def eval_d1_forward_word(seqs, model, dev, L, rng, freq_toks, method="independent"):
    """D1: Forward model word-level detection on forward-tok real data.

    Args:
      method: "independent" → random token position
              "same_word"   → pick a word, inject at subword within it
    """
    label = f"D1 fwd word {method}"
    print(f"\n{'─'*60}")
    print(f"{label}")
    print(f"{'─'*60}")

    n_seqs = len(seqs)
    all_word_labels = []
    all_word_scores = []
    n_rec1 = 0   # error word is #1 by surprise
    n_rec5 = 0   # error word is in top-5
    n_skipped = 0

    for si, s_orig in enumerate(tqdm(seqs, desc=f"  {label}", unit="seq", leave=False)):
        s = s_orig.clone()
        word_map = build_word_map(s.tolist())

        # Determine error position
        if method == "independent":
            t = int(rng.randint(L // 8, 7 * L // 8))
            error_word = word_map[t]
        else:  # same_word
            # Pick a word with at least 2 subword tokens (more interesting)
            word_to_positions = {}
            for pos, w in enumerate(word_map):
                if w >= 0:
                    word_to_positions.setdefault(w, []).append(pos)
            multi_token_words = [w for w, ps in word_to_positions.items()
                                if len(ps) >= 2
                                and min(ps) >= L // 8
                                and max(ps) <= 7 * L // 8]
            if not multi_token_words:
                # Fallback: any word in valid range
                valid_words = [w for w, ps in word_to_positions.items()
                              if min(ps) >= L // 8 and max(ps) <= 7 * L // 8]
                if not valid_words:
                    n_skipped += 1
                    continue
                error_word = int(rng.choice(valid_words))
            else:
                error_word = int(rng.choice(multi_token_words))
            positions = word_to_positions[error_word]
            t = int(rng.choice(positions))

        orig_token = int(s[t])

        # Inject random error
        c = int(rng.choice(freq_toks))
        while c == orig_token:
            c = int(rng.choice(freq_toks))
        s[t] = c

        # Forward model inference
        lf = lp_full(model, s, dev)
        P = np.arange(1, L - 1)
        xp = s.numpy()[P]
        ar = np.arange(len(P))
        token_surprise = -lf[P - 1][ar, xp]

        # Aggregate to words
        word_scores, word_labels = aggregate_to_words(token_surprise, word_map)
        word_labels[error_word] = 1

        labels, scores = word_arrays(word_scores, word_labels)
        all_word_labels.append(labels)
        all_word_scores.append(scores)

        # Per-sequence recall
        order = np.argsort(-scores)
        top5_words = order[:5]
        ep = np.where(labels == 1)[0]
        if len(ep):
            err_idx = ep[0]
            if err_idx == order[0]:
                n_rec1 += 1
            if err_idx in top5_words:
                n_rec5 += 1

    # Aggregate metrics across sequences
    flat_labels = np.concatenate(all_word_labels) if all_word_labels else np.array([])
    flat_scores = np.concatenate(all_word_scores) if all_word_scores else np.array([])
    n_valid = n_seqs - n_skipped

    return {
        "Word-AUC": roc_auc_score(flat_labels, flat_scores) if len(flat_labels) else 0.5,
        "Word-Recall@1": n_rec1 / max(n_valid, 1),
        "Word-Recall@5": n_rec5 / max(n_valid, 1),
        "n_words": len(flat_labels),
        "n_err_words": int(flat_labels.sum()),
        "n_seqs": n_seqs,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# D2: Tokflip backward 45M on forward-tok  (R→L, word-level)
# ---------------------------------------------------------------------------
def eval_d2_tokflip_word(seqs, model, dev, L, rng, freq_toks, method="independent"):
    """D2: Tokflip backward model word-level detection on forward-tok data.

    The model reads the sequence R→L (reverse=True), so surprise at
    original position t is extracted from the reversed output.
    """
    label = f"D2 tokflip word {method}"
    print(f"\n{'─'*60}")
    print(f"{label}")
    print(f"{'─'*60}")

    n_seqs = len(seqs)
    all_word_labels = []
    all_word_scores = []
    n_rec1, n_rec5 = 0, 0
    n_skipped = 0

    for si, s_orig in enumerate(tqdm(seqs, desc=f"  {label}", unit="seq", leave=False)):
        s = s_orig.clone()
        word_map = build_word_map(s.tolist())

        # Determine error position (same logic as D1)
        if method == "independent":
            t = int(rng.randint(L // 8, 7 * L // 8))
            error_word = word_map[t]
        else:  # same_word
            word_to_positions = {}
            for pos, w in enumerate(word_map):
                if w >= 0:
                    word_to_positions.setdefault(w, []).append(pos)
            multi_token_words = [w for w, ps in word_to_positions.items()
                                if len(ps) >= 2
                                and min(ps) >= L // 8
                                and max(ps) <= 7 * L // 8]
            if not multi_token_words:
                valid_words = [w for w, ps in word_to_positions.items()
                              if min(ps) >= L // 8 and max(ps) <= 7 * L // 8]
                if not valid_words:
                    n_skipped += 1
                    continue
                error_word = int(rng.choice(valid_words))
            else:
                error_word = int(rng.choice(multi_token_words))
            positions = word_to_positions[error_word]
            t = int(rng.choice(positions))

        orig_token = int(s[t])

        # Inject random error
        c = int(rng.choice(freq_toks))
        while c == orig_token:
            c = int(rng.choice(freq_toks))
        s[t] = c

        # Tokflip backward model (reverse=True: processes R→L)
        lb = lp_full(model, s, dev, reverse=True)
        P = np.arange(1, L - 1)
        xp = s.numpy()[P]
        # Map original position → reversed position in output
        bwd_pos = L - 2 - P   # original pos t → output index in reversed lp
        ar = np.arange(len(P))
        token_surprise = -lb[bwd_pos, xp]

        # Aggregate to words
        word_scores, word_labels = aggregate_to_words(token_surprise, word_map)
        word_labels[error_word] = 1

        labels, scores = word_arrays(word_scores, word_labels)
        all_word_labels.append(labels)
        all_word_scores.append(scores)

        order = np.argsort(-scores)
        ep = np.where(labels == 1)[0]
        if len(ep):
            err_idx = ep[0]
            if err_idx == order[0]:
                n_rec1 += 1
            if err_idx in order[:5]:
                n_rec5 += 1

    flat_labels = np.concatenate(all_word_labels) if all_word_labels else np.array([])
    flat_scores = np.concatenate(all_word_scores) if all_word_scores else np.array([])
    n_valid = n_seqs - n_skipped

    return {
        "Word-AUC": roc_auc_score(flat_labels, flat_scores) if len(flat_labels) else 0.5,
        "Word-Recall@1": n_rec1 / max(n_valid, 1),
        "Word-Recall@5": n_rec5 / max(n_valid, 1),
        "n_words": len(flat_labels),
        "n_err_words": int(flat_labels.sum()),
        "n_seqs": n_seqs,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# D3: Backrule backward 45M on backrule-tok  (R→L on reversed data, word-level)
# ---------------------------------------------------------------------------
def eval_d3_backrule_word(seqs, model, dev, L, rng, freq_toks, method="independent"):
    """D3: Backrule backward model word-level detection on backrule-tok data.

    The backrule-tok data is already tokenized & reversed.  The backrule
    backward model processes it L→R (no flip needed) — the reversal is
    baked into the training data, not the forward pass.
    """
    label = f"D3 backrule word {method}"
    print(f"\n{'─'*60}")
    print(f"{label}")
    print(f"{'─'*60}")

    n_seqs = len(seqs)
    all_word_labels = []
    all_word_scores = []
    n_rec1, n_rec5 = 0, 0
    n_skipped = 0

    for si, s_orig in enumerate(tqdm(seqs, desc=f"  {label}", unit="seq", leave=False)):
        s = s_orig.clone()
        word_map = build_word_map(s.tolist())

        # Determine error position
        if method == "independent":
            t = int(rng.randint(L // 8, 7 * L // 8))
            error_word = word_map[t]
        else:  # same_word
            word_to_positions = {}
            for pos, w in enumerate(word_map):
                if w >= 0:
                    word_to_positions.setdefault(w, []).append(pos)
            multi_token_words = [w for w, ps in word_to_positions.items()
                                if len(ps) >= 2
                                and min(ps) >= L // 8
                                and max(ps) <= 7 * L // 8]
            if not multi_token_words:
                valid_words = [w for w, ps in word_to_positions.items()
                              if min(ps) >= L // 8 and max(ps) <= 7 * L // 8]
                if not valid_words:
                    n_skipped += 1
                    continue
                error_word = int(rng.choice(valid_words))
            else:
                error_word = int(rng.choice(multi_token_words))
            positions = word_to_positions[error_word]
            t = int(rng.choice(positions))

        orig_token = int(s[t])

        # Inject random error
        c = int(rng.choice(freq_toks))
        while c == orig_token:
            c = int(rng.choice(freq_toks))
        s[t] = c

        # Backrule backward model — L→R on already-reversed data
        lb = lp_full(model, s, dev, reverse=False)
        P = np.arange(1, L - 1)
        xp = s.numpy()[P]
        ar = np.arange(len(P))
        token_surprise = -lb[P - 1][ar, xp]

        # Aggregate to words
        word_scores, word_labels = aggregate_to_words(token_surprise, word_map)
        word_labels[error_word] = 1

        labels, scores = word_arrays(word_scores, word_labels)
        all_word_labels.append(labels)
        all_word_scores.append(scores)

        order = np.argsort(-scores)
        ep = np.where(labels == 1)[0]
        if len(ep):
            err_idx = ep[0]
            if err_idx == order[0]:
                n_rec1 += 1
            if err_idx in order[:5]:
                n_rec5 += 1

    flat_labels = np.concatenate(all_word_labels) if all_word_labels else np.array([])
    flat_scores = np.concatenate(all_word_scores) if all_word_scores else np.array([])
    n_valid = n_seqs - n_skipped

    return {
        "Word-AUC": roc_auc_score(flat_labels, flat_scores) if len(flat_labels) else 0.5,
        "Word-Recall@1": n_rec1 / max(n_valid, 1),
        "Word-Recall@5": n_rec5 / max(n_valid, 1),
        "n_words": len(flat_labels),
        "n_err_words": int(flat_labels.sum()),
        "n_seqs": n_seqs,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# D4: Cross-tokenization ensemble  (D2 + D3 on aligned pairs)
# ---------------------------------------------------------------------------
def eval_d4_cross_ensemble(seqs_fwd, seqs_bwd, model_tokflip, model_backrule,
                           dev, L, rng, freq_toks):
    """D4: Ensemble word-level detection using aligned forward + backrule pairs.

    For each sequence pair at the same index, inject an error at the SAME
    word (by word index) in both tokenizations.  Then combine the word-level
    surprise from D2 (tokflip on forward-tok) and D3 (backrule on backrule-tok).

    Ensemble rule: word is flagged if EITHER model ranks it in top-K.
    """
    print(f"\n{'─'*60}")
    print("D4: Cross-tokenization ensemble (same word)")
    print(f"{'─'*60}")

    n_seqs = min(len(seqs_fwd), len(seqs_bwd))
    n_rec1_ens, n_rec5_ens = 0, 0
    n_rec1_tf, n_rec5_tf = 0, 0     # tokflip alone
    n_rec1_br, n_rec5_br = 0, 0     # backrule alone
    n_skipped = 0

    for si in tqdm(range(n_seqs), desc="  D4 ensemble", unit="seq", leave=False):
        s_fwd = seqs_fwd[si].clone()
        s_bwd = seqs_bwd[si].clone()

        wm_fwd = build_word_map(s_fwd.tolist())
        wm_bwd = build_word_map(s_bwd.tolist())

        num_w_fwd = max(wm_fwd) + 1 if wm_fwd else 0
        num_w_bwd = max(wm_bwd) + 1 if wm_bwd else 0
        if num_w_fwd < 2 or num_w_bwd < 2:
            n_skipped += 1
            continue

        # Pick a word index that is valid in BOTH tokenizations
        # (same word index ≈ same relative word position)
        max_w = min(num_w_fwd, num_w_bwd) - 1
        # Choose from middle words to avoid edge effects
        lo = max(1, max_w // 8)
        hi = min(max_w - 1, 7 * max_w // 8)
        if hi <= lo:
            n_skipped += 1
            continue
        shared_word = int(rng.randint(lo, hi + 1))

        # --- Forward-tok path (tokflip) ---
        fwd_positions = [p for p, w in enumerate(wm_fwd) if w == shared_word]
        if not fwd_positions:
            n_skipped += 1
            continue
        t_fwd = int(rng.choice(fwd_positions))
        orig_fwd = int(s_fwd[t_fwd])
        c = int(rng.choice(freq_toks))
        while c == orig_fwd:
            c = int(rng.choice(freq_toks))
        s_fwd[t_fwd] = c

        # Tokflip backward inference
        lb_tf = lp_full(model_tokflip, s_fwd, dev, reverse=True)
        P = np.arange(1, L - 1)
        bwd_pos = L - 2 - P
        token_surprise_tf = -lb_tf[bwd_pos, s_fwd.numpy()[P]]

        word_scores_tf, word_labels_tf = aggregate_to_words(token_surprise_tf, wm_fwd)
        word_labels_tf[shared_word] = 1

        # --- Backrule-tok path (backrule) ---
        bwd_positions = [p for p, w in enumerate(wm_bwd) if w == shared_word]
        if not bwd_positions:
            n_skipped += 1
            continue
        t_bwd = int(rng.choice(bwd_positions))
        orig_bwd = int(s_bwd[t_bwd])
        c = int(rng.choice(freq_toks))
        while c == orig_bwd:
            c = int(rng.choice(freq_toks))
        s_bwd[t_bwd] = c

        # Backrule backward inference
        lb_br = lp_full(model_backrule, s_bwd, dev, reverse=False)
        token_surprise_br = -lb_br[P - 1, s_bwd.numpy()[P]]

        word_scores_br, word_labels_br = aggregate_to_words(token_surprise_br, wm_bwd)
        word_labels_br[shared_word] = 1

        # --- Per-path word recall ---
        # Tokflip
        labels_tf, scores_tf = word_arrays(word_scores_tf, word_labels_tf)
        order_tf = np.argsort(-scores_tf)
        ep_tf = np.where(labels_tf == 1)[0]
        if len(ep_tf):
            eidx = ep_tf[0]
            if eidx == order_tf[0]:
                n_rec1_tf += 1
            if eidx in order_tf[:5]:
                n_rec5_tf += 1

        # Backrule
        labels_br, scores_br = word_arrays(word_scores_br, word_labels_br)
        order_br = np.argsort(-scores_br)
        ep_br = np.where(labels_br == 1)[0]
        if len(ep_br):
            eidx = ep_br[0]
            if eidx == order_br[0]:
                n_rec1_br += 1
            if eidx in order_br[:5]:
                n_rec5_br += 1

        # --- Ensemble: max surprise per word across both tokenizations ---
        # Since word indices are aligned (same word = same index), we can
        # simply take the max surprise for each word from either path.
        all_wids = sorted(set(word_scores_tf.keys()) | set(word_scores_br.keys()))
        ens_scores = []
        ens_labels = []
        for w in all_wids:
            s = max(word_scores_tf.get(w, -np.inf), word_scores_br.get(w, -np.inf))
            l = 1 if (word_labels_tf.get(w, 0) == 1 or word_labels_br.get(w, 0) == 1) else 0
            ens_scores.append(s)
            ens_labels.append(l)

        ens_scores = np.array(ens_scores)
        ens_labels = np.array(ens_labels)
        order_ens = np.argsort(-ens_scores)
        ep_ens = np.where(ens_labels == 1)[0]
        if len(ep_ens):
            eidx = ep_ens[0]
            if eidx == order_ens[0]:
                n_rec1_ens += 1
            if eidx in order_ens[:5]:
                n_rec5_ens += 1

    n_valid = n_seqs - n_skipped
    return {
        "ensemble_Word-Recall@1": n_rec1_ens / max(n_valid, 1),
        "ensemble_Word-Recall@5": n_rec5_ens / max(n_valid, 1),
        "tokflip_Word-Recall@1": n_rec1_tf / max(n_valid, 1),
        "tokflip_Word-Recall@5": n_rec5_tf / max(n_valid, 1),
        "backrule_Word-Recall@1": n_rec1_br / max(n_valid, 1),
        "backrule_Word-Recall@5": n_rec5_br / max(n_valid, 1),
        "n_seqs": n_seqs,
        "n_skipped": n_skipped,
    }


# ===================================================================
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser(description="V3: Word-level error detection")
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
        print("SMOKE TEST MODE (V3)")
        print("=" * 70)

    print("=" * 70)
    print("V3: Word-Level Error Detection with Dual BPE Tokenization")
    print(f"  device={dev}  L={L}  n_seqs={args.n}  vocab={VOCAB_SIZE:,}")
    print(f"  tokenizers available: {_HAVE_TOKENIZERS}")
    print("=" * 70)
    print()
    print("  D1: Forward 45M (L→R)  on forward-tok  [baseline]")
    print("  D2: Tokflip 45M (R→L)  on forward-tok")
    print("  D3: Backrule 45M (R→L) on backrule-tok  [bwd-freq errors]")
    print("  D3-CTRL: Backrule 45M (R→L) on backrule-tok  [fwd-freq errors — shows confound]")
    print("  D4: Cross-tok ensemble  [fwd-freq errors]")
    print("  D4-CTRL: Cross-tok ensemble  [bwd-freq errors in bwd path — control]")
    print("  Each with: independent / same_word injection")
    print("  Metrics: Word-AUC, Word-Recall@1, Word-Recall@5")
    print()

    # ── Load data ──
    print("[0] Load data & build vocab mapping")
    _ = get_id_to_str()  # pre-load vocab

    blocks_fwd = torch.load(str(FWD_TOK_VAL), map_location="cpu").to(torch.int64)
    n_full = blocks_fwd.numel() // L
    chunks_fwd = blocks_fwd[:n_full * L].view(n_full, L)
    seqs_fwd_all = [chunks_fwd[i].clone() for i in range(n_full)]
    rng.shuffle(seqs_fwd_all)
    seqs_fwd = seqs_fwd_all[:args.n]
    print(f"    fwd-tok:    {len(seqs_fwd)} sequences")

    blocks_rev = torch.load(str(BWD_BACKRULE_TOK_VAL), map_location="cpu").to(torch.int64)
    n_full_rev = blocks_rev.numel() // L
    chunks_rev = blocks_rev[:n_full_rev * L].view(n_full_rev, L)
    seqs_rev_all = [chunks_rev[i].clone() for i in range(n_full_rev)]
    rng.shuffle(seqs_rev_all)
    seqs_rev = seqs_rev_all[:args.n]
    print(f"    backrule-tok: {len(seqs_rev)} sequences")

    # Build word-map stats for a sample to verify
    sample_wm = build_word_map(seqs_fwd[0].tolist())
    n_words_fwd = max(sample_wm) + 1
    sample_wm_rev = build_word_map(seqs_rev[0].tolist())
    n_words_rev = max(sample_wm_rev) + 1
    print(f"    Word-map sanity: fwd seq has ~{n_words_fwd} words, "
          f"rev seq has ~{n_words_rev} words  (L={L})")

    # Frequent tokens for random error injection
    counts_fwd = torch.bincount(blocks_fwd.reshape(-1), minlength=VOCAB_SIZE).numpy()
    freq_toks = np.argsort(counts_fwd)[-500:]

    # Backrule-tok frequent tokens (for control experiment)
    counts_bwd = torch.bincount(blocks_rev.reshape(-1), minlength=VOCAB_SIZE).numpy()
    freq_toks_bwd = np.argsort(counts_bwd)[-500:]

    # Report overlap
    fwd_set = set(freq_toks.tolist())
    bwd_set = set(freq_toks_bwd.tolist())
    print(f"    Freq-tok overlap (top-500): {len(fwd_set & bwd_set)}/500")

    # ── Load 45M models ──
    print("\n[1] Load 45M evaluation models")
    fwd_45m = build_model(VOCAB_SIZE, dev, "45M")
    fwd_45m.load_state_dict(torch.load(str(FWD_45M_CKPT), map_location=dev))
    fwd_45m.eval()
    print(f"    Forward 45M:      {sum(p.numel() for p in fwd_45m.parameters())/1e6:.1f}M")

    bwd_tf_45m = build_model(VOCAB_SIZE, dev, "45M")
    bwd_tf_45m.load_state_dict(torch.load(str(BWD_TOKFLIP_45M_CKPT), map_location=dev))
    bwd_tf_45m.eval()
    print(f"    Tokflip 45M:      {sum(p.numel() for p in bwd_tf_45m.parameters())/1e6:.1f}M")

    bwd_br_45m = build_model(VOCAB_SIZE, dev, "45M")
    bwd_br_45m.load_state_dict(torch.load(str(BWD_BACKRULE_45M_CKPT), map_location=dev))
    bwd_br_45m.eval()
    print(f"    Backrule 45M:     {sum(p.numel() for p in bwd_br_45m.parameters())/1e6:.1f}M")

    # Sanity
    dummy = torch.randint(0, VOCAB_SIZE, (2, L), device=dev)
    assert fwd_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    assert bwd_tf_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    assert bwd_br_45m(input_ids=dummy).logits.shape == (2, L, VOCAB_SIZE)
    print("    Sanity check ✓")

    # ══════════════════════════════════════════════════════════════════
    # RUN ALL EVALUATIONS
    # ══════════════════════════════════════════════════════════════════
    t0 = time.time()
    all_results = {}

    # ---- D1: Forward baseline ----
    print("\n" + "=" * 70)
    print("D1: Forward 45M (L→R) word-level — baseline")
    print("=" * 70)
    all_results["D1_fwd_independent"] = eval_d1_forward_word(
        seqs_fwd, fwd_45m, dev, L, rng, freq_toks, method="independent")
    all_results["D1_fwd_sameword"] = eval_d1_forward_word(
        seqs_fwd, fwd_45m, dev, L, rng, freq_toks, method="same_word")

    # ---- D2: Tokflip backward ----
    print("\n" + "=" * 70)
    print("D2: Tokflip backward 45M (R→L) word-level")
    print("=" * 70)
    all_results["D2_tokflip_independent"] = eval_d2_tokflip_word(
        seqs_fwd, bwd_tf_45m, dev, L, rng, freq_toks, method="independent")
    all_results["D2_tokflip_sameword"] = eval_d2_tokflip_word(
        seqs_fwd, bwd_tf_45m, dev, L, rng, freq_toks, method="same_word")

    # ---- D3: Backrule backward ----
    print("\n" + "=" * 70)
    print("D3: Backrule backward 45M (R→L) word-level  [bwd-freq errors]")
    print("=" * 70)
    rng_state_before_d3 = rng.get_state()
    all_results["D3_backrule_independent"] = eval_d3_backrule_word(
        seqs_rev, bwd_br_45m, dev, L, rng, freq_toks_bwd, method="independent")
    all_results["D3_backrule_sameword"] = eval_d3_backrule_word(
        seqs_rev, bwd_br_45m, dev, L, rng, freq_toks_bwd, method="same_word")

    # ---- D3-control: SAME positions, forward-frequent error tokens (shows confound) ----
    print("\n" + "=" * 70)
    print("D3-CTRL: Backrule backward with forward-frequent error tokens")
    print("         (SAME error positions as D3 — shows frequency confound)")
    print("=" * 70)
    rng.set_state(rng_state_before_d3)  # exact same positions
    all_results["D3ctrl_backrule_independent"] = eval_d3_backrule_word(
        seqs_rev, bwd_br_45m, dev, L, rng, freq_toks, method="independent")
    all_results["D3ctrl_backrule_sameword"] = eval_d3_backrule_word(
        seqs_rev, bwd_br_45m, dev, L, rng, freq_toks, method="same_word")

    # ---- D4: Cross-tokenization ensemble ----
    print("\n" + "=" * 70)
    print("D4: Cross-tokenization ensemble (same-word across tok schemes)")
    print("=" * 70)
    rng_state_before_d4 = rng.get_state()
    all_results["D4_cross_ensemble"] = eval_d4_cross_ensemble(
        seqs_fwd, seqs_rev, bwd_tf_45m, bwd_br_45m, dev, L, rng, freq_toks)

    # ---- D4-control: SAME positions, backrule-frequent errors in bwd path ----
    print("\n" + "=" * 70)
    print("D4-CTRL: Cross-tok ensemble (SAME positions, bwd-freq errors in bwd path)")
    print("=" * 70)
    rng.set_state(rng_state_before_d4)
    all_results["D4ctrl_cross_ensemble"] = eval_d4_cross_ensemble(
        seqs_fwd, seqs_rev, bwd_tf_45m, bwd_br_45m, dev, L, rng, freq_toks_bwd)

    dt = time.time() - t0

    # Free GPU memory
    del fwd_45m, bwd_tf_45m, bwd_br_45m
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"RESULTS  ({dt:.1f}s)")
    print("=" * 70)

    # ---- D1-D3 table ----
    print(f"\n  {'─'*80}")
    print(f"  WORD-LEVEL DETECTION RESULTS")
    print(f"  {'─'*80}")
    hdr = (f"  {'Eval':<36} {'Word-AUC':>10}  "
           f"{'Word-R@1':>10}  {'Word-R@5':>10}  {'#words':>8}  {'#seqs':>6}")
    print(hdr)
    print("  " + "-" * len(hdr))
    for key in ["D1_fwd_independent", "D1_fwd_sameword",
                "D2_tokflip_independent", "D2_tokflip_sameword",
                "D3_backrule_independent", "D3_backrule_sameword",
                "D3ctrl_backrule_independent", "D3ctrl_backrule_sameword"]:
        m = all_results[key]
        print(f"  {key:<36} {m['Word-AUC']:>10.4f}  "
              f"{m['Word-Recall@1']:>10.3f}  {m['Word-Recall@5']:>10.3f}  "
              f"{m['n_words']:>8}  {m['n_seqs']:>6}")

    # ---- D4 cross-ensemble table ----
    for d4key in ["D4_cross_ensemble", "D4ctrl_cross_ensemble"]:
        d4 = all_results[d4key]
        print(f"\n  {'─'*80}")
        print(f"  CROSS-TOKENIZATION ENSEMBLE ({d4key})")
        print(f"  {'─'*80}")
        print(f"  {'Metric':<36} {'Value':>10}")
        print("  " + "-" * 48)
        for k, v in d4.items():
            if isinstance(v, float):
                print(f"  {k:<36} {v:>10.4f}")
            else:
                print(f"  {k:<36} {v:>10}")

    # ---- D3 vs D3ctrl comparison ----
    print(f"\n  {'─'*80}")
    print(f"  D3 vs D3-CTRL: Impact of error-token frequency source")
    print(f"  {'─'*80}")
    hdr2 = (f"  {'Metric':<36} {'D3 (bwd-freq)':>16}  {'D3ctrl (fwd-freq)':>18}  {'Delta':>10}")
    print(hdr2)
    print("  " + "-" * len(hdr2))
    for method in ["independent", "sameword"]:
        d3_key = f"D3_backrule_{method}"
        d3c_key = f"D3ctrl_backrule_{method}"
        for metric in ["Word-AUC", "Word-Recall@1", "Word-Recall@5"]:
            v3 = all_results[d3_key][metric]
            v3c = all_results[d3c_key][metric]
            delta = v3c - v3
            label = f"D3 {method} {metric}"
            print(f"  {label:<36} {v3:>16.4f}  {v3c:>18.4f}  {delta:>+10.4f}")
    # Also D4 comparison
    for metric in ["ensemble_Word-Recall@1", "ensemble_Word-Recall@5",
                   "backrule_Word-Recall@1", "backrule_Word-Recall@5"]:
        v4 = all_results["D4_cross_ensemble"][metric]
        v4c = all_results["D4ctrl_cross_ensemble"][metric]
        delta = v4c - v4
        label = f"D4 {metric}"
        print(f"  {label:<36} {v4:>16.4f}  {v4c:>18.4f}  {delta:>+10.4f}")

    # ---- Key comparisons ----
    print(f"\n  {'─'*80}")
    print(f"  KEY COMPARISON: Token-level (v2) vs Word-level (v3) Recall")
    print(f"  {'─'*80}")
    print(f"  V2 token-level Recall@1: 0.00  (all evals)")
    print(f"  V3 word-level Recall@1 ranges above — are any > 0?")

    # ── Save ──
    out_path = S_OUT / f"v3_results_L{L}_n{args.n}.json"
    out_path.write_text(json.dumps({
        "config": {"vocab_size": VOCAB_SIZE, "L": L, "n_seqs": args.n},
        "results": all_results,
    }, indent=2, default=float))
    print(f"\n[saved] {out_path}")
    print("\nDONE ✓")


if __name__ == "__main__":
    main()
