#!/usr/bin/env python3
"""Train forward/backward causal LMs with union-vocab BPE tokenizers.

Four model variants, all sharing the SAME union vocabulary:

  1. Forward LM  — tokenizer #1 (forward rules),  normal text
  2. Backward LM — tokenizer #2 (backward rules), text-level reversal
  3. Backward LM — tokenizer #1 (forward rules),  token-level flip
  4. Forward LM  — tokenizer #2 (backward rules), normal text   (forward read,
                    backward tokenization — no text reversal)

Flip modes:
  - text  (default): reverse raw text BEFORE encoding — for backward-rule
                     tokenizer (#2) or forward-rule tokenizer (#1)
  - token:          encode normal text, THEN reverse the token sequence — for
                     backward LM with forward-rule tokenizer (#3)
  - both:           reverse text → encode → reverse token ids — for forward LM
                     with backward-rule tokenizer (#4).  Text reversal ensures
                     backward merge rules match correctly; token reversal
                     restores forward reading order.

  Token-level flip: token internal order is preserved; only inter-token
  relative order is reversed.  "The cat" → [t1,t2,t3] → [t3,t2,t1].

All variants use the same GPT-2 architecture and hyperparameters as the
existing forward/backward training scripts.

Usage:
  # Forward LM with tokenizer #1 (forward rules)
  python train_union_lm.py \
      --tokenizer_dir outputs/tokenizer_union_fwd_v1_10000 \
      --direction forward --device cuda:0

  # Backward LM with tokenizer #2 (backward rules, text-level flip)
  python train_union_lm.py \
      --tokenizer_dir outputs/tokenizer_union_bwd_backrule_v1_10000 \
      --direction backward --device cuda:0

  # Backward LM with tokenizer #1 (forward rules, token-level flip → model #3)
  python train_union_lm.py \
      --tokenizer_dir outputs/tokenizer_union_fwd_v1_10000 \
      --direction backward --flip_mode token --device cuda:0

  # Forward LM with tokenizer #2 (backward rules, both-flip → model #4)
  python train_union_lm.py \
      --tokenizer_dir outputs/tokenizer_union_bwd_backrule_v1_10000 \
      --direction forward --flip_mode both --device cuda:0

  # Tokenize only first, then train
  python train_union_lm.py --tokenizer_dir ... --direction forward --phase tokenize
  python train_union_lm.py --tokenizer_dir ... --direction forward --phase train --device cuda:0
"""

from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
_DEPS = _HERE.parent / "_deps"
for _p in (str(_PROJECT), str(_DEPS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokenizers import Tokenizer as HFTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR = _HERE / "data"
# 65536\32768\16384
# Training hyperparameters
SEQ_LEN = 512          # context window (GPT-2 standard: 1024)
BT = 16384             # batch tokens (batch_size = BT // SEQ_LEN = 128)
LR = 3e-4              # peak learning rate (scaled up with batch size)
WARMUP = 200
N_STEPS = 10000
EVAL_EVERY = 500
CKPT_EVERY = 2000
LOG_EVERY = 50
GRAD_CLIP = 1.0
MODEL_SIZE = "45M"  # 6 layers, 512 dim, 8 heads
STRIDE = 64            # overlap stride for chunking (SEQ_LEN // 8)


# ===================================================================
# LR schedule
# ===================================================================

def _lr_at(step: int, lr: float, warmup: int, n_steps: int) -> float:
    """Linear warmup → cosine decay to 10% of peak LR."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, n_steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ===================================================================
# Path resolution
# ===================================================================

def resolve_paths(tokenizer_dir: str, direction: str,
                  tag: str = "",
                  flip_mode: str = "text") -> dict[str, Path]:
    """Resolve all output paths from tokenizer directory name.

    Derives tokenized_dir and model_dir from the tokenizer name so the four
    variants are automatically organised.  Output paths include ``_{direction}``
    so that forward and backward runs with the same tokenizer never collide.

    ``tag`` optionally appends a suffix to model_dir for versioning
    (e.g. --tag n20k → model_union_fwd_v1_10000_n20k).

    ``flip_mode`` ("text", "token", or "both") controls text/token reversal.
    When "token", a ``_tokflip`` suffix is appended.  When "both", a
    ``_bothflip`` suffix is appended (for forward LM with backward-rule
    tokenizer, where text is first reversed for correct merge matching,
    then token ids are reversed back to forward reading order).
    """
    tok_path = Path(tokenizer_dir)
    if not tok_path.is_absolute():
        tok_path = (_HERE / tok_path).resolve()
    tok_name = tok_path.name  # e.g. "tokenizer_union_fwd_v1_10000"

    # Derive output names: strip "tokenizer_" prefix → tokenized_* / model_*
    stem = tok_name[len("tokenizer_"):]  # e.g. "union_fwd_v1_10000"

    # Include direction in output paths so forward/backward variants of the
    # same tokenizer don't collide (e.g. backward-rule tok + forward direction).
    tokenized_dir = OUT_DIR / f"tokenized_{stem}_{direction}"
    model_stem = f"model_{stem}_{direction}"

    # Token-level flip uses the same tokenizer as forward but produces
    # different token sequences → needs its own output directories.
    # Guard: don't double-add _tokflip if the stem already contains it
    # (e.g. tokenizer_union_fwd_tokflip_v1_10000 already has it).
    if direction == "backward" and flip_mode == "token":
        if "_tokflip" not in stem:
            tokenized_dir = OUT_DIR / f"tokenized_{stem}_{direction}_tokflip"
            model_stem = f"model_{stem}_{direction}_tokflip"

    # Both-flip (text reversal + token flip) for forward LM with backward
    # rules — needs separate data from plain forward or backward runs.
    if flip_mode == "both":
        tokenized_dir = OUT_DIR / f"tokenized_{stem}_{direction}_bothflip"
        model_stem = f"model_{stem}_{direction}_bothflip"

    # Backward compat: reuse old-style paths (without _direction suffix) if
    # they exist, the new-style ones don't, and the old data was produced
    # with the same direction/flip_mode (checked via its meta.json).
    def _reuse_if_match(old_dir: Path, direction: str, flip_mode: str) -> Path | None:
        if not old_dir.exists():
            return None
        meta = old_dir / "wiki_train_meta.json"
        if not meta.exists():
            return None
        try:
            cfg = json.loads(meta.read_text(encoding="utf-8"))
            if cfg.get("direction") == direction and cfg.get("flip_mode") == flip_mode:
                return old_dir
        except Exception:
            pass
        return None

    if not tokenized_dir.exists():
        if reused := _reuse_if_match(OUT_DIR / f"tokenized_{stem}", direction, flip_mode):
            tokenized_dir = reused
    if direction == "backward" and flip_mode == "token" and not tokenized_dir.exists():
        if reused := _reuse_if_match(OUT_DIR / f"tokenized_{stem}_tokflip", direction, flip_mode):
            tokenized_dir = reused

    if tag:
        model_stem = f"{model_stem}_{tag}"
    model_dir = OUT_DIR / model_stem
    return {
        "tokenizer": tok_path,
        "tokenized": tokenized_dir,
        "model": model_dir,
    }


# ===================================================================
# Phase 1: Load HF tokenizer
# ===================================================================

def load_tokenizer(tokenizer_dir: Path) -> tuple[HFTokenizer, int]:
    """Load a HuggingFace tokenizer and return (tokenizer, vocab_size)."""
    tok_json = tokenizer_dir / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(
            f"{tok_json} not found — run build_union_tokenizers.py first"
        )
    tok = HFTokenizer.from_file(str(tok_json))
    vocab_size = tok.get_vocab_size()
    # Read merges from JSON (HF tokenizers may not expose via public API)
    with open(tok_json, encoding="utf-8") as f:
        _tok_data = json.load(f)
    merges_list = _tok_data.get("model", {}).get("merges", [])

    print(f"[Tokenizer] Loaded {tokenizer_dir.name}")
    print(f"  vocab={vocab_size:,}, merges={len(merges_list):,}")

    # Quick sanity: encode/decode roundtrip
    test = "Hello world"
    ids = tok.encode(test).ids
    decoded = tok.decode(ids)
    print(f"  Sanity: '{test}' → {len(ids)} ids → '{decoded}'")

    return tok, vocab_size


# ===================================================================
# Phase 2: Tokenize corpus
# ===================================================================

def tokenize_corpus(
    tok: HFTokenizer,
    tokenized_dir: Path,
    direction: str,
    text_dir: Path | None = None,
    force: bool = False,
    flip_mode: str = "text",
) -> dict[str, Path]:
    """Tokenize wiki_train.txt / wiki_val.txt and save as flat .pt tensors.

    flip_mode behaviour (independent of direction):
      - "text" (default):       text[::-1] → encode → save
      - "token":                normal text → encode → ids[::-1] per doc → save
      - "both":                 text[::-1] → encode → ids[::-1] per doc → save

    Variant mapping:
      - forward  + flip_mode=text      → variant #1 (fwd LM + fwd rules)
      - backward + flip_mode=text      → variant #2 (bwd LM + bwd rules)
      - backward + flip_mode=token     → variant #3 (bwd LM + fwd rules, tokflip)
      - forward  + flip_mode=both      → variant #4 (fwd LM + bwd rules:
                                          reverse text for correct merge matching,
                                          then reverse ids for forward reading order)

    Stores a fingerprint (tokenizer path + direction + flip_mode) in meta.json
    and compares on subsequent runs — if the tokenizer changed, data is
    automatically re-tokenized.

    Args:
        tok: HF tokenizer.
        tokenized_dir: Where to save train.pt / val.pt.
        direction: "forward" or "backward".
        text_dir: Directory containing wiki_train.txt / wiki_val.txt.
                  Defaults to OUT_DIR.
        force: If True, always re-tokenize regardless of fingerprint.
        flip_mode: "text" (default), "token", or "both".

    Returns:
        {"train": Path, "val": Path}
    """
    if text_dir is None:
        text_dir = OUT_DIR

    tokenized_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Fingerprint: which tokenizer+config produced these .pt files
    tok_fingerprint = f"{tokenized_dir.resolve()}|dir={direction}|flip={flip_mode}"

    for split in ("train", "val"):
        out_path = tokenized_dir / f"wiki_{split}.pt"
        meta_path = tokenized_dir / f"wiki_{split}_meta.json"

        # Check whether existing data matches current tokenizer
        reuse_ok = False
        if out_path.exists() and meta_path.exists() and not force:
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("tokenizer") == tok_fingerprint:
                    reuse_ok = True
                    print(f"[Tokenize] {out_path.name} up-to-date: "
                          f"{meta.get('n_tokens', '?'):,} tokens from "
                          f"{meta.get('n_docs', '?')} docs "
                          f"(tokenizer fingerprint matches)")
            except (json.JSONDecodeError, KeyError):
                pass

        if reuse_ok:
            paths[split] = out_path
            continue

        if out_path.exists():
            print(f"[Tokenize] {out_path.name} exists but tokenizer changed — re-tokenizing")
            out_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

        text_path = text_dir / f"wiki_{split}.txt"
        if not text_path.exists():
            text_path = text_dir / f"{split}.txt"
        if not text_path.exists():
            raise FileNotFoundError(
                f"Missing text file: {text_dir}/wiki_{split}.txt or {text_dir}/{split}.txt"
            )

        print(f"[Tokenize] Encoding {text_path} "
              f"(direction={direction}, flip_mode={flip_mode}) ...")
        with open(text_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        all_ids: list[int] = []
        reverse_text = (flip_mode in ("text", "both"))
        flip_tokens = (flip_mode in ("token", "both"))

        for i, text in enumerate(lines):
            try:
                if reverse_text:
                    text = text[::-1]  # character-level reversal
                ids = tok.encode(text).ids
                if flip_tokens:
                    ids = ids[::-1]  # token-level reversal (preserves token-internal order)
                all_ids.extend(ids)
            except Exception:
                continue
            if (i + 1) % 10000 == 0:
                print(f"  [{split}] {i+1}/{len(lines)} docs, {len(all_ids):,} tokens",
                      flush=True)

        tensor = torch.tensor(all_ids, dtype=torch.int64)
        torch.save(tensor, str(out_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": str(text_path),
                "tokenizer": tok_fingerprint,
                "direction": direction,
                "flip_mode": flip_mode,
                "n_docs": len(lines),
                "n_tokens": len(all_ids),
                "vocab_size": tok.get_vocab_size(),
            }, f, indent=2, ensure_ascii=False)
        print(f"  [{split}] {len(lines):,} docs → {len(all_ids):,} tokens → {out_path}")
        paths[split] = out_path

    return paths


# ===================================================================
# Phase 3: Model builder
# ===================================================================

def build_model(vocab_size: int, device: torch.device, model_size: str = "45M"):
    """Build from-scratch GPT-2 causal LM (dropout-free)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    size_presets = {
        "45M":  dict(n_layer=6, n_embd=512, n_head=8),
        "124M": dict(n_layer=12, n_embd=768, n_head=12),
    }
    p = size_presets[model_size]

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
# Data iterator
# ===================================================================

def make_data_iterator(
    token_path: Path, seq_len: int, batch_tokens: int, seed: int,
    stride: int = 32,
):
    """Infinite iterator yielding {"input_ids", "labels"} batches.

    Uses overlapping chunks (torch.unfold) with the given stride.
    """
    data = torch.load(str(token_path), map_location="cpu").to(torch.int64)
    n_total = data.numel()

    chunk_len = seq_len + 1
    if n_total < chunk_len:
        raise ValueError(
            f"Not enough tokens ({n_total}) for one chunk of length {chunk_len}"
        )
    chunks = data.unfold(0, chunk_len, stride)  # [n_chunks, chunk_len]
    n_chunks = chunks.size(0)

    all_inputs = chunks[:, :seq_len].clone()   # [n_chunks, seq_len]
    all_targets = chunks[:, 1:].clone()         # [n_chunks, seq_len]

    rng = np.random.default_rng(seed)
    indices = np.arange(n_chunks)
    rng.shuffle(indices)

    batch_size = batch_tokens // seq_len
    i = 0

    def iterate():
        nonlocal i
        while True:
            if i + batch_size > n_chunks:
                rng.shuffle(indices)
                i = 0
            batch_indices = indices[i:i + batch_size]
            i += batch_size
            yield {
                "input_ids": all_inputs[batch_indices].to(dtype=torch.int64),
                "labels": all_targets[batch_indices].to(dtype=torch.int64),
            }

    return iterate()


# ===================================================================
# Training loop
# ===================================================================

def train(
    model, device: torch.device,
    train_path: Path, val_path: Path,
    vocab_size: int, model_dir: Path,
    direction: str, seed: int = 46,
    n_steps: int = 10000,
    flip_mode: str = "text",
):
    """Standard causal LM training loop."""
    model_dir.mkdir(parents=True, exist_ok=True)

    if direction == "forward":
        if flip_mode == "both":
            direction_tag = "fwd_bothflip"
        else:
            direction_tag = "fwd"
    elif flip_mode == "token":
        direction_tag = "bwd_tokflip"
    else:
        direction_tag = "bwd"
    ckpt_path = model_dir / f"{direction_tag}_lm_s{seed}_n{n_steps}.pt"
    log_path = model_dir / f"train_{direction_tag}_lm.log"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Direction={direction}, flip_mode={flip_mode}, "
          f"tag={direction_tag}, model={MODEL_SIZE}, "
          f"vocab={vocab_size}, params={n_params/1e6:.1f}M, n_steps={n_steps}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0,
    )

    train_iter = make_data_iterator(train_path, SEQ_LEN, BT, seed, stride=STRIDE)
    val_iter = make_data_iterator(val_path, SEQ_LEN, BT, 999, stride=STRIDE)

    n_val_batches = 20
    val_batches = [next(val_iter) for _ in range(n_val_batches)]

    # Unigram baseline
    train_data = torch.load(str(train_path), map_location="cpu").to(torch.int64)
    counts = torch.bincount(train_data, minlength=vocab_size).float()
    p = counts / counts.sum()
    p = p[p > 0]
    unigram_ce = float(-(p * torch.log(p)).sum())
    print(f"[Train] Unigram CE={unigram_ce:.3f} ppl={math.exp(unigram_ce):.1f}")

    with open(log_path, "w") as f:
        f.write(f"# {direction_tag.upper()} LM (union vocab) train; "
                f"direction={direction} flip_mode={flip_mode} "
                f"n_steps={n_steps} "
                f"unigram_CE={unigram_ce:.4f} ppl={math.exp(unigram_ce):.2f}\n")
        f.write("step\tloss\tgrad_norm\tlr\ttok_per_s\teval_ce\teval_ppl\n")

    model.train()
    t0 = time.time()
    seen = 0
    best_eval_ce = float("inf")

    try:
        from tqdm import tqdm
        bar = tqdm(range(n_steps), total=n_steps)
    except ImportError:
        bar = range(n_steps)

    for step in bar:
        cur_lr = _lr_at(step, lr=LR, warmup=WARMUP, n_steps=n_steps)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr

        batch = next(train_iter)
        inputs = batch["input_ids"].to(device)
        targets = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            loss = model(input_ids=inputs, labels=targets).loss

        if not math.isfinite(float(loss)):
            raise RuntimeError(f"Non-finite loss at step {step}")

        optim.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        optim.step()
        seen += inputs.numel()

        eval_ce = float("nan")
        if step % EVAL_EVERY == 0 or step == n_steps - 1:
            eval_ce = evaluate(model, val_batches, device)
            eval_ppl = math.exp(min(eval_ce, 20))
            dt = time.time() - t0
            tps = seen / max(dt, 1e-6)
            with open(log_path, "a") as f:
                f.write(f"{step}\t{float(loss):.4f}\t{gn:.3f}\t{cur_lr:.2e}\t"
                        f"{tps:.0f}\t{eval_ce:.4f}\t{eval_ppl:.2f}\n")
            if hasattr(bar, "set_description"):
                bar.set_description(
                    f"loss={float(loss):.3f} gn={gn:.2f} lr={cur_lr:.1e} "
                    f"eval_CE={eval_ce:.3f} ppl={eval_ppl:.1f}")
                bar.write(
                    f"[Step {step}/{n_steps}] loss={float(loss):.4f} gn={gn:.3f} "
                    f"lr={cur_lr:.2e} tok/s={tps:.0f} eval_CE={eval_ce:.4f} "
                    f"ppl={eval_ppl:.2f} (unigram={math.exp(unigram_ce):.1f})")
            if eval_ce < best_eval_ce:
                best_eval_ce = eval_ce
                best_path = model_dir / f"{direction_tag}_lm_s{seed}_best.pt"
                torch.save(model.state_dict(), str(best_path))
                if hasattr(bar, "write"):
                    bar.write(f"  → saved best (CE={best_eval_ce:.4f}) → {best_path.name}")

        if step > 0 and step % CKPT_EVERY == 0:
            torch.save(model.state_dict(),
                      str(model_dir / f"{direction_tag}_lm_s{seed}_step{step}.pt"))

    torch.save(model.state_dict(), str(ckpt_path))
    final_ce = evaluate(model, val_batches, device)
    final_name = f"{direction_tag}_lm_s{seed}_final_ce{final_ce:.4f}.pt"
    torch.save(model.state_dict(), str(model_dir / final_name))
    wall = time.time() - t0
    print(f"\n[Train] Done. Final eval_CE={final_ce:.4f} ppl={math.exp(final_ce):.2f} "
          f"unigram_ppl={math.exp(unigram_ce):.2f} "
          f"ratio={math.exp(unigram_ce)/math.exp(final_ce):.2f}× wall={wall:.0f}s")
    print(f"  → saved final: {final_name}", flush=True)
    print(f"  → saved final: {ckpt_path.name}", flush=True)


@torch.no_grad()
def evaluate(model, batches, device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in batches:
        inputs = batch["input_ids"].to(device)
        targets = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            loss = model(input_ids=inputs, labels=targets).loss
        total_loss += float(loss) * inputs.numel()
        total_tokens += inputs.numel()
    model.train()
    return total_loss / max(total_tokens, 1)


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Train forward/backward LM with union-vocab BPE tokenizer")
    ap.add_argument("--tokenizer_dir", required=True,
                    help="Path to union-vocab tokenizer directory (e.g. "
                         "outputs/tokenizer_union_fwd_v1_10000)")
    ap.add_argument("--direction", required=True, choices=["forward", "backward"],
                    help="Forward: train left-to-right causal LM. "
                         "Backward: train right-to-left causal LM. "
                         "(Tokenization behaviour is controlled by --flip_mode.)")
    ap.add_argument("--flip_mode", choices=["text", "token", "both"], default="text",
                    help="'text' (default): reverse raw text before encoding "
                         "(for backward-rule tokenizer). "
                         "'token': encode normal text, then reverse token "
                         "sequence per doc (for backward LM with forward-rule tokenizer). "
                         "'both': reverse text → encode → reverse token ids "
                         "(for forward LM with backward-rule tokenizer #4; "
                         "text reversal enables correct merge matching, "
                         "token reversal restores forward reading order).")
    ap.add_argument("--phase", choices=["tokenize", "train", "all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--n_steps", type=int, default=N_STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--model_size", default=MODEL_SIZE, choices=["45M", "124M"])
    ap.add_argument("--tag", default="",
                    help="Optional suffix for model output directory "
                         "(e.g. --tag n20k → model_union_fwd_v1_10000_n20k). "
                         "Tokenized data is NOT affected.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    paths = resolve_paths(args.tokenizer_dir, args.direction,
                         tag=args.tag, flip_mode=args.flip_mode)

    print("=" * 60)
    print(f"Union LM Training — {args.direction.upper()}")
    print(f"  tokenizer:  {paths['tokenizer'].name}")
    print(f"  tokenized:  {paths['tokenized'].name}")
    print(f"  model:      {paths['model'].name}")
    print(f"  direction:  {args.direction}")
    if args.direction == "backward":
        print(f"  flip_mode:  {args.flip_mode}")
    print(f"  device={device}, seed={args.seed}, n_steps={args.n_steps}")
    print(f"  model={args.model_size}, lr={args.lr}")
    print(f"  seq_len={SEQ_LEN}, batch_tokens={BT}, stride={STRIDE}")
    print("=" * 60)

    # ── Load tokenizer ──
    tok, vocab_size = load_tokenizer(paths["tokenizer"])

    # ── Tokenize corpus ──
    if args.phase in ("tokenize", "all"):
        token_paths = tokenize_corpus(tok, paths["tokenized"], args.direction,
                                      flip_mode=args.flip_mode)
    else:
        token_paths = {
            "train": paths["tokenized"] / "wiki_train.pt",
            "val": paths["tokenized"] / "wiki_val.pt",
        }
        for split, p in token_paths.items():
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p} — run with --phase tokenize first")

    if args.phase == "tokenize":
        print(f"\n[Tokenize] Done. Data in {paths['tokenized']}/")
        return

    # ── Build model ──
    print(f"\n[Model] Building GPT-2 {args.model_size} (vocab={vocab_size}) ...")
    model = build_model(vocab_size, device, args.model_size)
    print(f"[Model] Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    dummy = torch.randint(0, vocab_size, (2, SEQ_LEN), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        print(f"[Model] Sanity: loss={model(input_ids=dummy, labels=dummy).loss:.4f} ✓")

    # ── Train ──
    train(model, device,
          token_paths["train"], token_paths["val"],
          vocab_size, paths["model"],
          direction=args.direction, seed=args.seed,
          n_steps=args.n_steps, flip_mode=args.flip_mode)
    print(f"\nDone → {paths['model']}/")


if __name__ == "__main__":
    main()
