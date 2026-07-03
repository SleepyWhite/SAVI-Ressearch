#!/usr/bin/env python3
"""Decode GPT-2-tokenized wiki .pt files back to raw text for custom BPE training."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import tiktoken

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "_deps"))

DATA_DIR = _HERE.parent / "data" / "wiki_reference"
OUT_DIR = _HERE / "data"
enc = tiktoken.get_encoding("gpt2")

for split in ("train", "val"):
    pt_path = DATA_DIR / f"{split}.pt"
    out_path = OUT_DIR / f"wiki_{split}.txt"
    if out_path.exists():
        print(f"[{split}] {out_path} already exists, skipping")
        continue

    print(f"[{split}] Loading {pt_path} ...")
    data = torch.load(str(pt_path), map_location="cpu")
    print(f"[{split}] Decoding {data.numel():,} tokens ...")
    # Decode in chunks for memory efficiency
    chunk_size = 100_000
    ids_list = data.tolist()
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(ids_list), chunk_size):
            chunk = ids_list[start:start + chunk_size]
            text = enc.decode(chunk)
            f.write(text)
            if start % 5_000_000 == 0 and start > 0:
                print(f"  [{split}] {start/1e6:.0f}M tokens decoded ...", flush=True)
    print(f"[{split}] Done → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
