"""featurizer — MuSR-cant Phase B hidden-state featurizer (Subtask 6, the model-facing part).

The ONLY new torch/transformers site in Phase B. It turns (prompt, generated-text) pairs into the
per-layer hidden-state features the ``verifiers.ProbeVerifier`` consumes:

  * ``featurize_last_token(prompts, texts) -> np.ndarray [N, L+1, H]`` — the per-layer hidden
    state at the LAST real token of ``prompt + text`` (the answer-level probe feature).
  * ``featurize_step_tokens(prompt, parsed_chain, text) -> np.ndarray [n_steps, L+1, H]`` — the
    per-layer hidden state at each BELIEF step-end token (the step-level probe feature).

DEPENDENCY INJECTION: ``run_musr_cant`` builds this lazily behind a ``featurizer_factory`` hook,
exactly like ``sc_core.HFEmitter`` behind ``emitter_factory``. The Phase B integration test injects
a pure ``FakeFeaturizer`` (deterministic numpy arrays, zero GPU); the b-stages only ever call the
two ``featurize_*`` methods, so nothing model-specific leaks into the core. This module imports
torch/transformers LAZILY inside ``__init__`` so importing ``featurizer`` stays CPU/GPU-free.

CODE SEPARATION: at import time this module pulls in ONLY stdlib + numpy + ``data_musr`` (for the
offline snapshot-symlink fix) + ``belief_schema`` (for the frozen BELIEF marker, reused so step
boundaries match ``parse_chain`` exactly). NEVER imports test code.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

import belief_schema as bs
import data_musr as dm

# Reuse the SAME belief-line marker the parser uses so featurized step boundaries line up with
# ``parse_chain``'s states (one feature per parsed BELIEF line, in order).
_BELIEF_MARKER_RE = bs._BELIEF_MARKER_RE

# Forward context cap (tokens). Narratives are audited <= dm.CONTEXT_TOKEN_LIMIT and a chain adds
# up to ~2k; 8192 comfortably covers prompt+text for Qwen3-4B while bounding a stray long input.
FEATURIZE_MAX_TOKENS = 8192


class HiddenStateFeaturizer:
    """Offline Qwen3-4B forward-only featurizer (per-layer hidden states). GPU site; injected.

    Loads with ``local_files_only=True`` + ``output_hidden_states=True`` + LEFT padding and the
    same chat template the emitter uses (``enable_thinking=False`` for Qwen3, tolerated-absent
    elsewhere). Not exercised by the CPU unit tests (``FakeFeaturizer`` substitutes); validated at
    L1 on the real GPU.
    """

    def __init__(self, model_name: str, device: str = "cuda", dtype=None,
                 max_tokens: int = FEATURIZE_MAX_TOKENS, batch: int = 8):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch                                    # lazy: keep the module CPU/GPU-free at import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:                                            # known refs/main-hash offline load fix
            dm._qwen_snapshot_symlink_fix()
        except Exception:
            pass

        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.batch = int(batch)                         # forward micro-batch (a memory lever)
        self.n_truncated = 0                            # count of inputs that hit the cap (observable)
        dtype = dtype if dtype is not None else torch.bfloat16

        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.tok.padding_side = "left"                  # last real token is at position -1 for all rows
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, local_files_only=True,
                attn_implementation="sdpa", output_hidden_states=True).to(device).eval()
        except (ValueError, ImportError):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, local_files_only=True,
                output_hidden_states=True).to(device).eval()

    # -- prompt rendering (chat-templated user turn, same convention as HFEmitter) -----
    def _render(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    # -- OOM / truncation guards -------------------------------------------------------
    def _is_oom(self, exc) -> bool:
        """True for a CUDA OOM, whether typed or surfaced as a generic 'out of memory' RuntimeError."""
        return (isinstance(exc, self._torch.cuda.OutOfMemoryError)
                or "out of memory" in str(exc).lower())

    def _note_truncation(self, strings, n_ids_after: int) -> None:
        """Warn + count when right-truncation ACTUALLY fired (a real input exceeded the cap).

        Only re-tokenizes (untruncated) when the encoded length hit the cap, so the extra work is
        confined to the rare near-cap case. A fired truncation means last-token / step-end offsets
        past the cut clamp to a mid-chain token — surfaced here rather than silently mislabelled."""
        if n_ids_after < self.max_tokens:
            return
        over = 0
        for s in strings:
            try:
                if len(self.tok(s, truncation=False)["input_ids"]) > self.max_tokens:
                    over += 1
            except Exception:
                pass
        if over:
            self.n_truncated += over
            print("[featurizer] WARNING: %d input(s) exceeded max_tokens=%d and were RIGHT-truncated"
                  " -> last-token/step-end features past the cut are unreliable (total truncated=%d)"
                  % (over, self.max_tokens, self.n_truncated), flush=True)

    # -- slice-FIRST forward helpers (index the wanted token(s) on bf16 BEFORE cast+move) ---
    def _forward_last_token(self, enc):
        """Per-layer hidden state at the LAST (left-padded) token -> numpy ``[B, L+1, H]``.

        Slices column -1 of each per-layer bf16 tensor BEFORE ``.float().cpu()``, so the fp32
        transient is ``[B, L+1, H]`` (no ``T`` factor) — avoids the ~25GB ``[B, T, L+1, H]`` peak."""
        torch = self._torch
        with torch.no_grad():
            out = self.model(**enc)
        feats = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)   # [B, L+1, H] (bf16)
        return feats.float().cpu().numpy()

    def _forward_step_rows(self, enc, tok_indices):
        """Per-layer hidden state at the given token indices (single sequence) -> ``[n_steps, L+1, H]``.

        Gathers only the ``tok_indices`` rows from each per-layer bf16 tensor BEFORE cast+move, so the
        fp32 transient is ``[n_steps, L+1, H]`` (no ``T`` factor). One empty_cache+retry on OOM."""
        torch = self._torch
        for attempt in (0, 1):
            try:
                with torch.no_grad():
                    out = self.model(**enc)
                dev = out.hidden_states[0].device
                idx = torch.as_tensor(tok_indices, device=dev, dtype=torch.long)
                rows = torch.stack([h[0].index_select(0, idx) for h in out.hidden_states], dim=1)
                return rows.float().cpu().numpy()        # [n_steps, L+1, H]
            except (self._torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not self._is_oom(exc) or attempt == 1:
                    raise
                self._torch.cuda.empty_cache()

    # -- answer-level feature: last real token, per layer -------------------------------
    def featurize_last_token(self, prompts: Sequence[str], texts: Sequence[str],
                             batch=None) -> np.ndarray:
        """Per-layer hidden state at the LAST real token of each ``render(prompt)+text``.

        Returns ``[N, L+1, H]`` (float32). With left padding the last real token is column -1 for
        every row. The forward micro-batch starts at ``batch`` (default ``self.batch``) and HALVES
        on a CUDA OOM (empty_cache + retry down to 1), so this hot path self-heals without any
        per-call-site wrapping. Only raises if a single-item forward still OOMs.
        """
        prompts = list(prompts)
        texts = list(texts)
        if not prompts:
            return np.zeros((0, 1, 1), dtype=np.float32)
        cur = int(batch or self.batch)
        feats = []
        i = 0
        while i < len(prompts):
            chunk_p = prompts[i:i + cur]
            chunk_t = texts[i:i + cur]
            fulls = [self._render(p) + (t or "") for p, t in zip(chunk_p, chunk_t)]
            try:
                enc = self.tok(fulls, return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_tokens)
                self._note_truncation(fulls, int(enc["input_ids"].shape[1]))
                enc = {k: v.to(self.device) for k, v in enc.items()}
                feats.append(self._forward_last_token(enc))    # [b, L+1, H]
                i += len(chunk_p)
            except (self._torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not self._is_oom(exc):
                    raise
                self._torch.cuda.empty_cache()
                if cur <= 1:
                    raise
                cur = max(1, cur // 2)
                print("[featurizer] OOM on last-token forward -> retrying at batch=%d" % cur,
                      flush=True)
        return np.concatenate(feats, axis=0).astype(np.float32)

    # -- step-level feature: each BELIEF step-end token, per layer -----------------------
    @staticmethod
    def _step_end_char_offsets(text: str, n_states: int):
        """FALLBACK step-end char offsets when ``ParsedChain.state_char_ends`` is unavailable
        (a hand-built chain). Enumerates the parser's marker; truncates/pads to ``n_states``.

        NOTE: this fallback CANNOT know which interior lines the parser dropped, so it is only
        correct when no interior BELIEF line was dropped. The primary path uses
        ``parsed_chain.state_char_ends`` (the actually-kept lines) precisely to avoid that bias.
        """
        ends = []
        for m in _BELIEF_MARKER_RE.finditer(text or ""):
            line_end = text.find("\n", m.end())
            if line_end == -1:
                line_end = len(text)
            ends.append(max(m.end(), line_end - 1))    # last char of the belief line
        if not ends:
            return []
        if len(ends) >= n_states:
            return ends[:n_states]
        return ends + [ends[-1]] * (n_states - len(ends))   # pad by repeating the last (defensive)

    def featurize_step_tokens(self, prompt: str, parsed_chain, text: str) -> np.ndarray:
        """Per-layer hidden state at each BELIEF step-end token of ``render(prompt)+text``.

        Returns ``[n_steps, L+1, H]`` with ``n_steps = len(parsed_chain.states)`` (empty chain ->
        ``[0, 1, 1]``). The step-end char position of state ``i`` is taken from
        ``parsed_chain.state_char_ends[i]`` — the last char of the ACTUALLY-KEPT BELIEF line — so a
        dropped interior line never shifts later states onto an earlier line's token. A chain without
        ``state_char_ends`` (hand-built) falls back to a marker scan. Positions map to token indices
        via the fast tokenizer's offset mapping; never raises (offsets clamped into range).
        """
        states = list(getattr(parsed_chain, "states", []) or [])
        n_states = len(states)
        if n_states == 0:
            return np.zeros((0, 1, 1), dtype=np.float32)

        rendered = self._render(prompt)
        full = rendered + (text or "")
        base = len(rendered)                           # char offset where ``text`` begins in ``full``
        # PRIMARY: the parser's per-kept-state char anchors (correct under interior drops).
        char_ends = list(getattr(parsed_chain, "state_char_ends", None) or [])
        if len(char_ends) != n_states:                 # FALLBACK: hand-built chain -> marker scan
            char_ends = self._step_end_char_offsets(text or "", n_states)
        # absolute char position (in ``full``) of each step's last char
        abs_chars = [base + c for c in char_ends] if char_ends else [len(full) - 1] * n_states

        enc = self.tok(full, return_tensors="pt", return_offsets_mapping=True,
                       truncation=True, max_length=self.max_tokens)
        offsets = enc.pop("offset_mapping")[0].tolist()   # [T, 2]
        n_ids = int(enc["input_ids"].shape[1])
        self._note_truncation([full], n_ids)
        tok_indices = [self._char_to_token(offsets, c, n_ids) for c in abs_chars]
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # slice-FIRST: gather only the n_steps step-end token rows (per layer) before cast+move.
        return self._forward_step_rows(enc, tok_indices)

    @staticmethod
    def _char_to_token(offsets, char_pos: int, T: int) -> int:
        """Token index whose span best covers ``char_pos``: the last token that starts at or before
        ``char_pos`` and has a non-empty span. Falls back to the last token. Clamped to [0, T-1]."""
        best = T - 1
        for j, (a, b) in enumerate(offsets):
            if b <= a:                                  # special/padding token (empty span)
                continue
            if a <= char_pos < b or (a <= char_pos and b <= char_pos):
                best = j
        return max(0, min(best, T - 1))
