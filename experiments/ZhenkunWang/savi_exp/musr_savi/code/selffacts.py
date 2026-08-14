"""selffacts — a3 self-facts control: the prompt layer for the three sampling arms
(design ``plans/2026-07-06-selffacts-a3.md`` §2 / Subtask 2).

The a3 control asks whether zero-oracle prompting (a self-extracted facts scaffold) can fix
the 26 certified commitment-failure MuSR items. Three arms share this module's prompts:

  * **S1** — a one-call scaffold: the ordinary question prompt, augmented to make the model
    first write an ``OBSERVATIONS:`` numbered list, then reason over it. ``format_s1_prompt``.
  * **S2** — a two-stage self-extraction: one greedy, question-agnostic extraction pass
    (``format_extract_prompt``) produces an observation list that is then injected into the
    EXACT same facts wrapper the oracle arm uses (``format_s2_prompt``).
  * **O** — the oracle re-verify arm; its prompt is ``sc_core.format_facts_prompt`` (not built
    here — S2 reuses its composition so the two are comparable byte-for-byte).

THE KEY INVARIANT (correctness of the whole comparison): ``format_s2_prompt`` composes with
``sc.FACTS_TEMPLATE`` + ``sc.FACTS_PREAMBLE`` VERBATIM, substituting only the facts content,
so that when the injected text equals ``sc.facts_block(item)`` the S2 prompt is byte-for-byte
identical to ``sc.format_facts_prompt(item)``. That makes S2-vs-O a pure "facts content"
contrast (self-extracted vs gold), holding the template/preamble/base-question fixed.

The S1 / EXTRACT templates are FROZEN before PREREG_a3: their sha256 is snapshot-tested here
(``sha256_text(S1_TEMPLATE) == S1_TEMPLATE_SHA256``) exactly as ``sc_core`` freezes its own
prompts. The token caps below are CANDIDATES, to be pinned by the L1 smoke in Subtask 4;
re-pinning them does not touch the template shas.

CODE SEPARATION: this module imports ONLY ``sc_core`` + stdlib (``hashlib``/``json``/``os``).
It NEVER imports test code and pulls in no torch/transformers at import (the extraction call is
routed through an injected Emitter by the runner, not made here). The per-story extraction
cache is an independent JSONL, keyed by ``story_id`` with last-write-wins semantics (mirrors
``sc.append_record`` / ``sc.load_cache``, but on the single ``story_id`` field rather than the
sample cache's 4-tuple key).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import sc_core as sc

# ======================================================================================
# Token caps — CANDIDATES, pinned by the L1 smoke in Subtask 4 (design §1/§7). Re-pinning
# these does NOT change the frozen template shas below (they gate generation, not the string).
# ======================================================================================
S1_MAX_NEW_TOKENS = 3072
EXTRACT_MAX_NEW_TOKENS = 1024

# ======================================================================================
# Frozen prompt templates (English — MuSR is English). sha256 snapshot-tested below.
# ======================================================================================
# S1: the standard question prompt (mirrors sc.PROMPT_TEMPLATE's narrative/question/choices
# layout) but asks the model to first surface an OBSERVATIONS numbered list, then reason over
# only those observations, and still finish with the frozen ANSWER contract.
S1_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    "First, under a heading \"OBSERVATIONS:\", list the observations you can confirm "
    "from the story, in the order they happened, as a numbered list. "
    "Then think step by step using only your listed observations, and finish your reply "
    'with a line "ANSWER: <option number>".'
)

# EXTRACT: a question-agnostic pass over the narrative ONLY — no question, no choices — so the
# same extracted observation list serves every question of a story (one extraction per story).
EXTRACT_TEMPLATE = (
    "{narrative}\n\n"
    "List the observations you can confirm from this story, in the order they happened, "
    "as a numbered list (one observation per line, like \"1. ...\"). "
    "Do not answer any question and do not add commentary; output only the numbered list."
)

# Frozen snapshots (computed from the template strings above; the snapshot tests assert these
# match). Changing a template MUST update its sha deliberately (freezing = testing).
S1_TEMPLATE_SHA256 = "c51f64681f48437f70520edf95cf7d5060e42a06373e1fcc953365fc30d79f46"
EXTRACT_TEMPLATE_SHA256 = "783b0ee7c15230baf24a8d2a9e303f583c1480b2234cec0695a22c7295281183"


def sha256_text(t: str) -> str:
    """sha256 hexdigest of ``t`` (utf-8). Same convention sc_core snapshot-tests its prompts."""
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


# ======================================================================================
# Prompt formatters
# ======================================================================================
def format_s1_prompt(item: dict) -> str:
    """S1 one-call scaffold prompt (frozen ``S1_TEMPLATE``). Choices rendered 1-based via the
    same ``sc.render_numbered_choices`` the ordinary prompt uses (so parse_answer's contract
    is unchanged)."""
    return S1_TEMPLATE.format(
        narrative=item["narrative"],
        question=item["question"],
        numbered_choices=sc.render_numbered_choices(item["choices"]),
    )


def format_extract_prompt(item: dict) -> str:
    """S2 stage-1 extraction prompt (frozen ``EXTRACT_TEMPLATE``). Narrative ONLY — deliberately
    excludes the question and every choice so the extraction is question-agnostic and can be
    cached and reused across all questions of the same story."""
    return EXTRACT_TEMPLATE.format(narrative=item["narrative"])


def format_s2_prompt(item: dict, extracted_text: Optional[str]) -> str:
    """S2 stage-2 prompt: the oracle arm's composition with the gold facts SWAPPED for the
    self-extracted ``extracted_text``.

    Reuses ``sc.FACTS_TEMPLATE`` + ``sc.FACTS_PREAMBLE`` verbatim and substitutes ONLY the facts
    content — structurally identical to ``sc.format_facts_prompt`` except for what goes in the
    facts slot. Consequently, when ``extracted_text == sc.facts_block(item)`` this returns the
    byte-for-byte same string as ``sc.format_facts_prompt(item)`` (the S2-vs-O invariant tested
    in ``test_s2_prompt_is_exact_oracle_composition_with_swapped_facts``).

    Note: the facts content is inserted RAW (no ``.strip()``). ``sc.format_facts_prompt`` inserts
    ``facts_block(item)`` raw, so stripping here would break the byte-equality whenever the facts
    content carried edge whitespace; ``facts_block`` is verified never to produce leading/trailing
    whitespace, so the two agree unconditionally on the invariant path. ``None``/empty extract ->
    an empty facts slot (never the literal string "None").
    """
    return sc.FACTS_TEMPLATE.format(
        facts_preamble=sc.FACTS_PREAMBLE,
        facts=(extracted_text if extracted_text else ""),
        base_prompt=sc.format_prompt(item),
    )


# ======================================================================================
# story_id — the per-story key for the extraction cache
# ======================================================================================
def story_id(item_id: str) -> str:
    """Map an item id ``"{subtask}-{NNNN}-q{N}"`` to its story id ``"{subtask}-{NNNN}"`` by
    stripping the trailing ``-q<N>`` question marker. Robust: rsplit on the LAST ``-q`` (subtask
    names never contain ``-q``); an id without a ``-q`` marker is returned unchanged."""
    return str(item_id).rsplit("-q", 1)[0]


# ======================================================================================
# Diagnostic heuristics (EXPLORATORY — never gate the verdict)
# ======================================================================================
def _is_numbered_line(line: str) -> bool:
    """Reasonable "numbered list line" test: the stripped line begins with a digit and has a
    ``.`` within its first few chars (so "1. ..", "2. ..", "10. .." all count; prose does not)."""
    s = line.strip()
    return bool(s) and s[0].isdigit() and "." in s[:4]


def scaffold_present(text_head: Optional[str]) -> bool:
    """S1 scaffold-compliance heuristic (diagnostic only): the sampled text head contains an
    ``OBSERVATIONS`` heading AND at least one numbered line. ``None``/empty -> False."""
    if not text_head:
        return False
    return "OBSERVATIONS" in text_head and any(
        _is_numbered_line(ln) for ln in text_head.splitlines())


def extraction_degenerate(text: Optional[str]) -> bool:
    """S2 extraction-degeneracy heuristic (diagnostic + L1 check, never gating): an extraction
    is degenerate iff it has FEWER than 2 numbered lines (empty/``None`` counts as degenerate)."""
    lines = [ln for ln in (text or "").splitlines() if _is_numbered_line(ln)]
    return len(lines) < 2


# ======================================================================================
# Per-story extraction cache (independent JSONL, keyed by story_id, last-write-wins)
# ======================================================================================
def append_extract(path: str, rec: dict) -> None:
    """Append one extraction record as a JSON line (atomic append + torn-tail heal + fsync,
    via ``sc.append_record``). ``rec`` carries at least ``story_id``/``subtask``/``text``/
    ``n_new_tokens``; identity for dedup on load is the single ``story_id`` field."""
    sc.append_record(path, rec)


def load_extracts(path: str) -> dict:
    """Load the extraction JSONL into ``{story_id: rec}`` with LAST-write-wins dedup (mirrors
    ``sc.load_cache`` but keyed by ``story_id`` instead of the 4-tuple sample key). Blank and
    malformed (torn) lines are skipped, never fatal; a missing/empty path -> ``{}``."""
    out: dict = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            sid = rec.get("story_id")
            if sid is None:
                continue
            out[sid] = rec        # last write wins
    return out
