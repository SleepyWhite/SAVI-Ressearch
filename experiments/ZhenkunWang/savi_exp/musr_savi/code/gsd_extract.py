"""gsd_extract — zero-oracle event extraction for stage g4 station D.

Station D's fact source, and the carrier of its deployment honesty: the model itself
extracts, from the NARRATIVE ALONE, "what moved where at each step". The parsed moves
replace the structural oracle's ``space.moves`` behind the mechanical event anchor line
(``gsd_score.render_event_line``) via the stages_g4 ``_SpaceView`` proxy, under the SAME
frozen scoring template. Extraction errors are never repaired, never re-prompted, and
never a reason to exclude an item — MISPARSE IS THE TREATMENT (a layer the model failed
to extract simply degrades to the E-none rendering: no event line at all).

==========================================================================================
LEAK BOUNDARY (zero-oracle discipline) — READ BEFORE TOUCHING THIS MODULE
==========================================================================================
The extraction prompt is built from ``item["narrative"]`` ONLY; the only other item field
this module ever touches is ``item["id"]`` (cache key). It must NEVER read the item's
task query, its answer options, or ANY gold-solution structure — those key names are
deliberately kept OUT of this source file in their entirety, and a unit test self-greps
the source to pin that (the gsd_score no-tree-reads idiom). Gold enters only downstream
and only as an EVALUATION read-out: ``extraction_quality`` compares parsed moves against
``space.moves`` (a product of the audited structural oracle) AFTER extraction, and its
result never feeds any model input.

==========================================================================================
FROZEN CONVENTIONS
==========================================================================================
* ``EXTRACT_TEMPLATE`` (sha256 exported as ``EXTRACT_TEMPLATE_SHA256``, snapshot-tested —
  freezing == testing): single ``{narrative}`` field; instructs the model to list every
  move, in story order, one line per move event, numbered from Event 1, in exactly the
  ``render_event_line`` wording ``Event t: the <object> is moved to the <location>.``
  (multi-object events: the parallel sentences share one Event line), and nothing else.
* Generation: ``greedy=True`` (deterministic readout), ONE prompt per item,
  ``max_new_tokens`` default 512. The seed handed to the emitter is
  ``sc_core.sample_seed(seed, EXTRACT_STAGE, item_id, 0)`` (default base seed 20260709) —
  inert under greedy decoding but pinned so the Emitter contract and shard invariance
  hold everywhere.
* Line parser (``parse_event_lines``): per stripped line, the frozen head regex
  ``^Event\\s+(\\d+):`` followed by a body that consists ENTIRELY of one or more frozen
  move sentences ``the <obj> is moved to the <loc>.`` — one sentence is the plan's
  anchored single-move regex, several sentences mirror ``render_event_line``'s multi-move
  format. Head and sentence matching are CASE-INSENSITIVE ("The ball is moved ..." is the
  same sentence — the same case-insensitive convention the quality read-out below
  freezes; captured obj/loc keep their verbatim casing). The body is validated in LINEAR
  TIME by consuming one ANCHORED sentence match at a time — never an ambiguous regex
  repetition over the whole body, so a long non-conforming line can never backtrack
  catastrophically — and the rule is ALL-OR-NOTHING: any junk between, inside, or after
  the sentences (e.g. ``... to the b.junk the c ...``) REJECTS the whole line, which is
  then skipped and consumes no index (FROZEN — a rejected line is never salvaged into
  partial moves). Anything else (chatter, headers, "nothing is moved" lines, partial
  matches) is likewise skipped and consumes NO index. Canonical ``t`` is POSITIONAL:
  accepted Event lines are renumbered 1..k in order of appearance; the model's own
  numbering is recorded verbatim in the cached ``raw_text`` but never trusted (FROZEN).
  A multi-move line yields several ``(t, obj, loc)`` sharing that line's canonical ``t``.
  Empty output -> no moves.
* JSONL content-key cache (``sc_core.append_record`` atomic-append idiom): key =
  ``(item_id, EXTRACT_STAGE, EXTRACT_TEMPLATE_SHA256)``, record stores the raw generated
  text. Last-write-wins on load; blank/torn lines skipped. A full hit leaves the emitter
  UNTOUCHED (``n_scored == 0``) and re-parses the cached text (parsing is deterministic,
  so the moves are byte-identical). ``n_scored`` = fresh generations THIS call (0 or 1).
* Alignment rule (``align_moves``, FROZEN): layer ``t`` (1 <= t < T_space) gets exactly
  the parsed moves whose canonical ``t`` equals it; a layer with no parsed entry maps to
  ``None`` — station D renders that layer with NO event line (the E-none degradation);
  parsed entries with ``t >= T_space`` are dropped. NO item is ever excluded for bad
  parsing.
* Quality read-out (``extraction_quality``): per layer t in 1..T-1, case-insensitive
  stripped ``(obj, loc)`` SET comparison of the parsed moves against the oracle moves at
  t; ``layer_match_rate`` = mean of the per-layer matches; ``all_exact`` = every layer
  matches. Evaluation-side only.

CODE SEPARATION: imports stdlib + ``sc_core`` (seed derivation / atomic append / GenOut
Emitter contract) only, so the module imports with ZERO torch/transformers. It NEVER
imports test code; the emitter is dependency-injected via the ``sc_core.Emitter``
protocol (real ``HFEmitter`` at L1, a scripted CPU emitter in the unit tests).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

import sc_core as sc

# ======================================================================================
# Frozen extraction template (sha256 snapshot-tested in tests/test_gsd_extract.py).
# The ONLY format field is {narrative} — zero-oracle by construction.
# ======================================================================================
EXTRACT_TEMPLATE = (
    "{narrative}\n\n"
    "Read the story above. List every time an object is moved to a new place, in the "
    "order the moves happen in the story. Write exactly one line per move event, "
    "numbered from Event 1, in exactly this format:\n"
    "Event 1: the <object> is moved to the <location>.\n"
    "Event 2: the <object> is moved to the <location>.\n"
    "If a single event moves several objects at once, put their \"the <object> is moved "
    "to the <location>.\" sentences together on that one Event line. Write the numbered "
    "list and nothing else.\n"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


EXTRACT_TEMPLATE_SHA256 = _sha256(EXTRACT_TEMPLATE)

# Frozen extraction constants (echoed in stages_g4.PREREG_G4).
EXTRACT_STAGE = "g4extract"
DEFAULT_EXTRACT_SEED = 20260709
DEFAULT_EXTRACT_MAX_NEW_TOKENS = 512

# ======================================================================================
# Frozen line parser
# ======================================================================================
# Head: "Event <digits>:" then the body. The model's <digits> are recorded in raw_text
# only — canonical t is positional (see module docstring). Case-insensitive (frozen: the
# same convention as the quality read-out).
_EVENT_HEAD_RE = re.compile(r"^Event\s+(\d+):\s*(.*)$", re.IGNORECASE)
# ONE frozen move sentence (render_event_line wording), matched ANCHORED at a position by
# ``_parse_move_body``. obj stops (lazily) at the first " is moved to the "; loc is a
# period-free run ending at the sentence's "." — both backtrack-bounded, and the
# sentence-at-a-time consumption below keeps the whole-body validation LINEAR (no
# ambiguous `(sentence)+` repetition -> no catastrophic backtracking on failing lines).
_MOVE_SENTENCE_RE = re.compile(r"the (.+?) is moved to the ([^.]+)\.", re.IGNORECASE)


def _parse_move_body(body: str) -> Optional[list]:
    """``[(obj, loc), ...]`` iff ``body`` is ENTIRELY one-or-more frozen move sentences
    (whitespace-separated), else None (all-or-nothing, FROZEN: junk between/inside/after
    the sentences rejects the WHOLE body — never a partial salvage). Linear time: one
    anchored sentence match is consumed per step; the first non-sentence residue fails.
    """
    pairs: list = []
    pos, n = 0, len(body)
    while pos < n:
        m = _MOVE_SENTENCE_RE.match(body, pos)
        if m is None:
            return None
        pairs.append((m.group(1), m.group(2)))
        pos = m.end()
        while pos < n and body[pos].isspace():
            pos += 1
    return pairs if pairs else None


def parse_event_lines(text: Optional[str]) -> list:
    """Parse a raw extraction reply into ``[(t, obj, loc), ...]`` (canonical positional t).

    Tolerant by SKIPPING whole lines: a line is accepted iff (after strip) it matches the
    frozen Event head AND its body consists entirely of frozen move sentences
    (case-insensitive, all-or-nothing — see ``_parse_move_body``); every other line —
    chatter, "nothing is moved" lines, partial matches, sentence runs with embedded junk
    — is skipped and consumes no index. Accepted lines are renumbered 1..k in order of
    appearance regardless of the model's own numbers; a multi-move line contributes
    several moves sharing its t.
    """
    moves: list = []
    if not text:
        return moves
    k = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        head = _EVENT_HEAD_RE.match(line)
        if head is None:
            continue
        pairs = _parse_move_body(head.group(2).strip())
        if pairs is None:
            continue
        k += 1
        for obj, loc in pairs:
            moves.append((k, obj, loc))
    return moves


# ======================================================================================
# Extraction cache (JSONL; sc_core atomic-append idiom, 3-part content key)
# ======================================================================================
def _extract_cache_key(rec: dict) -> tuple:
    return (rec["item_id"], rec["stage"], rec["template_sha"])


def _load_extract_cache(path: Optional[str]) -> dict:
    """``{3-part key: raw_text}`` with last-write-wins dedup; blank/torn lines skipped."""
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
                out[_extract_cache_key(rec)] = rec["raw_text"]
            except (ValueError, TypeError, KeyError):
                continue
    return out


# ======================================================================================
# The extraction pass
# ======================================================================================
def extract_events(item: dict, emitter, seed: int = DEFAULT_EXTRACT_SEED,
                   max_new_tokens: int = DEFAULT_EXTRACT_MAX_NEW_TOKENS,
                   cache_path: Optional[str] = None) -> dict:
    """Zero-oracle event extraction for one item. See the module docstring contract.

    Builds the frozen prompt from ``item["narrative"]`` ONLY, generates one greedy reply
    through ``emitter`` (or reloads it from the content-keyed cache: on a hit the emitter
    is untouched and ``n_scored == 0``), and parses it into canonical moves.

    Returns::

        {"moves":    [(t, obj, loc), ...],   # canonical positional t (parse_event_lines)
         "raw_text": str,                    # the verbatim model reply (cached)
         "n_scored": int}                    # fresh generations this call (0 == cache hit)
    """
    item_id = item["id"]
    key = (item_id, EXTRACT_STAGE, EXTRACT_TEMPLATE_SHA256)
    cache = _load_extract_cache(cache_path)
    if key in cache:
        raw_text = cache[key]
        n_scored = 0
    else:
        prompt = EXTRACT_TEMPLATE.format(narrative=item["narrative"])
        gen_seed = sc.sample_seed(seed, EXTRACT_STAGE, item_id, 0)
        outs = emitter.generate([prompt], [gen_seed], greedy=True,
                                max_new_tokens=max_new_tokens)
        assert len(outs) == 1, (               # loud, not a silent extraction loss
            "emitter returned %d outputs for 1 prompt (item %r)" % (len(outs), item_id))
        raw_text = outs[0].text
        rec = {"item_id": item_id, "stage": EXTRACT_STAGE,
               "template_sha": EXTRACT_TEMPLATE_SHA256, "raw_text": raw_text,
               "seed": gen_seed, "n_new_tokens": int(outs[0].n_new_tokens)}
        if cache_path:
            sc.append_record(cache_path, rec)
        n_scored = 1
    return {"moves": parse_event_lines(raw_text), "raw_text": raw_text,
            "n_scored": n_scored}


# ======================================================================================
# Frozen alignment rule (parsed moves -> per-layer injection table)
# ======================================================================================
def align_moves(parsed_moves: list, T_space: int) -> dict:
    """``{t: [(t, obj, loc), ...] | None for t in 1..T_space-1}`` (FROZEN rule).

    Layer ``t`` gets exactly the parsed moves whose canonical t == t; a layer with no
    parsed entry maps to ``None`` (station D renders it with NO event line — the E-none
    degradation, never an exclusion); parsed entries with ``t >= T_space`` are dropped.
    """
    out: dict = {t: None for t in range(1, T_space)}
    for mv in parsed_moves:
        t = mv[0]
        if 1 <= t < T_space:
            if out[t] is None:
                out[t] = []
            out[t].append(tuple(mv))
    return out


# ======================================================================================
# Extraction-quality read-out (EVALUATION side only — never feeds any model input)
# ======================================================================================
def _norm_pair(obj, loc) -> tuple:
    return (str(obj).strip().lower(), str(loc).strip().lower())


def extraction_quality(parsed_moves: list, space) -> dict:
    """Per-layer exact-match of parsed moves against the oracle ``space.moves``.

    For every layer t in 1..T-1 the parsed and oracle ``(obj, loc)`` pairs at t are
    compared as case-insensitive, stripped SETS (multi-move layers match iff the whole
    set does; a no-move layer matches iff nothing was parsed there). Returns::

        {"per_layer":        [{"t", "gold_moves", "parsed_moves", "match"}, ...],
         "layer_match_rate": float,   # mean of per-layer match over t = 1..T-1
         "all_exact":        bool}    # every layer matches
    """
    per_layer: list = []
    for t in range(1, space.T):
        gold = {_norm_pair(o, loc) for (tt, o, loc) in space.moves if tt == t}
        parsed = {_norm_pair(o, loc) for (tt, o, loc) in parsed_moves if tt == t}
        per_layer.append({"t": t, "gold_moves": sorted(gold),
                          "parsed_moves": sorted(parsed), "match": gold == parsed})
    n = len(per_layer)
    rate = (sum(1 for pl in per_layer if pl["match"]) / n) if n else 1.0
    return {"per_layer": per_layer, "layer_match_rate": rate,
            "all_exact": all(pl["match"] for pl in per_layer)}
