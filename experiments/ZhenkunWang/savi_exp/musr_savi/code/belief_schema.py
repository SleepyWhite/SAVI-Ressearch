"""belief_schema — MuSR-cant Phase B belief-table CoT template, parser, and canonicalization.

The substrate that turns a model's structured chain-of-thought into a *sequence of belief
states* — the nodes of the belief-state trellis and the units the step-level verifier scores.

The state representation mirrors the ground-truth gold shape exactly:

    item["tree_raw"]["intermediate_data"][0]["beliefs"][t]
        == {character: {object: location}}
        e.g. {"Danny": {"notebook": "producer's desk", "earphones": "recording booth"},
              "Emma": {...}, "Ricky": {...}}

so a parsed state and a gold state live in the same space and can be compared directly by the
facts oracle (Subtask 2) without any NL parsing.

CODE SEPARATION: this module imports ONLY stdlib (``hashlib``/``json``/``re``) + ``sc_core``.
It NEVER imports test code and pulls in no torch/transformers — it is pure, CPU-only, and
import-safe. The ``ANSWER: <option number>`` contract is delegated to ``sc_core`` (the same
contract Phase A froze), so ``sc.parse_answer`` reads model output identically here.


==========================================================================================
FROZEN CONVENTIONS
==========================================================================================
* Per-step line format shown to the model and emitted by ``render_belief_line``:
      ``BELIEF[t]: {json}``   with ``json`` = ``{character: {object: location}}``.
  ``t`` is the 0-based event index. One line per event.
* ``BELIEF_TEMPLATE`` is frozen; its sha256 (``BELIEF_TEMPLATE_SHA256``) is snapshot-tested
  and echoed into the Phase B PREREG config block (freezing == testing).
* Answer contract is IDENTICAL to ``sc.PROMPT_TEMPLATE`` (ends with a line
  ``ANSWER: <option number>``, 1-based), so ``sc.parse_answer`` maps it to a 0-based index.
* ``canon_state`` is the Phi-merge key: it strips and lowercases EVERY key and value —
  character keys, object keys, and location strings alike — then ``json.dumps(sort_keys=True)``.
  Two logically-identical states under any key ordering / whitespace / casing produce the
  byte-identical canonical string (case-folding names keeps trellis merge and equality-to-gold
  consistent — see ``canon_state``).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import sc_core as sc

# ======================================================================================
# Frozen belief-table CoT template.
#
# Filled with ``str.format`` (mirrors ``sc.PROMPT_TEMPLATE``): the only single-brace tokens
# are the named fields ``{narrative}`` / ``{question}`` / ``{numbered_choices}``; every literal
# JSON brace in the instructions/examples is escaped as ``{{`` / ``}}`` so ``.format`` renders
# it back to a real single brace. Ends with the SAME ``ANSWER: <option number>`` contract as
# Phase A so ``sc.parse_answer`` works unchanged on model output.
# ======================================================================================
BELIEF_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    "Solve this as a belief-tracking problem. First break the story into the events that "
    "happen, in the order they happen (event 0 first). After EACH event, write exactly one "
    "line that records what every character currently believes about where every tracked "
    "object is. Use this exact format, one line per event:\n"
    "BELIEF[t]: {{\"Character\": {{\"object\": \"location\"}}, ...}}\n"
    "where t is the 0-based event number and the JSON object maps each character to a map "
    "from each object to the location that character believes it is in. Include every "
    "character and every tracked object on every BELIEF line, and write each location exactly "
    "as it is worded in the options. For example, after the first event you might write:\n"
    "BELIEF[0]: {{\"Alice\": {{\"ball\": \"box\"}}, \"Bob\": {{\"ball\": \"box\"}}}}\n"
    "Update the beliefs event by event. After the final BELIEF line, finish your reply with a "
    "line \"ANSWER: <option number>\"."
)

BELIEF_TEMPLATE_SHA256 = hashlib.sha256(BELIEF_TEMPLATE.encode("utf-8")).hexdigest()


def format_belief_prompt(item: dict) -> str:
    """Fill ``BELIEF_TEMPLATE`` from a normalized item (keys: narrative/question/choices).

    Reuses ``sc.render_numbered_choices`` for the 1-based choice block so the option numbering
    the model sees is identical to Phase A. Substituted values are never re-scanned for braces
    by ``.format`` (only the template string is), so narratives containing ``{``/``}`` are safe.
    """
    return BELIEF_TEMPLATE.format(
        narrative=item["narrative"],
        question=item["question"],
        numbered_choices=sc.render_numbered_choices(item["choices"]),
    )


# ======================================================================================
# Parsed chain
# ======================================================================================
@dataclass
class ParsedChain:
    """The structured read-out of one model reply.

    * ``states``          — the ordered list of belief states, one per successfully-parsed BELIEF
                            line (each a ``{character: {object: location}}`` dict).
    * ``answer``          — the final option as a 0-BASED index (via ``sc.parse_answer``), or None.
    * ``parse_ok``        — ``len(states) >= 1 and answer is not None`` (a usable trellis chain).
    * ``state_char_ends`` — parallel to ``states`` (``len == len(states)``): the character offset of
                            the LAST char of the BELIEF line each KEPT state came from. This anchors
                            the step-end hidden-state featurization to the *actually-kept* lines, so a
                            dropped interior BELIEF line (unrecoverable JSON) does NOT shift every
                            later state's feature onto the wrong token (additive; empty for
                            hand-built chains — the featurizer falls back to a marker scan then).
    """
    states: list
    answer: Optional[int]
    parse_ok: bool
    state_char_ends: list = field(default_factory=list)


# Marker for a per-step belief line. Case-insensitive; tolerant of whitespace inside the
# bracket and around the (ASCII or full-width) colon. Requires a digit index so the literal
# ``BELIEF[t]`` example line (and the word "belief" in prose) is NOT matched.
_BELIEF_MARKER_RE = re.compile(r"BELIEF\s*\[\s*(\d+)\s*\]\s*[:：]", re.IGNORECASE)

# Light cleanup: drop a trailing comma directly before a closing } or ] (a common LLM slip).
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _extract_json_object(text: str, start: int, limit: int):
    """Return the first balanced ``{...}`` substring within
    ``text[start:limit]``, or None if none/unterminated within the bound.

    The search for the opening ``{`` AND the balanced scan are both bounded by ``limit`` — the
    caller passes the end of the current BELIEF marker's line, so a marker whose own line has
    no (or unterminated) JSON yields NO state instead of borrowing a LATER line's JSON (which
    would silently duplicate / mis-order trellis states). The per-step contract is one line per
    event (``render_belief_line`` emits single-line JSON), so line-bounding is exact. A depth
    counter that respects double-quoted strings prevents miscounting braces inside a value.
    No backtracking, no recursion.
    """
    i = text.find("{", start)
    if i == -1 or i >= limit:
        return None
    depth = 0
    in_str = False
    escape = False
    for j in range(i, limit):
        c = text[j]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None  # unterminated within the line -> give up on this marker


def _loads_lenient(js: str):
    """``json.loads`` with a light trailing-comma fallback. Returns the object or None.

    Deliberately does NOT attempt single-quote -> double-quote coercion: gold locations such
    as ``"producer's desk"`` contain apostrophes, so that "fix" would corrupt valid data.
    """
    for candidate in (js, _TRAILING_COMMA_RE.sub(r"\1", js)):
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


def parse_chain(text) -> ParsedChain:
    """Parse a model reply into a ``ParsedChain``. NEVER raises on malformed input.

    Extracts every ``BELIEF[t]: {...}`` line (markdown / surrounding prose / trailing commas
    tolerated) as a state dict, and the final answer via ``sc.parse_answer`` (0-based index).
    The JSON is read ONLY from the marker's own line, so a markerless-JSON line yields no state
    rather than borrowing a later line's JSON. A BELIEF line whose JSON can't be recovered — or
    that decodes to a non-dict — is silently dropped rather than raising.
    ``parse_ok = (len(states) >= 1 and answer is not None)``.
    """
    states: list = []
    state_char_ends: list = []
    answer: Optional[int] = None
    if isinstance(text, str) and text:
        try:
            answer = sc.parse_answer(text)
            for m in _BELIEF_MARKER_RE.finditer(text):
                line_end = text.find("\n", m.end())
                if line_end == -1:
                    line_end = len(text)
                js = _extract_json_object(text, m.end(), line_end)
                if js is None:
                    continue
                obj = _loads_lenient(js)
                if isinstance(obj, dict):
                    states.append(obj)
                    # Anchor the step-end feature at THIS kept line's last char, so a later
                    # dropped line never re-indexes this state onto an earlier line's token.
                    state_char_ends.append(max(m.end(), line_end - 1))
        except Exception:  # pragma: no cover - defensive: parse must never raise
            pass
    parse_ok = (len(states) >= 1 and answer is not None)
    return ParsedChain(states=states, answer=answer, parse_ok=parse_ok,
                       state_char_ends=state_char_ends)


# ======================================================================================
# Canonicalization — the Phi-merge key
# ======================================================================================
def _norm_leaf(x):
    """Normalize a location leaf: strip + lowercase strings; pass non-strings through."""
    if isinstance(x, str):
        return x.strip().lower()
    return x


def _norm_objects(v):
    """Normalize the per-character ``{object: location}`` map (object keys lowercased+stripped,
    location values normalized). A non-dict value (malformed state) is normalized as a leaf so
    canonicalization is total and deterministic rather than raising."""
    if isinstance(v, dict):
        return {str(o).strip().lower(): _norm_leaf(loc) for o, loc in v.items()}
    return _norm_leaf(v)


def canon_state(state: dict) -> str:
    """Deterministic canonical string for a belief state — the trellis Phi-merge key.

    Normalization (documented, frozen): EVERY key and value is ``.strip().lower()``-folded —
    character keys, object keys, and location values alike. Because canon_state is the SHARED
    key for both trellis Phi-merge AND equality-to-gold (the facts oracle compares parsed states
    to gold ``beliefs`` whose names are canonically capitalized), folding character case stops a
    model that varies name casing from under-merging in the trellis or silently missing gold.
    MuSR character names are distinct words, so there are no case-only collisions to worry about.
    Then ``json.dumps(..., sort_keys=True, ensure_ascii=False)`` makes key order irrelevant.
    Two logically-identical states (any key ordering / surrounding whitespace / casing) map to
    the byte-identical string; genuinely different states do not.
    """
    norm: dict = {}
    if isinstance(state, dict):
        for char, objs in state.items():
            norm[str(char).strip().lower()] = _norm_objects(objs)
    return json.dumps(norm, sort_keys=True, ensure_ascii=False)


# ======================================================================================
# Rendering — inverse of the per-line parse (round-trip)
# ======================================================================================
def render_belief_line(t: int, state: dict) -> str:
    """Render one ``BELIEF[t]: {json}`` line so ``parse_chain`` round-trips the state.

    The state is dumped verbatim (NOT canonicalized) so a parse of the rendered line returns
    the identical dict; ``sort_keys=True`` only makes the output deterministic. ``ensure_ascii``
    is False so non-ASCII locations render readably (JSON still round-trips them).
    """
    return "BELIEF[%d]: %s" % (int(t), json.dumps(state, ensure_ascii=False, sort_keys=True))
