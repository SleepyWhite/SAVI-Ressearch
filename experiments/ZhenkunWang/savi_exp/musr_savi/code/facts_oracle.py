"""facts_oracle — MuSR-cant Phase B structured GOLD-LABEL oracle (Subtask 2).

Turns the fully-structured gold reasoning data on each object_placements item into the three
labels Phase B needs, WITHOUT any natural-language parsing of the narrative:

  * step-level ground-truth belief states  -> training targets for the step verifier
    (``gold_beliefs`` / ``gold_final_state``);
  * the exact trellis ORACLE MASK           -> the solvability ceiling for gate G1
    (``solvable``);
  * the answer-correctness judge            -> ``is_goal_answer``.

All labels come from ``item["tree_raw"]["intermediate_data"][0]``, a dict:
    {"beliefs": [ {character: {object: location}}, ... ],   # gold belief trajectory (per step)
     "events":  [ [nl-event, ...], ... ],                   # NL event groups (unused here)
     "actual_locs": [ {object: location}, ... ]}            # true locations (unused here)
The parsed model states (``belief_schema.parse_chain``) live in the SAME
``{character: {object: location}}`` space, so a state and a gold state are compared directly.

CODE SEPARATION: imports ONLY stdlib (``re`` / ``math``). No torch/transformers, no test code;
pure, CPU-only, import-safe. Every function is TOTAL — on a missing/empty/malformed tree it
returns a documented sentinel (None / NaN) or a safe default, and NEVER raises.

CASE-INSENSITIVITY: every comparison of a character name, object, or location is done on
``str(x).strip().lower()`` (see ``_norm``). This module does its own per-pair normalization
rather than relying on ``belief_schema.canon_state`` so the label logic is independent of the
canonical-key policy and is case-insensitive on ALL of char / object / location.

QUESTION TEMPLATE (object_placements, frozen family, verified 256/256 on real data):
    "Which location is the most likely place <Observer> would look to find the <object>
     given the story?"
``question_target`` recovers ``(observer, object)`` from it; the correctness of that recovery
is proven by the cross-check in the test suite (``gold_final_state[observer][object]`` equals
``choices[gold_idx]`` on 256/256 items).
"""
from __future__ import annotations

import math
import re
from typing import Optional

# object_placements question template. Non-greedy observer/object groups anchored by the fixed
# "would look to find the ... given the story" spine, so multi-word observers ("Mr. Brown",
# "Captain Jake") and multi-word objects ("cup of coffee", "confidential financial report
# binder") are captured whole. Case-insensitive for robustness.
_QUESTION_RE = re.compile(
    r"place\s+(?P<observer>.+?)\s+would look to find the\s+(?P<object>.+?)\s+given the story",
    re.IGNORECASE,
)

# Sentinel for "case-insensitive lookup found nothing" — distinct from a stored None location.
_MISSING = object()

# Documented NaN sentinel returned by state_consistent when the gold reference is unavailable.
NAN = float("nan")


# ======================================================================================
# normalization + safe navigation helpers
# ======================================================================================
def _norm(x) -> str:
    """Canonical comparison key for a character / object / location: ``str(x).strip().lower()``."""
    return str(x).strip().lower()


def _ci_get(d, key):
    """Case-insensitive dict lookup. Returns the value, or ``_MISSING`` if ``d`` is not a dict
    or has no key matching (case-insensitively)."""
    if not isinstance(d, dict):
        return _MISSING
    nk = _norm(key)
    for k, v in d.items():
        if _norm(k) == nk:
            return v
    return _MISSING


def _state_loc(state, char, obj):
    """Case-insensitive ``state[char][obj]`` -> location value, or ``_MISSING`` if absent /
    malformed at any level. Never raises."""
    objs = _ci_get(state, char)
    if objs is _MISSING:
        return _MISSING
    return _ci_get(objs, obj)


def _intermediate0(item) -> Optional[dict]:
    """Return ``item["tree_raw"]["intermediate_data"][0]`` as a dict, or None if the tree is
    absent / intermediate_data is empty / the entry is not a dict. Never raises."""
    if not isinstance(item, dict):
        return None
    tree = item.get("tree_raw")
    if not isinstance(tree, dict):
        return None
    idata = tree.get("intermediate_data")
    if not idata or not isinstance(idata, (list, tuple)):
        return None
    d0 = idata[0]
    return d0 if isinstance(d0, dict) else None


# ======================================================================================
# gold belief states
# ======================================================================================
def gold_beliefs(item) -> Optional[list]:
    """The gold belief trajectory: ``intermediate_data[0]["beliefs"]`` (list of per-step
    ``{character: {object: location}}`` dicts), or None if the tree/beliefs are unavailable
    or empty. Never raises."""
    d0 = _intermediate0(item)
    if d0 is None:
        return None
    beliefs = d0.get("beliefs")
    if not beliefs or not isinstance(beliefs, list):
        return None
    return beliefs


def gold_final_state(item) -> Optional[dict]:
    """The last gold belief state ``gold_beliefs(item)[-1]``, or None if unavailable."""
    beliefs = gold_beliefs(item)
    if not beliefs:
        return None
    last = beliefs[-1]
    return last if isinstance(last, dict) else None


# ======================================================================================
# question target
# ======================================================================================
def question_target(item) -> Optional[tuple]:
    """Parse ``(observer, object)`` from the object_placements question, or None on no-match.

    Both strings are returned stripped, with their ORIGINAL casing (comparisons elsewhere are
    case-insensitive). Multi-word observers/objects are captured whole. Never raises."""
    if not isinstance(item, dict):
        return None
    q = item.get("question")
    if not isinstance(q, str):
        return None
    m = _QUESTION_RE.search(q)
    if not m:
        return None
    observer = m.group("observer").strip()
    obj = m.group("object").strip()
    if not observer or not obj:
        return None
    return (observer, obj)


# ======================================================================================
# state consistency vs gold
# ======================================================================================
def _consistency_fraction(state, gold) -> Optional[float]:
    """Fraction of gold ``(char, object) -> location`` assignments that ``state`` reproduces
    (case-insensitively). Denominator = number of (char, object) pairs in ``gold``.

    Returns None if ``gold`` is not a dict. If ``gold`` has zero (char, object) pairs the score
    is vacuously 1.0 (no assignment to violate). Malformed gold sub-entries (a character mapped
    to a non-dict) contribute no pairs. ``state`` may be anything: a missing char/object or a
    mismatched location simply counts as inconsistent."""
    if not isinstance(gold, dict):
        return None
    total = 0
    match = 0
    for char, objs in gold.items():
        if not isinstance(objs, dict):
            continue
        for obj, loc in objs.items():
            total += 1
            sv = _state_loc(state, char, obj)
            if sv is not _MISSING and _norm(sv) == _norm(loc):
                match += 1
    if total == 0:
        return 1.0
    return match / total


def state_consistent(state: dict, item: dict, ref="final") -> float:
    """Fraction in [0,1] of gold (char, object)->location assignments ``state`` matches.

    ``ref`` selects the gold reference state:
      * ``"final"``  (default) -> ``gold_final_state(item)``;
      * ``int t``              -> ``gold_beliefs(item)[t]`` (supports Python indexing; an
                                  out-of-range index yields the NaN sentinel);
      * ``"nearest"``          -> the best-matching gold step (max fraction over all steps).

    Comparison is case-insensitive on character / object / location. Missing or wrong pairs in
    ``state`` count as inconsistent; the denominator is the pair count of the gold reference.

    If the gold reference is unavailable (no tree / empty beliefs / out-of-range step) returns
    the documented sentinel ``float('nan')`` — callers should filter with ``math.isnan``. Never
    raises (a malformed ``state`` is simply scored as inconsistent)."""
    if isinstance(ref, str) and ref == "nearest":
        beliefs = gold_beliefs(item)
        if not beliefs:
            return NAN
        best = None
        for g in beliefs:
            frac = _consistency_fraction(state, g)
            if frac is None:
                continue
            if best is None or frac > best:
                best = frac
        return best if best is not None else NAN

    if isinstance(ref, str) and ref == "final":
        gold = gold_final_state(item)
    elif isinstance(ref, int) and not isinstance(ref, bool):
        beliefs = gold_beliefs(item)
        if not beliefs:
            return NAN
        try:
            gold = beliefs[ref]
        except (IndexError, TypeError):
            return NAN
    else:
        # unknown ref spec -> treat gold as unavailable
        return NAN

    frac = _consistency_fraction(state, gold)
    return frac if frac is not None else NAN


# ======================================================================================
# solvability — the trellis oracle mask
# ======================================================================================
def _gold_target_locations(item):
    """Return ``((observer, object), locset)`` where ``locset`` is the set of case-folded
    locations the question target takes across ALL gold belief steps
    (``{_norm(gold_beliefs[t][observer][object]) for every step t}``), or ``(None, None)`` if the
    question target or the gold trajectory is unavailable. Never raises.

    Because the gold final belief about the target equals ``choices[gold_idx]`` (verified 256/256
    on real data), the answer location is always a member of ``locset``; the intermediate,
    non-answer locations the target legitimately transits are members too."""
    tgt = question_target(item)
    if tgt is None:
        return None, None
    beliefs = gold_beliefs(item)
    if not beliefs:
        return None, None
    observer, obj = tgt
    locset = set()
    for g in beliefs:
        sv = _state_loc(g, observer, obj)
        if sv is not _MISSING:
            locset.add(_norm(sv))
    return tgt, locset


def _gold_pair_locations(item) -> dict:
    """Map ``(_norm(char), _norm(obj)) -> set`` of case-folded locations that (char, object) pair
    takes across ALL gold belief steps. Empty dict if the gold trajectory is unavailable. This is
    the per-pair generalization of ``_gold_target_locations`` used by the ``solvable`` fallback.
    Never raises."""
    beliefs = gold_beliefs(item)
    out: dict = {}
    if not beliefs:
        return out
    for g in beliefs:
        if not isinstance(g, dict):
            continue
        for char, objs in g.items():
            if not isinstance(objs, dict):
                continue
            for obj, loc in objs.items():
                out.setdefault((_norm(char), _norm(obj)), set()).add(_norm(loc))
    return out


def solvable(state: dict, item: dict) -> bool:
    """The trellis ORACLE MASK — a PERMISSIVE reachability ceiling (an UPPER bound).

    "Reachable" means ``state`` sits on SOME state consistent with the gold trajectory, not only
    with the terminal answer. A belief-tracking target legitimately passes through non-answer
    locations at intermediate steps before its final update, so the mask must keep those
    intermediate gold states — pruning them severs the gold path at an interior trellis layer and
    artificially depresses the G1 oracle ceiling (a spurious kill). We therefore err PERMISSIVE:
    over-permissiveness only raises the ceiling (counts more fixes), which is the SAFE direction
    for a kill-gate; over-strictness (the old ``== choices[gold_idx]`` rule) lowered it and risked
    spurious Phase-B termination. Correctness is still enforced downstream — the answer is read
    from the TERMINAL belief and G1 only counts a fix when ``is_goal_answer(decoded_answer)`` — so
    a permissive mask guides search without letting wrong-answer terminals score.

    Primary rule (question target available): let ``L`` = the set of locations the target
    ``(observer, object)`` takes across ALL gold belief steps (``_gold_target_locations``). True
    iff the target's belief in ``state`` is ABSENT (undecided → still reachable) or ∈ ``L``
    (case-insensitive). Only a target committed to a location that NEVER appears on the gold path
    is pruned.

    FALLBACK (question target unavailable): apply the same "on some gold step" test per pair. For
    each (char, object) pair assigned in ``state`` that also appears somewhere in the gold
    trajectory, require its location ∈ that pair's gold-location set (``_gold_pair_locations``);
    True iff no assigned pair violates. Pairs ``state`` does not assign, or that never appear in
    gold, impose no constraint. If the gold trajectory is unavailable there is nothing to
    contradict, so the default is True.

    Always returns a bool; never raises."""
    tgt, locset = _gold_target_locations(item)
    if tgt is not None and locset is not None:
        observer, obj = tgt
        sv = _state_loc(state, observer, obj)
        if sv is _MISSING:
            return True  # target undecided -> still reachable
        return _norm(sv) in locset

    # ---- fallback: per-pair "on some gold step" test ----
    pair_locs = _gold_pair_locations(item)
    if not pair_locs or not isinstance(state, dict):
        return True  # gold unavailable / non-dict state -> nothing to contradict
    for char, objs in state.items():
        if not isinstance(objs, dict):
            continue
        for obj, loc in objs.items():
            key = (_norm(char), _norm(obj))
            if key in pair_locs and _norm(loc) not in pair_locs[key]:
                return False
    return True


# ======================================================================================
# answer judge
# ======================================================================================
def is_goal_answer(answer_idx: Optional[int], item: dict) -> bool:
    """True iff ``answer_idx`` is a non-None, non-bool int equal to ``item["gold_idx"]``.

    ``bool`` is excluded explicitly (``True == 1`` / ``False == 0`` in Python, so an accidental
    ``True`` answer must not score as option 1). Never raises."""
    if answer_idx is None or isinstance(answer_idx, bool):
        return False
    if not isinstance(item, dict):
        return False
    return answer_idx == item.get("gold_idx")
