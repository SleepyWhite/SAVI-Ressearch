"""gsd_space — event-anchored reachable belief-state space enumerator (EXACT-GSD, R1 fix).

Builds, from the gold STRUCTURAL oracle of an object_placements item, the per-layer set of
reachable belief states and the witness-mask transitions between them: the graph substrate the
GSD scorer (gsd_score) and the exact DP decoder (gsd_decode) run on. The witness assignment —
who saw which move — is left COMPLETELY FREE: it is exactly the load-bearing difficulty of the
task, and the enumeration spans all of it.

==========================================================================================
LEAK BOUNDARY — READ BEFORE TOUCHING THIS MODULE
==========================================================================================
This module is ALLOWED to read exactly two things from the gold tree:

  * ``intermediate_data[0]["actual_locs"]``  — the true move sequence. This is the declared
    STRUCTURAL oracle (which object moved where, layer alignment); a3 proved event-level
    extraction is not the lesion, and a deployable variant could self-extract it.
  * ``gold_beliefs[0]``                      — the ROOT belief table only (the shared common
    ground before any witnessing happens).

``gold_beliefs[1:]`` — the per-character belief content after each event — IS the witness
outcome, i.e. THE TASK ANSWER. Any read of it from this module is leakage, full stop. The code
below touches ``beliefs`` in exactly one place (root construction, marked LEAK BOUNDARY) and
indexes only ``[0]``. ``events`` / ``narrative`` / ``choices`` / ``gold_idx`` are never read.
The leak-boundary unit test (test_no_witness_info_consumed) pins this: two items differing
only in ``gold_beliefs[1:]`` must enumerate byte-identical spaces.

==========================================================================================
FROZEN CONVENTIONS (verified against real data — all 256 items / 19 patients)
==========================================================================================
* ``T = len(actual_locs)`` (equal to ``len(beliefs)`` and ``len(events)`` on real data;
  T = 4 on all 19 certified patients). Layers are t = 0..T-1; layer t = the belief table
  AFTER event t. Event 0 is the story opening: ``actual_locs[0]`` is the INITIAL placement,
  so layer 0 is the root and moves are diffed from t = 1 on.
* ``moves = [(t, obj, new_loc), ...]``: per-cell diff of ``actual_locs[t-1] -> actual_locs[t]``
  for t = 1..T-1, t ascending, objects folded-sorted within a layer. Locations are compared
  case/whitespace-insensitively (a pure casing change is not a move); ``new_loc`` is stored
  verbatim. One layer may carry several moves (one event moving several objects) — a witness
  mask then governs ALL of them at once.
* Witness semantics: for each layer with moves, each subset of characters (the witness mask)
  yields one candidate successor — every character IN the mask updates its belief about EVERY
  object moved in this layer to the new location; every other cell is copied unchanged, so
  states are always COMPLETE char x object tables (mechanical readout is total).
  ``cand(s_prev, t) = {apply(s_prev, moves_at_t, subset) for subset in powerset(chars)}``.
  A no-move layer degenerates to the identity transition (cand = {s_prev}, mask = ()).
* Dedup: states are keyed by ``belief_schema.canon_state``. ``trans[(t, canon_prev)]`` holds
  ONE entry per distinct canon_next; when several masks collapse onto the same successor
  (e.g. a character whose belief already sits at the new location), the MINIMAL mask
  (smallest, then lexicographically first in chars order) is kept as the representative —
  masks are diagnostic; state identity is the graph's semantics.
* Determinism: ``chars`` = root character keys folded-sorted (original casing kept); layers
  and edge lists are sorted by canon string; two builds of the same item compare equal.
* ``path_count`` = number of distinct STATE paths from the root to the terminal layer
  (DP count over deduped edges). G-C scale assertion: ``path_count <= 2**(C*T)`` — violated
  means a construction bug, and unlike malformed input it raises (hard error by design).
* Sentinel semantics (mirrors facts_oracle): a malformed tree — actual_locs missing / empty /
  non-list / non-dict entries, beliefs missing / empty / beliefs[0] non-dict or empty,
  a character table non-dict or empty, characters disagreeing on the object universe, or a
  moved object outside the root universe (would break table completeness) — makes
  ``build_space`` return None. It NEVER raises on bad input.

CODE SEPARATION: imports ONLY stdlib (``itertools``/``dataclasses``/``typing``) +
``belief_schema`` (canon key) + ``facts_oracle`` (the ``_intermediate0`` tree-reading helper).
No torch / transformers / test code; pure, CPU-only, import-safe.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import belief_schema
from facts_oracle import _intermediate0

# Sentinel for "case-insensitive lookup found nothing" (same idiom as facts_oracle).
_MISSING = object()


def _norm(x) -> str:
    """Canonical comparison key: ``str(x).strip().lower()`` (facts_oracle convention)."""
    return str(x).strip().lower()


def _ci_get(d, key):
    """Case-insensitive dict lookup -> value, or ``_MISSING``."""
    nk = _norm(key)
    for k, v in d.items():
        if _norm(k) == nk:
            return v
    return _MISSING


def _ci_key(d, key):
    """The ACTUAL key of ``d`` matching ``key`` case-insensitively, or ``key`` itself."""
    nk = _norm(key)
    for k in d:
        if _norm(k) == nk:
            return k
    return key


@dataclass
class GsdSpace:
    """The enumerated reachable space of one item. See module docstring for conventions."""
    root: dict        # layer-0 belief table (complete: every char x object cell)
    root_flag: bool   # True iff gold_beliefs[0] characters disagree (root kept verbatim)
    moves: list       # [(t, obj, new_loc), ...] from the actual_locs per-cell diff, t >= 1
    layers: list      # layers[t] = [state dict, ...], canon-deduped + canon-sorted, t = 0..T-1
    trans: dict       # (t, canon_prev) -> [(canon_next, witness_mask), ...], t = 1..T-1
    chars: list       # character names, folded-sorted, original casing kept
    T: int            # number of layers = len(actual_locs)
    path_count: int   # distinct state paths root -> terminal; asserted <= 2**(C*T)


def _apply(state: dict, layer_moves: list, subset: tuple) -> dict:
    """One candidate successor: characters in ``subset`` see EVERY move of this layer (their
    belief about each moved object -> its new location); every other cell copies unchanged."""
    nxt = {c: dict(tbl) for c, tbl in state.items()}
    for char in subset:
        tbl = nxt[char]
        for _t, obj, new_loc in layer_moves:
            tbl[_ci_key(tbl, obj)] = new_loc
    return nxt


def build_space(item) -> Optional[GsdSpace]:
    """Enumerate the event-anchored reachable belief-state space of ``item``.

    Returns a ``GsdSpace``, or None on any malformed tree (sentinel semantics — never raises
    on bad input; the only raise is the G-C scale assertion, which signals a construction bug).
    """
    d0 = _intermediate0(item)
    if d0 is None:
        return None

    # ---- structural oracle: the move sequence -------------------------------------------
    actual_locs = d0.get("actual_locs")
    if not isinstance(actual_locs, list) or not actual_locs:
        return None
    if not all(isinstance(x, dict) for x in actual_locs):
        return None

    # ---- LEAK BOUNDARY: the ONLY read of gold beliefs — index [0], the root table only.
    # gold_beliefs[1:] is the witness outcome (the task answer) and must never be touched.
    beliefs = d0.get("beliefs")
    if not isinstance(beliefs, list) or not beliefs:
        return None
    root_raw = beliefs[0]
    # -------------------------------------------------------------------------------------

    if not isinstance(root_raw, dict) or not root_raw:
        return None
    for tbl in root_raw.values():
        if not isinstance(tbl, dict) or not tbl:
            return None
    # complete-table invariant: every character tracks the SAME object universe
    universes = {frozenset(_norm(o) for o in tbl) for tbl in root_raw.values()}
    if len(universes) != 1:
        return None
    universe = next(iter(universes))

    # root common ground: folded per-character tables must coincide, else flag (root verbatim)
    folded_tables = {
        tuple(sorted((_norm(o), _norm(loc)) for o, loc in tbl.items()))
        for tbl in root_raw.values()
    }
    root_flag = len(folded_tables) > 1
    root = {c: dict(tbl) for c, tbl in root_raw.items()}
    chars = sorted(root, key=_norm)
    T = len(actual_locs)

    # ---- moves: per-cell diff of actual_locs (layer 0 = initial placement, no diff) ------
    moves: list = []
    for t in range(1, T):
        prev, cur = actual_locs[t - 1], actual_locs[t]
        for obj in sorted(cur, key=_norm):
            if _norm(obj) not in universe:
                return None  # moved object no root table tracks -> completeness would break
            old = _ci_get(prev, obj)
            if old is _MISSING or _norm(old) != _norm(cur[obj]):
                moves.append((t, obj, cur[obj]))

    moves_by_t: dict = {}
    for mv in moves:
        moves_by_t.setdefault(mv[0], []).append(mv)

    # ---- layer-by-layer reachable enumeration (canon-deduped, deterministic order) -------
    # Subsets in (size, lex-in-chars-order): the FIRST subset realizing a successor is its
    # minimal representative mask; a no-move layer needs no special case (every subset maps
    # to the identity, deduped to the single edge with mask ()).
    subsets = [sub for r in range(len(chars) + 1)
               for sub in itertools.combinations(chars, r)]

    layers = [[root]]
    trans: dict = {}
    prev_by_canon = {belief_schema.canon_state(root): root}
    for t in range(1, T):
        layer_moves = moves_by_t.get(t, [])
        next_by_canon: dict = {}
        for canon_prev in sorted(prev_by_canon):
            s_prev = prev_by_canon[canon_prev]
            edges: dict = {}  # canon_next -> representative (minimal) witness mask
            for subset in subsets:
                s_next = _apply(s_prev, layer_moves, subset)
                canon_next = belief_schema.canon_state(s_next)
                if canon_next not in edges:
                    edges[canon_next] = subset
                    if canon_next not in next_by_canon:
                        next_by_canon[canon_next] = s_next
            trans[(t, canon_prev)] = sorted(edges.items())
        prev_by_canon = next_by_canon
        layers.append([next_by_canon[c] for c in sorted(next_by_canon)])

    # ---- path count + G-C scale assertion -------------------------------------------------
    counts = {belief_schema.canon_state(root): 1}
    for t in range(1, T):
        nxt_counts: dict = {}
        for canon_prev, n in counts.items():
            for canon_next, _mask in trans[(t, canon_prev)]:
                nxt_counts[canon_next] = nxt_counts.get(canon_next, 0) + n
        counts = nxt_counts
    path_count = sum(counts.values())
    bound = 2 ** (len(chars) * T)
    if path_count > bound:  # mathematically unreachable; a construction bug if it fires
        raise AssertionError(
            "G-C scale violation on item %r: path_count %d > 2**(C*T) = %d"
            % (item.get("id") if isinstance(item, dict) else None, path_count, bound))

    return GsdSpace(root=root, root_flag=root_flag, moves=moves, layers=layers,
                    trans=trans, chars=chars, T=T, path_count=path_count)
