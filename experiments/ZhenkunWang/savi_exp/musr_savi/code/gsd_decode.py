"""gsd_decode — exact two-arm Viterbi decode + mechanical readout + Δ-ledger (EXACT-GSD, R4/R5).

Runs the EXACT global argmax over a ``gsd_space.GsdSpace`` reachable graph under the edge
weights produced by ``gsd_score.score_transitions`` (the caller picks ONE normalization and
passes ``result["raw"]`` or ``result["lse"]``), reads the answer MECHANICALLY off the winning
terminal belief state (R5 — no ANSWER line, no voting), and decomposes the MAP-vs-best-gold
score gap layer by layer (the Δ-ledger — the data source of the demand curve, design §5
readout 1).

==========================================================================================
LEAK BOUNDARY — READ BEFORE TOUCHING THIS MODULE
==========================================================================================
  * ``"map"`` arm (λ=0) — NEVER touches ``facts_oracle.solvable`` / the gold trajectory /
    locset: its path and score depend ONLY on ``(space, A)``. ``item`` is consumed strictly
    AFTER the argmax — question/choices for the mechanical readout (task inputs) and gold_idx
    for the ``fixed`` field, which is the downstream JUDGE (``is_goal_answer``), never a
    decode input. The leak unit test monkeypatch-raises ``facts_oracle.solvable`` and the map
    arm must run unchanged (structural leak-proofing: ``solvable`` is only ever reached
    through the ``mode == "oracle"`` branch, called as a module attribute so the patch bites).
  * ``"oracle"`` arm — the SAME graph plus ``facts_oracle.solvable(state, item)`` as a HARD
    NODE MASK (the Phase B G1 construct, kept comparable): a masked STATE NODE has no
    in-edges, no out-edges, and cannot be a terminal; the root is a node like any other.
    Requires ``item``.
  * ``delta_ledger`` — analysis on the UNCONSTRAINED (map) graph; it consults gold_idx only
    to pick the gold-readout terminal set (never ``solvable``).

==========================================================================================
FROZEN CONVENTIONS
==========================================================================================
* Path score = ``A[(0, None, canon_root)]`` (the root path constant) +
  ``sum_{t=1..T-1} A[(t, canon_prev, canon_next)]``. Layer indices follow ``GsdSpace``
  (``layers[0]`` = the unique root layer). A missing edge key in ``A`` is a wiring bug and
  raises ``KeyError`` (``score_transitions`` scores every edge of the space).
* Terminal = the surviving nodes of the LAST layer (t = T-1); answer =
  ``readout(terminal_state, item)``.
* Ties: equal scores resolve to the LEXICOGRAPHICALLY SMALLEST canon — at the terminal
  argmax AND at every backpointer (realized by sorted iteration + strictly-greater updates),
  so the decode is fully deterministic, path included.
* All-dead (the oracle mask empties the root or any layer cuts every path) -> the sentinel
  ``{"answer_idx": None, "path_canon": [], "best_score": None, "fixed": False}``.
* ``readout``: ``facts_oracle.question_target(item) -> (observer, object)``; the state cell
  ``state[observer][object]`` is looked up case-insensitively and matched against
  ``item["choices"]`` with ``str(x).strip().lower()`` folding on BOTH sides (first match
  wins — deterministic even on duplicate choices). Any failure — target unparsed, missing
  cell, no matching choice — returns None (downstream: counts as not fixed). The gold side
  is verified well-defined 256/256, but an arbitrary reachable state may legitimately have
  no match.
* Δ-ledger: ``map_path`` = the ``gsd_decode(..., "map")`` path; ``best_gold_path`` = the
  terminal-constrained DP — the max-forward-score path among terminals whose readout equals
  gold_idx (interior nodes unconstrained). ``per_layer`` = ``[(t, lp_map, lp_gold, gap)]``
  with t = 0 the root row (both paths share the unique root, so gap is exactly 0.0);
  ``delta_total = map_score - gold_score`` and MUST equal ``sum(gap)`` — checked at abs tol
  1e-6, a violation is a construction bug and raises ``AssertionError``. ``fork_layer`` =
  the SMALLEST t attaining the max gap (coinciding paths give all-zero gaps and a degenerate
  fork_layer of 0). No gold-readout terminal -> ``{"delta_total": None, "per_layer": [],
  "fork_layer": None, "reachable_gold": False}`` (the zero-coverage subclass semantics).
  ADDITIVE extras for the g1 figures/tables (documented; the four spec keys are unchanged):
  ``map_path_canon`` / ``gold_path_canon`` / ``map_score`` / ``gold_score``.

CODE SEPARATION: imports ONLY stdlib (``math``/``typing``) + ``belief_schema`` (canon key)
+ ``facts_oracle`` (question_target / solvable / is_goal_answer). No torch / transformers /
test code; pure, CPU-only, import-safe, deterministic.
"""
from __future__ import annotations

import math
from typing import Optional

import belief_schema
import facts_oracle

# Sentinel for "case-insensitive lookup found nothing" (facts_oracle idiom).
_MISSING = object()


def _norm(x) -> str:
    """Canonical comparison key: ``str(x).strip().lower()`` (facts_oracle convention)."""
    return str(x).strip().lower()


def _ci_get(d, key):
    """Case-insensitive dict lookup -> value, or ``_MISSING`` (total on non-dicts)."""
    if not isinstance(d, dict):
        return _MISSING
    nk = _norm(key)
    for k, v in d.items():
        if _norm(k) == nk:
            return v
    return _MISSING


def _sentinel() -> dict:
    """The all-dead decode result (fresh dict per call)."""
    return {"answer_idx": None, "path_canon": [], "best_score": None, "fixed": False}


# ======================================================================================
# mechanical readout (R5)
# ======================================================================================
def readout(state, item) -> Optional[int]:
    """Mechanically read the answer of ``state``: 0-based choice index, or None.

    ``question_target(item) -> (observer, object)``; the cell ``state[observer][object]``
    (case-insensitive lookup) is folded with ``str.strip().lower()`` and matched against the
    equally-folded ``item["choices"]`` (first match wins). Target parse failure / missing
    cell / no matching choice -> None. Never raises.
    """
    tgt = facts_oracle.question_target(item)
    if tgt is None:
        return None
    observer, obj = tgt
    loc = _ci_get(_ci_get(state, observer), obj)
    if loc is _MISSING:
        return None
    choices = item.get("choices")
    if not isinstance(choices, (list, tuple)):
        return None
    nl = _norm(loc)
    for i, choice in enumerate(choices):
        if _norm(choice) == nl:
            return i
    return None


# ======================================================================================
# exact Viterbi forward / backtrack (shared by both arms and the Δ-ledger)
# ======================================================================================
def _states_by_canon(space) -> list:
    """Per-layer ``{canon: state dict}`` (the canon<->state bridge, from space.layers)."""
    return [{belief_schema.canon_state(s): s for s in layer} for layer in space.layers]


def _forward(space, A, alive: Optional[list]) -> list:
    """Exact Viterbi forward pass over the reachable graph.

    ``alive`` = None (no mask) or a per-layer list of surviving-canon sets (the oracle
    node mask: an edge is traversed only if BOTH endpoints survive; the root must too).
    Returns ``reach``: ``reach[t][canon] = (best_score, backpointer_canon_or_None)`` over
    surviving, reachable nodes. Tie policy: iterating predecessors in sorted canon order
    with strictly-greater updates keeps the smallest backpointer on ties.
    """
    root_canon = belief_schema.canon_state(space.root)
    layer0 = {}
    if alive is None or root_canon in alive[0]:
        layer0[root_canon] = (A[(0, None, root_canon)], None)
    reach = [layer0]
    for t in range(1, space.T):
        cur: dict = {}
        for canon_prev in sorted(reach[t - 1]):
            base = reach[t - 1][canon_prev][0]
            for canon_next, _mask in space.trans[(t, canon_prev)]:
                if alive is not None and canon_next not in alive[t]:
                    continue
                score = base + A[(t, canon_prev, canon_next)]
                if canon_next not in cur or score > cur[canon_next][0]:
                    cur[canon_next] = (score, canon_prev)
        reach.append(cur)
    return reach


def _best_terminal(reach_last: dict, restrict=None) -> Optional[tuple]:
    """Terminal argmax -> ``(score, canon)`` or None if no (restricted) survivor.

    Sorted iteration + strictly-greater updates = lexicographically smallest canon on ties.
    """
    best = None
    for canon in sorted(reach_last):
        if restrict is not None and canon not in restrict:
            continue
        score = reach_last[canon][0]
        if best is None or score > best[0]:
            best = (score, canon)
    return best


def _backtrack(reach: list, terminal_canon: str) -> list:
    """Follow backpointers from ``terminal_canon`` -> the full root-first canon path."""
    path = [terminal_canon]
    for t in range(len(reach) - 1, 0, -1):
        path.append(reach[t][path[-1]][1])
    path.reverse()
    return path


# ======================================================================================
# the two-arm decoder (R4)
# ======================================================================================
def gsd_decode(space, A, mode: str, item=None) -> dict:
    """Exact global argmax over ``space`` under edge weights ``A`` (single normalization).

    ``mode == "map"``: pure model dynamics — never touches solvable/gold (see LEAK
    BOUNDARY); ``item`` is optional and feeds only the post-hoc readout + ``fixed`` judge
    (without it: ``answer_idx`` None, ``fixed`` False).
    ``mode == "oracle"``: ``facts_oracle.solvable(state, item)`` as a hard node mask
    (requires ``item``).

    Returns ``{"answer_idx", "path_canon", "best_score", "fixed"}`` — or the all-dead
    sentinel (None / [] / None / False) when the mask leaves no surviving path.
    """
    if mode not in ("map", "oracle"):
        raise ValueError("unknown mode %r (want 'map' or 'oracle')" % (mode,))
    by_canon = _states_by_canon(space)
    alive = None
    if mode == "oracle":
        if item is None:
            raise ValueError("the oracle arm requires the item (solvable node mask)")
        # NOTE: module-attribute call — the leak test monkeypatches facts_oracle.solvable.
        alive = [{c for c, s in layer.items() if facts_oracle.solvable(s, item)}
                 for layer in by_canon]
    reach = _forward(space, A, alive)
    best = _best_terminal(reach[-1])
    if best is None:
        return _sentinel()
    best_score, terminal = best
    answer_idx = readout(by_canon[-1][terminal], item)
    return {
        "answer_idx": answer_idx,
        "path_canon": _backtrack(reach, terminal),
        "best_score": best_score,
        "fixed": facts_oracle.is_goal_answer(answer_idx, item),
    }


# ======================================================================================
# Δ-ledger (design §5 readout 1 — the demand-curve data source)
# ======================================================================================
def delta_ledger(space, A, item) -> dict:
    """Layerwise decomposition of score(MAP path) − score(best gold-readout path).

    ``map_path`` comes from ``gsd_decode(..., "map")``; ``best_gold_path`` from the
    terminal-constrained DP (max forward score among terminals reading out gold). See the
    module docstring for the frozen per_layer / fork_layer / sentinel semantics.
    """
    by_canon = _states_by_canon(space)
    map_res = gsd_decode(space, A, "map", item=item)
    map_path, map_score = map_res["path_canon"], map_res["best_score"]

    reach = _forward(space, A, None)
    gold_terminals = {c for c, s in by_canon[-1].items()
                      if facts_oracle.is_goal_answer(readout(s, item), item)}
    best_gold = _best_terminal(reach[-1], restrict=gold_terminals)
    if best_gold is None:
        return {"delta_total": None, "per_layer": [], "fork_layer": None,
                "reachable_gold": False, "map_path_canon": map_path,
                "gold_path_canon": [], "map_score": map_score, "gold_score": None}
    gold_score, gold_terminal = best_gold
    gold_path = _backtrack(reach, gold_terminal)

    # per-layer transition-score ledger; t = 0 is the root constant (shared unique root,
    # so map_path[0] == gold_path[0] and the gap is exactly 0.0 by construction).
    per_layer = []
    lp_map0, lp_gold0 = A[(0, None, map_path[0])], A[(0, None, gold_path[0])]
    per_layer.append((0, lp_map0, lp_gold0, lp_map0 - lp_gold0))
    for t in range(1, space.T):
        lp_map = A[(t, map_path[t - 1], map_path[t])]
        lp_gold = A[(t, gold_path[t - 1], gold_path[t])]
        per_layer.append((t, lp_map, lp_gold, lp_map - lp_gold))

    delta_total = map_score - gold_score
    gap_sum = sum(row[3] for row in per_layer)
    if not math.isclose(delta_total, gap_sum, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(
            "Δ-ledger decomposition broke on item %r: delta_total %r != sum(gaps) %r"
            % (item.get("id") if isinstance(item, dict) else None, delta_total, gap_sum))

    gaps = [row[3] for row in per_layer]
    fork_layer = gaps.index(max(gaps))  # smallest t attaining the max gap
    return {"delta_total": delta_total, "per_layer": per_layer, "fork_layer": fork_layer,
            "reachable_gold": True, "map_path_canon": map_path,
            "gold_path_canon": gold_path, "map_score": map_score,
            "gold_score": gold_score}
