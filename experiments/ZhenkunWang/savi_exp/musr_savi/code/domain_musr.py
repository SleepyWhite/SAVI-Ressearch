"""domain_musr — MuSR-cant Phase B chains-only belief-state trellis + SC / BoN arms (Subtask 4).

The DECODER for Phase B: it decodes a belief-state trellis built over an ALREADY-SAMPLED fixed
pool of parsed belief-chains, and provides the token-matched self-consistency (SC) and
best-of-N (BoN) baseline aggregators. All three arms — SAVI trellis / SC / BoN — consume the
SAME fixed pool of ``belief_schema.ParsedChain`` objects; nothing is regenerated. That fixed
pool is what makes the comparison ISO-TOKEN (token-matched, not iso-candidate): the trellis
spends no extra generation over the baselines.

WHY STANDALONE (we deliberately do NOT import decode_core)
---------------------------------------------------------
``decode_core.savi`` is GENERATIVE — it samples fresh N step-candidates PER NODE, which would
spend more tokens than the fixed-pool SC/BoN baselines and break token-matching; and
``decode_core.oracle`` is ``solvable(initial_state)``, trivially true for MuSR (not our ceiling).
So this module is a standalone chains-only trellis that REUSES decode_core's Phi-merge SEMANTICS
without any regeneration:

  * canonical-state-keyed nodes (``belief_schema.canon_state`` is the merge key), one layer per
    step index; an edge (t -> t+1) connects consecutive canonical states within a chain;
  * freq edge weight = ``log(count / N)`` where ``count`` is the edge multiplicity and ``N`` is
    the number of chains passing through the SOURCE node (its per-node candidate count — exactly
    decode_core's ``N`` in ``sample(node.state, N, ...)``, here the pool's own support);
  * Phi-merge: nodes sharing a canonical key at the SAME layer merge into one node, keeping the
    higher CUMULATIVE-score back-pointer (Viterbi max-product forward pass);
  * deterministic, stable order everywhere (sorted canonical keys / lowest index).

VIRTUAL START / END (the frequency-aware endpoints)
--------------------------------------------------
Unlike decode_core (one fixed initial state), a pool has many first states and many terminals of
varying length. To make the entry into layer 0 and the exit from each terminal frequency-aware
(so the modal start/answer wins), the forward pass uses a virtual START before layer 0 and a
virtual END after every terminal, both weighted with the SAME ``log(count/N)`` rule:
  * START -> layer-0 node ``v``:  ``log(start_count[v] / n_chains)``;
  * terminal ``u`` -> END:        ``log(term_count[u] / N_u)`` where ``N_u`` = chains through u.
These virtual nodes carry no canonical state and never appear in ``path_canon``. With this the
best complete path is the highest-probability belief trajectory under the pool's empirical
first-order Markov model, with proper start and termination mass — its terminal's stored answer
is read off as the decode.

CODE SEPARATION: imports ONLY stdlib (``math`` / ``collections`` / ``dataclasses``) + the three
pure CPU core modules ``belief_schema`` / ``facts_oracle`` / ``sc_core``. No torch/transformers,
no test code; pure, CPU-only, import-safe. Every function is TOTAL — an empty pool or an
all-``parse_ok=False`` pool yields a documented sentinel (answer None / best_score None), never
a crash.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import belief_schema as bs
import facts_oracle as fo
import sc_core as sc

# Clamp floor for soft verifier step-scores: keeps ``log(step_score)`` finite when a caller
# passes a score at (or numerically near) 0 despite the (0,1] contract. Deterministic.
_SCORE_FLOOR = 1e-12


# ======================================================================================
# Trellis object
# ======================================================================================
@dataclass
class Trellis:
    """The static chains-only belief-state trellis over a fixed pool of parse_ok chains.

    Nodes are keyed by ``(layer, canon)`` where ``layer`` is the 0-based step index and ``canon``
    is ``belief_schema.canon_state(states[layer])``. All structure below is built ONCE by
    ``build_trellis`` and consumed (read-only) by ``savi_decode`` for every mode — so the three
    decode modes share one trellis (iso-token).

    Fields
    ------
    node_state   : ``(layer, canon) -> representative state dict`` (any chain's state at that
                   node; all are canon-equal, used for the oracle mask ``solvable(state,item)``).
    out_counts   : ``(layer, canon) -> {succ_canon: edge_multiplicity}`` (chains continuing to
                   each successor at ``layer+1``).
    node_total   : ``(layer, canon) -> N_u`` = number of chain-steps passing THROUGH the node
                   (continuing + terminating) = the per-node freq denominator.
    term_answers : ``(layer, canon) -> {answer_idx: count}`` for chains TERMINATING at the node
                   (their final state); empty for non-terminal nodes.
    start_counts : ``{layer0_canon: count}`` = the START -> layer-0 edge multiplicities.
    n_chains     : number of parse_ok chains contributing to the trellis (= START denominator).
    chain_canon_seqs : per contributing chain, its full ``[canon(states[0]), ...]`` sequence —
                   used to decide ``marginal_not_in_single_chain``.
    layers       : sorted list of distinct layer indices present.
    n_steps      : total chain-steps (sum of ``len(states)`` over contributing chains).
    merge_hits   : chain-steps that landed on an ALREADY-EXISTING ``(layer, canon)`` node.
    merge_rate   : ``merge_hits / n_steps`` (0.0 for an empty pool) — feeds gate G2.
    """
    node_state: dict = field(default_factory=dict)
    out_counts: dict = field(default_factory=dict)
    node_total: dict = field(default_factory=dict)
    term_answers: dict = field(default_factory=dict)
    start_counts: dict = field(default_factory=dict)
    n_chains: int = 0
    chain_canon_seqs: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    n_steps: int = 0
    merge_hits: int = 0
    merge_rate: float = 0.0


def build_trellis(parsed_chains) -> Trellis:
    """Build the layered canonical-state trellis over the fixed pool (Phi-merge, freq counts).

    Only ``parse_ok`` chains contribute (a chain with no states or no answer is not a usable
    trellis chain — see ``belief_schema.ParsedChain``). For every contributing chain and every
    step index ``t`` a node ``(t, canon(states[t]))`` is created/merged, edges to the next step
    are tallied, and the final state records the chain's answer as a terminal. ``merge_rate`` is
    the fraction of chain-steps that landed on an already-existing node (gate G2). Never raises.
    """
    tr = Trellis()
    node_total: dict = {}
    out_counts: dict = defaultdict(lambda: defaultdict(int))
    term_answers: dict = defaultdict(lambda: defaultdict(int))
    node_state: dict = {}
    start_counts: dict = defaultdict(int)
    canon_seqs: list = []
    n_steps = 0
    merge_hits = 0

    for c in parsed_chains:
        if not getattr(c, "parse_ok", False):
            continue
        states = c.states
        if not states:
            continue
        seq = [bs.canon_state(s) for s in states]
        canon_seqs.append(seq)
        start_counts[seq[0]] += 1
        L = len(states)
        for t in range(L):
            key = (t, seq[t])
            if key in node_total:
                merge_hits += 1           # landed on an already-existing canonical node
            else:
                node_state[key] = states[t]  # first-seen representative (canon-equal to merges)
            node_total[key] = node_total.get(key, 0) + 1
            n_steps += 1
            if t + 1 < L:
                out_counts[key][seq[t + 1]] += 1   # continue -> successor
            else:
                term_answers[key][c.answer] += 1   # terminate here with this answer

    tr.node_state = node_state
    tr.out_counts = {k: dict(v) for k, v in out_counts.items()}
    tr.node_total = node_total
    tr.term_answers = {k: dict(v) for k, v in term_answers.items()}
    tr.start_counts = dict(start_counts)
    tr.n_chains = len(canon_seqs)
    tr.chain_canon_seqs = canon_seqs
    tr.layers = sorted({layer for (layer, _canon) in node_total})
    tr.n_steps = n_steps
    tr.merge_hits = merge_hits
    tr.merge_rate = (merge_hits / n_steps) if n_steps else 0.0
    return tr


# ======================================================================================
# SAVI decode (chains-only Viterbi forward pass, Phi-merged)
# ======================================================================================
def _pick_answer(counter) -> Optional[int]:
    """Argmax answer over a ``{answer_idx: count}`` counter with sc_core tie semantics: on a tie
    prefer a real option over ``None``, then the lowest index. Empty -> None."""
    if not counter:
        return None
    top = max(counter.values())
    winners = [a for a, cnt in counter.items() if cnt == top]
    winners.sort(key=lambda o: (o is None, o if o is not None else 0))
    return winners[0]


def savi_decode(trellis: Trellis, mode: str, lam: float = 0.0,
                step_scores=None, item=None) -> dict:
    """Decode the best belief path through the trellis and read off its answer.

    Modes
    -----
    * ``"none"``   : pure freq weighting — path score = sum of ``log(count/N)`` edge weights
                     (including the virtual START and END edges). The trellis's own aggregate;
                     the modal belief trajectory / answer over the merged pool.
    * ``"soft"``   : additionally adds ``lam * log(step_score)`` at each state visited, where
                     ``step_scores`` maps a canonical-state string to a verifier score in (0,1].
                     Pure DOWN-WEIGHTING, NO pruning; scores are clamped to ``_SCORE_FLOOR`` to
                     keep ``log`` finite. ``step_scores=None`` (or ``lam=0``) == ``"none"``.
    * ``"oracle"`` : hard mask (the G1 ceiling arm) — drop every node whose state is not
                     ``facts_oracle.solvable(state, item)``. A dropped node has no incoming and
                     no outgoing edges and cannot be a terminal, so any path committing the
                     question target to a wrong location is pruned. Needs ``item``.

    Returns
    -------
    ``{"answer_idx": int|None, "best_score": float|None, "path_canon": list[str],
       "marginal_not_in_single_chain": bool}``

    ``marginal_not_in_single_chain`` is True iff the decoded path's canonical-state sequence is
    NOT an index-aligned prefix of ANY single input chain — i.e. the trellis STITCHED it across
    chains. Because layers are aligned by step index, a within-a-single-chain path is exactly a
    prefix of that chain's canon sequence; anything else is a cross-chain stitch.

    Sentinel: an empty / all-parse_fail pool (no decodable terminal) returns
    ``answer_idx=None, best_score=None, path_canon=[]`` and ``marginal=False``. Never raises
    except on an unknown ``mode`` (``ValueError``, a programming error).
    """
    if mode not in ("none", "soft", "oracle"):
        raise ValueError(f"unknown mode: {mode!r}")

    empty = {"answer_idx": None, "best_score": None,
             "path_canon": [], "marginal_not_in_single_chain": False}
    if trellis.n_chains <= 0 or not trellis.node_total:
        return dict(empty)

    use_soft = (mode == "soft") and bool(step_scores) and lam != 0.0

    def emission(canon: str) -> float:
        """Per-state soft verifier contribution ``lam * log(clamped step_score)`` (0 otherwise)."""
        if not use_soft:
            return 0.0
        s = step_scores.get(canon, 1.0)
        if s < _SCORE_FLOOR:
            s = _SCORE_FLOOR
        return lam * math.log(s)

    def alive(key) -> bool:
        """Oracle mask: keep a node iff its state is still on a path to the gold answer."""
        if mode != "oracle":
            return True
        return fo.solvable(trellis.node_state[key], item)

    # ---- forward Viterbi over layers (Phi-merged: higher cumulative score kept) ----
    # reach[layer] = {canon: (best_score, backpointer_key_or_None)}
    reach: dict = defaultdict(dict)
    n_chains = trellis.n_chains

    # START -> layer 0 (frequency-weighted entry + this state's emission).
    for canon, cnt in sorted(trellis.start_counts.items()):
        key = (0, canon)
        if not alive(key):
            continue
        reach[0][canon] = (math.log(cnt / n_chains) + emission(canon), None)

    # Relax layer by layer; edges only go layer -> layer+1, so a layer is final once reached.
    for layer in trellis.layers:
        cur = reach.get(layer)
        if not cur:
            continue
        for u_canon in sorted(cur):
            u_score, _bp = cur[u_canon]
            key = (layer, u_canon)
            Nu = trellis.node_total[key]
            succ = trellis.out_counts.get(key, {})
            for v_canon in sorted(succ):
                v_key = (layer + 1, v_canon)
                if not alive(v_key):
                    continue                       # oracle drop of the successor
                edge = math.log(succ[v_canon] / Nu)
                cand = u_score + edge + emission(v_canon)
                prev = reach[layer + 1].get(v_canon)
                if prev is None or cand > prev[0]:
                    reach[layer + 1][v_canon] = (cand, key)

    # ---- best complete path: terminal -> END, freq-weighted ----
    best = None  # (complete_score, layer, canon)
    for layer in trellis.layers:
        for u_canon, (u_score, _bp) in reach.get(layer, {}).items():
            key = (layer, u_canon)
            ans_counter = trellis.term_answers.get(key)
            if not ans_counter:
                continue                            # not a terminal node
            term_count = sum(ans_counter.values())
            Nu = trellis.node_total[key]
            complete = u_score + math.log(term_count / Nu)  # END has no emission
            cand = (complete, layer, u_canon)
            if best is None or complete > best[0] or (
                    complete == best[0] and (layer, u_canon) < (best[1], best[2])):
                best = cand

    if best is None:
        return dict(empty)

    # ---- backtrack the winning path + read its terminal answer ----
    complete_score, blayer, bcanon = best
    path = []
    layer, canon = blayer, bcanon
    while True:
        path.append(canon)
        _score, bp = reach[layer][canon]
        if bp is None:
            break
        layer, canon = bp
    path.reverse()

    answer = _pick_answer(trellis.term_answers[(blayer, bcanon)])
    marginal = not any(seq[:len(path)] == path for seq in trellis.chain_canon_seqs)

    return {
        "answer_idx": answer,
        "best_score": complete_score,
        "path_canon": path,
        "marginal_not_in_single_chain": marginal,
    }


# ======================================================================================
# Baselines — SC (majority vote) and BoN (verifier argmax), over the SAME fixed pool
# ======================================================================================
def sc_answer(parsed_chains) -> Optional[int]:
    """Token-matched self-consistency: majority vote over the chains' ``answer`` fields.

    Delegates to ``sc_core.vote_at_rung`` so the majority / tie / None-bucket semantics are
    IDENTICAL to Phase A: ``None`` (a chain that failed to commit an answer) is a votable
    bucket; a tie resolves to a real option over ``None`` then the lowest index. Empty pool
    -> None (documented sentinel). Never raises.
    """
    records = [{"answer_idx": getattr(c, "answer", None), "sample_idx": i}
               for i, c in enumerate(parsed_chains)]
    return sc.vote_at_rung(records, len(records)).mode_idx


def bon_answer(parsed_chains, chain_scores) -> Optional[int]:
    """Best-of-N: the answer of the chain with the highest verifier ``chain_scores`` value.

    ``chain_scores`` is parallel to ``parsed_chains``. Ties resolve to the LOWEST index
    deterministically (strict ``>`` over the pool in order). Empty pool (or no usable scores)
    -> None. Never raises; a length mismatch is truncated to the shorter of the two.
    """
    best_idx = None
    best_score = None
    for i, (c, s) in enumerate(zip(parsed_chains, chain_scores)):
        if best_score is None or s > best_score:
            best_score = s
            best_idx = i
    if best_idx is None:
        return None
    return getattr(parsed_chains[best_idx], "answer", None)
