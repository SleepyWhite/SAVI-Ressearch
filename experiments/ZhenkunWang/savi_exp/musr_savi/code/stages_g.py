"""stages_g — EXACT-GSD analyzers. g0 half: G-A fidelity kill-gate + backtrace-readout
ablation + frozen-id / PREREG_g freeze; g1 half: per-item two-arm assembly, the six-readout
ledger, the results_g1 assembly, and the three figures
(design ``plans/2026-07-07-exact-gsd-design.md`` §2.4/§3/§4/§5).

Every analyzer here is a PURE function of already-loaded data structures (items, spaces,
parsed chains, an injected scorer object) and returns a deterministic, json-able dict — the
``stages.py`` / ``stages_b.py`` convention. NO file/GPU/model I/O lives in the analyzers;
the thin shell (``run_musr_cant.py --stage g0|g1``, Subtask 6) reads
``cache_b1_shard*.jsonl``, parses chains, builds spaces, constructs the real
``gsd_score.TFScorer``, and feeds them in. The only functions that touch the filesystem are
the explicit PREREG_g freeze pair (``write_prereg_g`` / ``check_prereg_g_roundtrip``) and
the three ``fig_*`` functions — each writes exactly ONE file at the explicit path it is
given (matplotlib, Agg backend, lazily imported; fixed figsize/dpi, no randomness).

==========================================================================================
LEAK / ORACLE BOUNDARY
==========================================================================================
g0 is an OFFLINE (zero-GPU-decode) unit; its two analyzers touch gold data only through the
DECLARED oracle constructs, never as decode inputs:

  * G-A fidelity gate — consumes ONLY model-generated b1 chains + the structural-oracle
    ``GsdSpace`` (whose own leak boundary is audited in gsd_space) + a TF scorer over the
    frozen GSD template. No gold beliefs, no solvable, no gold_idx anywhere in the gate.
  * backtrace-readout ablation — deliberately runs the Phase B ORACLE arm
    (``domain_musr.savi_decode(mode="oracle")``, the G1 construct kept comparable) and the
    downstream judge ``facts_oracle.is_goal_answer``; both are the declared oracle scope.

==========================================================================================
FROZEN CONVENTIONS (PREREG_G is the single source of truth; freezing == testing)
==========================================================================================
* G-A (design §4, the ONLY kill-gate; evidence set WIDENED by orchestrator ruling
  2026-07-07 — two pools + ALL s_prev groups — BEFORE the PREREG freeze):
    - evidence pools (``ga_evidence_pools``): the 19 certified patients (results_b1
      sticky_ids, seed_tag ``b1_patient``, N=256 chains each) PLUS the 40 deferred
      tuning items (seed_tag ``b1_tuning``, N=256; disjoint from the sticky set —
      verified against cache_b1). Fidelity is a METHOD property ("can TF likelihood
      stand in for on-policy behavior"), not a patient inference, so the tuning pool
      adds evidence without contamination. The train/calib pools (N=24,
      ``ga_excluded_pools``) stay out: too few chains per item and a third N-regime.
      The RUNNER builds the feed from exactly these two pools; the analyzers are
      pool-TRANSPARENT (an optional ``"pool"`` label on a feed entry is echoed into
      the rows, nothing more);
    - chains: every parsed chain contributing >= 1 state counts (``ga_chain_filter =
      "any_states"``) — a token-capped chain's early BELIEF lines are honest on-policy
      choices, so filtering to parse_ok would bias the frequency estimates;
    - layers: chain POSITION indices (the trellis layer convention), clamped to
      ``1..space.T-1`` (``ga_layer_range``) because the GSD prefix needs the mechanical
      event anchor line, which exists only for t < T. Free-generation layers beyond the
      gold event count are part of the cross-template fidelity question and are declared
      out of scope, not silently scored;
    - conditioning (``ga_conditioning = "all_sprev_groups"``): for EVERY observed
      layer-(t-1) state s_prev, the chains passing through it form one (item, t, s_prev)
      GROUP whose layer-t successor frequencies are tallied; each group is scored under
      its OWN prefix (TF scores and frequencies always share one prefix). The old
      modal-prev-only statistic survives as a DIAG field only (``is_modal_prev`` row
      flag; modal by pass count, ties -> lexicographically smallest canon,
      ``ga_modal_tie``; never part of the verdict);
    - the observed states may be OUTSIDE the event-anchored reachable space — the template
      renders any state dict, and that mismatch is part of the fidelity question (declared);
    - a group counts only with >= 3 DISTINCT observed successor states
      (``ga_min_distinct_states``); a group whose rank vectors are degenerate (all-tied
      frequencies / scores, or a lone observation) yields rho None and is EXCLUDED and
      counted in the diag — never coerced to a fake 0.0;
    - Spearman rho is hand-written (Pearson on average ranks, ties -> average rank; no
      scipy); aggregate = MEDIAN over all counted groups; three-valued verdict:
      median >= 0.5 -> "PASS"; 0.3 <= median < 0.5 -> "flag_continue"; < 0.3 -> "KILL";
      zero counted groups -> median None -> "KILL" (fail-closed: no evidence never PASSes).
* Backtrace-readout ablation (design §2.4, zero GPU): per patient, the b1 chains rebuild
  the OFFLINE trellis (``domain_musr.build_trellis`` — parse_ok chains only, by that
  module's own frozen contract) and decode the oracle arm ONCE; the two readouts of the
  SAME winner are reported side by side:
    - vote readout   = ``savi_decode``'s ``answer_idx`` (the Phase B ``_pick_answer``
      majority over the terminal's chain votes — what produced G1 = 3/19);
    - mechanical     = the winner terminal ``(layer, canon)`` -> ``trellis.node_state`` ->
      ``gsd_decode.readout(state, item)`` (the SAME implementation the g1 online decoder
      uses — one readout, two call sites). An offline state missing the (observer, object)
      cell reads out None and counts as NOT fixed (design-declared).
  ``fixed`` on both sides = ``facts_oracle.is_goal_answer``; ``mech_minus_baseline`` =
  mech_total − 3 (the Phase B G1 baseline) = the readout-artifact share (readout 5).
* Frozen ids: seeded draws (``random.Random(freeze_seed)``) over SORTED pools — knowledge
  controls = ``sample(sorted(results_a2 knowledge bucket), 3)``; each deep-dive id = a
  fresh ``Random(freeze_seed).choice(sorted(pool))`` (non-zero-coverage patients /
  zero-coverage patients / knowledge controls). The PREREG_G literals below were computed
  by exactly these calls over the real results files and are pinned by a unit test.
* ``write_prereg_g`` renders ``PREREG_g.md`` with a fenced ```json PREREG_G block;
  ``check_prereg_g_roundtrip`` parses it back and asserts equality with the code constant
  (json-normalized: tuples -> lists) — the stages_b ``prereg_roundtrip`` convention.

==========================================================================================
g1 HALF (Subtask 5) — six-readout ledger + figures (design §5, mechanically executed)
==========================================================================================
* ``g1_item_analysis`` — ONE patient/control item through both decode arms
  (``gsd_decode`` map + oracle) and the Δ-ledger, all on the ``lse`` MAIN-caliber scores
  the caller passes in (dual-caliber reporting, if wanted, is the runner re-invoking with
  ``raw``). ``lambda0_agrees_sc`` frozen None semantics: BOTH None -> agree (the two
  procedures abstained identically); ONE-SIDED None -> disagree.
* ``sc_from_cache`` — SC@N=256 recomputed from the cached b1 patient records; EXACTLY
  ``sc_core.vote_at_rung(records, N).mode_idx`` (prefix by sample_idx; tie -> prefer a
  real option over the None bucket, then the lowest option index).
* ``g1_readouts`` — decision lines 1/2/3/4/6 of design §5, every threshold read from the
  prereg dict PASSED IN (single source of truth; nothing hard-coded). R4's denominator =
  ALL rows passed in (an assembled row IS an effective patient row; items whose gold tree
  was malformed never produce a row — that exclusion roster is the runner's). Readout 5
  (backtrace ablation) is NOT computed here: it is a g0 product, and
  ``assemble_results_g1`` merges it from the g0 results dict.
* ``assemble_results_g1`` — the pure results_g1 assembly: six readouts (R5 merged from
  ``g0_results["ablation"]`` or a bare ablation dict; missing -> loud ValueError) +
  ``config`` echoed verbatim + an ``efficiency`` skeleton the runner overwrites + the
  per-item rows + the json-normalized prereg echo (+ optional ``sensitivity`` section).
* figures — ``fig_deepdive_trellis`` (layer x canon-sorted-state grid; edge grayscale =
  lse score, darker = likelier; MAP path solid red, best gold-readout path dashed green —
  distinct linestyles/markers + a legend so identity is never color-alone; oracle-masked
  nodes drawn as black X), ``fig_delta_ledger`` (per-patient delta_total sorted bars;
  unreachable-gold rows = gray hatched bars at the max reachable height, annotated), and
  ``fig_demand_curve`` (delta_total ascending scatter + step line, zero baseline). All
  three: Agg backend, fixed figsize/dpi, deterministic, write ONE png, return out_path.

CODE SEPARATION: imports ONLY stdlib (``json``/``math``/``random``) + the pure core modules
``belief_schema`` / ``domain_musr`` / ``facts_oracle`` / ``gsd_decode`` / ``gsd_score`` /
``sc_core`` (gsd_score's heavy deps are lazy — importing it is CPU-cheap; matplotlib is
imported lazily inside the fig functions). NEVER imports test code, torch, or transformers.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Optional, Sequence

import belief_schema as bs
import domain_musr as dm
import facts_oracle as fo
import gsd_decode as gd
import gsd_score
import sc_core as sc

# ======================================================================================
# PREREG_G — single source of truth for every g-stage threshold / frozen id (design §4/§5).
# Frozen BEFORE the g1 full run; round-trip-checked against outputs/PREREG_g.md.
# ======================================================================================
PREREG_G = {
    # ---- freeze seed for every frozen-id draw ------------------------------------------
    "freeze_seed": 20260707,
    # ---- G-A fidelity kill-gate (design §4; the ONLY kill-gate) ------------------------
    "ga_pass_min": 0.5,                  # median rho >= this            -> "PASS"
    "ga_kill_below": 0.3,                # median rho <  this            -> "KILL"
                                         # in between                    -> "flag_continue"
    "ga_min_distinct_states": 3,         # a group counts only with >= 3 distinct successors
    "ga_chain_filter": "any_states",     # every chain with >= 1 parsed state contributes
    "ga_layer_range": "1..T-1",          # chain position index, clamped to the space's T
    "ga_conditioning": "all_sprev_groups",   # one group per observed (item, t, s_prev)
    "ga_modal_tie": "smallest_canon",    # is_modal_prev DIAG flag tie rule (not the verdict)
    # ---- G-A evidence pools (both N=256; runner feeds EXACTLY these; analyzers are
    # pool-transparent). Tuning deferred items carry no patient inference -> no leak.
    "ga_evidence_pools": {
        "patient": {"seed_tag": "b1_patient", "id_source": "results_b1.sticky_ids",
                    "n_items": 19, "n_per_item": 256},
        "tuning": {"seed_tag": "b1_tuning", "id_source": "phase_b_split tuning (deferred)",
                   "n_items": 40, "n_per_item": 256},
    },
    "ga_excluded_pools": ("b1_train", "b1_calib"),   # N=24 regime: too thin, kept out
    # ---- transition-score normalization (dual-reported; this is the MAIN readout) ------
    "normalization_main": "lse",
    # ---- frozen scoring templates (sha256, imported from gsd_score — snapshot-tested) --
    "gsd_template_sha256": gsd_score.GSD_TEMPLATE_SHA256,
    "sensitivity_template_sha256": gsd_score.SENSITIVITY_TEMPLATE_SHA256,
    "gsd_root_template_sha256": gsd_score.GSD_ROOT_TEMPLATE_SHA256,
    # ---- readout decision lines (design §5, mechanically executed by g1) ---------------
    "phaseB_G1_baseline": 3,             # offline G1 oracle fixes (results_b2, 3/19)
    "readout_lines": {
        "1_delta_ledger": "descriptive",           # per-patient demand curve, no threshold
        "2_oracle_online_fixed_pass_gt": 3,        # online oracle fixes > 3 -> ceiling lifted
        "3_zero_cov_fixed_exist_min": 1,           # >= 1 zero-cov fix = existence proof
        "4_lambda0_sc_agreement_min": 0.8,         # lambda=0 vs SC@256 per-item agreement
        "5_ablation_baseline": 3,                  # mech_total - 3 = readout-artifact share
        "6_knowledge_control_flag_gt": 1,          # > 1/3 knowledge fixes -> specificity flag
    },
    # ---- frozen sets (sizes cross-checked against results_b1/b0) -----------------------
    "patient_sticky_n": 19,              # results_b1 sticky_ids (the certified patients)
    "primary_nonzero_n": 15,             # sticky minus zero-coverage
    "zero_coverage_ids": (               # results_b0 zero_coverage_ids (frozen verbatim)
        "object_placements-0014-q0",
        "object_placements-0021-q0",
        "object_placements-0042-q0",
        "object_placements-0055-q3",
    ),
    # knowledge controls: select_knowledge_controls(results_a2 ledger.knowledge_type)
    # with freeze_seed — recomputed and pinned by tests/test_stages_g.py.
    "knowledge_control_ids": (
        "object_placements-0020-q1",
        "object_placements-0034-q3",
        "object_placements-0035-q0",
    ),
    # deep-dive trio: select_deepdive(nonzero, zero_cov, knowledge_controls) @ freeze_seed.
    "deepdive_ids": {
        "nonzero": "object_placements-0018-q0",
        "zero_cov": "object_placements-0055-q3",
        "knowledge": "object_placements-0034-q3",
    },
}

# G-A verdict vocabulary (three-valued; "KILL" aborts the g1 likelihood route).
GA_PASS = "PASS"
GA_FLAG = "flag_continue"
GA_KILL = "KILL"


# ======================================================================================
# Spearman rho — hand-written (no scipy): Pearson on average ranks, ties -> average rank
# ======================================================================================
def _ranks(values: Sequence) -> list:
    """Average ranks (1-based) of ``values``; tied values share the mean of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence, ys: Sequence) -> Optional[float]:
    """Spearman rank correlation of two parallel sequences, or None when undefined.

    Undefined (returns None, caller EXCLUDES the layer): fewer than 2 pairs, or either
    rank vector has zero variance (an all-tied side carries no ordering information — a
    coerced 0.0 would drag the median with an artifact, so exclusion is the honest move).
    """
    if len(xs) != len(ys):
        raise ValueError("spearman_rho needs parallel sequences (%d vs %d)"
                         % (len(xs), len(ys)))
    n = len(xs)
    if n < 2:
        return None
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    dx = [r - mx for r in rx]
    dy = [r - my for r in ry]
    vx = sum(d * d for d in dx)
    vy = sum(d * d for d in dy)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(vx * vy)


def _median(values: Sequence) -> Optional[float]:
    """Plain median (mean of the two middles on even n); empty -> None."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    if n % 2:
        return float(vals[n // 2])
    return (vals[n // 2 - 1] + vals[n // 2]) / 2.0


# ======================================================================================
# G-A fidelity gate (design §4) — TF likelihood rank vs on-policy sampling frequency
# ======================================================================================
def ga_collect_groups(chains, t_max: Optional[int] = None) -> list:
    """Collect, per (layer t >= 1, observed s_prev), the successor-frequency group.

    ``chains`` — parsed chains (anything with ``.states``); every chain contributing at
    least one state counts (``ga_chain_filter = "any_states"``). Layer indices are chain
    POSITION indices; ``t_max`` (inclusive; the caller passes ``space.T - 1``) clamps the
    range to the layers the GSD template can render.

    For each t, EVERY canon observed at layer t-1 forms one group; the chains passing
    through it (and continuing to layer t) contribute their layer-t states to the tally
    (``ga_conditioning = "all_sprev_groups"``). The modal layer-(t-1) canon — by pass
    count over chains holding a state at t-1, ties -> lexicographically smallest canon —
    is only FLAGGED (``is_modal_prev``) for the diag statistic, never privileged.
    Returns (t ascending, then s_prev canon ascending)::

        [{"t": t, "s_prev": repr_state, "s_prev_canon": canon, "is_modal_prev": bool,
          "n_chains": int, "states": [{"canon": c, "state": repr_state, "freq": n}, ...]},
         ...]

    ``s_prev`` / ``state`` are first-seen representative dicts of their canon class (what
    the template renders); ``n_chains`` = sum of the freqs (the group's conditional
    support). Groups with no continuing chain are omitted. Never raises on malformed
    chains.
    """
    seqs = []
    for c in chains:
        states = getattr(c, "states", None)
        if states:
            seqs.append(states)
    if not seqs:
        return []
    hi = max(len(s) for s in seqs) - 1
    if t_max is not None:
        hi = min(hi, t_max)

    out = []
    for t in range(1, hi + 1):
        prev_counts: dict = {}
        prev_repr: dict = {}
        for states in seqs:
            if len(states) >= t:                       # the chain has a state at t-1
                cp = bs.canon_state(states[t - 1])
                prev_counts[cp] = prev_counts.get(cp, 0) + 1
                prev_repr.setdefault(cp, states[t - 1])
        if not prev_counts:
            continue
        top = max(prev_counts.values())
        modal = min(c for c, n in prev_counts.items() if n == top)

        freq: dict = {}          # sprev canon -> {succ canon: count}
        state_repr: dict = {}    # succ canon -> first-seen representative dict
        for states in seqs:
            if len(states) > t:
                cp = bs.canon_state(states[t - 1])
                cn = bs.canon_state(states[t])
                freq.setdefault(cp, {})
                freq[cp][cn] = freq[cp].get(cn, 0) + 1
                state_repr.setdefault(cn, states[t])
        for cp in sorted(freq):
            succ = freq[cp]
            out.append({
                "t": t,
                "s_prev": prev_repr[cp],
                "s_prev_canon": cp,
                "is_modal_prev": cp == modal,
                "n_chains": sum(succ.values()),
                "states": [{"canon": cn, "state": state_repr[cn], "freq": succ[cn]}
                           for cn in sorted(succ)],
            })
    return out


def _sha8(canon: str) -> str:
    """Short (8-hex) sha256 of a canon string — the compact group key in the rows."""
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def ga_item(item: dict, space, chains, scorer, template: str = "main",
            pool: Optional[str] = None) -> dict:
    """G-A statistics for ONE item: per counted (t, s_prev) group, Spearman(TF, freq).

    ``scorer`` — anything with ``score_batch(prefixes, target_lines, context="") ->
    list[float]`` (the real ``gsd_score.TFScorer`` in production; a stub in unit tests).
    All rows of one group share the SINGLE prefix built by the frozen GSD template from
    that group's s_prev; targets are the observed successor states canon-sorted
    (deterministic call order: t ascending, then s_prev canon ascending). The observed
    states may lie outside the reachable space — the template renders any state dict
    (declared: part of the fidelity question itself). ``pool`` is echoed verbatim into
    the rows (analyzer stays pool-transparent).

    Returns ``{"rows": [{"item_id", "t", "sprev_sha8", "rho", "n_states", "n_chains",
    "is_modal_prev", "pool"}, ...], "diag": {"n_groups_seen",
    "n_groups_lt_min_distinct", "n_groups_degenerate"}}``.
    """
    item_id = item.get("id") if isinstance(item, dict) else None
    min_distinct = PREREG_G["ga_min_distinct_states"]
    groups = ga_collect_groups(chains, t_max=space.T - 1)
    rows = []
    n_lt = 0
    n_degen = 0
    for grp in groups:
        if len(grp["states"]) < min_distinct:
            n_lt += 1
            continue
        prefix = gsd_score.build_prefix(item, space, grp["t"], grp["s_prev"], template)
        targets = [gsd_score.build_target(grp["t"], s["state"]) for s in grp["states"]]
        lps = scorer.score_batch([prefix] * len(targets), targets,
                                 context=" (item %r ga t=%d sprev=%s)"
                                 % (item_id, grp["t"], _sha8(grp["s_prev_canon"])))
        rho = spearman_rho(list(lps), [s["freq"] for s in grp["states"]])
        if rho is None:                    # degenerate rank vector -> excluded, counted
            n_degen += 1
            continue
        rows.append({"item_id": item_id, "t": grp["t"],
                     "sprev_sha8": _sha8(grp["s_prev_canon"]), "rho": rho,
                     "n_states": len(grp["states"]), "n_chains": grp["n_chains"],
                     "is_modal_prev": grp["is_modal_prev"], "pool": pool})
    return {"rows": rows, "diag": {"n_groups_seen": len(groups),
                                   "n_groups_lt_min_distinct": n_lt,
                                   "n_groups_degenerate": n_degen}}


def ga_gate(per_group: list) -> dict:
    """Three-valued G-A verdict over the counted (item, t, s_prev) group rows (design §4).

    median rho >= 0.5 -> "PASS"; 0.3 <= median < 0.5 -> "flag_continue" (prior lowered,
    g1 proceeds); median < 0.3 -> "KILL" (the likelihood-scoring route is dead — g1 must
    refuse to run and a diagnostic VERDICT goes to the user). ZERO counted groups ->
    median None -> "KILL" (fail-closed: absent evidence never certifies fidelity).
    """
    rows = list(per_group)
    median = _median([r["rho"] for r in rows])
    if median is None:
        verdict = GA_KILL
    elif median >= PREREG_G["ga_pass_min"]:
        verdict = GA_PASS
    elif median >= PREREG_G["ga_kill_below"]:
        verdict = GA_FLAG
    else:
        verdict = GA_KILL
    return {"median_rho": median, "n_groups_counted": len(rows),
            "per_group": rows, "verdict": verdict}


def ga_fidelity(feed: list, scorer, template: str = "main") -> dict:
    """The full G-A gate over the two-pool evidence feed (the runner's single entry point).

    ``feed`` — ``[{"item": item, "space": GsdSpace|None, "chains": [ParsedChain, ...],
    "pool": str|None}, ...]``. FEED CONTRACT (runner side, PREREG ``ga_evidence_pools``):
    exactly the 19 results_b1 sticky patients (seed_tag ``b1_patient``, N=256) plus the
    40 deferred tuning items (seed_tag ``b1_tuning``, N=256), each entry labelled
    ``"pool": "patient" | "tuning"``; train/calib (N=24) never enter. The analyzer itself
    is pool-TRANSPARENT: the label is echoed into the rows and tallied in the diag, with
    zero effect on the verdict. Entries with ``space`` None (malformed gold tree) are
    skipped and NAMED in the diag (the design's excluded-item roster).

    Returns the ``ga_gate`` dict (the results_g0 ``ga`` section) + an additive ``"diag"``
    block, including ``median_rho_modal_prev`` (the pre-widening modal-only statistic,
    diagnostic only) and ``n_groups_counted_by_pool``.
    """
    rows: list = []
    diag = {"n_groups_seen": 0, "n_groups_lt_min_distinct": 0,
            "n_groups_degenerate": 0, "items_skipped_no_space": []}
    for entry in feed:
        item = entry.get("item")
        space = entry.get("space")
        chains = entry.get("chains") or []
        if space is None:
            diag["items_skipped_no_space"].append(
                item.get("id") if isinstance(item, dict) else None)
            continue
        res = ga_item(item, space, chains, scorer, template=template,
                      pool=entry.get("pool"))
        rows.extend(res["rows"])
        for k in ("n_groups_seen", "n_groups_lt_min_distinct", "n_groups_degenerate"):
            diag[k] += res["diag"][k]
    out = ga_gate(rows)
    by_pool: dict = {}
    for r in rows:
        key = r["pool"] if r["pool"] is not None else "unlabelled"
        by_pool[key] = by_pool.get(key, 0) + 1
    diag["n_groups_counted_by_pool"] = by_pool
    diag["median_rho_modal_prev"] = _median(
        [r["rho"] for r in rows if r["is_modal_prev"]])
    out["diag"] = diag
    return out


# ======================================================================================
# Backtrace-readout ablation (design §2.4, readout 5) — zero GPU, pure CPU re-decode
# ======================================================================================
def ablation_backtrace_readout(feed: list) -> dict:
    """Re-decode the Phase B OFFLINE oracle arm per patient; report vote vs mechanical
    readout of the SAME winner terminal, side by side.

    ``feed`` — ``[{"item": item, "chains": [ParsedChain, ...]}, ...]`` in frozen id
    order. FEED CONTRACT (runner side): EXACTLY the 19 results_b1 sticky patients
    (seed_tag ``b1_patient``, N=256; never the 2 un-stuck N=64 leftovers, never tuning)
    — the b2 assembly the G1=3/19 baseline was measured on (``parse_chain`` texts;
    build_trellis applies its own parse_ok filter, keeping b2 parity). A missing item /
    empty pool yields an all-False row (b2's missing-recs convention; the patient still
    counts).

    Per item: ``build_trellis`` -> ``savi_decode(mode="oracle")`` ONCE; vote readout =
    its ``answer_idx`` (Phase B ``_pick_answer`` semantics); mechanical readout = winner
    terminal ``(len(path)-1, path[-1])`` -> ``node_state`` -> ``gsd_decode.readout``
    (None — e.g. a missing (observer, object) cell — counts as NOT fixed, by design).

    Returns ``{"per_item": [{"item_id", "vote_fixed", "mech_fixed", "vote_answer_idx",
    "mech_answer_idx"}, ...], "vote_total", "mech_total", "phaseB_G1_baseline",
    "mech_minus_baseline"}`` — the readout-5 artifact-share decomposition.
    """
    per_item = []
    for entry in feed:
        item = entry.get("item")
        chains = entry.get("chains") or []
        item_id = item.get("id") if isinstance(item, dict) else entry.get("item_id")
        vote_ans = mech_ans = None
        if item is not None and chains:
            trellis = dm.build_trellis(chains)
            res = dm.savi_decode(trellis, "oracle", item=item)
            vote_ans = res["answer_idx"]
            path = res["path_canon"]
            if path:                       # winner terminal = (last layer, last canon)
                state = trellis.node_state.get((len(path) - 1, path[-1]))
                if state is not None:
                    mech_ans = gd.readout(state, item)
        per_item.append({
            "item_id": item_id,
            "vote_fixed": fo.is_goal_answer(vote_ans, item),
            "mech_fixed": fo.is_goal_answer(mech_ans, item),
            "vote_answer_idx": vote_ans,
            "mech_answer_idx": mech_ans,
        })
    vote_total = sum(1 for r in per_item if r["vote_fixed"])
    mech_total = sum(1 for r in per_item if r["mech_fixed"])
    baseline = PREREG_G["phaseB_G1_baseline"]
    return {"per_item": per_item, "vote_total": vote_total, "mech_total": mech_total,
            "phaseB_G1_baseline": baseline,
            "mech_minus_baseline": mech_total - baseline}


# ======================================================================================
# Frozen-id selection (seeded, order-independent, reproducible)
# ======================================================================================
def select_knowledge_controls(knowledge_ids, seed: Optional[int] = None, k: int = 3) -> list:
    """The frozen knowledge-control draw: ``random.Random(seed).sample(sorted(pool), k)``,
    returned sorted. ``seed`` defaults to ``PREREG_G["freeze_seed"]``. Input order never
    matters (the pool is deduped + sorted first). Raises ValueError on a pool < k."""
    seed = PREREG_G["freeze_seed"] if seed is None else seed
    pool = sorted({str(x) for x in knowledge_ids})
    if len(pool) < k:
        raise ValueError("knowledge pool has %d ids; need >= %d" % (len(pool), k))
    return sorted(random.Random(seed).sample(pool, k))


def select_deepdive(nonzero_ids, zero_cov_ids, knowledge_control_ids,
                    seed: Optional[int] = None) -> dict:
    """The frozen deep-dive trio: ONE id per pool (non-zero-coverage patients /
    zero-coverage patients / knowledge controls), each drawn by a FRESH
    ``random.Random(seed).choice(sorted(pool))`` — order-independent, same-seed
    reproducible. Raises ValueError on any empty pool."""
    seed = PREREG_G["freeze_seed"] if seed is None else seed

    def pick(ids, name):
        pool = sorted({str(x) for x in ids})
        if not pool:
            raise ValueError("deep-dive pool %r is empty" % (name,))
        return random.Random(seed).choice(pool)

    return {"nonzero": pick(nonzero_ids, "nonzero"),
            "zero_cov": pick(zero_cov_ids, "zero_cov"),
            "knowledge": pick(knowledge_control_ids, "knowledge")}


# ======================================================================================
# PREREG_g freeze — write + round-trip check (stages_b prereg_roundtrip convention)
# ======================================================================================
def _prereg_diff(loaded: dict, code: dict) -> dict:
    """Top-level key-wise diff of two json-normalized PREREG dicts."""
    keys = sorted(set(loaded) | set(code))
    return {k: {"loaded": loaded.get(k, "<MISSING>"), "code": code.get(k, "<MISSING>")}
            for k in keys if loaded.get(k, "<MISSING>") != code.get(k, "<MISSING>")}


def write_prereg_g(path: str) -> str:
    """Freeze the g-stage pre-registration to ``path`` (PREREG_g.md): a fenced ```json
    PREREG_G block + the human-readable decision-line / frozen-id summary. Round-tripped
    by ``check_prereg_g_roundtrip`` (freezing == testing)."""
    payload = json.loads(json.dumps(PREREG_G))     # tuples -> lists (json-normalized)
    lines = PREREG_G["readout_lines"]
    dd = PREREG_G["deepdive_ids"]
    md = [
        "# MuSR-cant EXACT-GSD pre-registration (PREREG_g)",
        "",
        "Frozen BEFORE the g1 full run (g0/smoke may precede). Every threshold and frozen "
        "id below is the single source of truth in `stages_g.PREREG_G`; this file is "
        "round-trip checked against the code constant by `stages_g.check_prereg_g_roundtrip` "
        "(stages_b convention). Design: `plans/2026-07-07-exact-gsd-design.md` §4/§5.",
        "",
        "## Frozen constants (`stages_g.PREREG_G`)",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        "",
        "## G-A fidelity gate (the only kill-gate, design §4; evidence set widened "
        "2026-07-07 pre-freeze)",
        "",
        "Evidence set: TWO pools at N=256 — the %d results_b1 sticky patients (seed_tag "
        "`b1_patient`) + the %d deferred tuning items (seed_tag `b1_tuning`; disjoint "
        "from the sticky set; fidelity is a method property, so tuning evidence carries "
        "no patient-inference contamination). Train/calib pools (N=24) excluded. "
        "Conditioning: for EVERY observed layer-(t-1) state s_prev (1 <= t <= T-1, chain "
        "position index), the chains through it form one (item, t, s_prev) group; a group "
        "counts with >= %d DISTINCT observed successor states. Per counted group: "
        "Spearman rho of TF logprob vs on-policy successor frequency (one shared prefix "
        "per group); aggregate = median over ALL counted groups. Verdict: median >= %.1f "
        "-> PASS; [%.1f, %.1f) -> flag_continue (prior lowered, g1 proceeds); < %.1f -> "
        "KILL (likelihood scoring route dead; g1 refuses to run; diagnostic VERDICT "
        "only); zero counted groups -> KILL (fail-closed). The modal-prev-only median is "
        "kept as a diagnostic field, never the verdict. The b1-free-generation vs "
        "GSD-template mismatch IS the fidelity question (declared)."
        % (PREREG_G["ga_evidence_pools"]["patient"]["n_items"],
           PREREG_G["ga_evidence_pools"]["tuning"]["n_items"],
           PREREG_G["ga_min_distinct_states"], PREREG_G["ga_pass_min"],
           PREREG_G["ga_kill_below"], PREREG_G["ga_pass_min"],
           PREREG_G["ga_kill_below"]),
        "",
        "## Frozen scoring templates (sha256, `gsd_score`)",
        "",
        "- main transition template (`GSD_TEMPLATE`): `%s`" % PREREG_G["gsd_template_sha256"],
        "- sensitivity paraphrase (`SENSITIVITY_TEMPLATE`): `%s`"
        % PREREG_G["sensitivity_template_sha256"],
        "- root layer (`GSD_ROOT_TEMPLATE`): `%s`" % PREREG_G["gsd_root_template_sha256"],
        "",
        "## Normalization",
        "",
        "Transition scores are dual-reported (raw logprob sum + per-(t, s_prev) "
        "log-softmax); the MAIN readout normalization is `%s`."
        % PREREG_G["normalization_main"],
        "",
        "## Readout decision lines (design §5, mechanically executed)",
        "",
        "| # | readout | line |",
        "|---|---------|------|",
        "| 1 | Delta-ledger demand curve | %s |" % lines["1_delta_ledger"],
        "| 2 | online oracle fixes vs offline G1=%d/19 | > %d lifts the ceiling; <= %d "
        "-> construct-capped |" % (PREREG_G["phaseB_G1_baseline"],
                                   lines["2_oracle_online_fixed_pass_gt"],
                                   lines["2_oracle_online_fixed_pass_gt"]),
        "| 3 | zero-coverage sub-class (4 items) | >= %d fixed = state-creation existence "
        "proof |" % lines["3_zero_cov_fixed_exist_min"],
        "| 4 | lambda=0 vs SC@256 per-item agreement | >= %.1f -> belief-posterior lesion "
        "confirmed; below -> likelihood/frequency divergence, analyzed separately |"
        % lines["4_lambda0_sc_agreement_min"],
        "| 5 | backtrace-readout ablation | mech_total - %d = readout-artifact share of "
        "the offline G1 |" % lines["5_ablation_baseline"],
        "| 6 | knowledge controls (3 items) | > %d fixed -> specificity flag (structural-"
        "oracle leak suspicion) |" % lines["6_knowledge_control_flag_gt"],
        "",
        "## Frozen ids (seed %d)" % PREREG_G["freeze_seed"],
        "",
        "- zero-coverage sub-class (results_b0, verbatim): `%s`"
        % (list(PREREG_G["zero_coverage_ids"]),),
        "- knowledge controls (seeded sample of the results_a2 knowledge bucket): `%s`"
        % (list(PREREG_G["knowledge_control_ids"]),),
        "- deep-dive trio: non-zero-coverage `%s` / zero-coverage `%s` / knowledge `%s`"
        % (dd["nonzero"], dd["zero_cov"], dd["knowledge"]),
        "- patient set: the %d results_b1 sticky ids (%d non-zero-coverage + %d "
        "zero-coverage, reported separately)."
        % (PREREG_G["patient_sticky_n"], PREREG_G["primary_nonzero_n"],
           len(PREREG_G["zero_coverage_ids"])),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def load_prereg_g(path: str) -> dict:
    """Parse the fenced ```json PREREG_G block out of a PREREG_g.md -> dict."""
    with open(path) as f:
        text = f.read()
    start = text.find("```json")
    if start == -1:
        raise ValueError("load_prereg_g: no ```json block in %s" % path)
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise ValueError("load_prereg_g: unterminated ```json block in %s" % path)
    return json.loads(text[start:end])


def check_prereg_g_roundtrip(path: str) -> bool:
    """Assert the frozen PREREG_g.md still byte-matches ``PREREG_G`` (json-normalized:
    tuples -> lists). True on a match; AssertionError naming every drifted top-level key
    on a mismatch (the stages_b ``prereg_roundtrip`` convention)."""
    loaded = load_prereg_g(path)
    code = json.loads(json.dumps(PREREG_G))
    if loaded != code:
        raise AssertionError("PREREG_G mismatch: %s"
                             % json.dumps(_prereg_diff(loaded, code)))
    return True


# ======================================================================================
# ======================================================================================
# g1 half (Subtask 5) — per-item two-arm assembly, six-readout ledger, figures
# ======================================================================================
# ======================================================================================
def g1_item_analysis(item: dict, space, scores_lse: dict,
                     sc_answer_idx: Optional[int]) -> dict:
    """ONE item through both decode arms + the Δ-ledger (all on the lse MAIN caliber).

    ``scores_lse`` — the ``gsd_score.score_transitions(...)["lse"]`` edge weights (the
    PREREG ``normalization_main``); the caller may re-invoke with ``["raw"]`` for the
    dual-caliber report. ``sc_answer_idx`` — the item's SC@256 mode (``sc_from_cache``).

    ``lambda0_agrees_sc`` — frozen None semantics: the MAP (λ=0) answer and the SC mode
    AGREE when equal, INCLUDING both-None (the two procedures abstained identically);
    a ONE-SIDED None is a disagreement. ``sc_fixed`` = ``is_goal_answer`` on the SC mode
    (downstream judge, never a decode input).

    Returns the per-item row::

        {"item_id", "map": gsd_decode(...,"map"), "oracle": gsd_decode(...,"oracle"),
         "ledger": delta_ledger(...), "sc_answer_idx", "sc_fixed", "lambda0_agrees_sc"}
    """
    map_res = gd.gsd_decode(space, scores_lse, "map", item=item)
    oracle_res = gd.gsd_decode(space, scores_lse, "oracle", item=item)
    ledger = gd.delta_ledger(space, scores_lse, item)
    return {
        "item_id": item.get("id") if isinstance(item, dict) else None,
        "map": map_res,
        "oracle": oracle_res,
        "ledger": ledger,
        "sc_answer_idx": sc_answer_idx,
        "sc_fixed": fo.is_goal_answer(sc_answer_idx, item),
        # None == None -> True; None == int -> False: exactly the frozen semantics.
        "lambda0_agrees_sc": map_res["answer_idx"] == sc_answer_idx,
    }


def sc_from_cache(chains_records: Sequence[dict], n: int = 256) -> Optional[int]:
    """SC@``n`` majority answer recomputed from cached b1 records — EXACTLY the
    ``sc_core.vote_at_rung`` semantics (single implementation, one call).

    ``chains_records`` — the per-item b1 cache records (dicts with ``answer_idx`` +
    ``sample_idx``); the vote runs over the first ``n`` by sample_idx (prefix property).
    Returns ``mode_idx``: the majority option, None when the None (parse-fail) bucket
    wins or no records exist; ties prefer a REAL option over None, then the lowest
    option index (the Phase B convention the readout-4 comparison must match).
    """
    return sc.vote_at_rung(list(chains_records), n).mode_idx


# ======================================================================================
# the six-readout ledger (design §5; thresholds from the prereg dict, never hard-coded)
# ======================================================================================
def _r1_sort_key(entry: dict) -> tuple:
    """R1 table order: reachable rows by delta_total ascending, unreachable (None) rows
    LAST; item_id breaks ties (deterministic)."""
    d = entry["delta_total"]
    return (d is None, d if d is not None else 0.0, str(entry["item_id"]))


def g1_readouts(rows_patient: list, rows_knowledge: list, zero_cov_ids,
                prereg: dict) -> dict:
    """Mechanically execute design-§5 decision lines 1/2/3/4/6 over assembled g1 rows.

    ``rows_patient`` / ``rows_knowledge`` — ``g1_item_analysis`` rows for the certified
    patients / the frozen knowledge controls; ``zero_cov_ids`` — the frozen zero-coverage
    subset of the patients; ``prereg`` — the PREREG dict whose ``readout_lines`` /
    ``phaseB_G1_baseline`` carry EVERY threshold (pass ``PREREG_G`` in production; the
    mechanical-execution test passes a mutated copy and the verdicts must follow it).

    R4's denominator is ALL patient rows passed in: an assembled row is an effective row
    by construction (malformed-tree items never reach assembly — the runner's exclusion
    roster names them); zero rows -> rate None -> fail-closed False. Readout 5 is a g0
    product and is deliberately ABSENT here — ``assemble_results_g1`` merges it from the
    g0 ablation dict.

    Returns ``{"1_delta_ledger", "2_oracle_online", "3_zero_coverage",
    "4_lambda0_vs_sc", "6_knowledge_control"}`` (each block echoes the line it applied).
    """
    lines = prereg["readout_lines"]

    # ---- R1: Δ-ledger demand table (descriptive; verdict echoes the prereg label) ----
    table = sorted(({"item_id": r["item_id"],
                     "delta_total": r["ledger"]["delta_total"],
                     "fork_layer": r["ledger"]["fork_layer"],
                     "reachable_gold": r["ledger"]["reachable_gold"]}
                    for r in rows_patient), key=_r1_sort_key)
    r1 = {"verdict": lines["1_delta_ledger"], "per_patient": table,
          "n_unreachable_gold": sum(1 for e in table if not e["reachable_gold"])}

    # ---- R2: online oracle ceiling vs the offline G1 baseline ------------------------
    pass_gt = lines["2_oracle_online_fixed_pass_gt"]
    fixed_ids = sorted(str(r["item_id"]) for r in rows_patient if r["oracle"]["fixed"])
    r2 = {"n_fixed": len(fixed_ids), "n_patients": len(rows_patient),
          "fixed_ids": fixed_ids, "baseline": prereg["phaseB_G1_baseline"],
          "line_pass_gt": pass_gt,
          "verdict": "ceiling_raised" if len(fixed_ids) > pass_gt
          else "construct_ceiling_holds"}

    # ---- R3: zero-coverage sub-class existence proof ---------------------------------
    zc = {str(x) for x in zero_cov_ids}
    zc_rows = [r for r in rows_patient if str(r["item_id"]) in zc]
    zc_fixed = sorted(str(r["item_id"]) for r in zc_rows if r["oracle"]["fixed"])
    exist_min = lines["3_zero_cov_fixed_exist_min"]
    r3 = {"n_fixed": len(zc_fixed), "n_items": len(zc_rows), "fixed_ids": zc_fixed,
          "line_exist_min": exist_min,
          "existence_proof": len(zc_fixed) >= exist_min}

    # ---- R4: λ=0 vs SC per-item agreement --------------------------------------------
    n_rows = len(rows_patient)
    agree_min = lines["4_lambda0_sc_agreement_min"]
    n_agree = sum(1 for r in rows_patient if r["lambda0_agrees_sc"])
    rate = (n_agree / n_rows) if n_rows else None
    r4 = {"n_rows": n_rows, "n_agree": n_agree, "agreement_rate": rate,
          "line_min": agree_min,
          "posterior_dominated": rate is not None and rate >= agree_min,
          "disagree_ids": sorted(str(r["item_id"]) for r in rows_patient
                                 if not r["lambda0_agrees_sc"])}

    # ---- R6: knowledge-control specificity flag ---------------------------------------
    flag_gt = lines["6_knowledge_control_flag_gt"]
    k_fixed = sorted(str(r["item_id"]) for r in rows_knowledge if r["oracle"]["fixed"])
    r6 = {"n_fixed": len(k_fixed), "n_controls": len(rows_knowledge),
          "fixed_ids": k_fixed, "line_flag_gt": flag_gt,
          "specificity_flag": len(k_fixed) > flag_gt}

    return {"1_delta_ledger": r1, "2_oracle_online": r2, "3_zero_coverage": r3,
            "4_lambda0_vs_sc": r4, "6_knowledge_control": r6}


def assemble_results_g1(rows_patient: list, rows_knowledge: list, g0_results: dict,
                        config: dict, efficiency: Optional[dict] = None,
                        zero_cov_ids=None, prereg: Optional[dict] = None,
                        sensitivity: Optional[dict] = None) -> dict:
    """The pure results_g1 assembly: six readouts + config + efficiency (+ extras).

    ``g0_results`` — the results_g0 dict (R5 is merged from its ``"ablation"`` block) or
    the bare ``ablation_backtrace_readout`` dict itself; neither shape present is a wiring
    bug -> ValueError. R5's baseline comes from prereg line 5 (NOT from the g0 echo), so
    the decision stays mechanical; ``readout_artifact_share = mech_total - baseline``
    (design §5: the readout-artifact share of the offline G1; the remainder is the
    offline construct/support share). ``efficiency`` None -> the frozen skeleton
    ``{"scoring": {"gpu_hours": None, "n_scored": 0}, "total_gpu_hours": None}`` (the
    runner overwrites it with measured values). ``zero_cov_ids`` / ``prereg`` default to
    the PREREG_G frozen values; ``sensitivity`` (the deep-dive template-sensitivity
    section, runner-computed) is included only when given. Deterministic, json-able.
    """
    prereg = json.loads(json.dumps(PREREG_G if prereg is None else prereg))
    if zero_cov_ids is None:
        zero_cov_ids = prereg["zero_coverage_ids"]

    ablation = g0_results.get("ablation") if isinstance(g0_results, dict) else None
    if ablation is None and isinstance(g0_results, dict) and "mech_total" in g0_results:
        ablation = g0_results
    if not isinstance(ablation, dict) or "mech_total" not in ablation:
        raise ValueError("assemble_results_g1: g0_results carries no ablation block "
                         "(need results_g0['ablation'] or the bare ablation dict)")

    readouts = g1_readouts(rows_patient, rows_knowledge, zero_cov_ids, prereg)
    baseline = prereg["readout_lines"]["5_ablation_baseline"]
    readouts["5_backtrace_readout"] = {
        "vote_total": ablation.get("vote_total"),
        "mech_total": ablation["mech_total"],
        "baseline": baseline,
        "readout_artifact_share": ablation["mech_total"] - baseline,
        "source": "results_g0.ablation",
    }
    order = ["1_delta_ledger", "2_oracle_online", "3_zero_coverage",
             "4_lambda0_vs_sc", "5_backtrace_readout", "6_knowledge_control"]
    readouts = {k: readouts[k] for k in order}

    out = {
        "stage": "g1",
        "config": config,
        "prereg": prereg,
        "readouts": readouts,
        "per_item": {"patient": rows_patient, "knowledge": rows_knowledge},
        "efficiency": efficiency if efficiency is not None else
        {"scoring": {"gpu_hours": None, "n_scored": 0}, "total_gpu_hours": None},
    }
    if sensitivity is not None:
        out["sensitivity"] = sensitivity
    return out


# ======================================================================================
# figures (matplotlib, Agg, lazily imported; ONE png per call at the explicit out_path)
# ======================================================================================
# Semantic colors (frozen by the plan: MAP red / gold green / neutral grays); identity is
# never color-alone — the two paths also differ in linestyle AND marker, with a legend.
_COL_MAP = "#d62728"      # MAP (λ=0) path — solid, round markers
_COL_GOLD = "#2ca02c"     # best gold-readout path — dashed, square markers
_COL_BAR = "#4c78a8"      # Δ-ledger bars (single categorical hue)
_COL_UNREACH = "#c7c7c7"  # unreachable-gold bars (gray + hatch + annotation)
_COL_INK = "#444444"      # labels/annotations wear ink, never a series color
_FIG_DPI = 150


def _plt():
    """Lazy headless matplotlib (Agg set BEFORE pyplot import; project convention)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _edge_gray(lp: float, lo: float, hi: float) -> str:
    """Grayscale for an lse edge score: darker = likelier (single-hue sequential ramp,
    light 0.88 -> dark 0.15); a degenerate range renders mid-gray."""
    frac = 0.5 if hi <= lo else (lp - lo) / (hi - lo)
    g = 0.88 - 0.73 * frac
    return "%.3f" % g


def fig_deepdive_trellis(item: dict, space, scores_lse: dict, decode_map: dict,
                         decode_oracle: dict, ledger: dict, out_path: str) -> str:
    """The deep-dive trellis: layer x state grid with edge likelihoods and both paths.

    x = layer t; y = the layer's states in canon sort order. Nodes are labeled with the
    canon sha8; oracle-masked nodes (``facts_oracle.solvable`` False — the oracle-arm
    scope, drawn for diagnosis) get a black X. Edges are grayscale by lse score (darker
    = likelier, normalized over the drawn edges). The MAP path is a solid red line with
    round markers; the best gold-readout path (``ledger["gold_path_canon"]``) a dashed
    green line with square markers; an unreachable gold readout is annotated instead.
    Title carries the item id + both arms' answers. Returns ``out_path``.
    """
    plt = _plt()
    item_id = item.get("id") if isinstance(item, dict) else None
    canon_layers = [sorted(bs.canon_state(s) for s in layer) for layer in space.layers]
    ypos = [{c: i for i, c in enumerate(layer)} for layer in canon_layers]
    state_at = [{bs.canon_state(s): s for s in layer} for layer in space.layers]

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=_FIG_DPI)

    # ---- edges (recessive, under everything) -----------------------------------------
    edge_lps = [scores_lse[(t, cp, cn)]
                for t in range(1, space.T)
                for cp in canon_layers[t - 1]
                for cn, _m in space.trans.get((t, cp), ())]
    lo, hi = (min(edge_lps), max(edge_lps)) if edge_lps else (0.0, 0.0)
    for t in range(1, space.T):
        for cp in canon_layers[t - 1]:
            for cn, _mask in space.trans.get((t, cp), ()):
                lp = scores_lse[(t, cp, cn)]
                ax.plot([t - 1, t], [ypos[t - 1][cp], ypos[t][cn]],
                        color=_edge_gray(lp, lo, hi), lw=1.0, zorder=1)

    # ---- nodes: dots, sha8 labels, oracle-mask X --------------------------------------
    for t, layer in enumerate(canon_layers):
        for c in layer:
            y = ypos[t][c]
            ax.plot([t], [y], marker="o", ms=5, color="#888888", zorder=3)
            ax.annotate(_sha8(c), (t, y), textcoords="offset points",
                        xytext=(0, -11), ha="center", fontsize=5.5, color=_COL_INK)
            if not fo.solvable(state_at[t][c], item):
                ax.plot([t], [y], marker="x", ms=11, mew=2.0, color="black",
                        zorder=4, ls="none")

    # ---- the two paths (distinct color + linestyle + marker; legend below) -----------
    def _path_xy(path):
        return (list(range(len(path))), [ypos[t][c] for t, c in enumerate(path)])

    if decode_map.get("path_canon"):
        xs, ys = _path_xy(decode_map["path_canon"])
        ax.plot(xs, ys, color=_COL_MAP, lw=2.0, ls="-", marker="o", ms=7,
                mfc="none", zorder=5, label="MAP (λ=0) path")
    if ledger.get("gold_path_canon"):
        xs, ys = _path_xy(ledger["gold_path_canon"])
        ax.plot(xs, ys, color=_COL_GOLD, lw=2.0, ls="--", marker="s", ms=7,
                mfc="none", zorder=5, label="best gold-readout path")
    else:
        ax.annotate("gold readout unreachable", (0.02, 0.97),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=8, color=_COL_INK)
    ax.plot([], [], marker="x", ms=8, mew=2.0, color="black", ls="none",
            label="oracle-masked state")

    delta = ledger.get("delta_total")
    ax.set_title("GSD trellis — %s\nmap=%r  oracle=%r  Δ=%s  fork=%s"
                 % (item_id, decode_map.get("answer_idx"),
                    decode_oracle.get("answer_idx"),
                    "%.3f" % delta if delta is not None else "n/a",
                    ledger.get("fork_layer")), fontsize=9)
    ax.set_xlabel("layer t (belief after event t)", fontsize=8)
    ax.set_xticks(range(space.T))
    ax.set_xticklabels(["t=%d" % t for t in range(space.T)], fontsize=8)
    ax.set_yticks([])
    ax.set_ylim(-0.9, max(len(l) for l in canon_layers) - 0.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=3,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_delta_ledger(rows_patient: list, out_path: str) -> str:
    """Per-patient Δ-ledger bars: delta_total ascending, fork layer annotated per bar.

    Unreachable-gold rows (``reachable_gold`` False) sort last and are drawn as gray
    HATCHED bars at the max reachable height (a placeholder magnitude — the annotation
    "unreachable" carries the meaning, not the height; all-unreachable inputs use 1.0).
    Returns ``out_path``.
    """
    plt = _plt()
    entries = sorted(({"item_id": r["item_id"],
                       "delta_total": r["ledger"]["delta_total"],
                       "fork_layer": r["ledger"]["fork_layer"],
                       "reachable_gold": r["ledger"]["reachable_gold"]}
                      for r in rows_patient), key=_r1_sort_key)
    finite = [e["delta_total"] for e in entries if e["delta_total"] is not None]
    placeholder = max(finite) if finite else 1.0

    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=_FIG_DPI)
    for i, e in enumerate(entries):
        if e["delta_total"] is None:
            height = placeholder
            ax.bar(i, height, width=0.7, color=_COL_UNREACH, hatch="///",
                   edgecolor="#888888", lw=0.5)
            ax.annotate("unreachable", (i, height), rotation=90, fontsize=5.5,
                        ha="center", va="top", color=_COL_INK,
                        textcoords="offset points", xytext=(0, -3))
            fork = "—"
        else:
            height = e["delta_total"]
            ax.bar(i, height, width=0.7, color=_COL_BAR)
            fork = "t=%s" % e["fork_layer"]
        # fork label ABOVE the bar top, in ink (readable at any bar height)
        ax.annotate(fork, (i, height), textcoords="offset points", xytext=(0, 2),
                    ha="center", va="bottom", fontsize=5.5, color=_COL_INK)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([str(e["item_id"]).replace("object_placements-", "")
                        for e in entries], rotation=90, fontsize=6)
    ax.set_ylabel("Δ nats (MAP − best gold readout)", fontsize=8)
    ax.set_title("Δ-ledger — per-patient flip demand (fork layer on each bar)",
                 fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_demand_curve(rows_patient: list, out_path: str) -> str:
    """The demand curve: patients ranked by delta_total ascending (x) vs the nats the
    discriminative signal must supply to flip them (y); scatter + post-step line, zero
    baseline. Unreachable-gold rows carry no finite demand — they are counted in a
    corner annotation, never plotted. Returns ``out_path``.
    """
    plt = _plt()
    finite = sorted((r["ledger"]["delta_total"] for r in rows_patient
                     if r["ledger"]["delta_total"] is not None))
    n_unreach = sum(1 for r in rows_patient
                    if r["ledger"]["delta_total"] is None)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=_FIG_DPI)
    if finite:
        xs = list(range(1, len(finite) + 1))
        ax.plot(xs, finite, drawstyle="steps-post", color=_COL_BAR, lw=2.0, zorder=2)
        ax.plot(xs, finite, ls="none", marker="o", ms=8, color=_COL_BAR,
                mec="white", mew=1.0, zorder=3)
        ax.set_xticks(xs)
    ax.axhline(0, color="#888888", lw=1.0, ls="--", zorder=1)
    # baseline label sits just ABOVE the dashed line (x in axes fraction, y in data —
    # the get_yaxis_transform blend), right-aligned where the ascending curve leaves room
    ax.annotate("Δ=0 baseline (MAP already reads out gold)", (0.98, 0.0),
                xycoords=ax.get_yaxis_transform(), ha="right", va="bottom",
                textcoords="offset points", xytext=(0, 3),
                fontsize=7, color=_COL_INK)
    if n_unreach:
        ax.annotate("+%d unreachable-gold patient(s), off-curve" % n_unreach,
                    (0.02, 0.95), xycoords="axes fraction", fontsize=7,
                    color=_COL_INK, va="top")
    ax.set_xlabel("patients ranked by Δ (ascending)", fontsize=8)
    ax.set_ylabel("Δ nats demanded to flip (at fork)", fontsize=8)
    ax.set_title("Demand curve — discriminative nats required per patient", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
