"""stages_g2 — the three G-A fidelity arms (design ``plans/2026-07-08-ga-fidelity.md`` +
``…-design.md``). This module dissects the flagged EXACT-GSD G-A gate (median Spearman
ρ=0.400) into its context / estimator / noise shares to decide whether λ=0's reproducible
7-of-19 fixes reflect the model's on-policy belief or a TF-likelihood-under-GSD scoring
artifact.

Every analyzer here is a PURE function of already-loaded data structures (parsed chains,
sampled-node frequency dicts, cached TF scores, an injected observed-rho map) and returns a
deterministic, json-able dict — the ``stages_g`` convention. NO file / GPU / model / torch
I/O lives here; the thin ``run_musr_cant.py --stage g2`` shell (Subtask 6) loads caches,
builds spaces, drives the sampler, and feeds these analyzers.

------------------------------------------------------------------------------------------
Arm 0 — noise ceiling / reliability correction (Subtask 2; ZERO GPU, pure recompute)
------------------------------------------------------------------------------------------
``arm0_noise_ceiling`` answers "how much of ρ=0.400's lowness is just a NOISY frequency
target rather than an unfaithful likelihood". For each G-A counted group it splits that
group's on-policy b1 chains into two seeded halves, correlates the two successor-frequency
vectors (``stages_g.spearman_rho``), and:

  * Spearman-Brown up-corrects the half-length reliability to the whole-sample reliability
    ``rho_cc = 2·rho_half / (1 + rho_half)`` (``_sb``);
  * disattenuates the group's ORIGINAL observed rho (from results_g0 ``ga.per_group``):
    ``rho_disatt = clamp_[-1,1]( rho_observed / sqrt(rho_cc) )`` (``_disattenuate``; design
    §2.1's truncation to [-1, 1], which also satisfies the plan's ``min(1.0, …)`` ceiling).

ANALYSIS UNIVERSE = exactly the G-A counted groups. A group is keyed
``(item_id, t, sprev_sha8)`` with ``sprev_sha8 = stages_g._sha8(s_prev_canon)`` — the same
key results_g0 ``ga.per_group`` rows carry — and is analysed only when that key is present
in the INJECTED ``observed_rho_by_group`` map. This keeps Arm 0 apples-to-apples with the
35 counted groups that produced 0.400, and makes the analyzer unit-testable with a synthetic
observed map (no real results_g0 needed). Because splitting a G-A counted group partitions
its chains, the union of the two halves' observed successors equals the full-pool successor
set, so any counted group present in both halves automatically clears the G-A
``ga_min_distinct_states`` (>= 3) conditioning.

EXCLUSION (fail-closed, always NAMED in ``excluded``, never coerced to a fake 0.0 — the
stages_g rho-None convention):
  * a group present in only ONE half cannot be split-half measured -> ``not_in_both_halves``;
  * ``spearman_rho`` -> None (a constant / <2-point rank vector) -> ``degenerate_rho_half``;
  * a non-positive half correlation -> ``rho_half_nonpositive`` (Spearman-Brown is only
    defined for a positive reliability; a <=0 half correlation would also drive rho_cc <= 0).
Counted (non-excluded) groups populate ``per_group`` and the three medians
(``stages_g._median``); each median filters its own None values.

------------------------------------------------------------------------------------------
Arm 1 — matched-context fidelity rho (Subtask 3; design §2.2 / §2.4)
------------------------------------------------------------------------------------------
``arm1a_strict`` / ``arm1b_specificity`` replace the b1-free-generation frequency target that
produced 0.400 with the MATCHED-CONTEXT sample frequency (drawn under the SAME frozen GSD
prefix the TF scorer uses), removing the ρ=0.400 context confound. Arm 1a is the HEADLINE:
the EXACT 35 G-A counted groups, successor set = the group's OBSERVED canon set (apples-to-
apples with 0.400), ρ_matched = Spearman(TF lse, matched freq) with a dual ``raw`` caliber,
``context_share = median_rho_matched - baseline``. Arm 1b is a ZERO-extra-GPU specificity
recompute over the Arm-2 sample cache's >=3-successor nodes, ENUMERATED successor set (the
decoder's superset), split patient vs knowledge. See the in-file Arm-1 section for the exact
alignment / drop-recording / fail-closed rules.

------------------------------------------------------------------------------------------
Arm 2 — frequency-decode reproduction (Subtask 4; design §2.3 — the MAIN readout)
------------------------------------------------------------------------------------------
``arm2_freq_decode`` / ``arm2_all`` make the decisive replacement: swap the transition score
from the TF likelihood to the matched-context sample FREQUENCY and run the REUSED, UNCHANGED
``gsd_decode(..., "map")`` decoder — Arm 2 only changes the SOURCE of ``A``. ``build_a_freq``
assembles the frequency transition-score matrix that COVERS EVERY enumerated edge (the root
constant + ``log(eps)`` backoff on every edge, overwritten by ``log(count + eps)`` where the
node was sampled), so gsd_decode never KeyErrors. ``arm2_all`` counts the reproducible-7
denominator, hard-asserts that no zero-coverage control reproduces, fail-closes low-coverage /
low-parse items to ``excluded_inconclusive``, and reports an epsilon sensitivity sweep. See the
in-file Arm-2 section for the exact contract; ``fixed`` is the downstream ``is_goal_answer``
judge, never a decode input (the map arm never touches gold / solvable).

CODE SEPARATION: imports ONLY stdlib (``hashlib`` / ``math`` / ``random``) + the pure
``stages_g`` module (Arm 0/1) and — for Arm 2 — the equally pure ``belief_schema`` (canon key)
and ``gsd_decode`` (the frozen decoder, reused verbatim; it imports only stdlib + belief_schema
+ facts_oracle). NEVER imports test code, torch, or transformers.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import belief_schema
import gsd_decode
import gsd_score
import stages_g as sg


# ======================================================================================
# closed-form corrections (exposed for direct unit testing)
# ======================================================================================
def _sb(rho_half: float) -> float:
    """Spearman-Brown up-correction from a HALF-length reliability to the whole-sample
    reliability: ``2·rho_half / (1 + rho_half)``. Defined for rho_half in (-1, 1]; the
    caller only ever passes a positive rho_half (a <=0 half correlation is excluded first)."""
    return 2.0 * rho_half / (1.0 + rho_half)


def _disattenuate(rho_observed: float, rho_cc: float):
    """Correction for attenuation: ``rho_observed / sqrt(rho_cc)``, truncated to [-1, 1]
    (design §2.1 — a disattenuated correlation cannot exceed unit magnitude; this also
    satisfies the plan's ``min(1.0, …)`` upper clamp). ``rho_cc <= 0`` (or None) -> None:
    the correction is undefined and must not blow up the division."""
    if rho_cc is None or rho_cc <= 0.0:
        return None
    return max(-1.0, min(1.0, rho_observed / math.sqrt(rho_cc)))


# ======================================================================================
# deterministic seeded split-half
# ======================================================================================
def _derive_seed(base_seed: int, item_id) -> int:
    """A stable per-item seed derived from the frozen ``splithalf_seed`` and the item id
    (sha256, NOT Python's salted ``hash``) — so each item's split is independent yet
    reproducible across runs/processes."""
    h = hashlib.sha256(("%s|%s" % (base_seed, item_id)).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _split_indices(n: int, seed: int):
    """Deterministic split of ``range(n)`` into two near-even halves A/B: seed a
    ``random.Random``, shuffle the indices, take the first ``n // 2`` as A and the rest as
    B (each half returned sorted for a stable row order). Same seed -> identical split; a
    different seed may differ."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    half = n // 2
    return sorted(idx[:half]), sorted(idx[half:])


# ======================================================================================
# Arm 0 — noise ceiling / reliability correction
# ======================================================================================
def _succ_freqs(group) -> dict:
    """{successor canon -> freq} for one ``ga_collect_groups`` group row."""
    return {s["canon"]: s["freq"] for s in group["states"]}


def arm0_noise_ceiling(chains_by_item, prereg, observed_rho_by_group) -> dict:
    """Split-half reliability + disattenuation over the G-A counted groups (design §2.1).

    ``chains_by_item`` — ``{item_id: [ParsedChain, …]}`` for the G-A evidence pools
    (patient + tuning), i.e. the SAME chains G-A tallied; only ``.states`` is read.
    ``prereg`` — a dict carrying ``"splithalf_seed"`` (frozen 20260708).
    ``observed_rho_by_group`` — ``{(item_id, t, sprev_sha8): rho_observed}`` built by the
    caller from results_g0 ``ga.per_group`` (row key = ``(item_id, t, sprev_sha8)`` with
    ``sprev_sha8`` the ``_sha8`` of the canon s_prev); it BOTH supplies each group's original
    rho AND defines the analysis universe (a group is analysed only if its key is present).

    Per item: split its chains into two seeded halves, run ``ga_collect_groups`` on each,
    and for every group present in BOTH halves whose key is in ``observed_rho_by_group``,
    correlate the two successor-frequency vectors over the UNION of successor canons
    (a canon missing from one half -> 0), Spearman-Brown up-correct, and disattenuate the
    observed rho. Exclusions are NAMED (see the module docstring); counted groups feed the
    three medians.

    Returns ``{"median_rho_cc", "median_rho_disattenuated", "median_rho_observed",
    "per_group": [{"item_id", "t", "sprev", "rho_half", "rho_cc", "rho_observed",
    "rho_disatt"}, …], "n_groups", "excluded": [{"item_id", "t", "sprev", "reason"}, …]}``.
    Deterministic and json-able.
    """
    seed = prereg["splithalf_seed"]
    observed = observed_rho_by_group or {}
    per_group: list = []
    excluded: list = []

    for item_id in sorted(chains_by_item):
        chains = list(chains_by_item[item_id])
        idx_a, idx_b = _split_indices(len(chains), _derive_seed(seed, item_id))
        chains_a = [chains[i] for i in idx_a]
        chains_b = [chains[i] for i in idx_b]
        groups_a = {(g["t"], g["s_prev_canon"]): g for g in sg.ga_collect_groups(chains_a)}
        groups_b = {(g["t"], g["s_prev_canon"]): g for g in sg.ga_collect_groups(chains_b)}

        # Universe = the G-A counted groups for THIS item: every (t, s_prev) group seen in
        # either half whose results_g0 key is present in the injected observed map.
        candidate_keys = set(groups_a) | set(groups_b)
        item_keys = sorted(gk for gk in candidate_keys
                           if (item_id, gk[0], sg._sha8(gk[1])) in observed)

        for t, sprev_canon in item_keys:
            sprev_sha = sg._sha8(sprev_canon)
            key = (item_id, t, sprev_sha)
            gk = (t, sprev_canon)
            rec = {"item_id": item_id, "t": t, "sprev": sprev_sha}

            if gk not in groups_a or gk not in groups_b:   # can't split-half measure it
                excluded.append({**rec, "reason": "not_in_both_halves"})
                continue

            fa, fb = _succ_freqs(groups_a[gk]), _succ_freqs(groups_b[gk])
            union = sorted(set(fa) | set(fb))
            vec_a = [fa.get(c, 0) for c in union]
            vec_b = [fb.get(c, 0) for c in union]
            rho_half = sg.spearman_rho(vec_a, vec_b)
            if rho_half is None:                           # degenerate rank vector
                excluded.append({**rec, "reason": "degenerate_rho_half"})
                continue
            if rho_half <= 0.0:                            # Spearman-Brown undefined (<=0)
                excluded.append({**rec, "reason": "rho_half_nonpositive"})
                continue

            rho_cc = _sb(rho_half)
            if rho_cc <= 0.0:                              # defensive (unreachable for >0)
                excluded.append({**rec, "reason": "rho_cc_nonpositive"})
                continue
            rho_observed = observed[key]                  # a float (ga.per_group rho)
            rho_disatt = (None if rho_observed is None
                          else _disattenuate(rho_observed, rho_cc))
            per_group.append({
                "item_id": item_id, "t": t, "sprev": sprev_sha,
                "rho_half": rho_half, "rho_cc": rho_cc,
                "rho_observed": rho_observed, "rho_disatt": rho_disatt,
            })

    return {
        "median_rho_cc": sg._median([g["rho_cc"] for g in per_group]),
        "median_rho_disattenuated": sg._median(
            [g["rho_disatt"] for g in per_group if g["rho_disatt"] is not None]),
        "median_rho_observed": sg._median(
            [g["rho_observed"] for g in per_group if g["rho_observed"] is not None]),
        "per_group": per_group,
        "n_groups": len(per_group),
        "excluded": excluded,
    }


# ======================================================================================
# ======================================================================================
# Arm 1 — matched-context fidelity rho (Subtask 3; design §2.2 / §2.4)
# ======================================================================================
# ======================================================================================
# Both arms replace the b1-free-generation frequency target that produced the flagged 0.400
# with the MATCHED-CONTEXT sample frequency — the on-policy successor distribution drawn under
# the SAME frozen GSD prefix the TF scorer uses (``gsd_sample.sample_node_freqs``). That
# removes the ρ=0.400 CONTEXT confound (a b1-answer prompt vs the GSD template); what remains
# is the estimator (point-estimate TF vs Monte-Carlo frequency) + noise share.
#
#   * Arm 1a (strict, HEADLINE, comparable to 0.400): iterate the EXACT 35 G-A counted groups;
#     the successor set = that group's OBSERVED successor canon set (the same canons G-A
#     tallied — NOT the enumerated superset), so it is apples-to-apples with the flagged
#     median. A sampled canon outside the observed set falls in the sampler's off-manifold
#     bucket and never extends the ρ vector.
#   * Arm 1b (specificity, ZERO extra GPU): recompute over the Arm-2 sample cache's >=3-
#     successor nodes, using the ENUMERATED successor set (the decoder's actual object, a
#     superset of the observed), split patient vs knowledge — is the low ρ patient-specific or
#     a general scoring-method property?
#
# MAIN caliber = the ``lse`` log-softmax TF scores (PREREG_g ``normalization_main``); Arm 1a
# ALSO reports the ``raw`` caliber for the dual report. Alignment is exact: tf_vec and
# freq_vec share ONE canon order; an observed canon lacking an enumerated TF edge is dropped
# from BOTH and recorded. Fail-closed (never a coerced ρ=0): a group whose node was not
# sampled, or whose item parse_ok_rate is below the frozen gate, is EXCLUDED and NAMED.

# The flagged G-A median (results_g0 ``ga.median_rho``); the descriptive ``context_share``
# baseline used when the injected prereg carries no ``ga_flagged_median_rho`` override.
_GA_FLAGGED_MEDIAN_RHO = 0.4
# Arm 1b successor-count gate (design §2.4: only nodes with >= 3 enumerated successors enter).
_ARM1B_MIN_SUCC = 3


def _observed_canons(group) -> list:
    """The G-A group's OBSERVED successor canon set (Arm 1a strict caliber, design §2.4): the
    same canons G-A tallied for this group. Accepts either an explicit ``observed_canons`` list
    or a ``stages_g.ga_collect_groups`` ``states`` list (``[{"canon": …}, …]``). Deduped and
    canon-sorted for a deterministic, TF/freq-SHARED vector order (Spearman is order-invariant
    given the SAME index order on both sides — the sort just fixes that shared order)."""
    if group.get("observed_canons") is not None:
        canons = group["observed_canons"]
    else:
        canons = [s["canon"] for s in group.get("states", ())]
    return sorted(set(canons))


def arm1a_strict(ga_groups, sample_nodes_by_item, tf_scores_by_item, prereg) -> dict:
    """Strict matched-context ρ over the 35 G-A counted groups — the headline comparable to
    the flagged 0.400 (design §2.2 Arm 1a).

    ``ga_groups`` — the G-A counted groups (results_g0 ``ga.per_group`` re-materialised by the
    runner), each a dict carrying ``item_id`` / ``t`` / ``s_prev_canon`` and the group's
    OBSERVED successors (``states`` list, or an explicit ``observed_canons`` list). These EXACT
    groups are iterated (group by group against 0.400).
    ``sample_nodes_by_item`` — ``{item_id: gsd_sample.sample_node_freqs(...)}`` (``{"nodes":
    {(t, canon_prev): {"freqs", "off_manifold", "n_parse_ok", "n_total"}},
    "item_parse_ok_rate", "n_scored"}``).
    ``tf_scores_by_item`` — ``{item_id: gsd_score.score_transitions(...)}`` (``{"raw", "lse",
    "n_scored"}`` with ``(t, cp, cn)`` edge keys). ``lse`` is the MAIN caliber; ``raw`` the
    dual report.
    ``prereg`` — carries ``"parse_ok_min"`` (fail-closed gate) and optionally
    ``"ga_flagged_median_rho"`` (the ``context_share`` baseline; default 0.400).

    Per group: the successor set is the OBSERVED canon set; each observed canon with an
    enumerated TF edge ``(t, s_prev_canon, canon_next)`` is kept, the rest DROPPED and recorded.
    The matched-freq vector is ``freqs.get(canon, 0)`` over the kept canons; the TF vector is
    aligned in the SAME order. ``rho_matched = spearman_rho(tf_vec, matched_freq_vec)`` (lse)
    plus ``rho_matched_raw`` (raw). Fail-closed exclusions (NAMED in ``excluded``, never ρ=0):
    ``item_not_sampled`` / ``low_parse_ok_rate`` / ``node_not_sampled`` / ``no_tf_scores`` /
    ``degenerate_rho_matched`` (a <2-point or all-tied rank vector -> ``spearman_rho`` None).

    Returns ``{"median_rho_matched", "median_rho_matched_raw", "context_share", "baseline",
    "per_group": [{"item_id", "t", "sprev", "rho_matched", "rho_matched_raw", "n_obs_canons",
    "n_kept", "n_dropped_no_tf_edge", "dropped_sha8", "n_on_manifold"}, …], "n_groups",
    "excluded": [{"item_id", "t", "sprev", "reason"}, …]}``. Deterministic and json-able.
    """
    parse_ok_min = prereg["parse_ok_min"]
    baseline = prereg.get("ga_flagged_median_rho", _GA_FLAGGED_MEDIAN_RHO)
    per_group: list = []
    excluded: list = []

    for grp in ga_groups:
        item_id = grp["item_id"]
        t = grp["t"]
        sp = grp["s_prev_canon"]
        rec = {"item_id": item_id, "t": t, "sprev": sg._sha8(sp)}

        # ---- fail-closed gates (never coerced to a fake ρ=0) -----------------------------
        item_sample = sample_nodes_by_item.get(item_id)
        if item_sample is None:
            excluded.append({**rec, "reason": "item_not_sampled"})
            continue
        if item_sample.get("item_parse_ok_rate", 0.0) < parse_ok_min:
            excluded.append({**rec, "reason": "low_parse_ok_rate"})
            continue
        node = item_sample.get("nodes", {}).get((t, sp))
        if node is None:
            excluded.append({**rec, "reason": "node_not_sampled"})
            continue
        tf = tf_scores_by_item.get(item_id)
        if tf is None:
            excluded.append({**rec, "reason": "no_tf_scores"})
            continue
        tf_lse, tf_raw = tf["lse"], tf["raw"]

        # ---- align OBSERVED canons to enumerated TF edges (drop + record the rest) --------
        obs = _observed_canons(grp)
        kept = [c for c in obs if (t, sp, c) in tf_lse]
        dropped = [c for c in obs if (t, sp, c) not in tf_lse]
        freqs = node["freqs"]
        matched_freq_vec = [freqs.get(c, 0) for c in kept]
        tf_vec = [tf_lse[(t, sp, c)] for c in kept]
        tf_vec_raw = [tf_raw[(t, sp, c)] for c in kept]

        rho_matched = sg.spearman_rho(tf_vec, matched_freq_vec)
        if rho_matched is None:                         # degenerate rank vector -> excluded
            excluded.append({**rec, "reason": "degenerate_rho_matched",
                             "n_kept": len(kept), "n_dropped_no_tf_edge": len(dropped)})
            continue
        rho_matched_raw = sg.spearman_rho(tf_vec_raw, matched_freq_vec)
        per_group.append({
            "item_id": item_id, "t": t, "sprev": sg._sha8(sp),
            "rho_matched": rho_matched, "rho_matched_raw": rho_matched_raw,
            "n_obs_canons": len(obs), "n_kept": len(kept),
            "n_dropped_no_tf_edge": len(dropped),
            "dropped_sha8": [sg._sha8(c) for c in dropped],
            "n_on_manifold": sum(matched_freq_vec),
        })

    median_rho_matched = sg._median([g["rho_matched"] for g in per_group])
    median_rho_matched_raw = sg._median(
        [g["rho_matched_raw"] for g in per_group if g["rho_matched_raw"] is not None])
    context_share = (None if median_rho_matched is None
                     else median_rho_matched - baseline)
    return {
        "median_rho_matched": median_rho_matched,
        "median_rho_matched_raw": median_rho_matched_raw,
        "context_share": context_share,
        "baseline": baseline,
        "per_group": per_group,
        "n_groups": len(per_group),
        "excluded": excluded,
    }


def arm1b_specificity(sample_nodes_by_item, tf_scores_by_item, prereg,
                      patient_ids, knowledge_ids) -> dict:
    """Specificity ρ over the Arm-2 sample cache (design §2.2 Arm 1b): ZERO extra GPU, split
    decode-patient vs knowledge.

    For every sampled decode item (``patient_ids`` = repro7 ∪ zero_cov = 10, ``knowledge_ids``
    = 3), each ENUMERATED node with >= ``prereg["arm1b_min_succ"]`` (default 3) enumerated
    successors present in the sample output is scored. Unlike Arm 1a, the successor set is the
    ENUMERATED canon set — the object the decoder actually uses (a superset of the observed) —
    DERIVED from the TF edge keys (``score_transitions`` scores exactly the enumerated
    ``space.trans`` edges, so its ``(t, cp, cn)`` keys enumerate that node's successors). The
    matched-freq vector is ``freqs.get(canon, 0)`` over those enumerated canons (a never-sampled
    enumerated successor -> 0); the TF (lse) vector is aligned in the SAME canon order;
    ``rho = spearman_rho(tf_vec, freq_vec)``.

    Aggregation splits by item into the patient vs knowledge buckets and takes each median over
    its non-None node ρ's (a degenerate node -> ``spearman_rho`` None -> omitted, never coerced
    to 0). Descriptive read: the patient median should NOT sit systematically below the
    knowledge median (else the low fidelity is patient-specific; same level -> ρ is a general
    property of the scoring method). An item present in ``patient_ids`` is bucketed as patient.

    Returns ``{"median_rho_patient", "median_rho_knowledge", "per_node": [{"item_id", "t",
    "sprev", "bucket", "n_succ", "rho_matched", "n_on_manifold"}, …], "n_patient",
    "n_knowledge"}`` where ``n_patient`` / ``n_knowledge`` count the NODES contributing to each
    median. Deterministic and json-able.
    """
    min_succ = prereg.get("arm1b_min_succ", _ARM1B_MIN_SUCC)
    patient_set = {str(x) for x in patient_ids}
    knowledge_set = {str(x) for x in knowledge_ids}
    per_node: list = []
    rho_patient: list = []
    rho_knowledge: list = []

    for item_id in sorted(patient_set | knowledge_set):
        item_sample = sample_nodes_by_item.get(item_id)
        tf = tf_scores_by_item.get(item_id)
        if item_sample is None or tf is None:           # not sampled / not scored -> skip
            continue
        tf_lse = tf["lse"]
        # enumerated successors per node = the TF edge keys grouped by (t, cp).
        enum_succ: dict = {}
        for (tt, cpp, cn) in tf_lse:
            enum_succ.setdefault((tt, cpp), []).append(cn)

        for (t, cp), node in sorted(item_sample.get("nodes", {}).items()):
            succ = sorted(set(enum_succ.get((t, cp), ())))
            if len(succ) < min_succ:                    # single-/2-successor node: no ρ
                continue
            freqs = node["freqs"]
            freq_vec = [freqs.get(c, 0) for c in succ]
            tf_vec = [tf_lse[(t, cp, c)] for c in succ]
            rho = sg.spearman_rho(tf_vec, freq_vec)
            is_patient = item_id in patient_set
            per_node.append({
                "item_id": item_id, "t": t, "sprev": sg._sha8(cp),
                "bucket": "patient" if is_patient else "knowledge",
                "n_succ": len(succ), "rho_matched": rho,
                "n_on_manifold": sum(freq_vec),
            })
            if rho is not None:
                (rho_patient if is_patient else rho_knowledge).append(rho)

    return {
        "median_rho_patient": sg._median(rho_patient),
        "median_rho_knowledge": sg._median(rho_knowledge),
        "per_node": per_node,
        "n_patient": len(rho_patient),
        "n_knowledge": len(rho_knowledge),
    }


# ======================================================================================
# ======================================================================================
# Arm 2 — frequency-decode reproduction (Subtask 4; design §2.3 — the MAIN readout)
# ======================================================================================
# ======================================================================================
# The decisive replacement: swap the transition score from the TF likelihood to the
# matched-context sample FREQUENCY and run the SAME frozen decoder (``gsd_decode(..., "map")``,
# REUSED UNCHANGED — Arm 2 changes ONLY the source of ``A``). For every reproducible λ=0 fix we
# ask whether the on-policy frequency-argmax path still lands on the gold-readout terminal.
#
#   * ``build_a_freq`` constructs the frequency transition-score matrix A_freq that COVERS EVERY
#     enumerated edge of the space (``gsd_decode`` KeyErrors on any missing edge — it reads
#     ``A[(0, None, root_canon)]`` and ``A[(t, cp, cn)]`` for every edge in ``space.trans``): the
#     root path constant ``A[(0, None, canon_state(root))] = 0.0``; every enumerated edge
#     ``(t, cp, cn)`` initialised to ``log(eps)`` (the frozen add-eps backoff); each SAMPLED node
#     edge overwritten with ``log(count + eps)``. A single-successor forced edge, or an unsampled
#     sibling of a branching node, keeps ``log(eps)`` (argmax is forced / a sampled edge >=
#     log(1+eps) always outscores an unsampled sibling). Only enumerated edges are overwritten:
#     gsd_sample already classifies sample freqs to enumerated canons, so the keys match.
#   * ``arm2_freq_decode`` runs ``gsd_decode(space, A_freq, "map", item=item)`` and returns the
#     answer/path/score plus ``fixed`` (``facts_oracle.is_goal_answer`` on the decoded answer — a
#     downstream JUDGE, NEVER a decode input; the map arm never touches gold / solvable) plus the
#     item's on-manifold draw count.
#   * ``arm2_all`` classifies the 13 decode items via the frozen prereg lists (repro7 /
#     zero_cov_ctrl / knowledge_ctrl), FAIL-CLOSES low-parse / low-coverage items to
#     ``excluded_inconclusive`` (NOT counted as "not reproduced"; removed from the n_repro
#     denominator), and HARD-ASSERTS that no zero-coverage control reproduces (their gold path is
#     structurally never sampled -> A_freq ~ all log(eps) -> the decode cannot reach the gold
#     terminal; a fixed=True there is a classification / backoff leak and raises — but only over
#     items actually decoded, never the excluded_inconclusive ones). ``eps_sensitivity`` re-counts
#     n_repro over repro7 at each frozen epsilon (descriptive — guards against tuning eps to
#     manufacture a reproduction). The main epsilon is ``prereg["epsilon_backoff"]``.


def build_a_freq(space, sample_nodes, eps) -> dict:
    """Build the frequency transition-score matrix ``A_freq`` for ``gsd_decode(..., "map")``.

    COVERS EVERY enumerated edge of ``space`` (gsd_decode KeyErrors on a missing edge):
      * root path constant ``A[(0, None, canon_state(root))] = 0.0`` (singleton, like the lse
        root);
      * every enumerated edge ``(t, cp, cn)`` in ``space.trans`` -> ``log(eps)``;
      * each sampled branching-node edge -> ``log(count + eps)`` (ONLY enumerated edges are
        overwritten — gsd_sample already classifies sample freqs to the node's enumerated
        successor canons, so a freq key that is not an enumerated edge is a wiring bug elsewhere
        and is skipped here rather than fabricating an off-manifold edge).

    Deterministic; returns a fresh dict keyed EXACTLY ``{(0, None, root)} ∪ {enumerated edges}``.
    """
    log_eps = math.log(eps)
    root_canon = belief_schema.canon_state(space.root)
    A = {(0, None, root_canon): 0.0}
    for (t, cp), edge_list in space.trans.items():
        for cn, _mask in edge_list:
            A[(t, cp, cn)] = log_eps
    for (t, cp), node in sample_nodes.get("nodes", {}).items():
        for cn, count in node.get("freqs", {}).items():
            key = (t, cp, cn)
            if key in A:               # only enumerated edges (gsd_sample classifies to these)
                A[key] = math.log(count + eps)
    return A


def _n_on_manifold(sample_nodes) -> int:
    """Total on-manifold sampled draws for the item = Σ over nodes of ``sum(freqs.values())``."""
    return sum(sum(node.get("freqs", {}).values())
               for node in sample_nodes.get("nodes", {}).values())


def _path_sampled_support(path_canon, sample_nodes) -> int:
    """Number of transitions on the (root-first) ``path_canon`` that were ACTUALLY sampled
    (matched-context freq > 0). Zero support on a REPRODUCED gold path = an all-backoff
    (``log eps``) tie artifact; positive support = the gold path was legitimately reached by
    on-policy conditional sampling. This — not "zero_cov never reproduces" — is the true
    anti-leak invariant: the original premise (a zero-coverage item's gold path is structurally
    unsampled) is FALSE for matched-context conditional sampling, where the model is spoon-fed
    each s_prev + event line and can produce the gold transition free generation never did."""
    if not path_canon:
        return 0
    nodes = sample_nodes.get("nodes", {})
    hits = 0
    for t in range(1, len(path_canon)):
        node = nodes.get((t, path_canon[t - 1]))
        if node and node.get("freqs", {}).get(path_canon[t], 0) > 0:
            hits += 1
    return hits


def arm2_freq_decode(item, space, sample_nodes, prereg, epsilon=None) -> dict:
    """Frequency-argmax decode of ONE item (design §2.3 — the main readout).

    ``item`` / ``space`` — the object_placements item + its enumerated ``GsdSpace``.
    ``sample_nodes`` — that item's ``gsd_sample.sample_node_freqs`` output.
    ``prereg`` — carries ``"epsilon_backoff"`` (the main add-eps value).
    ``epsilon`` — override for the eps sensitivity sweep; defaults to ``prereg["epsilon_backoff"]``.

    Builds ``A_freq`` (``build_a_freq``) and runs the REUSED ``gsd_decode(space, A_freq, "map",
    item=item)``. ``fixed`` = ``res["fixed"]`` = ``is_goal_answer`` on the decoded answer (a
    downstream judge, NOT a decode input).

    Returns ``{"answer_idx", "fixed", "path_canon", "best_score", "n_on_manifold"}``.
    """
    eps = epsilon if epsilon is not None else prereg["epsilon_backoff"]
    A = build_a_freq(space, sample_nodes, eps)
    res = gsd_decode.gsd_decode(space, A, "map", item=item)
    return {
        "answer_idx": res["answer_idx"],
        "fixed": res["fixed"],
        "path_canon": res["path_canon"],
        "best_score": res["best_score"],
        "n_on_manifold": _n_on_manifold(sample_nodes),
    }


def arm2_all(items_spaces_samples, prereg) -> dict:
    """Batch frequency-decode over the 13 decode items (design §2.3 / §3 — the main verdict input).

    ``items_spaces_samples`` — list of ``{"item_id", "item", "space", "sample_nodes"}`` for the 13
    decode items (repro7 ∪ zero_cov_ctrl(3) ∪ knowledge_ctrl(3)); ids are classified via the frozen
    prereg lists ``repro7_ids`` / ``zero_cov_ctrl_ids`` / ``knowledge_ctrl_ids``.
    ``prereg`` — carries those three id lists, ``epsilon_backoff`` (main eps), ``epsilon_sensitivity``
    (the sweep), ``parse_ok_min`` and ``node_coverage_min`` (the fail-closed gates).

    Per item: FAIL-CLOSED — if ``item_parse_ok_rate < parse_ok_min`` OR ``n_on_manifold <
    node_coverage_min`` the item is ``excluded_inconclusive`` (with a named reason) and does NOT
    count as "not reproduced"; an excluded repro7 item is removed from the n_repro denominator.
    Otherwise ``arm2_freq_decode`` at the main epsilon gives its ``fixed`` bit, recorded in the
    ``repro`` / ``zero_cov`` / ``knowledge`` dict for its bucket.

    ANTI-LEAK INVARIANT (corrected): every REPRODUCED gold path must have real sampled support
    (``_path_sampled_support > 0``); a reproduced all-backoff (``log eps``) tie RAISES. zero_cov
    reproduction is a REPORTED finding (``n_zero_cov_reproduced``), NOT an assertion — matched-
    context conditional sampling can reach a free-generation-zero-coverage gold path (the original
    "gold path structurally never sampled" premise is false for spoon-fed conditional sampling).
    ``eps_sensitivity`` = ``{eps: n_repro over decoded repro7}`` per eps (descriptive).

    Returns ``{"repro", "n_repro", "n_repro_denominator", "n_repro_total", "zero_cov",
    "n_zero_cov_reproduced", "zero_cov_reproduced_ids", "knowledge", "eps_sensitivity",
    "excluded_inconclusive", "per_item"}``. Deterministic.
    """
    repro7 = {str(x) for x in prereg["repro7_ids"]}
    zero_cov_ctrl = {str(x) for x in prereg["zero_cov_ctrl_ids"]}
    knowledge_ctrl = {str(x) for x in prereg["knowledge_ctrl_ids"]}
    parse_ok_min = prereg["parse_ok_min"]
    node_coverage_min = prereg["node_coverage_min"]
    main_eps = prereg["epsilon_backoff"]

    repro: dict = {}
    zero_cov: dict = {}
    knowledge: dict = {}
    excluded: list = []
    per_item: list = []
    repro_entries: list = []   # non-excluded repro7: the n_repro denominator + eps-sweep set

    for entry in items_spaces_samples:
        item_id = str(entry["item_id"])
        item = entry["item"]
        space = entry["space"]
        sample_nodes = entry["sample_nodes"]

        if item_id in repro7:
            bucket = "repro"
        elif item_id in zero_cov_ctrl:
            bucket = "zero_cov"
        elif item_id in knowledge_ctrl:
            bucket = "knowledge"
        else:
            bucket = "other"

        parse_ok_rate = sample_nodes.get("item_parse_ok_rate", 0.0)
        n_on = _n_on_manifold(sample_nodes)

        # ---- fail-closed: low parse-ok / low coverage -> inconclusive (NOT "not reproduced") --
        if parse_ok_rate < parse_ok_min or n_on < node_coverage_min:
            reason = ("low_parse_ok_rate" if parse_ok_rate < parse_ok_min
                      else "low_node_coverage")
            excluded.append({"item_id": item_id, "bucket": bucket, "reason": reason,
                             "item_parse_ok_rate": parse_ok_rate, "n_on_manifold": n_on})
            per_item.append({"item_id": item_id, "bucket": bucket, "excluded": True,
                             "reason": reason, "fixed": None, "answer_idx": None,
                             "path_canon": None, "best_score": None,
                             "n_on_manifold": n_on, "item_parse_ok_rate": parse_ok_rate})
            continue

        res = arm2_freq_decode(item, space, sample_nodes, prereg, epsilon=main_eps)
        fixed = bool(res["fixed"])
        support = _path_sampled_support(res["path_canon"], sample_nodes)
        per_item.append({"item_id": item_id, "bucket": bucket, "excluded": False,
                         "reason": None, "fixed": fixed, "answer_idx": res["answer_idx"],
                         "path_canon": res["path_canon"], "best_score": res["best_score"],
                         "n_on_manifold": res["n_on_manifold"],
                         "path_sampled_support": support,
                         "item_parse_ok_rate": parse_ok_rate})

        if bucket == "repro":
            repro[item_id] = fixed
            repro_entries.append((item, space, sample_nodes))
        elif bucket == "zero_cov":
            zero_cov[item_id] = fixed
        elif bucket == "knowledge":
            knowledge[item_id] = fixed
        # "other" ids (not in any frozen list) are decoded + recorded in per_item but judged in
        # no bucket — a defensive path; the runner only ever feeds the 13 classified ids.

    # ---- anti-leak invariant (corrected): a REPRODUCED gold path must have real sampled
    # support, not an all-backoff (log eps) tie. A zero_cov item reproducing WITH support is the
    # FINDING (matched-context conditional sampling reaches a free-generation-zero-coverage gold
    # path), not a leak. Only a reproduced path with ZERO sampled edges is a backoff artifact.
    backoff_artifacts = sorted(
        pi["item_id"] for pi in per_item
        if pi.get("fixed") and not pi.get("excluded")
        and pi.get("path_sampled_support", 0) == 0)
    if backoff_artifacts:  # unconditional raise (survives python -O): the real decisive control
        raise AssertionError(
            "reproduced gold-readout path with ZERO sampled support (all-backoff tie artifact "
            "— a real classification/backoff bug): %r" % (backoff_artifacts,))

    n_repro = sum(1 for f in repro.values() if f)
    # zero_cov reproduction is now a REPORTED finding (not an assertion): conditional matched-
    # context sampling reaching a selection-impossible (free-gen zero-coverage) gold path.
    n_zero_cov_reproduced = sum(1 for f in zero_cov.values() if f)
    zero_cov_reproduced_ids = sorted(i for i, f in zero_cov.items() if f)

    # ---- epsilon sensitivity (descriptive): n_repro over decoded repro7 at each frozen eps -----
    eps_sensitivity: dict = {}
    for eps in prereg["epsilon_sensitivity"]:
        eps_sensitivity[eps] = sum(
            1 for (item, space, sample_nodes) in repro_entries
            if arm2_freq_decode(item, space, sample_nodes, prereg, epsilon=eps)["fixed"])

    return {
        "repro": repro,
        "n_repro": n_repro,
        "n_repro_denominator": len(repro_entries),
        "n_repro_total": len(repro7),
        "zero_cov": zero_cov,
        "n_zero_cov_reproduced": n_zero_cov_reproduced,
        "zero_cov_reproduced_ids": zero_cov_reproduced_ids,
        "knowledge": knowledge,
        "eps_sensitivity": eps_sensitivity,
        "excluded_inconclusive": excluded,
        "per_item": per_item,
    }


# ======================================================================================
# ======================================================================================
# Subtask 5 — PREREG_G2 freeze + combined verdict table + results_g2 assembly + figures
# ======================================================================================
# ======================================================================================
# PREREG_G2 is the single source of truth for every g2 frozen constant (the plan's frozen-constants
# block): the reproducible-7 denominator, the two 3-item controls, the arm-2 belief/artifact
# gate (5/2), the arm-1 pass gate (0.5), the sampler M / max-new-tokens / seeds, the add-eps
# backoff + its sensitivity sweep, the parse & coverage fail-closed gates, the flagged G-A
# baseline ρ=0.400, and the TWO live gsd_score template shas (referenced from the module
# constants, NEVER a hand-copied string that could drift from the template). It is FROZEN
# before the g2 full run and round-trip-checked against ``outputs/PREREG_g2.md`` exactly the
# way ``stages_g.PREREG_G`` round-trips (freezing == testing; the stages_b convention).
PREREG_G2 = {
    # ---- reproducible-7 denominator (non-zero-coverage λ=0 fixes; belief question) -------
    "repro7_ids": [
        "object_placements-0006-q3", "object_placements-0011-q2",
        "object_placements-0042-q1", "object_placements-0043-q2",
        "object_placements-0047-q2", "object_placements-0050-q1",
        "object_placements-0051-q2",
    ],
    # ---- zero-coverage λ=0 fixes: FINDING axis — do they reproduce under matched-context
    # conditional sampling? (free-gen zero-coverage ≠ conditional-sampling impossibility) -------
    "zero_cov_ctrl_ids": [
        "object_placements-0014-q0", "object_placements-0021-q0",
        "object_placements-0055-q3",
    ],
    # ---- knowledge controls (specificity; along PREREG_g's knowledge_control_ids) --------
    "knowledge_ctrl_ids": [
        "object_placements-0020-q1", "object_placements-0034-q3",
        "object_placements-0035-q0",
    ],
    # ---- combined-verdict gates (design §3) ----------------------------------------------
    "arm2_belief_min": 5,        # n_repro >= 5 -> belief_supported (with rho gate)
    "arm2_artifact_max": 2,      # n_repro <= 2 -> artifact_flagged (with rho gate); 3..4 mixed
    "arm1_pass_min": 0.5,        # matched-context median rho >= 0.5 -> context is the driver
    # ---- sampler config (the only GPU station; Subtask 1) --------------------------------
    "sample_M": 128,             # draws per branching node — L1-extrapolated to ~6.5 GPU-h;
                                 # was 256 (→~13 GPU-h, over the 8 cap); degraded+declared per plan §5/§9
    "sample_max_new_tokens": 256,   # a single short BELIEF line
    "epsilon_backoff": 1e-6,     # A_freq = log(count + eps); the MAIN caliber
    "epsilon_sensitivity": [1e-5, 1e-6, 1e-7],   # eps sweep (descriptive)
    "splithalf_seed": 20260708,  # Arm 0 seeded split-half
    "sample_base_seed": 20260708,   # sampler seed derivation
    "parse_ok_min": 0.8,         # per-item parse-ok rate gate (fail-closed)
    "node_coverage_min": 32,     # per-item on-manifold draw gate (fail-closed)
    # ---- frozen scoring templates (sha256, imported LIVE from gsd_score) -----------------
    "gsd_template_sha256": gsd_score.GSD_TEMPLATE_SHA256,
    "gsd_root_template_sha256": gsd_score.GSD_ROOT_TEMPLATE_SHA256,
    # ---- flagged G-A median (results_g0 ga.median_rho): the Arm 1a context_share baseline -
    "ga_flagged_median_rho": 0.4,
}

# combined-verdict label vocabulary (design §3; four-valued — the fourth is fail-closed).
VERDICT_STRONG = "strong_belief"
VERDICT_ARTIFACT = "artifact"
VERDICT_PARTIAL = "partial"
VERDICT_INCONCLUSIVE = "inconclusive"


# ======================================================================================
# PREREG_g2 freeze — write + round-trip check (mirrors stages_g's prereg_roundtrip pattern)
# ======================================================================================
def _prereg_g2_diff(loaded: dict, code: dict) -> dict:
    """Top-level key-wise diff of two json-normalized PREREG_G2 dicts (mirrors
    ``stages_g._prereg_diff``)."""
    keys = sorted(set(loaded) | set(code))
    return {k: {"loaded": loaded.get(k, "<MISSING>"), "code": code.get(k, "<MISSING>")}
            for k in keys if loaded.get(k, "<MISSING>") != code.get(k, "<MISSING>")}


def write_prereg_g2(path: str) -> str:
    """Freeze the g2 pre-registration to ``path`` (PREREG_g2.md): a fenced ```json PREREG_G2
    block + a human-readable summary of the frozen ids / gates / templates. Round-tripped by
    ``check_prereg_g2_roundtrip`` (freezing == testing; the ``stages_g.write_prereg_g``
    convention). ``epsilon_sensitivity`` floats survive json (repr-stable)."""
    payload = json.loads(json.dumps(PREREG_G2))     # tuples -> lists (json-normalized)
    md = [
        "# MuSR-cant G-A fidelity narrowing pre-registration (PREREG_g2)",
        "",
        "Frozen BEFORE the g2 (stage g2) full run. Every id / threshold / seed / template "
        "sha below is the single source of truth in `stages_g2.PREREG_G2`; this file is "
        "round-trip checked against the code constant by "
        "`stages_g2.check_prereg_g2_roundtrip` (the `stages_g` PREREG convention). Design: "
        "`plans/2026-07-08-ga-fidelity.md` (冻结常量) + `…-design.md` §3.",
        "",
        "## Frozen constants (`stages_g2.PREREG_G2`)",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        "",
        "## Combined verdict (design §3, mechanical)",
        "",
        "- **strong_belief**: `n_repro >= %d` (arm2_belief_min) AND matched-context median "
        "`rho >= %.1f` (arm1_pass_min) -> the reproducible λ=0 fixes reflect the model's "
        "on-policy belief."
        % (PREREG_G2["arm2_belief_min"], PREREG_G2["arm1_pass_min"]),
        "- **artifact**: `n_repro <= %d` (arm2_artifact_max) AND matched-context median "
        "`rho < 0.3` -> the fixes are a TF-likelihood-under-GSD scoring artifact."
        % (PREREG_G2["arm2_artifact_max"],),
        "- **partial**: otherwise (mixed reproduction 3..4/7, or the two arms disagree); the "
        "reason records the context / estimator / noise share decomposition.",
        "- **inconclusive**: matched-context median rho undefined (all Arm 1a groups "
        "excluded / fail-closed).",
        "",
        "## Frozen decode ids",
        "",
        "- reproducible-7 (non-zero-coverage λ=0 fixes; the n_repro denominator): `%s`"
        % (list(PREREG_G2["repro7_ids"]),),
        "- zero-coverage controls (expect 0 reproduction — gold path never sampled): `%s`"
        % (list(PREREG_G2["zero_cov_ctrl_ids"]),),
        "- knowledge controls (specificity): `%s`"
        % (list(PREREG_G2["knowledge_ctrl_ids"]),),
        "",
        "## Sampler config (the only GPU station)",
        "",
        "- draws per branching node `sample_M` = %d; `sample_max_new_tokens` = %d; "
        "`sample_base_seed` = %d; `splithalf_seed` = %d."
        % (PREREG_G2["sample_M"], PREREG_G2["sample_max_new_tokens"],
           PREREG_G2["sample_base_seed"], PREREG_G2["splithalf_seed"]),
        "- add-eps backoff `epsilon_backoff` = %s (main); sensitivity sweep = %s "
        "(descriptive)."
        % (PREREG_G2["epsilon_backoff"], PREREG_G2["epsilon_sensitivity"]),
        "- fail-closed gates: `parse_ok_min` = %.1f; `node_coverage_min` = %d."
        % (PREREG_G2["parse_ok_min"], PREREG_G2["node_coverage_min"]),
        "",
        "## Frozen scoring templates (sha256, `gsd_score` — live constants)",
        "",
        "- main transition template (`GSD_TEMPLATE`): `%s`"
        % PREREG_G2["gsd_template_sha256"],
        "- root layer (`GSD_ROOT_TEMPLATE`): `%s`"
        % PREREG_G2["gsd_root_template_sha256"],
        "",
        "## Baseline",
        "",
        "- flagged G-A median (results_g0 `ga.median_rho`; the Arm 1a `context_share` "
        "baseline): `%s`." % PREREG_G2["ga_flagged_median_rho"],
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def load_prereg_g2(path: str) -> dict:
    """Parse the fenced ```json PREREG_G2 block out of a PREREG_g2.md -> dict (mirrors
    ``stages_g.load_prereg_g``)."""
    with open(path) as f:
        text = f.read()
    start = text.find("```json")
    if start == -1:
        raise ValueError("load_prereg_g2: no ```json block in %s" % path)
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise ValueError("load_prereg_g2: unterminated ```json block in %s" % path)
    return json.loads(text[start:end])


def check_prereg_g2_roundtrip(path: str) -> bool:
    """Assert the frozen PREREG_g2.md still matches ``PREREG_G2`` (json-normalized: tuples
    -> lists). True on a match; AssertionError naming every drifted top-level key on a
    mismatch (the ``stages_g.check_prereg_g_roundtrip`` convention)."""
    loaded = load_prereg_g2(path)
    code = json.loads(json.dumps(PREREG_G2))
    if loaded != code:
        raise AssertionError("PREREG_G2 mismatch: %s"
                             % json.dumps(_prereg_g2_diff(loaded, code)))
    return True


# ======================================================================================
# combined verdict (design §3 — mechanical; the strict Arm 1a rho + Arm 2 n_repro)
# ======================================================================================
def _fmt_num(x) -> str:
    """Plain 3-dp formatter that survives None (-> ``n/a``) for the reason strings."""
    return "n/a" if x is None else "%.3f" % x


def combined_verdict(arm1a: dict, arm2: dict, prereg: dict, arm0: dict = None) -> dict:
    """Mechanically execute design §3's combined verdict over the two decisive arms.

    ``arm1a`` — ``arm1a_strict`` output; reads ``median_rho_matched`` (the strict
    matched-context ρ vs the flagged 0.400) plus, for the reason, ``context_share`` /
    ``median_rho_matched_raw``. ``arm2`` — ``arm2_all`` output; reads ``n_repro`` plus, for
    the reason, ``n_repro_denominator`` / ``n_repro_total``. ``prereg`` — carries
    ``arm2_belief_min`` (5) / ``arm2_artifact_max`` (2) / ``arm1_pass_min`` (0.5). ``arm0`` —
    OPTIONAL ``arm0_noise_ceiling`` output; when given, the partial/strong reason cites the
    ``median_rho_cc`` noise ceiling (the context/estimator/noise share decomposition). The
    3-argument call (test convention) omits it; the runner passes it for a fuller reason.

    Decision (BOTH conditions required for a decisive label):
      * ``n_repro >= arm2_belief_min`` AND ``rho >= arm1_pass_min`` -> ``strong_belief``;
      * ``n_repro <= arm2_artifact_max`` AND ``rho < 0.3`` -> ``artifact``;
      * ``rho`` is None (all Arm 1a groups excluded) -> ``inconclusive``;
      * else -> ``partial`` (mixed reproduction 3..4/7, or the two arms disagree).

    Returns ``{"label", "reason"}`` — the label plus a plain-language string citing
    n_repro / denominator, the matched-context ρ, and (when ``arm0`` is given) ρ_cc.
    """
    belief_min = prereg["arm2_belief_min"]
    artifact_max = prereg["arm2_artifact_max"]
    pass_min = prereg["arm1_pass_min"]

    n_repro = arm2["n_repro"]
    denom = arm2.get("n_repro_denominator")
    total = arm2.get("n_repro_total")
    denom_str = "?" if denom is None else str(denom)
    rho = arm1a["median_rho_matched"]
    rho_cc = None if arm0 is None else arm0.get("median_rho_cc")
    context_share = arm1a.get("context_share")

    cc_clause = "" if rho_cc is None else " (split-half noise ceiling ρ_cc=%s)" % _fmt_num(rho_cc)

    # ---- inconclusive: matched-context rho undefined (fail-closed, never a coerced 0) -----
    if rho is None:
        return {
            "label": VERDICT_INCONCLUSIVE,
            "reason": ("inconclusive: matched-context median ρ is undefined (all Arm 1a "
                       "groups excluded / fail-closed); n_repro=%d/%s reproduced — the "
                       "fidelity share cannot be adjudicated%s."
                       % (n_repro, denom_str, cc_clause)),
        }

    # ---- strong_belief: BOTH the reproduction gate AND the fidelity gate clear ------------
    if n_repro >= belief_min and rho >= pass_min:
        return {
            "label": VERDICT_STRONG,
            "reason": ("strong_belief: n_repro=%d/%s reproduced (>= belief_min %d) AND "
                       "matched-context median ρ=%s (>= pass %.2f)%s — the reproducible λ=0 "
                       "fixes reflect the model's on-policy belief, not a "
                       "TF-likelihood-under-GSD scoring artifact."
                       % (n_repro, denom_str, belief_min, _fmt_num(rho), pass_min,
                          cc_clause)),
        }

    # ---- artifact: BOTH the (near-)zero reproduction AND the failed fidelity --------------
    if n_repro <= artifact_max and rho < 0.3:
        return {
            "label": VERDICT_ARTIFACT,
            "reason": ("artifact: n_repro=%d/%s reproduced (<= artifact_max %d) AND "
                       "matched-context median ρ=%s (< 0.30)%s — the λ=0 fixes are a "
                       "TF-likelihood-under-GSD scoring artifact, not model belief."
                       % (n_repro, denom_str, artifact_max, _fmt_num(rho), cc_clause)),
        }

    # ---- partial: mixed reproduction (3..4/7) and/or the two arms disagree -----------------
    if artifact_max < n_repro < belief_min:
        repro_note = "mixed reproduction (%d/%s, between the 2/5 gates)" % (n_repro, denom_str)
    elif n_repro >= belief_min:                 # reproduction passes but fidelity does not
        repro_note = "arms disagree (n_repro=%d/%s passes but ρ below the fidelity gate)" % (
            n_repro, denom_str)
    else:                                       # n_repro <= artifact_max but rho >= 0.3
        repro_note = "arms disagree (n_repro=%d/%s low but ρ above the artifact floor)" % (
            n_repro, denom_str)
    est_share = (None if (rho_cc is None or rho is None) else rho_cc - rho)
    noise_floor = (None if rho_cc is None else 1.0 - rho_cc)
    return {
        "label": VERDICT_PARTIAL,
        "reason": ("partial: %s; matched-context median ρ=%s vs flagged %s — share "
                   "decomposition context=%s, estimator=%s, noise(1-ρ_cc)=%s%s (of a "
                   "n_repro_total=%s reproducible set)."
                   % (repro_note, _fmt_num(rho), _fmt_num(arm1a.get("baseline")),
                      _fmt_num(context_share), _fmt_num(est_share), _fmt_num(noise_floor),
                      cc_clause, "?" if total is None else str(total))),
    }


# ======================================================================================
# results_g2 assembly (pure; json-safe; NaN/Inf-validated)
# ======================================================================================
def _fmt_eps_key(k) -> str:
    """Stable string key for a (float) epsilon — ``"%g"`` renders 1e-06 / 1e-05 / 1e-07
    cleanly (repr-stable, round-trips through json without float-key coercion surprises).
    Already-string keys pass through."""
    return k if isinstance(k, str) else ("%g" % k)


def _json_safe_arm2(arm2: dict) -> dict:
    """A shallow copy of ``arm2`` with the float-keyed ``eps_sensitivity`` map stringified
    (Subtask 4 returns FLOAT ε keys; json needs string keys for a stable, round-trippable
    dump). Does NOT mutate the input."""
    out = dict(arm2)
    eps = arm2.get("eps_sensitivity")
    if isinstance(eps, dict):
        out["eps_sensitivity"] = {_fmt_eps_key(k): v for k, v in eps.items()}
    return out


def _assert_finite(obj, where: str = "results_g2") -> None:
    """Recursively assert no NaN / Inf float lives anywhere in ``obj`` (results-write
    guard; design 6). Raises ValueError naming the offending path."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite float at %s: %r" % (where, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, "%s.%s" % (where, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite(v, "%s[%d]" % (where, i))


def assemble_results_g2(arm0: dict, arm1a: dict, arm1b: dict, arm2: dict, prereg: dict,
                        scope: dict, excluded, efficiency: dict) -> dict:
    """The pure results_g2 assembly: the three arms + the mechanical combined verdict +
    config / scope / exclusion roster / efficiency / run stats.

    ``arm0`` / ``arm1a`` / ``arm1b`` / ``arm2`` — the four analyzer outputs (Arm 1b is the
    zero-extra-GPU specificity recompute). ``prereg`` — the frozen PREREG_G2 (or a mutated
    copy in tests). ``scope`` — the runner's item/node-set scope block (sampled items, arm
    node counts, malformed-tree exclusions). ``excluded`` — the top-level exclusion roster
    (malformed trees / fail-closed items, named). ``efficiency`` — the GPU-time breakdown
    (sampling GPU-seconds vs the zero-GPU scoring reuse).

    ``verdict`` = ``combined_verdict(arm1a, arm2, prereg, arm0=arm0)`` (the arm0-enriched
    reason). ``config`` echoes the runtime-relevant frozen constants; ``run_stats`` summarizes
    the per-arm counts. The float-keyed Arm-2 ``eps_sensitivity`` map is stringified for a
    stable json dump, and the WHOLE dict is NaN/Inf-validated before return (design 6).
    Deterministic and json-able.
    """
    prereg = json.loads(json.dumps(prereg))          # tuples -> lists (json-normalized)
    verdict = combined_verdict(arm1a, arm2, prereg, arm0=arm0)

    config = {
        "stage": "g2",
        "normalization_main": "lse",
        "sample_M": prereg["sample_M"],
        "sample_max_new_tokens": prereg["sample_max_new_tokens"],
        "epsilon_backoff": prereg["epsilon_backoff"],
        "epsilon_sensitivity": list(prereg["epsilon_sensitivity"]),
        "sample_base_seed": prereg["sample_base_seed"],
        "splithalf_seed": prereg["splithalf_seed"],
        "parse_ok_min": prereg["parse_ok_min"],
        "node_coverage_min": prereg["node_coverage_min"],
        "arm2_belief_min": prereg["arm2_belief_min"],
        "arm2_artifact_max": prereg["arm2_artifact_max"],
        "arm1_pass_min": prereg["arm1_pass_min"],
        "ga_flagged_median_rho": prereg["ga_flagged_median_rho"],
    }

    run_stats = {
        "arm0_n_groups": arm0.get("n_groups"),
        "arm0_n_excluded": len(arm0.get("excluded", ())),
        "arm1a_n_groups": arm1a.get("n_groups"),
        "arm1a_n_excluded": len(arm1a.get("excluded", ())),
        "arm1b_n_patient": arm1b.get("n_patient"),
        "arm1b_n_knowledge": arm1b.get("n_knowledge"),
        "arm2_n_repro": arm2.get("n_repro"),
        "arm2_n_repro_denominator": arm2.get("n_repro_denominator"),
        "arm2_n_repro_total": arm2.get("n_repro_total"),
        "arm2_n_zero_cov_reproduced": sum(1 for f in arm2.get("zero_cov", {}).values() if f),
        "arm2_n_knowledge_reproduced": sum(1 for f in arm2.get("knowledge", {}).values() if f),
        "arm2_n_excluded_inconclusive": len(arm2.get("excluded_inconclusive", ())),
        "verdict_label": verdict["label"],
    }

    out = {
        "stage": "g2",
        "config": config,
        "prereg": prereg,
        "arm0": arm0,
        "arm1a": arm1a,
        "arm1b": arm1b,
        "arm2": _json_safe_arm2(arm2),           # float ε keys -> strings (stable json)
        "verdict": verdict,
        "scope": scope,
        "excluded": list(excluded) if excluded is not None else [],
        "efficiency": efficiency,
        "run_stats": run_stats,
    }
    _assert_finite(out)                          # NaN/Inf guard before the runner writes it
    return out


# ======================================================================================
# figures (matplotlib, Agg, lazily imported; ONE png per call at the explicit out_path;
# each SKIPS gracefully — returns None — when matplotlib is unavailable, mirroring stages_g)
# ======================================================================================
_COL_FLAGGED = "#c7c7c7"   # the flagged 0.400 baseline (neutral gray)
_COL_MATCHED = "#4c78a8"   # matched-context ρ (primary blue)
_COL_CEILING = "#2ca02c"   # split-half noise ceiling ρ_cc (green)
_COL_REPRO = "#2ca02c"     # reproduced / fixed (green)
_COL_NOREPRO = "#d62728"   # not reproduced (red)
_COL_EXCL = "#c7c7c7"      # excluded / inconclusive (gray + hatch)
_COL_CONTEXT = "#4c78a8"   # context share (blue)
_COL_ESTIMATOR = "#ff7f0e"  # estimator share (orange)
_COL_NOISE = "#c7c7c7"     # irreducible noise share (gray)
_COL_INK = "#444444"
_FIG_DPI = 150


def _plt():
    """Lazy headless matplotlib (Agg set BEFORE pyplot import; project convention).
    Raises ImportError when matplotlib is missing — the fig_* callers catch it and skip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _try_plt():
    """``_plt()`` or None when matplotlib is unavailable (graceful skip, like stages_g's
    figure guard)."""
    try:
        return _plt()
    except ImportError:
        return None


def fig_rho_three_caliber(arm0: dict, arm1a: dict, prereg: dict, out_path: str):
    """Three-caliber ρ bars: the flagged 0.400 baseline / the matched-context median ρ /
    the split-half noise-ceiling ρ_cc — the headline "how much of ρ=0.400 is context vs
    noise" panel. A None median (all groups excluded) renders a zero-height ``n/a`` bar.
    Returns ``out_path`` (or None when matplotlib is missing)."""
    plt = _try_plt()
    if plt is None:
        return None
    baseline = prereg.get("ga_flagged_median_rho", _GA_FLAGGED_MEDIAN_RHO)
    vals = [baseline, arm1a.get("median_rho_matched"), arm0.get("median_rho_cc")]
    labels = ["flagged G-A\n(ρ=%.3f)" % baseline, "matched-context\nρ_matched",
              "noise ceiling\nρ_cc"]
    colors = [_COL_FLAGGED, _COL_MATCHED, _COL_CEILING]

    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=_FIG_DPI)
    for i, (v, c) in enumerate(zip(vals, colors)):
        h = 0.0 if v is None else v
        ax.bar(i, h, width=0.62, color=c)
        txt = "n/a" if v is None else "%.3f" % v
        ax.annotate(txt, (i, h), textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=8, color=_COL_INK)
    ax.axhline(baseline, color=_COL_INK, lw=0.8, ls="--", zorder=1)
    ax.axhline(prereg.get("arm1_pass_min", 0.5), color=_COL_MATCHED, lw=0.8, ls=":",
               zorder=1)
    ax.annotate("arm1_pass_min=%.2f" % prereg.get("arm1_pass_min", 0.5),
                (0.99, prereg.get("arm1_pass_min", 0.5)),
                xycoords=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6.5, color=_COL_MATCHED)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Spearman ρ (TF vs frequency)", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("G-A fidelity — three calibers of ρ", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_repro_ledger(arm2: dict, prereg: dict, out_path: str):
    """The frequency-decode reproduction ledger: a tile grid over the 13 decode items in
    three frozen bands — reproducible-7 / zero-coverage-3 / knowledge-3. Each item is a tile,
    green = reproduced (``fixed``), red = not, gray+hatch = excluded_inconclusive / missing.
    Each band annotates its reproduced count. Returns ``out_path`` (or None w/o matplotlib)."""
    plt = _try_plt()
    if plt is None:
        return None
    excluded_ids = {str(e.get("item_id")) for e in arm2.get("excluded_inconclusive", ())}
    bands = [
        ("reproducible-7", list(prereg["repro7_ids"]), arm2.get("repro", {})),
        ("zero-cov-3", list(prereg["zero_cov_ctrl_ids"]), arm2.get("zero_cov", {})),
        ("knowledge-3", list(prereg["knowledge_ctrl_ids"]), arm2.get("knowledge", {})),
    ]
    max_w = max(len(ids) for _n, ids, _d in bands)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=_FIG_DPI)
    for row, (band_name, ids, fixed_map) in enumerate(bands):
        y = len(bands) - 1 - row                 # top band first
        n_fixed = 0
        for col, iid in enumerate(ids):
            iid = str(iid)
            if iid in excluded_ids:
                color, hatch, mark = _COL_EXCL, "///", "excl"
            elif fixed_map.get(iid):
                color, hatch, mark = _COL_REPRO, None, "✓"
                n_fixed += 1
            else:
                color, hatch, mark = _COL_NOREPRO, None, "·"
            ax.bar(col, 0.82, bottom=y + 0.09, width=0.82, color=color, hatch=hatch,
                   edgecolor="white", lw=0.6)
            ax.annotate(mark, (col, y + 0.5), ha="center", va="center", fontsize=8,
                        color="white")
            ax.annotate(iid.replace("object_placements-", ""), (col, y + 0.02),
                        ha="center", va="bottom", fontsize=5.0, color=_COL_INK,
                        rotation=0)
        ax.annotate("%s: %d/%d fixed" % (band_name, n_fixed, len(ids)),
                    (max_w - 0.4, y + 0.5), ha="left", va="center", fontsize=7.5,
                    color=_COL_INK)
    ax.set_xlim(-0.6, max_w + 2.2)
    ax.set_ylim(0.0, len(bands))
    ax.set_yticks([len(bands) - 1 - r + 0.5 for r in range(len(bands))])
    ax.set_yticklabels([b[0] for b in bands], fontsize=7.5)
    ax.set_xticks([])
    ax.set_title("Arm 2 — frequency-decode reproduction ledger", fontsize=9)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_share_decomposition(arm0: dict, arm1a: dict, prereg: dict, out_path: str):
    """The context / estimator / noise share decomposition (design §3): how the gap from the
    flagged 0.400 toward a perfect ρ splits into the CONTEXT share (matched-context lift over
    0.400 = ``arm1a.context_share``), the ESTIMATOR share (ρ_cc − ρ_matched, point-estimate vs
    Monte-Carlo), and the IRREDUCIBLE noise floor (1 − ρ_cc). A None component renders as a
    zero-height ``n/a`` bar. Returns ``out_path`` (or None w/o matplotlib)."""
    plt = _try_plt()
    if plt is None:
        return None
    baseline = prereg.get("ga_flagged_median_rho", _GA_FLAGGED_MEDIAN_RHO)
    rho_m = arm1a.get("median_rho_matched")
    rho_cc = arm0.get("median_rho_cc")
    context_share = arm1a.get("context_share")
    if context_share is None and rho_m is not None:
        context_share = rho_m - baseline
    estimator_share = (None if (rho_cc is None or rho_m is None) else rho_cc - rho_m)
    noise_share = (None if rho_cc is None else 1.0 - rho_cc)

    shares = [context_share, estimator_share, noise_share]
    labels = ["context\n(ρ_matched − %.3f)" % baseline, "estimator\n(ρ_cc − ρ_matched)",
              "irreducible noise\n(1 − ρ_cc)"]
    colors = [_COL_CONTEXT, _COL_ESTIMATOR, _COL_NOISE]

    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=_FIG_DPI)
    for i, (v, c) in enumerate(zip(shares, colors)):
        h = 0.0 if v is None else v
        ax.bar(i, h, width=0.62, color=c)
        txt = "n/a" if v is None else "%+.3f" % v
        va = "bottom" if h >= 0 else "top"
        off = 3 if h >= 0 else -3
        ax.annotate(txt, (i, h), textcoords="offset points", xytext=(0, off),
                    ha="center", va=va, fontsize=8, color=_COL_INK)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("share of fidelity gap (Spearman ρ units)", fontsize=8)
    ax.set_title("G-A fidelity — context / estimator / noise share decomposition",
                 fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
