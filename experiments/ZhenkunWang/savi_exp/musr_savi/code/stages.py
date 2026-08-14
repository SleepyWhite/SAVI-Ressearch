"""stages — MuSR-cant Phase A three-level funnel analyzers.

The mechanical verdict layer of the falsification funnel. Every function here is a PURE
function of per-item sample records (the ``sc_core`` RECORD SCHEMA): it consumes evidence
and emits a deterministic classification. NO file/network/GPU I/O, and the ONLY import is
``sc_core`` (never the test modules) — voting / Wilson CI / sticky are imported and reused,
never reimplemented.

Three levels (design 2026-07-05-musr-cant-design.md §3–§6):
  * A0 ``triage``              — per-subtask qualification + arena selection (§3).
  * A1 ``a1_escalation_set``   — provisional-survivor set for budget deepening (§4).
  * A2 four gates + ``certify`` — statistics / knowledge-vs-commitment / scale-ladder /
    seed, then the four-gate conjunction and the three-valued continue/terminate gate (§5, §6).

Every tunable threshold lives in the module-level ``PREREG`` dict (single source of truth;
NO scattered magic numbers). ``PREREG`` is frozen before the A0 full run (design §9) and its
sha256 is round-trip-checked against ``outputs/PREREG_phaseA.md`` by the runner.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import sc_core as sc

# ======================================================================================
# PREREG — single source of truth for every tunable threshold (design §3–§6).
# ======================================================================================
PREREG = {
    # ---- A0 triage (design §3) -------------------------------------------------------
    "a0_rung": 32,                              # SC@32 used for every A0 triage statistic
    "triage_parse_rate_min": 0.80,             # parse_rate >= this
    "triage_sc_acc_margin_over_chance": 0.05,  # SC acc lower bound = chance_mean + this
    "triage_sc_acc_max": 0.90,                 # SC acc upper bound (avoid saturated arenas)
    "triage_sticky_density_min": 0.08,         # sticky-item fraction >= this
    "triage_top2_budget_gpuh_max": 25.0,       # admit a 2nd arena iff A1 budget proj <= this
    # ---- A1 attrition (design §4) ----------------------------------------------------
    "a1_escalation_rung": 256,                 # N=256 mode still wrong -> escalate
    "a1_deepen_rungs": (512, 1024),            # deepening rungs for the escalation set
    # ---- A2 four gates (design §5) ---------------------------------------------------
    "a2_stats_rung": 1024,                     # Wilson sticky gate at the 1024-equiv rung
    "a2_facts_rung": 8,                        # SC@8 over the facts-fed records
    "a2_ladder_rung": 32,                      # each scale-ladder model voted at N=32
    "a2_seed_rung": 64,                        # second-seed recheck at N=64
    "ladder_sample_cap": 60,                   # survivors > cap -> subsample ladder+seed arms
    "subsample_base_seed": 20260705,           # deterministic ladder/seed subsample seed
    # ---- continue/terminate gate (design §6) -----------------------------------------
    "certify_greenlight_min": 20,              # >=20 certified -> PhaseB greenlight
    "certify_user_decision_min": 8,            # 8..19 -> user decision; <8 -> line dead
}


# ======================================================================================
# A0 — triage
# ======================================================================================
@dataclass
class TriageVerdict:
    """Mechanical A0 triage read-out (design §3)."""
    early_negative: bool                       # True iff ZERO subtasks qualify (MuSR line closes)
    selected: list                             # chosen arena(s), sticky-density ranked (0..2)
    qualifying: list                           # all qualifying subtasks, sticky-density desc
    top2_admitted: bool                        # was a 2nd arena admitted (budget allowed)?
    budget_projection: Optional[float]         # echoed A1 GPU-h projection used for the top-2 call
    per_subtask: dict                          # per-subtask diagnostics (see _triage_subtask_stats)


def _triage_subtask_stats(items, chance_mean):
    """Compute the four triage quantities + concentration diagnostic for one subtask.

    ``items`` = list of per-item sample-record lists (each item's N=32 records).
    """
    rung = PREREG["a0_rung"]
    n_items = len(items)
    n_samples = 0
    n_parsed = 0
    n_mode_correct = 0
    n_sticky = 0
    mode_shares = []
    n_options_dist = Counter()

    for recs in items:
        n_samples += len(recs)
        n_parsed += sum(1 for r in recs if r.get("answer_idx") is not None)
        vr = sc.vote_at_rung(recs, rung)
        if vr.mode_correct:
            n_mode_correct += 1
        if sc.sticky(recs, rung):
            n_sticky += 1
        if vr.n > 0:
            mode_shares.append(vr.mode_count / vr.n)
            n_options_dist[recs[0].get("n_options")] += 1

    parse_rate = (n_parsed / n_samples) if n_samples else 0.0
    sc_acc = (n_mode_correct / n_items) if n_items else 0.0
    sticky_density = (n_sticky / n_items) if n_items else 0.0
    sc_acc_lo = chance_mean + PREREG["triage_sc_acc_margin_over_chance"]
    sc_acc_hi = PREREG["triage_sc_acc_max"]

    qualifies = (
        parse_rate >= PREREG["triage_parse_rate_min"]
        and sc_acc_lo <= sc_acc <= sc_acc_hi
        and sticky_density >= PREREG["triage_sticky_density_min"]
    )
    return {
        "parse_rate": parse_rate,
        "sc_acc": sc_acc,
        "sticky_density": sticky_density,
        "chance_mean": chance_mean,
        "sc_acc_window": (sc_acc_lo, sc_acc_hi),
        "qualifies": qualifies,
        "n_items": n_items,
        "n_samples": n_samples,
        # concentration diagnostic (answer-space; flags binary-degeneration risk, design §3.4).
        "mode_share_mean": (sum(mode_shares) / len(mode_shares)) if mode_shares else 0.0,
        "n_options_dist": dict(n_options_dist),
    }


def triage(a0_by_subtask, budget_projection=None):
    """A0 triage over the three MuSR subtasks (design §3 — pre-registered, mechanical).

    ``a0_by_subtask`` maps ``subtask -> {"items": [<item's N=32 records>, ...],
    "chance_mean": float}`` (``chance_mean`` from ``data_musr.chance_line_subtask_mean``).

    Qualify iff parse_rate >= 0.80 AND SC@32 acc in [chance_mean+0.05, 0.90] AND
    sticky_density >= 0.08. Rank qualifiers by sticky_density desc (subtask-name tie-break);
    pick top-1; admit top-2 ONLY IF ``budget_projection`` (est A1 GPU-h on both) <= 25.
    ZERO qualify -> ``early_negative=True`` (MuSR line closes).
    """
    per_subtask = {}
    for sub, block in a0_by_subtask.items():
        per_subtask[sub] = _triage_subtask_stats(block["items"], block["chance_mean"])

    # Rank qualifiers by sticky_density desc, then subtask name asc (deterministic).
    qualifying = sorted(
        [s for s, d in per_subtask.items() if d["qualifies"]],
        key=lambda s: (-per_subtask[s]["sticky_density"], s),
    )

    if not qualifying:
        return TriageVerdict(
            early_negative=True, selected=[], qualifying=[], top2_admitted=False,
            budget_projection=budget_projection, per_subtask=per_subtask,
        )

    selected = [qualifying[0]]
    top2_admitted = False
    if (len(qualifying) >= 2 and budget_projection is not None
            and budget_projection <= PREREG["triage_top2_budget_gpuh_max"]):
        selected.append(qualifying[1])
        top2_admitted = True

    return TriageVerdict(
        early_negative=False, selected=selected, qualifying=qualifying,
        top2_admitted=top2_admitted, budget_projection=budget_projection,
        per_subtask=per_subtask,
    )


# ======================================================================================
# A1 — escalation set
# ======================================================================================
def a1_escalation_set(a1_by_item):
    """Provisional-survivor set: items whose N=256 majority vote is STILL wrong (design §4).

    ``a1_by_item`` maps ``item_id -> [sample records]``. Returns the sorted list of item ids
    whose ``vote_at_rung(records, 256).mode_correct`` is False (they get deepened to N=512/1024).
    """
    rung = PREREG["a1_escalation_rung"]
    return sorted(
        item_id for item_id, recs in a1_by_item.items()
        if not sc.vote_at_rung(recs, rung).mode_correct
    )


# ======================================================================================
# A2 — the four gates
# ======================================================================================
@dataclass
class StatsGate:
    """Gate 1 (design §5.1): Wilson sticky gate at the 1024-equiv rung (excludes p~0.5 luck)."""
    passed: bool               # True iff sc_core.sticky(records, N) — confidently stuck-wrong
    corr_ci: tuple             # correctness Wilson CI at the rung (for the VERDICT ledger)
    n: int                     # samples actually voted


@dataclass
class FactsGate:
    """Gate 2 (design §5.2): knowledge/commitment split by SC@8 over the facts-fed records."""
    passed: bool                       # True == COMMITMENT type (patient)
    kind: str                          # "commitment" | "knowledge"
    mode_correct: bool                 # SC@8 mode correct (the driving quantity)
    greedy_correct: Optional[bool]     # optional greedy readout — recorded, NON-driving


@dataclass
class LadderGate:
    """Gate 3 (design §5.3): scale/model-class robustness across the ladder models."""
    passed: bool               # True == HARDCORE (no ladder model solved it)
    classification: str        # "hardcore" | "parameter_maskable"
    solved_by: list            # ladder-model names that solved it (mode correct & not sticky)
    all_sticky: bool           # diagnostic: were ALL ladder models still sticky? (design's "whole ladder sticky")


@dataclass
class SeedGate:
    """Gate 4 (design §5.4): second-seed recheck of the mode direction at N=64."""
    passed: bool                       # True iff seed-2 mode == primary mode (consistent)
    seed2_mode_idx: Optional[int]      # seed-2 majority option
    primary_mode_idx: Optional[int]    # the primary-seed majority option compared against


def a2_gate_stats(item_records, N=None):
    """Gate 1 — the Wilson statistics gate (design §5.1).

    Passes iff the item is confidently stuck on a wrong answer at the N=1024-equiv rung
    (``sc_core.sticky`` True — mode-wrong CI clean), which excludes p~0.5 luck-type stickiness.
    Returns the gate result WITH the correctness Wilson CI for the ledger.
    """
    N = PREREG["a2_stats_rung"] if N is None else N
    vr = sc.vote_at_rung(item_records, N)
    corr_ci = sc.wilson_ci(vr.k_correct, vr.n)
    return StatsGate(passed=sc.sticky(item_records, N), corr_ci=corr_ci, n=vr.n)


def a2_gate_facts(facts_records, greedy_records=None, N=None):
    """Gate 2 — the knowledge/commitment split (design §5.2).

    SC@8 mode over the gold-facts-fed records: mode correct -> COMMITMENT (patient, passes);
    mode wrong -> KNOWLEDGE (excluded, listed separately). A greedy readout may be recorded
    (``greedy_records``) but does NOT drive the decision (design §5.2).
    """
    N = PREREG["a2_facts_rung"] if N is None else N
    vr = sc.vote_at_rung(facts_records, N)
    passed = vr.mode_correct
    greedy_correct = None
    if greedy_records:
        greedy_correct = sc.vote_at_rung(greedy_records, len(greedy_records)).mode_correct
    return FactsGate(
        passed=passed,
        kind=("commitment" if passed else "knowledge"),
        mode_correct=vr.mode_correct,
        greedy_correct=greedy_correct,
    )


def a2_gate_ladder(item_id, ladder_records_by_model, N=None):
    """Gate 3 — scale/model-class robustness (design §5.3).

    For each ladder model, vote at N=32. A model "solved it" iff its mode is correct with a
    CLEAN CI (design §5.3: "mode correct and CI clean"): ``mode_correct`` True AND the correctness
    Wilson lower bound > 0.5 — symmetric to ``sc_core.sticky``'s "correctness Wilson upper < 0.5"
    for a confidently-wrong item. This requires the bigger model to RELIABLY solve it, so a single
    lucky/wobbly success (e.g. 17/32, CI straddling 0.5) does NOT discard a genuine patient.
    If ANY ladder model solved it -> PARAMETER-MASKABLE (out of hardcore); else it passes as
    hardcore. PREREG pins this BINARY reading (parameter-maskable ⟺ a bigger model solves it);
    ``all_sticky`` (were ALL ladder models still sticky) is reported as the design's stricter
    "whole-ladder-sticky" secondary diagnostic only.
    """
    N = PREREG["a2_ladder_rung"] if N is None else N
    solved_by = []
    all_sticky = True
    for model, recs in ladder_records_by_model.items():
        vr = sc.vote_at_rung(recs, N)
        stuck = sc.sticky(recs, N)
        corr_lo, _ = sc.wilson_ci(vr.k_correct, vr.n)
        if vr.mode_correct and corr_lo > 0.5:          # confidently solved (clean CI)
            solved_by.append(model)
        if not stuck:
            all_sticky = False
    solved_by.sort()
    passed = len(solved_by) == 0
    return LadderGate(
        passed=passed,
        classification=("hardcore" if passed else "parameter_maskable"),
        solved_by=solved_by,
        all_sticky=all_sticky,
    )


def a2_gate_seed(seed2_records, primary_mode_idx, N=None):
    """Gate 4 — second-seed recheck (design §5.4).

    Passes iff the second sampling seed's mode direction (majority option at N=64) is
    consistent with the primary seed's mode (``primary_mode_idx``) — i.e. the same stuck
    answer reappears under an independent seed (not seed luck).
    """
    N = PREREG["a2_seed_rung"] if N is None else N
    vr = sc.vote_at_rung(seed2_records, N)
    return SeedGate(
        passed=(vr.mode_idx == primary_mode_idx),
        seed2_mode_idx=vr.mode_idx,
        primary_mode_idx=primary_mode_idx,
    )


# ======================================================================================
# A2 — certification (four-gate conjunction + continue/terminate gate)
# ======================================================================================
@dataclass
class CertifyVerdict:
    """Final A2 certification ledger + the three-valued continue/terminate verdict (design §6)."""
    commitment_certified: list                 # passed all four gates -> the certified can't set
    knowledge_type: list                       # facts gate: not fixed by gold facts
    parameter_maskable: list                   # ladder gate: a bigger model solves it
    luck_type: list                            # stats gate: p~0.5, not confidently stuck
    seed_unstable: list                        # seed gate: mode flips under a second seed
    n_certified: int                           # == len(commitment_certified)
    verdict: str                               # PhaseB_greenlight | user_decision | MuSR_line_dead
    sampling: dict = field(default_factory=dict)  # ladder/seed subsample record (design §5)


def ladder_seed_subsample(survivors, cap, base_seed):
    """The deterministic <=cap subset of survivors that gets the ladder+seed arms (design §5).

    Sorts the survivors (canonical order), then draws ``cap`` with a fixed-seed RNG and returns
    them sorted. len(survivors) <= cap -> all survivors (no subsampling). Pure + reproducible.
    """
    ordered = sorted(survivors)
    if len(ordered) <= cap:
        return list(ordered)
    picked = random.Random(base_seed).sample(ordered, cap)
    return sorted(picked)


def _continue_terminate(n_certified):
    if n_certified >= PREREG["certify_greenlight_min"]:
        return "PhaseB_greenlight"
    if n_certified >= PREREG["certify_user_decision_min"]:
        return "user_decision"
    return "MuSR_line_dead"


def certify(survivors, stats_gates, facts_gates, ladder_gates, seed_gates, *,
            ladder_sample_cap=None, base_seed=None):
    """Four-gate conjunction over A1 survivors + the continue/terminate gate (design §5, §6).

    ``survivors``   : list of item ids that reached A2.
    ``stats_gates`` : {item_id: StatsGate}   — full set (statistics arm stays full).
    ``facts_gates`` : {item_id: FactsGate}   — full set (facts arm stays full).
    ``ladder_gates``: {item_id: LadderGate}  — only the subsampled items need entries.
    ``seed_gates``  : {item_id: SeedGate}    — only the subsampled items need entries.

    When ``len(survivors) > ladder_sample_cap`` the expensive ladder+seed arms are run on a
    deterministic ``cap``-item subsample only (design §5, guards R1-Distill long-CoT budget);
    the stats+facts arms stay full-set. Items outside the subsample that clear stats+facts are
    NOT certified (their ladder/seed evidence was never gathered) — they are recorded honestly
    under ``sampling["deferred"]``, never mislabeled into an exclusion bucket.

    Classification priority (each survivor lands in exactly one place): stats -> facts ->
    (subsample? -> ladder -> seed -> certified : deferred). Emits the ledger + a three-valued
    verdict (>=20 greenlight / 8-19 user decision / <8 line dead).
    """
    if ladder_sample_cap is None:
        ladder_sample_cap = PREREG["ladder_sample_cap"]
    if base_seed is None:
        base_seed = PREREG["subsample_base_seed"]

    survivors = list(survivors)
    subset = set(ladder_seed_subsample(survivors, ladder_sample_cap, base_seed))

    commitment_certified, knowledge_type, parameter_maskable = [], [], []
    luck_type, seed_unstable, deferred = [], [], []

    for item_id in sorted(survivors):
        # stats + facts are FULL-SET arms (every survivor must have an entry). A missing entry is
        # an assembly bug, not a science outcome — raise rather than silently mislabel it as an
        # exclusion (which would corrupt the certified count). ladder/seed may legitimately be
        # absent (>cap subsample) and are routed to `deferred` below.
        sg = stats_gates.get(item_id)
        if sg is None:
            raise KeyError("certify: missing full-set stats gate for survivor %r "
                           "(assembly bug — all survivors must be stats-gated)" % item_id)
        if not sg.passed:
            luck_type.append(item_id)
            continue
        fg = facts_gates.get(item_id)
        if fg is None:
            raise KeyError("certify: missing full-set facts gate for survivor %r "
                           "(assembly bug — all survivors must be facts-gated)" % item_id)
        if not fg.passed:
            knowledge_type.append(item_id)
            continue
        if item_id not in subset:
            deferred.append(item_id)               # stats+facts pass; ladder/seed never run
            continue
        lg = ladder_gates.get(item_id)
        seg = seed_gates.get(item_id)
        if lg is None or seg is None:
            deferred.append(item_id)               # subsampled but arm evidence missing
            continue
        if not lg.passed:
            parameter_maskable.append(item_id)
            continue
        if not seg.passed:
            seed_unstable.append(item_id)
            continue
        commitment_certified.append(item_id)

    n_certified = len(commitment_certified)
    subset_sorted = sorted(subset)
    return CertifyVerdict(
        commitment_certified=commitment_certified,
        knowledge_type=knowledge_type,
        parameter_maskable=parameter_maskable,
        luck_type=luck_type,
        seed_unstable=seed_unstable,
        n_certified=n_certified,
        verdict=_continue_terminate(n_certified),
        sampling={
            "subsampled": len(survivors) > ladder_sample_cap,
            "cap": ladder_sample_cap,
            "base_seed": base_seed,
            "n_survivors": len(survivors),
            "ladder_seed_subset": subset_sorted,
            "deferred": sorted(deferred),
        },
    )


# ======================================================================================
# a3 — self-facts control (the zero-oracle prompting threat test; design 2026-07-06-selffacts-a3.md)
# ======================================================================================
# Overall hypothesis (threat H-S): most of the 26 certified commitment-failure items can be
# stably fixed by a zero-oracle self-extraction scaffold (S1 one-call / S2 two-stage). The main
# read-out is the RESIDUAL patient count — certified items NOT stably fixed by S1 OR S2. A
# three-valued gate turns that count into a Phase-B decision, and a validity tripwire on the 46
# knowledge-type negative controls guards against non-specific answer-shifting. Every threshold
# lives here in PREREG_A3 (single source of truth), and voting / Wilson CI are reused from
# ``sc_core`` (never reimplemented), exactly as the A0–A2 layer above.
PREREG_A3 = {
    "a3_fix_rung": 64,                    # S1/S2/O all voted at N=64
    "a3_residual_greenlight_min": 20,     # residual >=20 -> threat excluded
    "a3_residual_user_decision_min": 8,   # 8..19 -> user decision; <8 -> prompting dominates
    "a3_knowledge_tripwire": 0.20,        # any self-extraction arm knowledge fix-rate >= this -> invalid
    "a3_expected_counts": (26, 46),       # (certified, knowledge) completeness; mismatch handled by runner (not here)
}

_A3_FIX_ARMS = ("selffacts_s1", "selffacts_s2")   # the two self-extraction arms whose union = "fixed"


def a3_stable_fix(recs, N=None):
    """"Stably fixed" ⟺ mode correct AND correctness Wilson lower bound > 0.5 (at N=a3_fix_rung).

    This is the SAME "solved" reading as ``a2_gate_ladder``'s clean-CI solve — the exact reverse
    symmetry of ``sc_core.sticky``'s "correctness Wilson UPPER < 0.5" for a confidently-wrong item.
    A mode-correct-but-wobbly readout (e.g. 34/64, CI straddling 0.5) does NOT count as a stable
    fix. Reuses ``sc_core.vote_at_rung`` + ``sc_core.wilson_ci`` (no reimplemented voting/CI).
    """
    N = PREREG_A3["a3_fix_rung"] if N is None else N
    vr = sc.vote_at_rung(recs, N)
    lo, _ = sc.wilson_ci(vr.k_correct, vr.n)
    return bool(vr.mode_correct and lo > 0.5)


@dataclass
class A3Verdict:
    """a3 self-facts ledger + the three-valued gate / tripwire verdict (design §3)."""
    residual: list                 # MAIN read-out: certified ids NOT stably fixed by S1 OR S2
    fixed_by: dict                 # {certified item_id -> [self-extraction arms that stably fix it]}
    knowledge_fix_rate: dict       # {"selffacts_s1": r, "selffacts_s2": r} over the knowledge controls
    tripwire_fired: bool           # max(the two knowledge rates) >= a3_knowledge_tripwire
    fragile_fix: list              # certified ids whose ORACLE (O) records fail a3_stable_fix (triage, non-gating)
    verdict: str                   # threat_excluded | user_decision | prompting_dominates | mechanism_invalid
    n_residual: int                # == len(residual)


def _a3_require(arm_recs, item_id, arm_label):
    """Missing-arm guard — mirror ``certify``'s full-set KeyError: an in-scope item lacking a
    required arm's records is an ASSEMBLY BUG (would silently corrupt the ledger), NOT a science
    exclusion. Raise rather than skip."""
    if item_id not in arm_recs:
        raise KeyError("a3_verdict: missing %s records for in-scope item %r (assembly bug)"
                       % (arm_label, item_id))
    return arm_recs[item_id]


def a3_verdict(certified, knowledge, s1_recs, s2_recs, o_recs, prereg=None):
    """Residual-patient ledger + three-valued gate + knowledge tripwire (design §3).

    ``s1_recs`` / ``s2_recs`` : {item_id -> records}, expected to cover certified ∪ knowledge.
    ``o_recs``                : {item_id -> records}, expected to cover certified only (oracle arm).

    An in-scope item missing a required arm's records raises ``KeyError`` (mirrors ``certify``);
    it is never silently skipped. ``fixed_by`` / ``residual`` are computed only over the certified
    set, from the S1∨S2 union (O is oracle — it feeds ``fragile_fix`` only, not the fix union).
    The knowledge tripwire (either self-extraction arm stably "fixing" >= 20% of the knowledge
    negative controls) OVERRIDES the three-valued residual gate to ``mechanism_invalid``.
    """
    prereg = PREREG_A3 if prereg is None else prereg

    # --- certified set: fixed_by (S1∨S2 union) + residual + fragile (O) ----------------
    fixed_by = {}
    residual = []
    fragile_fix = []
    for cid in certified:
        s1 = _a3_require(s1_recs, cid, "selffacts_s1")
        s2 = _a3_require(s2_recs, cid, "selffacts_s2")
        o = _a3_require(o_recs, cid, "oracle")
        arms = []
        if a3_stable_fix(s1):
            arms.append("selffacts_s1")
        if a3_stable_fix(s2):
            arms.append("selffacts_s2")
        fixed_by[cid] = arms
        if not arms:                       # not fixed by S1 OR S2 -> residual patient
            residual.append(cid)
        if not a3_stable_fix(o):           # oracle fix that doesn't hold at N=64 -> fragile
            fragile_fix.append(cid)
    residual.sort()
    fragile_fix.sort()

    # --- knowledge negative controls: per-arm fix-rate + validity tripwire -------------
    n_know = len(knowledge)
    know_fixes = {"selffacts_s1": 0, "selffacts_s2": 0}
    for kid in knowledge:
        s1 = _a3_require(s1_recs, kid, "selffacts_s1")
        s2 = _a3_require(s2_recs, kid, "selffacts_s2")
        if a3_stable_fix(s1):
            know_fixes["selffacts_s1"] += 1
        if a3_stable_fix(s2):
            know_fixes["selffacts_s2"] += 1
    knowledge_fix_rate = {
        arm: (know_fixes[arm] / n_know if n_know else 0.0) for arm in _A3_FIX_ARMS
    }
    tripwire_fired = (
        max(knowledge_fix_rate.values()) >= prereg["a3_knowledge_tripwire"]
    )

    # --- verdict: tripwire OVERRIDES the three-valued residual gate ---------------------
    n_residual = len(residual)
    if tripwire_fired:
        verdict = "mechanism_invalid"
    elif n_residual >= prereg["a3_residual_greenlight_min"]:
        verdict = "threat_excluded"
    elif n_residual >= prereg["a3_residual_user_decision_min"]:
        verdict = "user_decision"
    else:
        verdict = "prompting_dominates"

    return A3Verdict(
        residual=residual,
        fixed_by=fixed_by,
        knowledge_fix_rate=knowledge_fix_rate,
        tripwire_fired=tripwire_fired,
        fragile_fix=fragile_fix,
        verdict=verdict,
        n_residual=n_residual,
    )
