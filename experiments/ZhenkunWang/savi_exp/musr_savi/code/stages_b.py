"""stages_b — MuSR-cant Phase B three-valued gate analyzers.

The mechanical verdict layer for Phase B's five-gate funnel + the H-B main readout
(design ``plans/2026-07-06-phaseB-design.md`` §4 decision table). Every function here is a
PURE function of the Subtask-6 feed (per-sample / per-item score & result dicts) and emits a
deterministic, json-able classification. NO file / network / GPU I/O, no model, no torch.

Self-contained by design: the ONLY imports are stdlib, ``numpy`` and ``sklearn`` (and,
optionally, ``sc_core`` for its Wilson CI — never imported by default, and NEVER any test
module). The bootstrap statistics are implemented LOCALLY here (seeded, reproducible) rather
than borrowed from other experiment dirs, so this analyzer stands alone.

Three-valued scheme (mirrors ``stages.py``'s dataclass-verdict style): each gate returns a
``GateResult(status, detail)`` with ``status`` drawn from
``{"PASS", "INSUFFICIENT", "FALSE", "TERMINATE"}`` — only the subset each gate can emit:

  * ``gate_g5a``          — PASS / INSUFFICIENT           (answer-level info-gate, never kills)
  * ``gate_preservation`` — PASS / INSUFFICIENT / TERMINATE   (hard-kill at sticky < 8)
  * ``gate_g5b``          — PASS / INSUFFICIENT           (step-level info-gate, never kills)
  * ``gate_g1_oracle``    — PASS / TERMINATE              (hard-kill at oracle-fixed <= 2)
  * ``gate_g2_merge``     — PASS / INSUFFICIENT           (INSUFFICIENT renames readout "soft-rerank")

The H-B three-valued readout (``readout_hb``) reports PASS / INSUFFICIENT / FALSE mapping to
phaseB_positive / null / negative in ``verdict_b``. ``gate_g3_marginal`` and
``select_lambda`` are diagnostics (no pass/fail). Every threshold lives in the module-level
``PREREG_B`` dict (single source of truth; NO scattered magic numbers), round-trip-checked by
``prereg_roundtrip`` (a3 convention).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

# ======================================================================================
# PREREG_B — single source of truth for every Phase B threshold (design §4 decision table).
# ======================================================================================
PREREG_B = {
    # ---- deterministic bootstrap config (fixed seed -> reproducible CIs) --------------
    "bootstrap_seed": 20260706,        # np.random.default_rng seed for EVERY bootstrap here
    "n_boot": 2000,                    # bootstrap resamples (design's CI protocol)
    "ci_alpha": 0.05,                  # two-sided 95% CI
    # ---- G5a answer-level discrimination info-gate (B0; NEVER terminates) -------------
    "g5a_auc_ci_floor": 0.5,           # pooled sample-level AUC CI lower must exceed this
    "g5a_net_fix_min": 2,              # verifier-argmax vs SC-mode net fix >= this
    # ---- preservation pre-check (B1 @ N=64; HARD-KILL below shrink_min) ---------------
    "preservation_pass_min": 15,       # sticky_count >= this -> PASS
    "preservation_shrink_min": 8,      # shrink_min..pass_min-1 -> INSUFFICIENT(shrink); < -> TERMINATE
    # ---- lambda grid (tuning pool; lambda* = argmax net_fix, tie -> smallest) ---------
    "lambda_grid": (0.1, 0.5, 1.0, 2.0),
    # ---- G5b step-level discrimination (B2; NEVER terminates) -------------------------
    "g5b_auc_ci_floor": 0.5,           # pooled step-level AUC CI lower must exceed this
    # ---- G1 oracle upper bound (B2; HARD-KILL at <= 2) --------------------------------
    "g1_oracle_fixed_min": 3,          # oracle-fixed patients >= this -> PASS; <= this-1 -> TERMINATE
    # ---- G2 merge validity (median < min -> rename readout "soft-rerank") -------------
    "g2_merge_median_min": 0.01,       # per-patient cross-chain merge-rate median >= this -> PASS
    # ---- H-B main readout (paired CI > floor AND net_fix >= min) ----------------------
    "hb_ci_floor": 0.0,                # paired bootstrap CI lower must exceed this
    "hb_net_fix_min": 3,              # SAVI-SC net fix over the primary set >= this
    # ---- patient set (a3 leftover; frozen fragile ids + expected sizes; diagnostics) --
    "fragile_ids": ("0032-q0", "0058-q2"),   # a3 fragile-fix ids -> H-B reported with/without
    "patient_residual_n": 21,          # residual patient count from a3 (completeness)
    "primary_nonzero_n": 17,           # non-zero-coverage primary readout size
    "zero_coverage_n": 4,              # zero-coverage sub-class size (reported separately)
}

# Status vocabulary (consistent three-valued scheme; "TERMINATE" where a gate can hard-kill).
PASS = "PASS"
INSUFFICIENT = "INSUFFICIENT"
FALSE = "FALSE"
TERMINATE = "TERMINATE"


@dataclass
class GateResult:
    """Three-valued gate verdict + a json-able ``detail`` payload (mirrors stages.py style)."""
    status: str                                   # PASS | INSUFFICIENT | FALSE | TERMINATE
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"status": self.status, "detail": self.detail}


# ======================================================================================
# Local, seeded statistics helpers (self-contained — NOT imported from other exp dirs)
# ======================================================================================
def roc_auc(scores, labels) -> float:
    """Pooled ROC-AUC of ``scores`` against boolean ``labels`` (sklearn ``roc_auc_score``).

    Degenerate inputs never raise: an empty input or an all-one-class label vector (AUC
    undefined) returns the chance value 0.5. Use ``_auc_degenerate(labels)`` to learn WHY a
    0.5 came back (empty / single-class) when a gate needs to flag it.
    """
    labels = [1 if bool(x) else 0 for x in labels]
    if len(labels) == 0 or len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, list(scores)))


def _auc_degenerate(labels) -> bool:
    """True iff ``labels`` cannot yield a defined AUC (empty or single-class)."""
    labs = [1 if bool(x) else 0 for x in labels]
    return len(labs) == 0 or len(set(labs)) < 2


def _rng(seed):
    return np.random.default_rng(PREREG_B["bootstrap_seed"] if seed is None else seed)


def bootstrap_ci(values, stat_fn, n_boot=None, seed=None, alpha=None):
    """Nonparametric percentile bootstrap CI of ``stat_fn`` over resamples of ``values``.

    ``values`` is any indexable sequence of elements (e.g. ``(score, label)`` pairs);
    ``stat_fn(resampled_list) -> float`` computes the statistic on each resample. Deterministic
    for a fixed ``seed`` (defaults to ``PREREG_B['bootstrap_seed']``). Empty ``values`` returns
    the documented sentinel ``(nan, nan)`` (a NaN CI never clears any ``> floor`` gate, so empty
    evidence is treated as INSUFFICIENT, never a spurious PASS).
    """
    n_boot = PREREG_B["n_boot"] if n_boot is None else n_boot
    alpha = PREREG_B["ci_alpha"] if alpha is None else alpha
    vals = list(values)
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = _rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        stats[b] = stat_fn([vals[i] for i in idx[b]])
    lo = float(np.percentile(stats, 100.0 * alpha / 2.0))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def paired_bootstrap_ci(deltas, n_boot=None, seed=None, alpha=None):
    """Percentile bootstrap CI of the NET FIX (sum of paired per-item ``deltas``).

    ``deltas`` are per-item paired differences in ``{-1, 0, +1}`` (e.g. ``savi - sc``); the
    statistic is their SUM (== net_fix), so the CI is on the same integer-scale quantity the
    H-B gate thresholds. Vectorised + deterministic for a fixed ``seed``. Empty -> ``(nan, nan)``.
    """
    n_boot = PREREG_B["n_boot"] if n_boot is None else n_boot
    alpha = PREREG_B["ci_alpha"] if alpha is None else alpha
    d = np.asarray(list(deltas), dtype=float)
    n = d.shape[0]
    if n == 0:
        return (float("nan"), float("nan"))
    rng = _rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = d[idx].sum(axis=1)
    lo = float(np.percentile(stats, 100.0 * alpha / 2.0))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def _auc_ci(scores, labels, seed=None):
    """Bootstrap 95% CI for the pooled AUC by resampling (score, label) pairs together."""
    pairs = list(zip(list(scores), [1 if bool(x) else 0 for x in labels]))

    def _stat(sample):
        return roc_auc([p[0] for p in sample], [p[1] for p in sample])

    return bootstrap_ci(pairs, _stat, seed=seed)


# ======================================================================================
# Gate 0 — zero-coverage sub-class (design §2: frozen out of the main readout)
# ======================================================================================
def zero_coverage_ids(patient_cache) -> list:
    """Item ids with ZERO correct samples in the Phase A-style cache (design §2, the 4 zero-cov).

    ``patient_cache`` = per-sample records, each at least ``{"item_id": str, "correct": bool}``
    (extra fields ignored). Returns the sorted ids that never produced a single correct sample
    — the sub-class no reranker can touch (fixing one is a "changed generation" finding).
    """
    seen = set()
    any_correct = set()
    for r in patient_cache:
        iid = r["item_id"]
        seen.add(iid)
        if r.get("correct"):
            any_correct.add(iid)
    return sorted(iid for iid in seen if iid not in any_correct)


# ======================================================================================
# G5a — answer-level discrimination (B0 info-gate; never TERMINATEs)
# ======================================================================================
def gate_g5a(sample_rows, sc_modes, seed=None) -> GateResult:
    """Answer-level verifier discrimination on the patient pool (design §4 / B0).

    ``sample_rows`` : list of ``{"item_id": str, "score": float, "correct": bool}`` — one row
                      per patient sample (the verifier's score + that sample's correctness).
    ``sc_modes``    : ``{item_id: sc_mode_correct(bool)}`` — SC's argmax-answer correctness.

    Two quantities: (i) pooled sample-level ROC-AUC (score vs correct) with a bootstrap 95% CI;
    (ii) ``net_fix`` = (#items whose verifier-argmax answer is correct) - (#items SC gets right),
    where the verifier-argmax answer per item is the HIGHEST-SCORED sample's ``correct``.

    PASS iff ``AUC CI lower > 0.5`` AND ``net_fix >= 2``; otherwise INSUFFICIENT. This gate is an
    INFO gate — it NEVER TERMINATEs; an INSUFFICIENT only lowers the downstream prior (recorded
    as ``detail['prior_downgraded'] = True``), and B1/B2 proceed regardless.
    """
    scores = [r["score"] for r in sample_rows]
    labels = [bool(r["correct"]) for r in sample_rows]
    auc = roc_auc(scores, labels)
    lo, hi = _auc_ci(scores, labels, seed=seed)

    # verifier-argmax answer per item = the correctness of the item's highest-scored sample.
    best = {}   # item_id -> (best_score, correct)
    order = {}  # first-seen order for deterministic tie-break
    for k, r in enumerate(sample_rows):
        iid = r["item_id"]
        order.setdefault(iid, k)
        if iid not in best or r["score"] > best[iid][0]:
            best[iid] = (r["score"], bool(r["correct"]))
    n_verifier = sum(1 for v in best.values() if v[1])
    n_sc = sum(1 for iid in best if sc_modes.get(iid, False))
    net_fix = n_verifier - n_sc

    auc_pass = (not math.isnan(lo)) and lo > PREREG_B["g5a_auc_ci_floor"]
    netfix_pass = net_fix >= PREREG_B["g5a_net_fix_min"]
    status = PASS if (auc_pass and netfix_pass) else INSUFFICIENT
    return GateResult(status=status, detail={
        "auc": auc,
        "auc_ci": (lo, hi),
        "auc_degenerate": _auc_degenerate(labels),
        "auc_pass": auc_pass,
        "net_fix": int(net_fix),
        "n_verifier_correct": int(n_verifier),
        "n_sc_correct": int(n_sc),
        "netfix_pass": netfix_pass,
        "n_items": len(best),
        "n_samples": len(sample_rows),
        # INFO gate: an INSUFFICIENT only downgrades the prior; B1/B2 still proceed.
        "prior_downgraded": status == INSUFFICIENT,
    })


# ======================================================================================
# Preservation pre-check (B1 @ N=64; HARD-KILL below shrink_min)
# ======================================================================================
def gate_preservation(pool_rows) -> GateResult:
    """Structured-template SC@64 patient-preservation pre-check (design §4 / B1, HARD-KILL gate).

    ``pool_rows`` : list of ``{"item_id": str, "sc_mode_correct": bool}`` (SC@64 under the frozen
    belief-table template). ``sticky_count`` = #items STILL wrong (the surviving patients).

    PASS iff ``sticky_count >= 15``; INSUFFICIENT if ``8 <= sticky_count <= 14`` (the main set
    SHRINKS to the sticky survivors — listed in ``detail['sticky_ids']``); TERMINATE if
    ``sticky_count < 8`` (the structured template un-stuck too many — line re-scoped by the user).
    The surviving sticky ids are always returned so the runner can shrink the primary set.
    """
    sticky_ids = sorted(r["item_id"] for r in pool_rows if not r["sc_mode_correct"])
    sticky_count = len(sticky_ids)
    if sticky_count >= PREREG_B["preservation_pass_min"]:
        status = PASS
    elif sticky_count >= PREREG_B["preservation_shrink_min"]:
        status = INSUFFICIENT
    else:
        status = TERMINATE
    return GateResult(status=status, detail={
        "sticky_count": sticky_count,
        "sticky_ids": sticky_ids,           # surviving patients (== shrunk main set when INSUFFICIENT)
        "n_items": len(pool_rows),
        "shrink": status == INSUFFICIENT,   # main set shrinks to sticky_ids
    })


# ======================================================================================
# select_lambda — tuning-pool net-fix argmax (tie -> smallest lambda, deterministic)
# ======================================================================================
def select_lambda(tuning_rows) -> dict:
    """Pick lambda* = the grid value maximising net_fix on the deferred tuning pool.

    ``tuning_rows`` : ``{lambda(float): net_fix(number)}``. Ties are broken deterministically by
    the SMALLEST lambda (least aggressive verifier weight). Returns
    ``{"lambda_star": <float>, "grid": {...}}``. Empty grid -> ``lambda_star`` None.
    """
    grid = dict(tuning_rows)
    if not grid:
        return {"lambda_star": None, "grid": grid}
    best = max(grid.values())
    lambda_star = min(lam for lam, nf in grid.items() if nf == best)   # tie -> smallest lambda
    return {"lambda_star": lambda_star, "grid": grid}


# ======================================================================================
# G5b — step-level discrimination (B2 info-gate; never TERMINATEs)
# ======================================================================================
def gate_g5b(step_rows, seed=None) -> GateResult:
    """Step-level verifier discrimination on the patient-pool steps (design §4 / B2).

    ``step_rows`` : list of ``{"score": float, "consistent": bool}`` — the step verifier's score
    vs the gold-step consistency label (from the facts tree). Pooled ROC-AUC + bootstrap 95% CI.

    PASS iff ``AUC CI lower > 0.5``; otherwise INSUFFICIENT. Like G5a this is an INFO gate — the
    main H-B readout still proceeds on an INSUFFICIENT (recorded as ``prior_downgraded``); it
    NEVER TERMINATEs.
    """
    scores = [r["score"] for r in step_rows]
    labels = [bool(r["consistent"]) for r in step_rows]
    auc = roc_auc(scores, labels)
    lo, hi = _auc_ci(scores, labels, seed=seed)
    auc_pass = (not math.isnan(lo)) and lo > PREREG_B["g5b_auc_ci_floor"]
    status = PASS if auc_pass else INSUFFICIENT
    return GateResult(status=status, detail={
        "auc": auc,
        "auc_ci": (lo, hi),
        "auc_degenerate": _auc_degenerate(labels),
        "n_steps": len(step_rows),
        "prior_downgraded": status == INSUFFICIENT,
    })


# ======================================================================================
# G1 — oracle upper bound (B2; HARD-KILL at oracle-fixed <= 2)
# ======================================================================================
def gate_g1_oracle(oracle_rows) -> GateResult:
    """Exact-mask trellis oracle upper bound on the patient pool (design §4 / B2, HARD-KILL gate).

    ``oracle_rows`` : list of ``{"item_id": str, "oracle_fixed": bool}`` — whether the EXACT
    (facts-tree) verifier + trellis fixes each patient. ``n_fixed`` = sum.

    PASS iff ``n_fixed >= 3`` (a learnable ceiling exists, quantified for the record); TERMINATE
    if ``n_fixed <= 2`` (the wall is on the GENERATOR side, not discrimination — terminate and
    emit the diagnostic; per design the full readout is still computed and carried in verdict_b).
    """
    fixed_ids = sorted(r["item_id"] for r in oracle_rows if r["oracle_fixed"])
    n_fixed = len(fixed_ids)
    status = PASS if n_fixed >= PREREG_B["g1_oracle_fixed_min"] else TERMINATE
    return GateResult(status=status, detail={
        "n_fixed": n_fixed,
        "fixed_ids": fixed_ids,
        "n_items": len(oracle_rows),
    })


# ======================================================================================
# G2 — merge validity (median merge rate; INSUFFICIENT renames readout "soft-rerank")
# ======================================================================================
def _merge_rate(x):
    return float(x["merge_rate"]) if isinstance(x, dict) else float(x)


def gate_g2_merge(merge_rows) -> GateResult:
    """Cross-chain state-merge validity (design §4 / B2).

    ``merge_rows`` : per-patient cross-chain merge rate — a list of floats (fraction of chain
    steps that Phi-merged), or dicts carrying a ``"merge_rate"`` key. ``median`` over patients.

    PASS iff ``median >= 0.01`` (the trellis is MORE than a reranker). INSUFFICIENT if
    ``median < 0.01`` (or no data): the trellis degenerated to a reranker, so the readout is
    honestly renamed — ``detail['rename_to'] = "soft-rerank"`` — and B2 proceeds under that name.
    """
    rates = [_merge_rate(x) for x in merge_rows]
    if not rates:
        median = float("nan")
        status = INSUFFICIENT
    else:
        median = float(np.median(rates))
        status = PASS if median >= PREREG_B["g2_merge_median_min"] else INSUFFICIENT
    detail = {"median": median, "n_items": len(rates)}
    if status == INSUFFICIENT:
        detail["rename_to"] = "soft-rerank"      # honest downgrade name (readout proceeds)
    return GateResult(status=status, detail=detail)


# ======================================================================================
# G3 — marginal coverage (diagnostic; no pass/fail)
# ======================================================================================
def gate_g3_marginal(marginal_rows) -> dict:
    """Marginal coverage among oracle-fixed patients (design §4 / B2, G0-style diagnostic).

    ``marginal_rows`` : per-fixed-patient booleans ``not_in_any_single_chain`` — whether the
    correct path the oracle-trellis reached lies OUTSIDE every single sampled chain. Returns
    ``{"marginal_coverage_rate": fraction, "n": int}`` (pure diagnostic; empty -> rate 0.0).
    """
    flags = [bool(x) for x in marginal_rows]
    n = len(flags)
    rate = (sum(flags) / n) if n else 0.0
    return {"marginal_coverage_rate": rate, "n": n}


# ======================================================================================
# readout_hb — the H-B main readout (three-valued PASS/INSUFFICIENT/FALSE)
# ======================================================================================
def _paired_block(rows, key_a, key_b, seed=None):
    """Net fix (sum of per-item ``a - b``) + paired bootstrap 95% CI over the rows."""
    deltas = [int(bool(r[key_a])) - int(bool(r[key_b])) for r in rows]
    net = int(sum(deltas))
    lo, hi = paired_bootstrap_ci(deltas, seed=seed)
    return net, (lo, hi)


def readout_hb(patient_results, zero_cov_ids, fragile_ids, seed=None) -> dict:
    """H-B main readout: SAVI@lambda* vs token-matched SC (+ SAVI vs BoN ablation) (design §4).

    ``patient_results`` : per-item ``{"item_id", "savi_correct", "sc_correct", "bon_correct"}``
                          over the full residual patient set.
    ``zero_cov_ids``    : the zero-coverage sub-class — EXCLUDED from the primary readout and
                          reported separately (SAVI fixes there == a "changed generation" finding).
    ``fragile_ids``     : a3 fragile-fix ids inside the primary set — the readout is reported
                          BOTH with and without them (sensitivity dual, zero extra compute).

    Primary readout = residual items minus ``zero_cov_ids``. Reports net_fix + paired bootstrap
    95% CI for SAVI-SC (headline) and SAVI-BoN (ablation), the fragile dual, and the zero-cov
    sub-class. ``hb_status``: PASS iff (SAVI-SC CI lower > 0 AND net_fix >= 3); FALSE iff the CI
    lies entirely below 0 (SC decisively wins); INSUFFICIENT otherwise (CI straddles 0).
    """
    zero_set = set(zero_cov_ids)
    frag_set = set(fragile_ids)

    primary = [r for r in patient_results if r["item_id"] not in zero_set]
    zero_rows = [r for r in patient_results if r["item_id"] in zero_set]
    without_frag = [r for r in primary if r["item_id"] not in frag_set]

    net_sc, ci_sc = _paired_block(primary, "savi_correct", "sc_correct", seed=seed)
    net_bon, ci_bon = _paired_block(primary, "savi_correct", "bon_correct", seed=seed)
    net_wo, ci_wo = _paired_block(without_frag, "savi_correct", "sc_correct", seed=seed)

    lo_sc, hi_sc = ci_sc
    hb_pass = (not math.isnan(lo_sc)) and lo_sc > PREREG_B["hb_ci_floor"] \
        and net_sc >= PREREG_B["hb_net_fix_min"]
    if hb_pass:
        hb_status = PASS
    elif (not math.isnan(hi_sc)) and hi_sc < 0:
        hb_status = FALSE                            # CI entirely below 0 -> SC decisively wins
    else:
        hb_status = INSUFFICIENT                     # CI straddles 0 (or net_fix < min)

    return {
        "primary": {
            "n_items": len(primary),
            "item_ids": sorted(r["item_id"] for r in primary),
            "net_fix_savi_sc": net_sc,
            "ci_savi_sc": ci_sc,
            "net_fix_savi_bon": net_bon,
            "ci_savi_bon": ci_bon,
        },
        "hb_status": hb_status,
        "hb_pass": hb_pass,
        "fragile_dual": {
            "fragile_ids_present": sorted(r["item_id"] for r in primary if r["item_id"] in frag_set),
            "with_fragile": {"n_items": len(primary), "net_fix": net_sc, "ci": ci_sc},
            "without_fragile": {"n_items": len(without_frag), "net_fix": net_wo, "ci": ci_wo},
        },
        "zero_coverage": {
            "n_items": len(zero_rows),
            "item_ids": sorted(r["item_id"] for r in zero_rows),
            # SAVI fixes among the zero-cov sub-class (expected 0 — any fix is a big finding).
            "savi_fixed": int(sum(1 for r in zero_rows if r["savi_correct"])),
        },
    }


# ======================================================================================
# verdict_b — assemble every gate into an overall three-valued verdict (design §4)
# ======================================================================================
def _prior(gate):
    if gate is None:
        return "absent"
    if gate.status == PASS:
        return "supported"
    return "downgraded"          # INSUFFICIENT -> prior lowered (info gates never terminate)


def verdict_b(gates: dict) -> dict:
    """Assemble the five gates + H-B readout into the overall three-valued Phase B verdict.

    ``gates`` keys (all optional; absent ones are skipped): ``"g5a"``, ``"preservation"``,
    ``"g5b"``, ``"g1_oracle"``, ``"g2_merge"`` -> ``GateResult``; ``"g3_marginal"`` -> dict;
    ``"readout"`` -> the ``readout_hb`` dict.

    Termination logic (design §4): if preservation TERMINATEs OR G1 TERMINATEs, the overall
    verdict is ``"terminated"`` with the reason(s) — BUT the full readout is still carried when it
    was computed (G1 terminates AFTER B2's readout, so its termination still emits the readout).
    Otherwise the verdict mirrors the H-B status: PASS -> ``phaseB_positive``,
    INSUFFICIENT -> ``phaseB_null``, FALSE -> ``phaseB_negative`` — annotated with the G5a/G5b
    priors and the G2 rename flag.
    """
    g5a = gates.get("g5a")
    pres = gates.get("preservation")
    g5b = gates.get("g5b")
    g1 = gates.get("g1_oracle")
    g2 = gates.get("g2_merge")
    readout = gates.get("readout")

    gate_status = {}
    for name in ("g5a", "preservation", "g5b", "g1_oracle", "g2_merge"):
        gr = gates.get(name)
        if gr is not None:
            gate_status[name] = gr.status

    priors = {"g5a": _prior(g5a), "g5b": _prior(g5b)}
    g2_rename = (g2 is not None and g2.status == INSUFFICIENT)

    out = {
        "gate_status": gate_status,
        "priors": priors,
        "g2_rename": g2_rename,     # True -> readout honestly renamed "soft-rerank vs SC"
    }

    # --- termination (hard-kill gates); design: G1-terminate still emits the full readout ------
    reasons = []
    if pres is not None and pres.status == TERMINATE:
        reasons.append("preservation_terminated_sticky_below_%d" % PREREG_B["preservation_shrink_min"])
    if g1 is not None and g1.status == TERMINATE:
        reasons.append("g1_oracle_terminated_fixed_below_%d" % PREREG_B["g1_oracle_fixed_min"])
    if reasons:
        out["verdict"] = "terminated"
        out["termination_reasons"] = reasons
        if readout is not None:
            out["readout"] = readout        # carry whatever readout was computed (G1 case)
            out["hb"] = _hb_echo(readout)
        return out

    # --- non-terminated: verdict mirrors the H-B status ---------------------------------------
    if readout is None:
        out["verdict"] = "incomplete"       # no readout and no termination -> nothing to decide
        return out
    hb_status = readout["hb_status"]
    out["verdict"] = {
        PASS: "phaseB_positive",
        INSUFFICIENT: "phaseB_null",
        FALSE: "phaseB_negative",
    }[hb_status]
    out["readout"] = readout
    out["hb"] = _hb_echo(readout)
    return out


def _hb_echo(readout):
    prim = readout["primary"]
    return {
        "status": readout["hb_status"],
        "net_fix_savi_sc": prim["net_fix_savi_sc"],
        "ci_savi_sc": prim["ci_savi_sc"],
    }


# ======================================================================================
# prereg_roundtrip — consistency guard (a3 convention)
# ======================================================================================
def _prereg_diff(a, b):
    """Key-wise diff of two json-normalized PREREG dicts (missing marked ``<MISSING>``)."""
    keys = sorted(set(a) | set(b))
    return {k: {"loaded": a.get(k, "<MISSING>"), "code": b.get(k, "<MISSING>")}
            for k in keys if a.get(k, "<MISSING>") != b.get(k, "<MISSING>")}


def prereg_roundtrip(prereg_dict, live_prereg=None) -> bool:
    """Assert a loaded PREREG dict's thresholds byte-match ``PREREG_B`` (freeze == testing).

    JSON has no tuples, so both sides are compared through a ``json.dumps/loads`` normalization
    (tuples -> lists), exactly like the a3 round-trip test. Returns True on a match; raises
    ``AssertionError`` naming every diverging key on a mismatch.
    """
    live = PREREG_B if live_prereg is None else live_prereg
    loaded = json.loads(json.dumps(dict(prereg_dict)))
    code = json.loads(json.dumps(dict(live)))
    if loaded != code:
        raise AssertionError("PREREG_B mismatch: %s" % json.dumps(_prereg_diff(loaded, code)))
    return True
