"""stages_g3 — the g3 event-line ablation freeze + station-A analyzers (design
``plans/2026-07-09-eventline-ablation-design.md`` §1/§3 + plan
``plans/2026-07-09-eventline-ablation.md``). g2 ended VERDICT=partial because the GSD
template bundles TWO ingredients — the belief scaffold (``BELIEF[t-1]``) and a per-step
fact feed (the ``Event t: …`` line). g3 removes the second ingredient (arms ``anchor`` =
a content-free index-only anchor line, ``noevent`` = the event-line block deleted) and
re-reads two stations:

  * **Station A** (this module's analyzers) audits the EXACT-GSD headline: of λ=0
    TF-MAP's 10/19 fixes (results_g1, E-full), how many SURVIVE event-line removal?
    ``survival = |map-fixed(noevent) ∩ original10_ids|``; ≥7 -> ``scaffold_carries``,
    ≤3 -> ``facts_carry``, 4..6 -> ``mixed``. A map-fixed knowledge control in ANY arm
    raises a NAMED anomaly flag (three-arm knowledge expectation is 0/3; the flag never
    changes the label).
  * **Station B** (Subtask 4) re-runs the g2 sampling probe under the ablated templates
    via the REUSED ``stages_g2.arm2_all`` — which is why ``PREREG_G3`` carries the exact
    prereg keys arm2_all reads (``repro7_ids`` / ``zero_cov_ctrl_ids`` /
    ``knowledge_ctrl_ids`` / ``epsilon_backoff`` / ``epsilon_sensitivity`` /
    ``parse_ok_min`` / ``node_coverage_min``).

``PREREG_G3`` is the single source of truth for every g3 frozen constant (the plan's
frozen-constants block) and round-trips through ``outputs/PREREG_g3.md`` exactly the way
``stages_g2.PREREG_G2`` round-trips (freezing == testing). The four template shas are
frozen LITERALS asserted at import time against the live ``gsd_score`` constants — a
template edit makes this module unimportable rather than silently drifting the freeze.

Every analyzer here is a PURE function of already-loaded data structures (the per-item
rows produced by ``stages_g.g1_item_analysis``, a loaded results_g1 dict) and returns a
deterministic, json-able dict — the ``stages_g`` convention. NO file / GPU / model /
torch I/O lives here beyond the prereg markdown write/load; the thin
``run_musr_cant.py --stage g3`` shell (Subtask 5) builds spaces, drives scorer/sampler,
and feeds these analyzers.

CODE SEPARATION: imports ONLY stdlib (``hashlib`` / ``json`` / ``math``) + the pure
``gsd_score`` module-level sha constants (gsd_score imports torch lazily inside its
scorer class, never at module scope) + the equally pure ``stages_g`` (``_median``) and
``stages_g2`` (the REUSED ``arm2_all`` — station B changes NOTHING about the decode).
matplotlib only ever loads lazily inside the ``fig_*`` functions (the ``stages_g``
``_plt`` convention). NEVER imports test code, torch, or transformers.
"""
from __future__ import annotations

import hashlib
import json
import math

import gsd_score
import stages_g as sg
import stages_g2


# ======================================================================================
# PREREG_G3 — the g3 freeze (the plan's frozen-constants block; frozen BEFORE the g3 full run)
# ======================================================================================
PREREG_G3 = {
    # ---- arms & frozen templates (sha LITERALS; E-full = g1/g2 reuse, never re-run) ------
    "arms": ["anchor", "noevent"],
    "template_shas": {
        "anchor": "aac31c1da8ca9741be1de5c0621d6eacbedc8155fec6c2aeb5b36cab7e68d9e2",
        "noevent": "15e4a6cc0f4f5a51720f44673fe68dab60cc1c664711202c6da650919616ba5e",
        "main": "fccd448b715ccd781bae624dd2109eb322fe42fd8494543924c4570dbb21a74a",
        "root": "80c48de4e1d1b92e67ed7930249e4deef18282c80c5886ac2f29208400a0330f",
    },
    # ---- station A: the 10 λ=0 map-fixed ids frozen from results_g1 (survival numerator
    # universe; = repro7 ∪ zero_cov — the 3 zero-coverage controls are λ=0 fixes too) ------
    "original10_ids": [
        "object_placements-0006-q3", "object_placements-0011-q2",
        "object_placements-0014-q0", "object_placements-0021-q0",
        "object_placements-0042-q1", "object_placements-0043-q2",
        "object_placements-0047-q2", "object_placements-0050-q1",
        "object_placements-0051-q2", "object_placements-0055-q3",
    ],
    "stationA_scaffold_min": 7,   # survival >= 7 -> scaffold_carries
    "stationA_facts_max": 3,      # survival <= 3 -> facts_carry; 4..6 -> mixed
    # ---- station B id lists (verbatim from outputs/PREREG_g2.md; arm2_all classifies) ----
    "repro7_ids": [
        "object_placements-0006-q3", "object_placements-0011-q2",
        "object_placements-0042-q1", "object_placements-0043-q2",
        "object_placements-0047-q2", "object_placements-0050-q1",
        "object_placements-0051-q2",
    ],
    "zero_cov_ctrl_ids": [
        "object_placements-0014-q0", "object_placements-0021-q0",
        "object_placements-0055-q3",
    ],
    "knowledge_ctrl_ids": [
        "object_placements-0020-q1", "object_placements-0034-q3",
        "object_placements-0035-q0",
    ],
    # ---- station B ordered gate (design §3; first hit stops) -----------------------------
    "stationB_confound_min": 2,   # 1) knowledge_repro >= 2 -> confound_persists
    "stationB_facts_max": 2,      # 2) n_repro <= 2         -> facts_carry_probe
    "stationB_scaffold_min": 5,   # 3) n_repro >= 5 AND knowledge_repro <= 1
                                  #    -> scaffold_specific; 4) else mixed
    # ---- sampler config (station B; the only GPU-heavy station) --------------------------
    "sample_M": 64,               # g2's 128 halved (budget); --smoke -> 8
    "sample_base_seed": 20260708,
    "seed_tag": "g3sample",       # BOTH arms share one seed sequence = paired sampling
    "sample_max_new_tokens": 256,
    "parse_ok_min": 0.8,          # per-item parse-ok rate gate (fail-closed)
    "node_coverage_min": 16,      # 32@M=128 scaled to the same 25% at M=64
    "epsilon_backoff": 1e-6,      # A_freq = log(count + eps); the MAIN caliber
    "epsilon_sensitivity": [1e-5, 1e-6, 1e-7],   # eps sweep (descriptive)
    # ---- frozen continuous-readout definitions (Subtask 4 implements these verbatim) -----
    "onehot_def": "freqs 恰有 1 个非零后继且 n_parse_ok > 0",
    "entropy_def": ("Shannon(freq/sum(freq)) nats;逐节点,"
                    "报 pool×arm×层型(move/no-move) 中位数"),
    # ---- E-full baselines (preflight: recomputed from results_g1/results_g2 and asserted
    # byte-for-byte equal — guards against baseline drift) ---------------------------------
    "baseline_stationA": {"map_fixed": 10, "oracle_fixed": 15, "knowledge_fixed": 0},
    "baseline_stationB": {"n_repro": 7, "zero_cov": 3, "knowledge": 2},
}

# The frozen sha LITERALS must equal the live gsd_score template constants — fail-loud at
# import (plain raise, survives ``python -O``): a template edit invalidates the freeze.
for _name, _live in (("anchor", gsd_score.E_ANCHOR_TEMPLATE_SHA256),
                     ("noevent", gsd_score.E_NONE_TEMPLATE_SHA256),
                     ("main", gsd_score.GSD_TEMPLATE_SHA256),
                     ("root", gsd_score.GSD_ROOT_TEMPLATE_SHA256)):
    if PREREG_G3["template_shas"][_name] != _live:
        raise AssertionError(
            "PREREG_G3 template sha drift for %r: frozen %s != live gsd_score %s"
            % (_name, PREREG_G3["template_shas"][_name], _live))
del _name, _live

# station-A label vocabulary (design §3; three-valued, mechanical).
LABEL_SCAFFOLD_CARRIES = "scaffold_carries"
LABEL_FACTS_CARRY = "facts_carry"
LABEL_MIXED = "mixed"


# ======================================================================================
# PREREG_g3 freeze — write + round-trip check (mirrors stages_g2's prereg pattern)
# ======================================================================================
def _prereg_g3_diff(loaded: dict, code: dict) -> dict:
    """Top-level key-wise diff of two json-normalized PREREG_G3 dicts (mirrors
    ``stages_g2._prereg_g2_diff``)."""
    keys = sorted(set(loaded) | set(code))
    return {k: {"loaded": loaded.get(k, "<MISSING>"), "code": code.get(k, "<MISSING>")}
            for k in keys if loaded.get(k, "<MISSING>") != code.get(k, "<MISSING>")}


def write_prereg_g3(path: str) -> str:
    """Freeze the g3 pre-registration to ``path`` (PREREG_g3.md): a fenced ```json
    PREREG_G3 block + a human-readable summary of the frozen ids / gates / templates.
    Round-tripped by ``check_prereg_g3_roundtrip`` (freezing == testing; the
    ``stages_g2.write_prereg_g2`` convention)."""
    payload = json.loads(json.dumps(PREREG_G3))     # tuples -> lists (json-normalized)
    md = [
        "# MuSR-cant event-line ablation pre-registration (PREREG_g3)",
        "",
        "Frozen BEFORE the g3 (stage g3) full run. Every id / threshold / seed / template "
        "sha below is the single source of truth in `stages_g3.PREREG_G3`; this file is "
        "round-trip checked against the code constant by "
        "`stages_g3.check_prereg_g3_roundtrip` (the `stages_g2` PREREG convention). "
        "Design: `plans/2026-07-09-eventline-ablation.md` (冻结常量) + `…-design.md` §3.",
        "",
        "## Frozen constants (`stages_g3.PREREG_G3`)",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        "",
        "## Station A verdict (design §3, mechanical; decision arm = noevent)",
        "",
        "- `survival = |map-fixed(E-none) ∩ original10_ids|` (denominator 10).",
        "- **scaffold_carries**: `survival >= %d` (stationA_scaffold_min)."
        % (PREREG_G3["stationA_scaffold_min"],),
        "- **facts_carry**: `survival <= %d` (stationA_facts_max)."
        % (PREREG_G3["stationA_facts_max"],),
        "- **mixed**: 4..6.",
        "- Knowledge controls stay 0/3 in EVERY arm; any map-fixed knowledge item raises "
        "the anomaly flag (named ids; the label is untouched).",
        "",
        "## Station B verdict (design §3, ordered — first hit stops; main arm = noevent)",
        "",
        "- 1) **confound_persists**: `knowledge_repro >= %d` (stationB_confound_min)."
        % (PREREG_G3["stationB_confound_min"],),
        "- 2) **facts_carry_probe**: `n_repro <= %d` (stationB_facts_max)."
        % (PREREG_G3["stationB_facts_max"],),
        "- 3) **scaffold_specific**: `n_repro >= %d` (stationB_scaffold_min) AND "
        "`knowledge_repro <= 1`." % (PREREG_G3["stationB_scaffold_min"],),
        "- 4) **mixed**: otherwise. zero_cov reproduction is a REPORTED finding, not an "
        "assertion (the g2 correction); fail-closed exclusions are named and leave the "
        "denominator.",
        "",
        "## Frozen ids",
        "",
        "- original λ=0 map fixes (station-A survival universe): `%s`"
        % (list(PREREG_G3["original10_ids"]),),
        "- reproducible-7 (station-B n_repro denominator): `%s`"
        % (list(PREREG_G3["repro7_ids"]),),
        "- zero-coverage controls (reported, not asserted): `%s`"
        % (list(PREREG_G3["zero_cov_ctrl_ids"]),),
        "- knowledge controls (specificity): `%s`"
        % (list(PREREG_G3["knowledge_ctrl_ids"]),),
        "",
        "## Sampler config (station B)",
        "",
        "- draws per branching node `sample_M` = %d (g2's 128 halved); "
        "`sample_max_new_tokens` = %d; `sample_base_seed` = %d; `seed_tag` = `%s` "
        "(both arms share one seed sequence = paired sampling)."
        % (PREREG_G3["sample_M"], PREREG_G3["sample_max_new_tokens"],
           PREREG_G3["sample_base_seed"], PREREG_G3["seed_tag"]),
        "- add-eps backoff `epsilon_backoff` = %s (main); sensitivity sweep = %s "
        "(descriptive)."
        % (PREREG_G3["epsilon_backoff"], PREREG_G3["epsilon_sensitivity"]),
        "- fail-closed gates: `parse_ok_min` = %.1f; `node_coverage_min` = %d "
        "(32@M=128 scaled to the same 25%%)."
        % (PREREG_G3["parse_ok_min"], PREREG_G3["node_coverage_min"]),
        "",
        "## Frozen templates (sha256 literals, asserted at import against `gsd_score`)",
        "",
        "- arm `anchor` (`E_ANCHOR_TEMPLATE`): `%s`"
        % PREREG_G3["template_shas"]["anchor"],
        "- arm `noevent` (`E_NONE_TEMPLATE`): `%s`"
        % PREREG_G3["template_shas"]["noevent"],
        "- E-full baseline (`GSD_TEMPLATE`): `%s`" % PREREG_G3["template_shas"]["main"],
        "- root layer (`GSD_ROOT_TEMPLATE`, shared by all arms): `%s`"
        % PREREG_G3["template_shas"]["root"],
        "",
        "## E-full baselines (preflight byte-for-byte assertions)",
        "",
        "- station A (results_g1): `%s`."
        % (json.dumps(PREREG_G3["baseline_stationA"], sort_keys=True),),
        "- station B (results_g2 arm2): `%s`."
        % (json.dumps(PREREG_G3["baseline_stationB"], sort_keys=True),),
        "",
        "## Frozen continuous-readout definitions (station B, Subtask 4)",
        "",
        "- one-hot: `%s`" % (PREREG_G3["onehot_def"],),
        "- entropy: `%s`" % (PREREG_G3["entropy_def"],),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def load_prereg_g3(path: str) -> dict:
    """Parse the fenced ```json PREREG_G3 block out of a PREREG_g3.md -> dict (mirrors
    ``stages_g2.load_prereg_g2``)."""
    with open(path) as f:
        text = f.read()
    start = text.find("```json")
    if start == -1:
        raise ValueError("load_prereg_g3: no ```json block in %s" % path)
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise ValueError("load_prereg_g3: unterminated ```json block in %s" % path)
    return json.loads(text[start:end])


def check_prereg_g3_roundtrip(path: str) -> bool:
    """Assert the frozen PREREG_g3.md still matches ``PREREG_G3`` (json-normalized).
    True on a match; AssertionError naming every drifted top-level key on a mismatch
    (the ``stages_g2.check_prereg_g2_roundtrip`` convention)."""
    loaded = load_prereg_g3(path)
    code = json.loads(json.dumps(PREREG_G3))
    if loaded != code:
        raise AssertionError("PREREG_G3 mismatch: %s"
                             % json.dumps(_prereg_g3_diff(loaded, code)))
    return True


# ======================================================================================
# station A — per-arm tally + mechanical survival verdict + E-full baseline preflight
# ======================================================================================
def stationA_arm(rows, prereg: dict = PREREG_G3) -> dict:
    """Tally ONE station-A arm from its 22 per-item rows (19 patients + 3 knowledge).

    ``rows`` — ``stages_g.g1_item_analysis`` outputs for the arm (the integration layer
    re-scores + re-decodes each item under the arm's template and calls g1_item_analysis;
    only ``item_id`` / ``map.fixed`` / ``oracle.fixed`` are read here). ``prereg`` —
    supplies ``knowledge_ctrl_ids`` (knowledge membership; defaults to the frozen
    ``PREREG_G3``, injectable for tests).

    ``n_map_fixed`` / ``n_oracle_fixed`` count over the 19 PATIENTS ONLY — knowledge
    controls never enter these counts (they mirror results_g1's 10/19 and 15/19 axes).
    ``knowledge_fixed_ids`` names the map-fixed knowledge ids — the ``stationA_verdict``
    anomaly criterion (an oracle-fixed knowledge item is visible in ``per_item``).

    Returns ``{"per_item": [{"item_id", "bucket", "map_fixed", "oracle_fixed"}, …]
    (item_id-sorted), "n_map_fixed", "n_oracle_fixed", "knowledge_fixed_ids"}``.
    Deterministic and json-able.
    """
    knowledge_set = {str(x) for x in prereg["knowledge_ctrl_ids"]}
    per_item = sorted(
        ({"item_id": str(r["item_id"]),
          "bucket": "knowledge" if str(r["item_id"]) in knowledge_set else "patient",
          "map_fixed": bool(r["map"]["fixed"]),
          "oracle_fixed": bool(r["oracle"]["fixed"])} for r in rows),
        key=lambda p: p["item_id"])
    patients = [p for p in per_item if p["bucket"] == "patient"]
    return {
        "per_item": per_item,
        "n_map_fixed": sum(1 for p in patients if p["map_fixed"]),
        "n_oracle_fixed": sum(1 for p in patients if p["oracle_fixed"]),
        "knowledge_fixed_ids": sorted(p["item_id"] for p in per_item
                                      if p["bucket"] == "knowledge" and p["map_fixed"]),
    }


def stationA_verdict(arm_results_by_arm: dict, prereg: dict) -> dict:
    """Mechanically execute design §3's station-A verdict over the per-arm tallies.

    ``arm_results_by_arm`` — ``{arm_name: stationA_arm(...)}``; MUST contain the decision
    arm ``"noevent"`` (E-anchor is a descriptive secondary readout; E-full lives in
    results_g1 and is never re-run). ``prereg`` — carries ``original10_ids`` (the frozen
    survival universe) + ``stationA_scaffold_min`` (7) / ``stationA_facts_max`` (3).

    ``survival`` = |{noevent map-fixed item ids} ∩ original10_ids|; a noevent fix OUTSIDE
    the original 10 never counts. Label: survival >= scaffold_min -> ``scaffold_carries``;
    <= facts_max -> ``facts_carry``; else ``mixed``. ``oracle_ceiling_by_arm`` = each
    arm's 19-patient oracle-fixed count (the 15/19 upper-bound axis, descriptive).
    ``knowledge_anomaly_flag`` = True iff ANY arm map-fixed ANY knowledge control
    (expected 0/3 everywhere); the offending ids are NAMED per arm in
    ``knowledge_anomaly_ids`` and the label is never altered by the flag.

    Returns ``{"survival", "survived_ids", "lost_ids", "label", "oracle_ceiling_by_arm",
    "knowledge_anomaly_flag", "knowledge_anomaly_ids"}`` with sorted id lists.
    Deterministic and json-able.
    """
    if "noevent" not in arm_results_by_arm:
        raise ValueError("stationA_verdict: decision arm 'noevent' missing from "
                         "arm_results_by_arm (got %r)" % (sorted(arm_results_by_arm),))
    original10 = {str(x) for x in prereg["original10_ids"]}
    noevent_fixed = {p["item_id"] for p in arm_results_by_arm["noevent"]["per_item"]
                     if p["map_fixed"]}
    survived_ids = sorted(original10 & noevent_fixed)
    lost_ids = sorted(original10 - noevent_fixed)
    survival = len(survived_ids)

    if survival >= prereg["stationA_scaffold_min"]:
        label = LABEL_SCAFFOLD_CARRIES
    elif survival <= prereg["stationA_facts_max"]:
        label = LABEL_FACTS_CARRY
    else:
        label = LABEL_MIXED

    knowledge_anomaly_ids = {arm: list(res["knowledge_fixed_ids"])
                             for arm, res in sorted(arm_results_by_arm.items())}
    return {
        "survival": survival,
        "survived_ids": survived_ids,
        "lost_ids": lost_ids,
        "label": label,
        "oracle_ceiling_by_arm": {arm: res["n_oracle_fixed"]
                                  for arm, res in sorted(arm_results_by_arm.items())},
        "knowledge_anomaly_flag": any(knowledge_anomaly_ids.values()),
        "knowledge_anomaly_ids": knowledge_anomaly_ids,
    }


def stationA_baseline_check(results_g1: dict, prereg: dict) -> None:
    """Preflight: assert the E-full station-A baseline recomputed from a loaded
    results_g1 dict equals ``prereg["baseline_stationA"]`` byte-for-byte (anti-drift;
    plan Evaluation design's hard preflight).

    Recomputes ``map_fixed`` / ``oracle_fixed`` over ``per_item.patient`` (the 19-patient
    ``map.fixed`` / ``oracle.fixed`` counts) and ``knowledge_fixed`` over
    ``per_item.knowledge`` (``map.fixed`` OR ``oracle.fixed`` — expected 0). Returns None
    when all three match; raises ValueError naming every mismatched count otherwise.
    """
    baseline = prereg["baseline_stationA"]
    patients = results_g1["per_item"]["patient"]
    knowledge = results_g1["per_item"]["knowledge"]
    recomputed = {
        "map_fixed": sum(1 for r in patients if r["map"]["fixed"]),
        "oracle_fixed": sum(1 for r in patients if r["oracle"]["fixed"]),
        "knowledge_fixed": sum(1 for r in knowledge
                               if r["map"]["fixed"] or r["oracle"]["fixed"]),
    }
    mismatches = {k: {"prereg": baseline[k], "results_g1": recomputed[k]}
                  for k in sorted(baseline) if baseline[k] != recomputed[k]}
    if mismatches:
        raise ValueError(
            "stationA_baseline_check: results_g1 E-full baseline drifted from "
            "PREREG_G3 baseline_stationA: %s" % json.dumps(mismatches, sort_keys=True))
    return None


# ======================================================================================
# ======================================================================================
# Subtask 4 — station B (probe rescue) + continuous readout + figures
# ======================================================================================
# ======================================================================================
# Station B re-runs the g2 sampling probe under the ablated templates. The DECODE machine
# is the frozen ``stages_g2.arm2_all`` reused verbatim (``stationB_arm`` is a thin
# delegation — PREREG_G3 carries the exact prereg keys arm2_all reads); the NEW instrument
# is the continuous readout: the strict one-hot share and the per-node successor entropy
# (frozen definitions in ``PREREG_G3["onehot_def"]`` / ``["entropy_def"]``). In g2
# (E-full), ~52% of patient-pool and ~55% of knowledge-pool nodes were strict one-hot —
# if the event line carried the facts, the knowledge-pool one-hot share should collapse
# under E-none while the patient pool holds (design §3's directional prediction).

# station-B label vocabulary (design §3; ORDERED gates — first hit stops; the fourth
# label reuses LABEL_MIXED).
LABEL_CONFOUND_PERSISTS = "confound_persists"
LABEL_FACTS_CARRY_PROBE = "facts_carry_probe"
LABEL_SCAFFOLD_SPECIFIC = "scaffold_specific"


def _sha256(text: str) -> str:
    """Full sha256 hex of a canon string (the ``gsd_score._sha256`` hashing idiom; the
    per_node ``sprev_sha`` key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stationB_arm(items_spaces_samples, prereg: dict = PREREG_G3) -> dict:
    """ONE station-B arm = the REUSED ``stages_g2.arm2_all`` on that arm's sample cache
    (design §2.3: station B changes ONLY the sampling template; the frequency-decode
    machine, fail-closed gates, anti-leak invariant and eps sweep are g2's, unchanged).

    ``items_spaces_samples`` — ``[{"item_id", "item", "space", "sample_nodes"}, …]`` for
    the 13 decode items, where ``sample_nodes`` was drawn under THIS arm's template.
    ``prereg`` — defaults to the frozen ``PREREG_G3``, which carries the exact keys
    arm2_all reads (``repro7_ids`` / ``zero_cov_ctrl_ids`` / ``knowledge_ctrl_ids`` /
    ``epsilon_backoff`` / ``epsilon_sensitivity`` / ``parse_ok_min`` /
    ``node_coverage_min``). Returns the arm2_all dict verbatim (fail-closed exclusions
    already leave the n_repro denominator inside arm2_all).
    """
    return stages_g2.arm2_all(items_spaces_samples, prereg)


def _stationB_headline(arm2: dict) -> dict:
    """One arm's headline numbers for the verdict's ``per_arm`` block: reproduction over
    the (exclusion-shrunk) denominator, the zero-cov / knowledge reproduced counts, and
    the NAMED fail-closed exclusions."""
    return {
        "n_repro": arm2["n_repro"],
        "n_repro_denominator": arm2["n_repro_denominator"],
        "n_zero_cov_reproduced": sum(1 for f in arm2["zero_cov"].values() if f),
        "knowledge_repro": sum(1 for f in arm2["knowledge"].values() if f),
        "n_excluded": len(arm2["excluded_inconclusive"]),
        "excluded_ids": sorted(str(e["item_id"]) for e in arm2["excluded_inconclusive"]),
    }


def stationB_verdict(arm2_by_arm: dict, prereg: dict) -> dict:
    """Mechanically execute design §3's station-B verdict — ORDERED gates, first hit
    stops — over the per-arm ``arm2_all`` outputs.

    ``arm2_by_arm`` — ``{arm_name: stationB_arm(...)}``; MUST contain the decision arm
    ``"noevent"`` (E-anchor is a descriptive secondary readout). ``prereg`` — carries
    ``stationB_confound_min`` (2) / ``stationB_facts_max`` (2) / ``stationB_scaffold_min``
    (5). ``knowledge_repro`` = fixed=True count over the arm's ``knowledge`` dict;
    ``n_repro`` = the arm's headline reproduction count (fail-closed exclusions already
    left the denominator inside arm2_all).

    Ordered gates on the noevent arm (design §3 — the ORDER is load-bearing: a confounded
    probe, gate 1, is unreadable and must preempt every downstream interpretation):
      1. ``knowledge_repro >= stationB_confound_min``            -> ``confound_persists``
      2. ``n_repro <= stationB_facts_max``                       -> ``facts_carry_probe``
      3. ``n_repro >= stationB_scaffold_min AND knowledge <= 1`` -> ``scaffold_specific``
      4. otherwise                                               -> ``mixed``

    Returns ``{"label", "reason", "per_arm"}`` — the label, a plain string naming the
    decision numbers, and BOTH arms' headline numbers (``_stationB_headline``).
    Deterministic and json-able.
    """
    if "noevent" not in arm2_by_arm:
        raise ValueError("stationB_verdict: decision arm 'noevent' missing from "
                         "arm2_by_arm (got %r)" % (sorted(arm2_by_arm),))
    confound_min = prereg["stationB_confound_min"]
    facts_max = prereg["stationB_facts_max"]
    scaffold_min = prereg["stationB_scaffold_min"]

    per_arm = {arm: _stationB_headline(res) for arm, res in sorted(arm2_by_arm.items())}
    dec = per_arm["noevent"]
    n_repro, denom = dec["n_repro"], dec["n_repro_denominator"]
    knowledge_repro = dec["knowledge_repro"]
    n_knowledge = len(arm2_by_arm["noevent"]["knowledge"])

    # ---- ORDERED gates: the first hit stops (design §3) ----------------------------------
    if knowledge_repro >= confound_min:                              # gate 1
        label = LABEL_CONFOUND_PERSISTS
        reason = ("confound_persists: noevent knowledge_repro=%d/%d (>= confound_min %d) "
                  "— the sampling probe stays confounded even without the event line "
                  "(n_repro=%d/%d is unreadable as belief evidence)."
                  % (knowledge_repro, n_knowledge, confound_min, n_repro, denom))
    elif n_repro <= facts_max:                                       # gate 2
        label = LABEL_FACTS_CARRY_PROBE
        reason = ("facts_carry_probe: noevent n_repro=%d/%d (<= facts_max %d) with "
                  "knowledge_repro=%d/%d — the event line carried the probe's "
                  "reproductions; the fixes are fact injection, not scaffold."
                  % (n_repro, denom, facts_max, knowledge_repro, n_knowledge))
    elif n_repro >= scaffold_min and knowledge_repro <= 1:           # gate 3
        label = LABEL_SCAFFOLD_SPECIFIC
        reason = ("scaffold_specific: noevent n_repro=%d/%d (>= scaffold_min %d) AND "
                  "knowledge_repro=%d/%d (<= 1) — the belief scaffold carries the probe "
                  "and its specificity is restored."
                  % (n_repro, denom, scaffold_min, knowledge_repro, n_knowledge))
    else:                                                            # gate 4
        label = LABEL_MIXED
        reason = ("mixed: noevent n_repro=%d/%d and knowledge_repro=%d/%d fall between "
                  "the frozen gates (confound_min %d / facts_max %d / scaffold_min %d)."
                  % (n_repro, denom, knowledge_repro, n_knowledge,
                     confound_min, facts_max, scaffold_min))

    return {"label": label, "reason": reason, "per_arm": per_arm}


# ======================================================================================
# continuous readout — strict one-hot share + per-node successor entropy (frozen defs)
# ======================================================================================
def continuous_readout(sample_nodes_by_item, space_by_item, pool_of) -> dict:
    """The load-bearing quantitative instrument (design §3's continuous secondary readout): per sampled
    node, the strict one-hot bit and the successor-frequency Shannon entropy, rolled up
    by pool × layer kind.

    ``sample_nodes_by_item`` — ``{item_id: sample_node_freqs(...)["nodes"]}`` i.e.
    ``{(t, canon_prev): {"freqs": {canon_next: int}, "off_manifold", "n_parse_ok",
    "n_total"}}`` (NOTE: the ``"nodes"`` dict, not the full sampler output).
    ``space_by_item`` — ``{item_id: GsdSpace}`` (supplies ``moves`` for layer_kind and
    ``trans`` for n_succ — the gold tree is NEVER read).
    ``pool_of`` — callable ``item_id -> pool name`` (``'patient'`` | ``'knowledge'``);
    the caller derives it from the frozen prereg lists (repro7 ∪ zero_cov = patient,
    knowledge_ctrl = knowledge).

    Frozen per-node definitions (``PREREG_G3["onehot_def"]`` / ``["entropy_def"]``):
      * ``onehot`` — ``freqs`` has EXACTLY 1 nonzero successor count AND ``n_parse_ok >
        0`` (freqs counts only on-manifold draws, so a parse-ok off-manifold-only node is
        NOT one-hot);
      * ``entropy`` — Shannon entropy in nats of ``freqs`` normalized by its sum; exactly
        ``0.0`` for a one-hot node; ``None`` when the freqs sum to 0 (undefined — never a
        coerced 0.0);
      * ``layer_kind`` — ``'move'`` iff ``space.moves`` carries any move with layer == t,
        else ``'nomove'``.

    Returns ``{"per_node": [{"item_id", "t", "sprev_sha" (full sha256 of canon_prev),
    "pool", "layer_kind", "n_succ", "onehot", "entropy", "n_parse_ok", "n_on_manifold"},
    …], "summary": {pool: {layer_kind|'all': {"onehot_share", "entropy_median",
    "n_nodes"}}}}``. ``onehot_share`` is over ALL nodes of the cell (a zero-on-manifold
    node counts as NOT one-hot); ``entropy_median`` is over the cell's non-None entropies
    (None when empty); each pool also carries an ``'all'`` layer-kind rollup over both
    kinds. ``per_node`` is (item_id, t, sprev_sha)-sorted. Deterministic and json-able
    (the (t, canon_prev) tuple keys are flattened into row fields).
    """
    per_node: list = []
    for item_id in sorted(sample_nodes_by_item, key=str):
        nodes = sample_nodes_by_item[item_id]
        space = space_by_item[item_id]
        pool = str(pool_of(item_id))
        move_layers = {mv[0] for mv in space.moves}
        for (t, cp), node in nodes.items():
            freqs = node.get("freqs", {})
            n_parse_ok = node.get("n_parse_ok", 0)
            n_on = sum(freqs.values())
            nonzero = [v for v in freqs.values() if v > 0]
            if n_on <= 0:
                entropy = None                       # undefined, never a coerced 0.0
            elif len(nonzero) == 1:
                entropy = 0.0                        # exact zero for a one-hot node
            else:
                entropy = -sum((v / n_on) * math.log(v / n_on) for v in nonzero)
            per_node.append({
                "item_id": str(item_id),
                "t": t,
                "sprev_sha": _sha256(cp),
                "pool": pool,
                "layer_kind": "move" if t in move_layers else "nomove",
                "n_succ": len(space.trans.get((t, cp), ())),
                "onehot": bool(len(nonzero) == 1 and n_parse_ok > 0),
                "entropy": entropy,
                "n_parse_ok": n_parse_ok,
                "n_on_manifold": n_on,
            })
    per_node.sort(key=lambda r: (r["item_id"], r["t"], r["sprev_sha"]))

    cells: dict = {}
    for row in per_node:
        for kind in (row["layer_kind"], "all"):      # 'all' = the per-pool rollup cell
            cells.setdefault((row["pool"], kind), []).append(row)
    summary: dict = {}
    for (pool, kind) in sorted(cells):
        rows = cells[(pool, kind)]
        entropies = [r["entropy"] for r in rows if r["entropy"] is not None]
        summary.setdefault(pool, {})[kind] = {
            "onehot_share": sum(1 for r in rows if r["onehot"]) / len(rows),
            "entropy_median": sg._median(entropies),
            "n_nodes": len(rows),
        }
    return {"per_node": per_node, "summary": summary}


# ======================================================================================
# figures (matplotlib, Agg, lazily imported; ONE png per call at the explicit out_path;
# each SKIPS gracefully — returns None — when matplotlib is unavailable; EMPTY inputs
# still write a "no data" figure, never a crash)
# ======================================================================================
_COL_SURVIVED = "#2ca02c"   # station-A survived / kept (green)
_COL_LOST = "#d62728"       # station-A lost (red)
_COL_INK = "#444444"        # labels/annotations wear ink, never a series color
_ARM_COLORS = {"full": "#c7c7c7", "anchor": "#ff7f0e", "noevent": "#4c78a8"}
_ARM_FALLBACK = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_FIG_DPI = 150

# fig_fix_matrix column spec: (row-dict key, column label, frozen denominator). The
# denominators are the frozen readout universes (19 patients / repro7 / two 3-controls).
_FIX_MATRIX_COLS = [
    ("map_fixed", "stn-A map\n/19", 19),
    ("oracle_fixed", "stn-A oracle\n/19", 19),
    ("n_repro", "stn-B repro\n/7", 7),
    ("zero_cov", "zero-cov\n/3", 3),
    ("knowledge", "knowledge\n/3", 3),
]


def _plt():
    """Lazy headless matplotlib (Agg set BEFORE pyplot import; project convention)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _try_plt():
    """``_plt()`` or None when matplotlib is unavailable (graceful skip, the stages_g2
    figure-guard convention)."""
    try:
        return _plt()
    except ImportError:
        return None


def _arm_color(arm: str, i: int) -> str:
    """The frozen arm color (full gray / anchor orange / noevent blue); unknown arm
    names fall back to a stable cycle."""
    return _ARM_COLORS.get(arm, _ARM_FALLBACK[i % len(_ARM_FALLBACK)])


def _no_data(ax, note: str = "no data") -> None:
    """The defensive empty-input rendering: a centered note on a blank axes."""
    ax.annotate(note, (0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                fontsize=10, color=_COL_INK)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_fix_matrix(matrix_rows, out_path: str):
    """The two-station fix/reproduction matrix: rows = arms (E-full baseline / anchor /
    noevent), cols = the five frozen readouts (``_FIX_MATRIX_COLS``). ``matrix_rows`` is
    a PREPARED ``[(row_label, {col_key: int|None}), …]`` — the caller assembles the
    E-full row from the PREREG_G3 baselines + g2 numbers, keeping this function pure.
    Cell shade = value / frozen denominator; a None/missing readout renders ``n/a``.
    Empty rows -> a "no data" figure. Returns ``out_path`` (None w/o matplotlib)."""
    plt = _try_plt()
    if plt is None:
        return None
    rows = list(matrix_rows)
    n_cols = len(_FIX_MATRIX_COLS)
    fig, ax = plt.subplots(figsize=(7.4, 1.6 + 0.75 * max(1, len(rows))), dpi=_FIG_DPI)
    if not rows:
        _no_data(ax)
    else:
        for r, (label, vals) in enumerate(rows):
            y = len(rows) - 1 - r                    # top row first
            for c, (key, _clabel, denom) in enumerate(_FIX_MATRIX_COLS):
                v = (vals or {}).get(key)
                if v is None:
                    color, txt = "#eeeeee", "n/a"
                else:
                    frac = max(0.0, min(1.0, v / float(denom)))
                    color = (1.0 - 0.55 * frac, 1.0 - 0.25 * frac, 1.0)   # white -> blue
                    txt = "%d/%d" % (v, denom)
                ax.add_patch(plt.Rectangle((c + 0.03, y + 0.03), 0.94, 0.94,
                                           facecolor=color, edgecolor="white", lw=0.8))
                ax.annotate(txt, (c + 0.5, y + 0.5), ha="center", va="center",
                            fontsize=8, color=_COL_INK)
        ax.set_xlim(0.0, n_cols)
        ax.set_ylim(0.0, len(rows))
        ax.set_xticks([c + 0.5 for c in range(n_cols)])
        ax.set_xticklabels([cl for _k, cl, _d in _FIX_MATRIX_COLS], fontsize=7.5)
        ax.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
        ax.set_yticklabels([str(label) for label, _v in rows], fontsize=8)
        ax.tick_params(length=0)
    ax.set_title("g3 event-line ablation — fix / reproduction matrix (arms x readouts)",
                 fontsize=9)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_onehot_share(summary_by_arm: dict, out_path: str):
    """Grouped one-hot-share bars: x = the pool × layer_kind cells (incl. each pool's
    'all' rollup), series = arms (``summary_by_arm = {arm: continuous_readout(...)
    ["summary"]}``; the caller computes the 'full' baseline from the g2 cache). A cell
    missing from an arm renders a zero-height bar. Empty input -> "no data". Returns
    ``out_path`` (None w/o matplotlib)."""
    plt = _try_plt()
    if plt is None:
        return None
    arms = sorted(summary_by_arm)
    cells = sorted({(pool, kind) for arm in arms
                    for pool, kinds in summary_by_arm[arm].items()
                    for kind in kinds})
    fig, ax = plt.subplots(figsize=(max(5.8, 1.15 * len(cells) + 2.0), 4.0),
                           dpi=_FIG_DPI)
    if not arms or not cells:
        _no_data(ax)
    else:
        width = 0.8 / len(arms)
        for i, arm in enumerate(arms):
            xs, ys = [], []
            for j, (pool, kind) in enumerate(cells):
                cell = summary_by_arm[arm].get(pool, {}).get(kind, {})
                share = cell.get("onehot_share")
                xs.append(j - 0.4 + (i + 0.5) * width)
                ys.append(0.0 if share is None else share)
            ax.bar(xs, ys, width=width * 0.92, color=_arm_color(arm, i), label=arm)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(["%s\n%s" % (pool, kind) for pool, kind in cells],
                           fontsize=7.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("strict one-hot share", fontsize=8)
        ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("station B — one-hot share by pool x layer kind (series = arm)",
                 fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_entropy_deepdive(per_node_rows_by_arm: dict, item_ids, out_path: str):
    """Per-layer entropy deep-dive for the chosen items (design §3: one patient + one
    knowledge — the localisation prediction): one panel per item, x = layer t, y = the
    median non-None per-node entropy at that layer, one line per arm
    (``per_node_rows_by_arm = {arm: continuous_readout(...)["per_node"]}``). A panel
    with no defined entropies (or empty inputs) renders a "no data" note. Returns
    ``out_path`` (None w/o matplotlib)."""
    plt = _try_plt()
    if plt is None:
        return None
    item_ids = [str(i) for i in item_ids]
    arms = sorted(per_node_rows_by_arm)
    n_panels = max(1, len(item_ids))
    fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 3.6), dpi=_FIG_DPI,
                             squeeze=False)
    axes = axes[0]
    if not item_ids:
        _no_data(axes[0])
    else:
        for ax, iid in zip(axes, item_ids):
            plotted = False
            all_ts: set = set()
            for i, arm in enumerate(arms):
                by_t: dict = {}
                for r in per_node_rows_by_arm[arm]:
                    if r["item_id"] == iid and r["entropy"] is not None:
                        by_t.setdefault(r["t"], []).append(r["entropy"])
                ts = sorted(by_t)
                if not ts:
                    continue
                ax.plot(ts, [sg._median(by_t[t]) for t in ts], marker="o", ms=4,
                        lw=1.5, color=_arm_color(arm, i), label=arm)
                all_ts.update(ts)
                plotted = True
            if plotted:
                ax.set_xticks(sorted(all_ts))        # layer t is an integer axis
                ax.set_xlabel("layer t", fontsize=8)
                ax.set_ylabel("median entropy (nats)", fontsize=8)
                ax.legend(fontsize=7, frameon=False)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            else:
                _no_data(ax)
            ax.set_title(iid, fontsize=8)
    fig.suptitle("station B — per-layer sampled-successor entropy (deep-dive)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_survival_waterfall(stationA_verdict_res: dict, out_path: str):
    """Station-A survival waterfall: one bar per original-10 λ=0-fixed id — green =
    survived (noevent map-fixed), red = lost — ordered survived-then-lost from the
    verdict's sorted id lists. An empty verdict renders a "no data" note. Returns
    ``out_path`` (None w/o matplotlib)."""
    plt = _try_plt()
    if plt is None:
        return None
    survived = [str(x) for x in stationA_verdict_res.get("survived_ids", ())]
    lost = [str(x) for x in stationA_verdict_res.get("lost_ids", ())]
    ids = survived + lost
    fig, ax = plt.subplots(figsize=(max(6.6, 0.66 * len(ids) + 1.6), 3.4), dpi=_FIG_DPI)
    if not ids:
        _no_data(ax)
    else:
        survived_set = set(survived)
        for i, iid in enumerate(ids):
            kept = iid in survived_set
            ax.bar(i, 1.0, width=0.78,
                   color=_COL_SURVIVED if kept else _COL_LOST)
            ax.annotate("kept" if kept else "lost", (i, 0.5), ha="center", va="center",
                        fontsize=6.5, color="white", rotation=90)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([iid.replace("object_placements-", "") for iid in ids],
                           fontsize=6.5, rotation=45, ha="right")
        ax.set_yticks([])
    survival = stationA_verdict_res.get("survival")
    label = stationA_verdict_res.get("label")
    ax.set_title("station A — noevent survival of the original λ=0 fixes: %s/%d (%s)"
                 % ("n/a" if survival is None else survival, max(1, len(ids)) if ids
                    else 0, label if label else "n/a"), fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
