"""stages_g4 — the g4 step-accuracy + self-parsed-event-line freeze and the station-C
analyzers (design ``plans/2026-07-09-g4-step-parsed-design.md`` §1/§2 + plan
``plans/2026-07-09-g4-step-parsed.md`` Subtask 2). g3 split the old headline in two:
whole-path fixes lean on the per-step fact feed, but the SINGLE-step belief update runs
near-deterministically on the scaffold alone — and only on the commitment-failure items.
g4 asks the two follow-ups:

  * **Station C** (this module's analyzers, ZERO GPU): "near-deterministic" is not
    "correct". On the GOLD-PREFIX branching nodes — sampled nodes (t, canon_prev) whose
    canon_prev equals the canon of ``facts_oracle.gold_beliefs(item)[t-1]`` — how often
    does the modal sampled successor equal the gold successor (``modal_acc``), and how
    much sampled mass sits on it (``mass_on_gold``)? Read per pool (patient/knowledge)
    per arm (full/anchor/noevent, all replayed from the frozen g2/g3 sample caches).
    Decision gate on the noevent arm: Δ_acc = modal_acc(patient) − modal_acc(knowledge);
    ≥ 0.2 -> ``step_specific``, ≤ 0.05 -> ``no_step_signal``, else ``mixed``.
    The gold trajectory is an EVALUATION-side readout only — it never enters any model
    input (non-leak; the sample caches were drawn long before this module existed).
  * **Station D** (analyzers below): the self-parsed event-line rerun. Its frozen
    constants (extract template sha, seed/greedy/max_new_tokens, the alignment rule,
    stage ``g4d_parsed``, the 7/3 survival gate, the quality def) are frozen HERE — the
    whole prereg freezes at once (``extract_template_sha`` was deferred to Subtask 3 so
    it could freeze the REAL ``gsd_extract`` literal, never a placeholder). The
    mechanical pieces: ``_SpaceView`` (the moves-injection proxy behind
    ``gsd_score.build_prefix``), ``stationD_prefix_plan`` (the frozen per-layer E-none
    degradation plan for unparsed layers), ``stationD_verdict`` (survival =
    |map-fixed ∩ original10|, gate 7/3, plus the ``gained_ids`` churn readout) and
    ``quality_recovery_table`` (the extraction-quality × recovery cross table that
    decides which explanation carries).

``PREREG_G4`` is the single source of truth for every g4 frozen constant (the plan's
frozen-constants block) and round-trips through ``outputs/PREREG_g4.md`` exactly the way
``stages_g3.PREREG_G3`` round-trips (freezing == testing). The id literals shared with
earlier stages (original10 / knowledge3 / decode13 = repro7 ∪ zero_cov ∪ knowledge / the
g1 baseline) are frozen LITERALS asserted at import time against ``stages_g3.PREREG_G3``
— a drift makes this module unimportable rather than silently forking the freeze.
``patients19_ids`` is frozen from the results_g1 ``per_item.patient`` scope (read once,
hard-coded; never re-read at import time).

Every analyzer here is a PURE function of already-loaded data structures (the per-item
``sample_node_freqs(...)["nodes"]`` dicts, item dicts, ``GsdSpace``-shaped objects) and
returns a deterministic, json-able dict — the ``stages_g`` convention. NO file / GPU /
model / torch I/O lives here beyond the prereg markdown write/load; the thin
``run_musr_cant.py --stage g4`` shell (Subtask 4) builds spaces, replays caches, and
feeds these analyzers.

CODE SEPARATION: imports ONLY stdlib (``hashlib`` / ``json`` / ``math``) + the pure
``belief_schema`` (``canon_state`` — the shared canon key) + the pure ``facts_oracle``
(``gold_beliefs`` — the evaluation-side gold source) + the equally pure ``stages_g``
(``spearman_rho``) and ``stages_g3`` (the id-literal cross-assertions only).
``gsd_extract`` (itself stdlib + sc_core only — torch/transformers-free) is imported
ONLY inside the ``extract_template_sha`` import-time assertion, keeping this module's
declared dependency list to the freeze surfaces. matplotlib only ever loads lazily
inside the ``fig_*`` functions (the ``stages_g`` ``_plt`` convention). NEVER imports
test code, torch, or transformers.
"""
from __future__ import annotations

import hashlib
import json
import math

import belief_schema
import facts_oracle
import stages_g as sg
import stages_g3


# ======================================================================================
# PREREG_G4 — the g4 freeze (the plan's frozen-constants block; frozen BEFORE the g4 full run)
# ======================================================================================
PREREG_G4 = {
    # ---- station C: arms + the EXACT frozen cache coordinates (pure replay; the g4 run
    # re-opens these caches with a raising-stub emitter and hard-asserts n_scored==0) ----
    "stationC_arms": ["full", "anchor", "noevent"],   # decision arm = noevent
    "stationC_cache_coords": {
        "full": {"cache": "cache_g2_sample.jsonl", "M": 128, "stage": "g2",
                 "seed_tag": "g2sample", "template": "main"},
        "anchor": {"cache": "cache_g3_sample_anchor.jsonl", "M": 64,
                   "stage": "g3_anchor", "seed_tag": "g3sample", "template": "anchor"},
        "noevent": {"cache": "cache_g3_sample_noevent.jsonl", "M": 64,
                    "stage": "g3_noevent", "seed_tag": "g3sample",
                    "template": "noevent"},
    },
    # ---- station C: frozen readout definitions + the Δ_acc gate -------------------------
    "modal_acc_def": ("金标前缀分支节点上 argmax freq 后继 canon == 金标后继 canon 的"
                      "节点比例(并列取 canon 字典序最小,冻结;freq 全零节点记不中)"),
    "mass_on_gold_def": "freq[gold_succ]/Σfreq 的节点均值(freq 全零节点记 0)",
    "stationC_delta_min": 0.2,    # Δ_acc >= 0.2  -> step_specific
    "stationC_delta_null": 0.05,  # Δ_acc <= 0.05 -> no_step_signal; in between -> mixed
    # ---- station D: frozen NOW too (the whole prereg freezes at once; analyzers below).
    # 'extract_template_sha' — the frozen sha256 LITERAL of gsd_extract.EXTRACT_TEMPLATE,
    # asserted at import time against the live constant (the stages_g3 template-sha
    # pattern). Its freeze was DEFERRED from Subtask 2 to Subtask 3: gsd_extract was
    # created concurrently (Subtask 1) and a placeholder would have been a fake freeze.
    "extract_template_sha":
        "26660e0568e51ebaa2c68b968a95d2f2882e2457a3817e44b866025c5ef277d3",
    "extract_seed": 20260709,
    "extract_greedy": True,
    "extract_max_new_tokens": 512,
    "alignment_rule": "层 t 用解析第 t 条;缺 -> 该层 E-none(无事件行);多余丢弃;不剔题",
    "stationD_stage": "g4d_parsed",   # 6-tuple cache-key isolation from g1 (template sha == main)
    "stationD_deploy_min": 7,     # survival >= 7 -> parse_deployable
    "stationD_wall_max": 3,       # survival <= 3 -> parse_insufficient; in between -> mixed
    "quality_def": ("逐层 (obj,loc) 大小写无关 exact-match vs space.moves;"
                    "逐题 all-layers-exact bool"),
    # ---- shared id literals (verbatim from PREREG_g2/PREREG_g3; asserted vs stages_g3
    # at import time). decode13 = repro7 + zero_cov3 + knowledge3, in that frozen order. --
    "decode13_ids": [
        "object_placements-0006-q3", "object_placements-0011-q2",
        "object_placements-0042-q1", "object_placements-0043-q2",
        "object_placements-0047-q2", "object_placements-0050-q1",
        "object_placements-0051-q2",
        "object_placements-0014-q0", "object_placements-0021-q0",
        "object_placements-0055-q3",
        "object_placements-0020-q1", "object_placements-0034-q3",
        "object_placements-0035-q0",
    ],
    "knowledge_ctrl_ids": [
        "object_placements-0020-q1", "object_placements-0034-q3",
        "object_placements-0035-q0",
    ],
    "original10_ids": [
        "object_placements-0006-q3", "object_placements-0011-q2",
        "object_placements-0014-q0", "object_placements-0021-q0",
        "object_placements-0042-q1", "object_placements-0043-q2",
        "object_placements-0047-q2", "object_placements-0050-q1",
        "object_placements-0051-q2", "object_placements-0055-q3",
    ],
    # ---- patients19: the results_g1 per_item.patient scope (frozen from
    # outputs/results_g1.json, read ONCE and hard-coded; never re-read at import) --------
    "patients19_ids": [
        "object_placements-0000-q2", "object_placements-0003-q3",
        "object_placements-0006-q2", "object_placements-0006-q3",
        "object_placements-0011-q2", "object_placements-0013-q1",
        "object_placements-0014-q0", "object_placements-0018-q0",
        "object_placements-0021-q0", "object_placements-0032-q0",
        "object_placements-0042-q0", "object_placements-0042-q1",
        "object_placements-0043-q2", "object_placements-0045-q0",
        "object_placements-0046-q0", "object_placements-0047-q2",
        "object_placements-0050-q1", "object_placements-0051-q2",
        "object_placements-0055-q3",
    ],
    # ---- preflight baselines (byte-for-byte assertions against loaded results files) ----
    "baseline_g1": {"map_fixed": 10, "oracle_fixed": 15, "knowledge_fixed": 0},
    "baseline_g3_noevent": {
        "survival": 1,
        "map_fixed_ids": ["object_placements-0000-q2", "object_placements-0003-q3",
                          "object_placements-0051-q2"],
    },
}

# The shared id literals must equal the g3 freeze — fail-loud at import (plain raise,
# survives ``python -O``): a fork of the frozen lists invalidates every cross-stage
# comparison. patients19 has no g3 counterpart; it gets internal-consistency checks.
_G3 = stages_g3.PREREG_G3
for _key, _expected in (
        ("original10_ids", _G3["original10_ids"]),
        ("knowledge_ctrl_ids", _G3["knowledge_ctrl_ids"]),
        ("decode13_ids", _G3["repro7_ids"] + _G3["zero_cov_ctrl_ids"]
                         + _G3["knowledge_ctrl_ids"]),
        ("baseline_g1", _G3["baseline_stationA"])):
    if PREREG_G4[_key] != _expected:
        raise AssertionError(
            "PREREG_G4 id-literal drift for %r: frozen %r != stages_g3.PREREG_G3 %r"
            % (_key, PREREG_G4[_key], _expected))
del _key, _expected
if (len(PREREG_G4["patients19_ids"]) != 19
        or PREREG_G4["patients19_ids"] != sorted(PREREG_G4["patients19_ids"])
        or not set(PREREG_G4["original10_ids"]) <= set(PREREG_G4["patients19_ids"])
        or set(PREREG_G4["knowledge_ctrl_ids"]) & set(PREREG_G4["patients19_ids"])):
    raise AssertionError(
        "PREREG_G4 patients19_ids inconsistent: need 19 sorted patient ids containing "
        "original10 and disjoint from the knowledge controls; got %r"
        % (PREREG_G4["patients19_ids"],))
if (len(set(PREREG_G4["baseline_g3_noevent"]["map_fixed_ids"])
        & set(PREREG_G4["original10_ids"]))
        != PREREG_G4["baseline_g3_noevent"]["survival"]):
    raise AssertionError(
        "PREREG_G4 baseline_g3_noevent inconsistent: survival %r != |map_fixed_ids ∩ "
        "original10_ids|" % (PREREG_G4["baseline_g3_noevent"],))


def _assert_extract_template_sha() -> None:
    """The frozen ``extract_template_sha`` LITERAL must equal the live
    ``gsd_extract.EXTRACT_TEMPLATE_SHA256`` — fail-loud at import (plain raise, survives
    ``python -O``): an extraction-template edit invalidates the station-D freeze. The
    import is DELIBERATELY inside this function (run once, right below): gsd_extract is
    itself pure (stdlib + sc_core, zero torch/transformers at module scope), and keeping
    it out of the module-scope import list keeps this module's declared dependencies to
    the freeze surfaces only (the stages_g3 template-sha pattern, one module over)."""
    import gsd_extract
    if PREREG_G4["extract_template_sha"] != gsd_extract.EXTRACT_TEMPLATE_SHA256:
        raise AssertionError(
            "PREREG_G4 extract_template_sha drift: frozen %s != live gsd_extract %s"
            % (PREREG_G4["extract_template_sha"],
               gsd_extract.EXTRACT_TEMPLATE_SHA256))


_assert_extract_template_sha()

# station-C label vocabulary (design §2, mechanical; mixed shared with stages_g3).
LABEL_STEP_SPECIFIC = "step_specific"
LABEL_NO_STEP_SIGNAL = "no_step_signal"
LABEL_MIXED = stages_g3.LABEL_MIXED    # "mixed"

# station-D label vocabulary (design §2, mechanical; mixed shared with station C/g3).
LABEL_PARSE_DEPLOYABLE = "parse_deployable"
LABEL_PARSE_INSUFFICIENT = "parse_insufficient"

# the two NAMED goldprefix exclusion reasons (design §2: neither is an error)
EXCLUDE_NO_GOLD = "no_gold_trajectory"
EXCLUDE_NOT_SAMPLED = "gold_prefix_not_sampled"


# ======================================================================================
# PREREG_g4 freeze — write + round-trip check (mirrors stages_g3's prereg pattern)
# ======================================================================================
def _prereg_g4_diff(loaded: dict, code: dict) -> dict:
    """Top-level key-wise diff of two json-normalized PREREG_G4 dicts (mirrors
    ``stages_g3._prereg_g3_diff``)."""
    keys = sorted(set(loaded) | set(code))
    return {k: {"loaded": loaded.get(k, "<MISSING>"), "code": code.get(k, "<MISSING>")}
            for k in keys if loaded.get(k, "<MISSING>") != code.get(k, "<MISSING>")}


def write_prereg_g4(path: str) -> str:
    """Freeze the g4 pre-registration to ``path`` (PREREG_g4.md): a fenced ```json
    PREREG_G4 block + a human-readable summary of the frozen ids / gates / coordinates.
    Round-tripped by ``check_prereg_g4_roundtrip`` (freezing == testing; the
    ``stages_g3.write_prereg_g3`` convention). Key-set agnostic — Subtask 3's addition
    of ``extract_template_sha`` re-freezes through the same machinery."""
    payload = json.loads(json.dumps(PREREG_G4))     # tuples -> lists (json-normalized)
    md = [
        "# MuSR-cant g4 step-accuracy + self-parsed event-line pre-registration "
        "(PREREG_g4)",
        "",
        "Frozen BEFORE the g4 (stage g4) full run. Every id / threshold / seed / cache "
        "coordinate below is the single source of truth in `stages_g4.PREREG_G4`; this "
        "file is round-trip checked against the code constant by "
        "`stages_g4.check_prereg_g4_roundtrip` (the `stages_g3` PREREG convention). "
        "Design: `plans/2026-07-09-g4-step-parsed.md` (冻结常量) + `…-design.md` §1/§2. "
        "NOTE: `extract_template_sha` was deferred to Subtask 3 (gsd_extract landed "
        "concurrently; a placeholder would have been a fake freeze) — it is now frozen "
        "below and asserted at import time against the live "
        "`gsd_extract.EXTRACT_TEMPLATE_SHA256`.",
        "",
        "## Frozen constants (`stages_g4.PREREG_G4`)",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        "",
        "## Station C verdict (design §2, mechanical; decision arm = noevent)",
        "",
        "- gold-prefix branching nodes: sampled (t, canon_prev) with canon_prev == "
        "canon(gold_beliefs[t-1]); gold successor = canon(gold_beliefs[t]). The gold "
        "trajectory is an evaluation-side readout ONLY — it never enters a model input.",
        "- `modal_acc`: %s" % (PREREG_G4["modal_acc_def"],),
        "- `mass_on_gold`: %s" % (PREREG_G4["mass_on_gold_def"],),
        "- `delta_acc = modal_acc(patient) - modal_acc(knowledge)` on the noevent arm.",
        "- **step_specific**: `delta_acc >= %s` (stationC_delta_min)."
        % (PREREG_G4["stationC_delta_min"],),
        "- **no_step_signal**: `delta_acc <= %s` (stationC_delta_null)."
        % (PREREG_G4["stationC_delta_null"],),
        "- **mixed**: otherwise.",
        "- named exclusions (NOT errors): `no_gold_trajectory` / "
        "`gold_prefix_not_sampled`.",
        "",
        "## Station C cache coordinates (pure replay; 0 fresh draws hard-asserted)",
        "",
    ] + [
        "- arm `%s`: `%s`" % (arm, json.dumps(
            PREREG_G4["stationC_cache_coords"][arm], sort_keys=True))
        for arm in PREREG_G4["stationC_arms"]
    ] + [
        "",
        "## Station D verdict (design §2, mechanical; decision arm = the parsed rerun)",
        "",
        "- extraction: greedy=%s, seed=%d, max_new_tokens=%d; input = narrative ONLY "
        "(zero oracle)."
        % (PREREG_G4["extract_greedy"], PREREG_G4["extract_seed"],
           PREREG_G4["extract_max_new_tokens"]),
        "- extraction template (`gsd_extract.EXTRACT_TEMPLATE`, sha256 asserted at "
        "import against the live constant): `%s`"
        % (PREREG_G4["extract_template_sha"],),
        "- alignment rule: %s" % (PREREG_G4["alignment_rule"],),
        "- scoring stage: `%s` (cache-key isolation; template sha == main)."
        % (PREREG_G4["stationD_stage"],),
        "- **parse_deployable**: `survival >= %d` (stationD_deploy_min)."
        % (PREREG_G4["stationD_deploy_min"],),
        "- **parse_insufficient**: `survival <= %d` (stationD_wall_max)."
        % (PREREG_G4["stationD_wall_max"],),
        "- **mixed**: 4..6. survival = |map-fixed(parsed) ∩ original10_ids|.",
        "- extraction quality: %s" % (PREREG_G4["quality_def"],),
        "",
        "## Frozen ids",
        "",
        "- decode13 (station-C node universe; repro7 + zero_cov3 + knowledge3): `%s`"
        % (list(PREREG_G4["decode13_ids"]),),
        "- patients19 (station-D scope; the results_g1 patient universe): `%s`"
        % (list(PREREG_G4["patients19_ids"]),),
        "- original λ=0 map fixes (station-D survival universe): `%s`"
        % (list(PREREG_G4["original10_ids"]),),
        "- knowledge controls (specificity): `%s`"
        % (list(PREREG_G4["knowledge_ctrl_ids"]),),
        "",
        "## Baselines (preflight byte-for-byte assertions)",
        "",
        "- results_g1 (E-full): `%s`."
        % (json.dumps(PREREG_G4["baseline_g1"], sort_keys=True),),
        "- results_g3 noevent (station A): `%s`."
        % (json.dumps(PREREG_G4["baseline_g3_noevent"], sort_keys=True),),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def load_prereg_g4(path: str) -> dict:
    """Parse the fenced ```json PREREG_G4 block out of a PREREG_g4.md -> dict (mirrors
    ``stages_g3.load_prereg_g3``)."""
    with open(path) as f:
        text = f.read()
    start = text.find("```json")
    if start == -1:
        raise ValueError("load_prereg_g4: no ```json block in %s" % path)
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise ValueError("load_prereg_g4: unterminated ```json block in %s" % path)
    return json.loads(text[start:end])


def check_prereg_g4_roundtrip(path: str) -> bool:
    """Assert the frozen PREREG_g4.md still matches ``PREREG_G4`` (json-normalized).
    True on a match; AssertionError naming every drifted top-level key on a mismatch
    (the ``stages_g3.check_prereg_g3_roundtrip`` convention)."""
    loaded = load_prereg_g4(path)
    code = json.loads(json.dumps(PREREG_G4))
    if loaded != code:
        raise AssertionError("PREREG_G4 mismatch: %s"
                             % json.dumps(_prereg_g4_diff(loaded, code)))
    return True


# ======================================================================================
# station C — gold-prefix node selection + per-arm accuracy tally + mechanical verdict
# ======================================================================================
def _sha256(text: str) -> str:
    """Full sha256 hex of a canon string (the ``stages_g3._sha256`` hashing idiom; keeps
    the long canon strings out of the json rows)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def goldprefix_nodes(item, space, sample_nodes) -> dict:
    """Select the GOLD-PREFIX branching nodes of one item among its SAMPLED nodes.

    For every layer t in 1..T-1 (T = ``space.T``, the enumerated layer count): the
    gold prefix state is ``facts_oracle.gold_beliefs(item)[t-1]`` and the gold successor
    is ``gold_beliefs(item)[t]``, both canonicalized via ``belief_schema.canon_state``
    (the SHARED trellis/gold canon key — matching is casing/ordering-proof). The layer
    yields a node iff ``(t, canon(gold_prev))`` is a key of ``sample_nodes`` (the
    ``sample_node_freqs(...)["nodes"]`` dict — only branching nodes with >= 2 successors
    were ever sampled).

    NAMED exclusions (neither is an error, design §2):
      * ``no_gold_trajectory`` — ``gold_beliefs(item)`` is None/unavailable (ALL layers
        excluded), or the trajectory is shorter than the layer index;
      * ``gold_prefix_not_sampled`` — the gold-prefix canon is not among the sampled
        nodes at that t (a gold-prefix node may have < 2 successors or be unreachable
        in the enumerated space).

    Returns ``{"nodes": [{"t", "canon_prev", "gold_succ_canon"}, …] (t-ascending),
    "excluded": [{"t", "reason"}, …]}``. Deterministic and json-able.
    """
    gold = facts_oracle.gold_beliefs(item)
    nodes: list = []
    excluded: list = []
    for t in range(1, int(space.T)):
        if gold is None or t >= len(gold):
            excluded.append({"t": t, "reason": EXCLUDE_NO_GOLD})
            continue
        canon_prev = belief_schema.canon_state(gold[t - 1])
        if (t, canon_prev) not in sample_nodes:
            excluded.append({"t": t, "reason": EXCLUDE_NOT_SAMPLED})
            continue
        nodes.append({"t": t, "canon_prev": canon_prev,
                      "gold_succ_canon": belief_schema.canon_state(gold[t])})
    return {"nodes": nodes, "excluded": excluded}


def _modal_canon(freqs: dict):
    """The FROZEN modal-successor rule (``PREREG_G4["modal_acc_def"]``): argmax over the
    POSITIVE counts; ties broken by the lexicographic-min canon among the argmax set;
    None when every count is zero (or freqs is empty)."""
    positive = {cn: v for cn, v in freqs.items() if v > 0}
    if not positive:
        return None
    vmax = max(positive.values())
    return min(cn for cn, v in positive.items() if v == vmax)


def stationC_arm(sample_nodes_by_item, space_by_item, gold_by_item, pool_of) -> dict:
    """Tally ONE station-C arm over its items' gold-prefix nodes (design §2's readout).

    ``sample_nodes_by_item`` — ``{item_id: sample_node_freqs(...)["nodes"]}`` i.e.
    ``{(t, canon_prev): {"freqs": {canon_next: int}, "off_manifold", "n_parse_ok",
    "n_total"}}`` replayed from THIS arm's frozen cache. ``space_by_item`` —
    ``{item_id: GsdSpace}`` (only ``T`` is read). ``gold_by_item`` — ``{item_id: item
    dict}`` (the gold source: ``facts_oracle.gold_beliefs`` reads it — evaluation-side
    only). ``pool_of`` — callable ``item_id -> 'patient' | 'knowledge'`` (the caller
    derives it from the frozen decode13/knowledge_ctrl lists).

    Frozen per-node readouts (``PREREG_G4["modal_acc_def"]`` / ``["mass_on_gold_def"]``):
      * ``modal_hit`` — modal sampled successor canon == gold successor canon; ties by
        lexicographic-min canon; all-zero freqs -> False;
      * ``mass_on_gold`` — ``freqs.get(gold_succ_canon, 0) / sum(freqs)``; 0.0 when the
        freqs sum to 0.

    Returns ``{"per_node": [{"item_id", "t", "sprev_sha", "gold_succ_sha", "modal_sha",
    "pool", "modal_hit", "mass_on_gold", "n_on_manifold", "n_parse_ok"}, …]
    ((item_id, t, sprev_sha)-sorted), "per_pool": {pool: {"modal_acc", "mass_on_gold",
    "n_nodes"}} (over the pool's nodes; only pools with >= 1 node appear), "per_item":
    [{"item_id", "pool", "n_nodes", "modal_acc", "mass_on_gold", "n_excluded"}, …]
    (item_id-sorted; the accuracies are None for a 0-node item), "excluded":
    [{"item_id", "t", "reason"}, …]}``. Deterministic and json-able.
    """
    per_node: list = []
    per_item: list = []
    excluded: list = []
    for item_id in sorted(sample_nodes_by_item, key=str):
        sample_nodes = sample_nodes_by_item[item_id]
        space = space_by_item[item_id]
        item = gold_by_item[item_id]
        pool = str(pool_of(item_id))
        gp = goldprefix_nodes(item, space, sample_nodes)
        item_rows: list = []
        for nd in gp["nodes"]:
            node = sample_nodes[(nd["t"], nd["canon_prev"])]
            freqs = node.get("freqs", {})
            total = sum(freqs.values())
            modal = _modal_canon(freqs)
            mass = (freqs.get(nd["gold_succ_canon"], 0) / total) if total > 0 else 0.0
            item_rows.append({
                "item_id": str(item_id),
                "t": nd["t"],
                "sprev_sha": _sha256(nd["canon_prev"]),
                "gold_succ_sha": _sha256(nd["gold_succ_canon"]),
                "modal_sha": None if modal is None else _sha256(modal),
                "pool": pool,
                "modal_hit": bool(modal is not None
                                  and modal == nd["gold_succ_canon"]),
                "mass_on_gold": mass,
                "n_on_manifold": total,
                "n_parse_ok": node.get("n_parse_ok", 0),
            })
        per_node.extend(item_rows)
        excluded.extend({"item_id": str(item_id), "t": ex["t"], "reason": ex["reason"]}
                        for ex in gp["excluded"])
        n = len(item_rows)
        per_item.append({
            "item_id": str(item_id),
            "pool": pool,
            "n_nodes": n,
            "modal_acc": (sum(1 for r in item_rows if r["modal_hit"]) / n) if n else None,
            "mass_on_gold": (sum(r["mass_on_gold"] for r in item_rows) / n) if n else None,
            "n_excluded": len(gp["excluded"]),
        })
    per_node.sort(key=lambda r: (r["item_id"], r["t"], r["sprev_sha"]))
    per_item.sort(key=lambda r: r["item_id"])
    excluded.sort(key=lambda e: (e["item_id"], e["t"]))

    pools: dict = {}
    for row in per_node:
        pools.setdefault(row["pool"], []).append(row)
    per_pool = {
        pool: {
            "modal_acc": sum(1 for r in rows if r["modal_hit"]) / len(rows),
            "mass_on_gold": sum(r["mass_on_gold"] for r in rows) / len(rows),
            "n_nodes": len(rows),
        }
        for pool, rows in sorted(pools.items())
    }
    return {"per_node": per_node, "per_pool": per_pool, "per_item": per_item,
            "excluded": excluded}


def stationC_verdict(arm_results_by_arm: dict, prereg: dict) -> dict:
    """Mechanically execute design §2's station-C verdict over the per-arm tallies.

    ``arm_results_by_arm`` — ``{arm_name: stationC_arm(...)}``; MUST contain the
    decision arm ``"noevent"`` with a NON-EMPTY finite ``modal_acc`` for BOTH pools
    (full/anchor are descriptive secondary readouts) — anything else raises ValueError
    (fail-loud, never a silent label). ``prereg`` — carries ``stationC_delta_min``
    (0.2) / ``stationC_delta_null`` (0.05).

    ``delta_acc = modal_acc(patient) − modal_acc(knowledge)`` on the noevent arm;
    ``>= delta_min`` -> ``step_specific``; ``<= delta_null`` -> ``no_step_signal``;
    else ``mixed``. Returns ``{"delta_acc", "label", "reason", "per_arm_pools"}``
    (``per_arm_pools`` = every arm's per_pool block, the descriptive rollup).
    Deterministic and json-able.
    """
    if "noevent" not in arm_results_by_arm:
        raise ValueError("stationC_verdict: decision arm 'noevent' missing from "
                         "arm_results_by_arm (got %r)" % (sorted(arm_results_by_arm),))
    per_arm_pools = {arm: res["per_pool"]
                     for arm, res in sorted(arm_results_by_arm.items())}
    dec = per_arm_pools["noevent"]
    accs = {}
    for pool in ("patient", "knowledge"):
        cell = dec.get(pool)
        acc = None if cell is None else cell.get("modal_acc")
        if (cell is None or not cell.get("n_nodes")
                or acc is None or not math.isfinite(acc)):
            raise ValueError(
                "stationC_verdict: decision arm 'noevent' has no usable %r pool "
                "(need >= 1 gold-prefix node with a finite modal_acc; got %r)"
                % (pool, cell))
        accs[pool] = acc
    delta = accs["patient"] - accs["knowledge"]

    if delta >= prereg["stationC_delta_min"]:
        label = LABEL_STEP_SPECIFIC
        reason = ("step_specific: noevent delta_acc=%.4f (>= delta_min %s) — "
                  "patient modal_acc=%.4f vs knowledge modal_acc=%.4f: given the "
                  "correct history the single-step update is not just deterministic "
                  "but CORRECT, and specifically so on the commitment-failure pool."
                  % (delta, prereg["stationC_delta_min"], accs["patient"],
                     accs["knowledge"]))
    elif delta <= prereg["stationC_delta_null"]:
        label = LABEL_NO_STEP_SIGNAL
        reason = ("no_step_signal: noevent delta_acc=%.4f (<= delta_null %s) — "
                  "patient modal_acc=%.4f vs knowledge modal_acc=%.4f: the one-hot "
                  "differential was a determinism difference, not a correctness one."
                  % (delta, prereg["stationC_delta_null"], accs["patient"],
                     accs["knowledge"]))
    else:
        label = LABEL_MIXED
        reason = ("mixed: noevent delta_acc=%.4f falls between the frozen gates "
                  "(delta_null %s < delta < delta_min %s); patient modal_acc=%.4f vs "
                  "knowledge modal_acc=%.4f."
                  % (delta, prereg["stationC_delta_null"],
                     prereg["stationC_delta_min"], accs["patient"], accs["knowledge"]))
    return {"delta_acc": delta, "label": label, "reason": reason,
            "per_arm_pools": per_arm_pools}


# ======================================================================================
# station D — parsed-moves injection view + E-none prefix plan + recovery verdict +
# quality × recovery table (design §2 station D; the g3 station-A survival machinery
# supplies the arm tally — stages_g3.stationA_arm — this station only re-reads it)
# ======================================================================================
class _SpaceView:
    """A ``GsdSpace`` proxy whose ``moves`` are the INJECTED parsed moves; every other
    attribute (``T``, ``layers``, ``trans``, ``root``, ``chars``, ``path_count``, …)
    delegates to the real space.

    Consumed ONLY for prefix rendering — ``gsd_score.render_event_line`` /
    ``build_prefix`` read exactly ``space.moves`` and ``space.T`` for the event line —
    so the injection changes what the model is TOLD, byte-for-byte the event line and
    nothing else. Enumeration / decode / candidate targets MUST use the real space:
    the parsed moves never define the state space, only the anchor text.

    The one Python subtlety, pinned here and in the unit tests: ``__getattr__`` fires
    only for attributes NOT found by normal lookup, so ``moves`` is set as an INSTANCE
    attribute in ``__init__`` (instance-dict lookup wins) and everything else falls
    through to the real space. The injected list is copied — later mutation of the
    caller's list never reaches the view.
    """

    def __init__(self, space, moves):
        self._space = space
        self.moves = list(moves)

    def __getattr__(self, name):
        return getattr(self._space, name)


def stationD_prefix_plan(aligned: dict, T: int) -> dict:
    """The FROZEN per-layer template plan realizing the alignment rule's E-none
    degradation: ``{t: 'main' | 'noevent' for t in 1..T-1}`` — ``'main'`` iff
    ``aligned[t]`` is a non-empty move list, ``'noevent'`` otherwise (None, an empty
    list, or a missing key — under ``gsd_extract.align_moves`` a missing layer is
    always ``None``, never ``[]``, but the empty list is frozen to the same fate).

    WHY a plan exists at all: ``gsd_score.render_event_line`` renders a moveless layer
    as the POSITIVE claim "Event t: nothing is moved in this step." — fact content the
    model never asserted. A layer the model failed to parse must degrade to the ABSENCE
    of an event line (template ``'noevent'``), which only a template switch can express.

    WIRING NOTE (Subtask 4's decision, documented here): ``TFScorer.score_transitions``
    scores EVERY layer of a space under ONE template per call, so a mixed per-layer plan
    cannot be executed in a single call — the integration either scores per-template
    layer subsets or otherwise assembles per-layer scores; this helper is only the pure
    per-layer plan (deterministic, json-able).
    """
    return {t: ("main" if aligned.get(t) else "noevent") for t in range(1, int(T))}


def stationD_verdict(arm_results: dict, prereg: dict) -> dict:
    """Mechanically execute design §2's station-D verdict over the parsed arm's tally.

    ``arm_results`` — ONE ``stages_g3.stationA_arm``-shaped dict (the single parsed
    arm; the integration re-scores the 22 items under the injected event lines and
    reuses the g3 station-A machinery verbatim). Only ``per_item`` (``item_id`` /
    ``bucket`` / ``map_fixed``) and ``knowledge_fixed_ids`` are read. ``prereg`` —
    carries ``original10_ids`` (the frozen survival universe) + ``stationD_deploy_min``
    (7) / ``stationD_wall_max`` (3).

    ``survival`` = |{map-fixed patients} ∩ original10_ids|. Label: >= deploy_min ->
    ``parse_deployable``; <= wall_max -> ``parse_insufficient`` (a3's self-extraction
    wall, quantified in the decode setting); else ``mixed``. ``gained_ids`` = map-fixed
    PATIENTS outside the original 10 — the g3 churn readout (noevent gained 0000-q2 /
    0003-q3); never counted toward survival. ``knowledge_anomaly_flag`` = True iff any
    knowledge control is map-fixed (expected 0/3); the offenders are NAMED in
    ``knowledge_anomaly_ids`` and the flag NEVER changes the label (g3 convention).

    Returns ``{"survival", "survived_ids", "lost_ids", "gained_ids", "label", "reason",
    "knowledge_anomaly_flag", "knowledge_anomaly_ids"}`` with sorted id lists.
    Deterministic and json-able.
    """
    original10 = {str(x) for x in prereg["original10_ids"]}
    patient_fixed = {p["item_id"] for p in arm_results["per_item"]
                     if p["bucket"] == "patient" and p["map_fixed"]}
    survived_ids = sorted(original10 & patient_fixed)
    lost_ids = sorted(original10 - patient_fixed)
    gained_ids = sorted(patient_fixed - original10)
    survival = len(survived_ids)
    denom = len(original10)

    if survival >= prereg["stationD_deploy_min"]:
        label = LABEL_PARSE_DEPLOYABLE
        reason = ("parse_deployable: survival=%d/%d (>= deploy_min %d; gained=%d) — "
                  "the model's OWN zero-oracle event lines recover the λ=0 fixes: "
                  "structured stepwise fact injection is a genuinely deployable method."
                  % (survival, denom, prereg["stationD_deploy_min"], len(gained_ids)))
    elif survival <= prereg["stationD_wall_max"]:
        label = LABEL_PARSE_INSUFFICIENT
        reason = ("parse_insufficient: survival=%d/%d (<= wall_max %d; gained=%d) — "
                  "a3's self-extraction wall reproduces quantitatively in the decode "
                  "setting: fact-feed QUALITY is the bottleneck of the fixes."
                  % (survival, denom, prereg["stationD_wall_max"], len(gained_ids)))
    else:
        label = LABEL_MIXED
        reason = ("mixed: survival=%d/%d falls between the frozen gates (wall_max %d < "
                  "survival < deploy_min %d; gained=%d)."
                  % (survival, denom, prereg["stationD_wall_max"],
                     prereg["stationD_deploy_min"], len(gained_ids)))

    knowledge_anomaly_ids = sorted(str(x) for x
                                   in arm_results["knowledge_fixed_ids"])
    return {
        "survival": survival,
        "survived_ids": survived_ids,
        "lost_ids": lost_ids,
        "gained_ids": gained_ids,
        "label": label,
        "reason": reason,
        "knowledge_anomaly_flag": bool(knowledge_anomaly_ids),
        "knowledge_anomaly_ids": knowledge_anomaly_ids,
    }


def quality_recovery_table(per_item_quality: dict, per_item_map_fixed: dict,
                           prereg: dict = PREREG_G4) -> dict:
    """The extraction-quality × λ=0-recovery cross table (design §2: quality × recovery cross table —
    the table deciding WHICH explanation carries: if recovery concentrates in the
    all-exact bucket, fact-feed quality is the bottleneck; recovery despite misparses
    would say the anchor structure suffices).

    ``per_item_quality`` — ``{item_id: gsd_extract.extraction_quality(...) dict}``
    (only ``all_exact`` and ``layer_match_rate`` are read); ``per_item_map_fixed`` —
    ``{item_id: bool}``; both keyed over the SAME item universe (the 19 patients) —
    a key-set mismatch raises ValueError naming the difference, never a silent zip.
    ``prereg`` supplies ``original10_ids`` (injectable for tests; defaults to the
    frozen ``PREREG_G4``).

    Buckets: ``"exact"`` = items with ``all_exact`` True, ``"inexact"`` = the rest.
    Per bucket: ``n_items`` (ALL bucket items), ``n_recovered`` (map_fixed AND in
    original10 — a fixed NON-original10 item is churn, read via
    ``stationD_verdict.gained_ids``, not recovery), ``ids_recovered`` / ``ids_lost``
    (the bucket's original10 members, split by map_fixed; sorted). Plus
    ``spearman_quality_vs_fixed`` — ``stages_g.spearman_rho`` of layer_match_rate vs
    the map_fixed indicator over ALL items (None when degenerate: < 2 items or a
    zero-variance side). Deterministic and json-able.
    """
    if set(per_item_quality) != set(per_item_map_fixed):
        raise ValueError(
            "quality_recovery_table: item universes differ (quality-only %r, "
            "fixed-only %r)"
            % (sorted(str(k) for k in
                      set(per_item_quality) - set(per_item_map_fixed)),
               sorted(str(k) for k in
                      set(per_item_map_fixed) - set(per_item_quality))))
    original10 = {str(x) for x in prereg["original10_ids"]}
    buckets = {name: {"n_items": 0, "n_recovered": 0,
                      "ids_recovered": [], "ids_lost": []}
               for name in ("exact", "inexact")}
    rates: list = []
    fixed_flags: list = []
    for iid in sorted(per_item_quality, key=str):
        quality = per_item_quality[iid]
        fixed = bool(per_item_map_fixed[iid])
        bucket = buckets["exact" if quality["all_exact"] else "inexact"]
        bucket["n_items"] += 1
        sid = str(iid)
        if sid in original10:                # recovery is defined over the original 10
            if fixed:
                bucket["n_recovered"] += 1
                bucket["ids_recovered"].append(sid)
            else:
                bucket["ids_lost"].append(sid)
        rates.append(float(quality["layer_match_rate"]))
        fixed_flags.append(1 if fixed else 0)
    return {"exact": buckets["exact"], "inexact": buckets["inexact"],
            "spearman_quality_vs_fixed": sg.spearman_rho(rates, fixed_flags)}


# ======================================================================================
# figures (matplotlib, Agg, lazily imported; ONE png per call at the explicit out_path;
# SKIPS gracefully — returns None — when matplotlib is unavailable; EMPTY inputs still
# write a "no data" figure, never a crash — the stages_g3 figure conventions)
# ======================================================================================
_COL_INK = "#444444"        # labels/annotations wear ink, never a series color
_COL_FIXED = "#2ca02c"      # station-D map-fixed (green; the stages_g3 survived color)
_COL_NOT_FIXED = "#d62728"  # station-D not fixed (red; the stages_g3 lost color)
_ARM_COLORS = {"full": "#c7c7c7", "anchor": "#ff7f0e", "noevent": "#4c78a8"}
_ARM_FALLBACK = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_FIG_DPI = 150

# the two frozen station-C readout panels: (per_pool key, panel title)
_STEP_ACC_PANELS = [("modal_acc", "modal accuracy vs gold successor"),
                    ("mass_on_gold", "sampled mass on gold successor")]


def _plt():
    """Lazy headless matplotlib (Agg set BEFORE pyplot import; project convention)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _try_plt():
    """``_plt()`` or None when matplotlib is unavailable (graceful skip, the stages_g3
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


def fig_step_accuracy(per_pool_by_arm: dict, out_path: str):
    """The station-C readout figure: TWO panels (modal_acc | mass_on_gold), each with
    grouped bars — x = pools, series = arms (``per_pool_by_arm = {arm:
    stationC_arm(...)["per_pool"]}``). A pool missing from an arm renders a zero-height
    bar; each bar is annotated with its pool's n_nodes once (first panel). Empty input
    -> a "no data" figure. Returns ``out_path`` (None w/o matplotlib)."""
    plt = _try_plt()
    if plt is None:
        return None
    arms = sorted(per_pool_by_arm)
    pools = sorted({pool for arm in arms for pool in per_pool_by_arm[arm]})
    fig, axes = plt.subplots(1, 2, figsize=(max(7.6, 2.1 * len(pools) + 4.2), 3.8),
                             dpi=_FIG_DPI)
    if not arms or not pools:
        for ax in axes:
            _no_data(ax)
    else:
        width = 0.8 / len(arms)
        for p, (key, title) in enumerate(_STEP_ACC_PANELS):
            ax = axes[p]
            for i, arm in enumerate(arms):
                xs, ys = [], []
                for j, pool in enumerate(pools):
                    cell = per_pool_by_arm[arm].get(pool, {})
                    val = cell.get(key)
                    x = j - 0.4 + (i + 0.5) * width
                    xs.append(x)
                    ys.append(0.0 if val is None else val)
                    if p == 0:                       # annotate n once, on the first panel
                        n = cell.get("n_nodes")
                        if n is not None:
                            ax.annotate("n=%d" % n, (x, ys[-1]), ha="center",
                                        va="bottom", fontsize=6, color=_COL_INK)
                ax.bar(xs, ys, width=width * 0.92, color=_arm_color(arm, i), label=arm)
            ax.set_xticks(range(len(pools)))
            ax.set_xticklabels(pools, fontsize=8)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel(key, fontsize=8)
            ax.set_title(title, fontsize=8.5)
            ax.legend(fontsize=7.5, frameon=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.suptitle("station C — gold-prefix step accuracy by pool (series = arm)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_parse_recovery(table: dict, per_item_rows, out_path: str):
    """The station-D quality × recovery figure: one point per patient item — x =
    extraction ``layer_match_rate``, y = the map_fixed categorical (0/1, DETERMINISTIC
    jitter: items spread evenly within their y-group in item_id order — no RNG), marker
    = original-10 membership (circle = an original λ=0 fix, triangle = other patient),
    color = map_fixed (green fixed / red not). ``per_item_rows`` — ``[{"item_id",
    "layer_match_rate", "map_fixed", "in_original10"}, …]``; ``table`` — the
    ``quality_recovery_table`` output, rendered as the title's bucket summary line.
    Empty rows -> a "no data" figure (still a PNG, never a crash). Returns ``out_path``
    (None w/o matplotlib)."""
    plt = _try_plt()
    if plt is None:
        return None
    rows = sorted((dict(r) for r in per_item_rows),
                  key=lambda r: str(r["item_id"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=_FIG_DPI)
    if not rows:
        _no_data(ax)
    else:
        groups: dict = {}
        for r in rows:
            groups.setdefault(int(bool(r["map_fixed"])), []).append(r)
        for y0, grp in sorted(groups.items()):
            n = len(grp)
            for k, r in enumerate(grp):
                jitter = 0.0 if n == 1 else -0.16 + 0.32 * k / (n - 1)
                in10 = bool(r.get("in_original10"))
                ax.scatter([float(r["layer_match_rate"])], [y0 + jitter],
                           marker="o" if in10 else "^", s=44 if in10 else 36,
                           color=_COL_FIXED if y0 else _COL_NOT_FIXED,
                           edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["not map-fixed", "map-fixed"], fontsize=8)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xlabel("extraction layer_match_rate (vs space.moves)", fontsize=8)
        # marker-convention legend (ink proxies — color is the y axis, not the legend)
        ax.scatter([], [], marker="o", color=_COL_INK, label="original-10 λ=0 fix")
        ax.scatter([], [], marker="^", color=_COL_INK, label="other patient")
        ax.legend(fontsize=7, frameon=False, loc="center left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    exact = (table or {}).get("exact", {})
    inexact = (table or {}).get("inexact", {})
    rho = (table or {}).get("spearman_quality_vs_fixed")
    summary = ("all-exact: rec %s of %s items | inexact: rec %s of %s items | "
               "spearman(rate, fixed) = %s"
               % (exact.get("n_recovered", 0), exact.get("n_items", 0),
                  inexact.get("n_recovered", 0), inexact.get("n_items", 0),
                  "n/a" if rho is None else "%.3f" % (rho,)))
    ax.set_title("station D — extraction quality x λ=0 recovery\n%s" % summary,
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
