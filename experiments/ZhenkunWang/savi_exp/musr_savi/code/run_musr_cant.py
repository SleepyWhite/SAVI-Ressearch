#!/usr/bin/env python
"""run_musr_cant — MuSR-cant Phase A single entry point (Subtask 4, the [INTEGRATION] deliverable).

Assembles ``data_musr`` (acquisition/audit), ``sc_core`` (sampling/voting/stats) and ``stages``
(triage/attrition/certification) into one CLI with four stages:

    --stage audit   download+normalize the 3 subtasks, run data_musr.audit, write data_audit.json
    --stage a0      triage: N=32 SC over the frozen triage ids -> stages.triage -> results_a0.json
    --stage a1      attrition: N=256 (+deepen 512/1024 on survivors) on the selected arena;
                    attrition + pass@N coverage curves + token ledger -> results_a1.json
    --stage a2      certify A1 survivors: stats / facts / ladder / seed gates -> stages.certify
                    -> results_a2.json (classification ledger + Phase-B interface)

Phase A does NOT test any decode method — it is a falsification-first funnel (design §0). This
is eval-only: no training loop, so cadence/checkpoint/MFU are N/A (justified in the plan).

DEPENDENCY INJECTION: the only GPU/torch site is ``sc_core.HFEmitter``, built lazily by the
default emitter factory. Every stage takes ``emitter_factory`` (``model_name -> Emitter``) via
DI so ``tests/test_integration.py`` drives the whole funnel on CPU with a FakeEmitter and this
module NEVER imports the tests (code-separation rule).

Observability / robustness (design §7): per-stage phase banner; tqdm over items; a per-item log
line (id/rung/mode_correct/tokens/seconds); an end-of-run summary + efficiency section (GPU-h,
inst/min, mean tokens/sec per sample); per-item JSONL resume; unparseable -> counted wrong (in
sc_core, never crashes); OOM -> one retry then ``oom_skipped`` recorded; results.json is checked
free of NaN/Inf and RAISES before writing; every results.json carries a config block with all
seeds, the PROMPT/facts template shas, the model names, the data sha256s and a PREREG snapshot.
Multi-shard: ``merge_shards`` concatenates per-shard JSONL and asserts (item,sample_idx) key
disjointness across shards, yielding a single analysis input identical to a single process.

Sharded full-run workflow (Subtask 5): every stage invocation SAMPLES its shard (into
``cache_<stage>_shard<i>.jsonl``) and then ANALYSES the MERGED cache of all shard files present
(``load_stage_cache`` globs ``cache_<stage>*.jsonl``). Because the seed protocol is
shard-invariant and sampling is resume-idempotent, the canonical results come from ONE final
un-sharded consolidation pass (``--stage <s>`` with no ``--shard``) after the parallel shards
finish: it re-draws nothing (all cache-hit) and writes the complete ``results_<stage>.json``
from the fully-merged cache.

EXACT-GSD stages (``--stage g0|g1``, plans/2026-07-07-exact-gsd.md Subtask 6): the zero-
approximation decode pipeline over the Subtask 1-5 components (``gsd_space`` enumeration /
``gsd_score`` TF likelihood / ``gsd_decode`` exact two-arm DP + Δ-ledger / ``stages_g``
analyzers). g0 = PREREG_g freeze + the G-A fidelity kill-gate over the two-pool b1 evidence
feed + the zero-GPU backtrace-readout ablation -> results_g0.json; g1 = preflight (PREREG_g
round-trip, results_g0 exists, ga verdict != KILL, G-C scale assertion) + per-item
enumerate -> score (dual caliber, cached) -> two-arm decode -> Δ-ledger + SC@256 from the b1
cache + the deep-dive template-sensitivity rerun + the six-readout assembly + figures ->
results_g1.json. The SOLE GPU site is the TF scorer, dependency-injected exactly like the
emitter (``stage_g0(args, scorer=None)`` / ``stage_g1(args, scorer=None)``; ``None`` loads
the real bf16 model via ``gsd_score.load_scorer``, tests inject a tiny CPU fake). Scoring is
JSONL-cached (``cache_g0_scores*.jsonl`` content-keyed for the G-A pass;
``cache_g1_scores*.jsonl`` via ``TFScorer.score_transitions``'s own 6-part key), so reruns
score nothing and ``--shard i/M`` parallelizes by item with the same one final un-sharded
consolidation pass as the a/b stages. g-stage log lines are TEED to
``outputs/g{0,1}[_shard<i>].log`` in addition to stdout.

G-A fidelity narrowing (``--stage g2``, plans/2026-07-08-ga-fidelity.md Subtask 6): the
three-arm dissection of the flagged EXACT-GSD G-A gate (median Spearman ρ=0.400), deciding
whether λ=0's reproducible 7-of-19 fixes reflect the model's on-policy belief or a
TF-likelihood-under-GSD scoring artifact. It reuses the Subtask 1-5 components (``gsd_sample``
per-branching-node conditional next-BELIEF sampler / ``stages_g2`` Arm 0 split-half noise
ceiling + Arm 1 matched-context ρ + Arm 2 frequency-decode + the combined verdict). Preflight
= PREREG_g2 round-trip (frozen on first run), results_g1 exists, the g0 G-A verdict != KILL,
and the G-C scale assertion; then the item set (the results_g0 ``ga.per_group`` group items ∪
the 13 frozen decode items) is enumerated, Arm 0 recomputes the split-half reliability over the
b1 chains (zero GPU), a SINGLE sampling pass (the SOLE GPU station — the Arm 1a G-A group nodes
∪ every Arm 2 decode branching node) draws M next-BELIEF lines per node under the frozen GSD
template, Arm 1 correlates the matched-context frequency with the cached (g1) / freshly-scored
(tuning) TF likelihood, Arm 2 builds ``A_freq`` and runs the reused ``gsd_decode(..., "map")``,
and the mechanical combined verdict lands in {strong_belief / artifact / partial / inconclusive}
-> results_g2.json + VERDICT_ga.md + figures/gsd_g2_*.png. Both heavy sites are
dependency-injected exactly like g0/g1: ``stage_g2(args, scorer=None, emitter=None)`` — the TF
scorer AND the sampler emitter (None builds the real bf16 model / a lazily-built HFEmitter so a
full-cache-hit consolidation loads nothing; tests inject CPU fakes). Sampling is JSONL
content-key cached (``cache_g2_sample*.jsonl``), TF scores reuse ``cache_g1_scores*.jsonl`` +
``cache_g2_scores*.jsonl``, so reruns re-draw / re-score nothing and ``--shard i/M`` parallelizes
by item with one final un-sharded consolidation pass. Log lines tee to
``outputs/g2[_shard<i>].log``.

Event-line ablation (``--stage g3``, plans/2026-07-09-eventline-ablation.md Subtask 5): one
template change (the per-step ``Event t:`` fact feed removed — arm ``anchor`` = an index-only
constant line, arm ``noevent`` = the block deleted), two stations re-read. Preflight = PREREG_g3
freeze + round-trip, PREREG_g2.md / results_g1.json / results_g2.json exist, and the E-full
baselines are asserted byte-for-byte (station A: results_g1 map 10/19 + oracle 15/19 + knowledge
0/3 via ``stages_g3.stationA_baseline_check``; station B: results_g2 arm2 7/3/2). Station A
(cheap, first) re-scores g1's 22 items per arm (``TFScorer.score_transitions(template=arm,
stage='g3a_<arm>')``, cached in ``cache_g3_scores*.jsonl``; SC read from results_g1, never
recomputed) and re-decodes via the reused ``stages_g.g1_item_analysis`` -> survival verdict.
Station B (the GPU cost) re-samples the 13 decode items' branching nodes per arm
(``gsd_sample.sample_node_freqs(template=arm, stage='g3_<arm>', seed_tag='g3sample')``, per-ARM
caches ``cache_g3_sample_{anchor,noevent}*.jsonl`` — the sample key carries no template, so arm
isolation = unique stage string + separate file), frequency-decodes via the reused
``stages_g2.arm2_all`` -> ordered probe-rescue verdict + the continuous one-hot/entropy readout,
with the E-full continuous baseline RECOMPUTED from ``cache_g2_sample.jsonl`` at zero GPU (a
raising stub guarantees zero fresh draws). Both heavy sites are dependency-injected
(``stage_g3(args, scorer=None, emitter=None)``). ``--shard i/M`` = a GEN-ONLY pass (shard caches
only, no analysis/results); the final un-sharded pass consolidates and writes results_g3.json +
VERDICT_g3.md + figures/g3_*.png. ``--g3-station {A,B,all}`` runs one or both stations;
``--g3-sample-m`` overrides the frozen M (smoke -> 8). A --smoke pass that would overwrite a
REAL results_g3.json redirects its artifacts to ``outputs/backup_smoke_g3/`` (the g-line
backup_smoke convention, automated). Log lines tee to ``outputs/g3[_shard<i>].log``.

Step accuracy + self-parsed event lines (``--stage g4``, plans/2026-07-09-g4-step-parsed.md
Subtask 4): the two g3 follow-ups, two stations. Preflight = the reviewer-mandated
extract-template-sha hard-assert (PREREG_G4['extract_template_sha'] == sha256 of the LIVE
``gsd_extract.EXTRACT_TEMPLATE``, recomputed here independently of the stages_g4 import-time
check), PREREG_g4 freeze + round-trip (with ONE sanctioned in-place rewrite: a STALE file
from an earlier partial pass whose only diff is the documented Subtask-3
``extract_template_sha`` key addition — anything else hard-raises), upstream files exist
(results_g1 / results_g3 / cache_g2_sample / cache_g3_sample_{anchor,noevent} /
cache_g3_scores), and the frozen baselines asserted byte-for-byte (g1 10/15/0 via
``stages_g3.stationA_baseline_check``; g3 noevent survival==1 + the 3 map-fixed ids).
Station C (ZERO GPU, first): the 13 decode items' gold-prefix branching nodes read per arm
(full/anchor/noevent) by PURE REPLAY of the frozen g2/g3 sample caches at their exact frozen
coordinates behind a raising-stub emitter (any fresh draw fails loudly; ``n_scored == 0``
hard-asserted per item x arm) -> ``stages_g4.stationC_arm`` modal_acc / mass_on_gold ->
the Δ_acc verdict (gates 0.2 / 0.05). Station D (the ONLY GPU): the 22 g1 items each get a
zero-oracle extraction pass (``gsd_extract.extract_events``, greedy, narrative-only prompt,
cache_g4_extract.jsonl) -> alignment -> the frozen per-layer template plan
(``stationD_prefix_plan``: parsed layer -> 'main', unparsed -> the E-none degradation);
scoring runs AT MOST two ``score_transitions`` calls per item — template='main' over the
``_SpaceView``-injected parsed moves at stage 'g4d_parsed', plus (only when some layer
degraded) template='noevent' over the REAL space at stage 'g3a_noevent', which DELIBERATELY
matches g3's station-A noevent coordinates so every such edge full-hits cache_g3_scores.jsonl
(the noevent template ignores space.moves, so the prefixes are byte-identical; the scorer is
primed from the g3 caches) — the per-layer MERGE (``_g4_merge_lse``) then splices whole
lse groups by plan (both calls share identical candidate sets, so no normalization group is
ever mixed) and the reused ``stages_g.g1_item_analysis`` decodes over the REAL space ->
``stages_g3.stationA_arm`` -> the survival verdict (gates 7/3) + the quality x recovery
cross table. Both heavy sites are dependency-injected (``stage_g4(args, scorer=None,
emitter=None)``). No ``--shard`` (<= 0.5 GPU-h single card; a --shard hard-raises).
``--g4-station {C,D,all}`` runs one or both stations; ``--smoke`` scopes BOTH stations to
2 items (repro7[0] + knowledge3[0]) on the same code path, with the g3-style smoke-clobber
guard (``outputs/backup_smoke_g4/``). Log lines tee to ``outputs/g4.log``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

# --- path bootstrap: sibling core modules (importlib-friendly, no package/conftest) ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import data_musr as dm      # noqa: E402
import sc_core as sc        # noqa: E402
import selffacts as sf      # noqa: E402
import stages              # noqa: E402
# --- Phase B core modules (all CPU/torch-free at import; featurizer lazy-imports torch) --------
import belief_schema as bs  # noqa: E402
import facts_oracle as fo   # noqa: E402
import verifiers as vf      # noqa: E402
import domain_musr as dmn   # noqa: E402
import stages_b as sb       # noqa: E402
import phase_b_split as pbs  # noqa: E402
import featurizer as fz     # noqa: E402
# --- EXACT-GSD core modules (CPU/torch-free at import; gsd_score lazy-imports torch) -----------
import gsd_space as gsp     # noqa: E402
import gsd_score            # noqa: E402
import gsd_decode as gdec   # noqa: E402
import gsd_sample           # noqa: E402
import gsd_extract          # noqa: E402
import stages_g as sg       # noqa: E402
import stages_g2 as sg2     # noqa: E402
import stages_g3 as sg3     # noqa: E402
import stages_g4 as sg4     # noqa: E402

# ======================================================================================
# Frozen model roster (design §2). HF ids; loaded offline from the local cache at L1/full.
# ======================================================================================
MAIN_MODEL = "Qwen/Qwen3-4B"
LADDER_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
# Single small ladder model used under --smoke (keeps L1 under 10 min; the full roster runs
# only in the real a2 full run).
SMOKE_LADDER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

STAGES = ("audit", "a0", "a1", "a2", "a3", "b0", "b1", "b2", "g0", "g1", "g2", "g3", "g4")
DEFAULT_OUT_DIR = os.path.join(_HERE, "outputs")
DEFAULT_DATA_DIR = os.path.join(_HERE, "data")

# ---- Phase B constants (design plans/2026-07-06-phaseB.md §Subtask 6) -------------------------
PHASE_B_ARENA = ["object_placements"]      # the only certified-patient subtask (design §2)
# Store the (effectively full) generated text on each Phase B sample record so the featurizer /
# critic can rederive features on resume WITHOUT regeneration. A realistic 2048-token chain is
# ~10k chars, so this cap never truncates a real reply (it only bounds a pathological run-on).
B_TEXT_KEEP = 200000
# Per-stage Phase B sample budgets (design §3/§5; smoke overrides in _apply_smoke).
DEFAULT_B0_PATIENT_N = 256
DEFAULT_B0_TRAIN_N = 64
DEFAULT_B1_PATIENT_N = 64
DEFAULT_B1_DEEPEN_N = 1024
DEFAULT_B1_TUNING_N = 256
DEFAULT_B1_TRAIN_N = 24
DEFAULT_B_MAX_NEW_TOKENS = 2048
B_SPLIT_SEED = 20260706                    # frozen story-split seed (design §2)
PREREG_B_PATH = os.path.join(DEFAULT_OUT_DIR, "PREREG_b.md")
# BoN whole-chain score for a parse-fail chain: below the [0,1] step-verifier range so a
# non-parse_ok chain is never argmax-picked unless the whole pool failed to parse.
_BON_PARSEFAIL_SCORE = -1.0

# ---- EXACT-GSD constants (design plans/2026-07-07-exact-gsd-design.md §2/§4/§6) ---------------
# G-C scale ceiling on the certified patients (2**(C*T) with C=3, T=4 — verified on real data);
# violated at g1 preflight = a provenance / construction error, hard-raised.
G_PATH_COUNT_MAX = 4096
DEFAULT_GSD_BATCH = 8                       # TF scorer forward batch (gsd_score halves on OOM)

# Per-stage default sample budgets (design §3-§5). smoke / CLI can override.
DEFAULT_A0_RUNG = 32
DEFAULT_A1_TOP = 256
DEFAULT_A1_DEEPEN = list(stages.PREREG["a1_deepen_rungs"])   # (512, 1024)
DEFAULT_A2_FACTS = stages.PREREG["a2_facts_rung"]            # 8
DEFAULT_A2_LADDER = stages.PREREG["a2_ladder_rung"]          # 32
DEFAULT_A2_SEED = stages.PREREG["a2_seed_rung"]              # 64
DEFAULT_A3_N = stages.PREREG_A3["a3_fix_rung"]               # 64 (S1/S2/O all vote at N=64)
# A1 selection-budget prefix rungs for the attrition + pass@N curves (E1 convention).
A1_PREFIX_RUNGS = (1, 2, 4, 8, 16, 32, 64, 256)


# ======================================================================================
# small utilities (log / banner / tqdm / json)
# ======================================================================================
def _now():
    return time.strftime("%H:%M:%S")


# When set (g stages), every log line is TEED to this file handle in addition to stdout
# (the production requirement "per-item logs -> stdout + outputs/g{0,1}.log"). The a/b stages
# leave it None (their log files come from shell redirection, the established convention).
_LOG_FH = None


def log(msg):
    line = "[%s] %s" % (_now(), msg)
    print(line, flush=True)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
        except Exception:
            pass                                          # observability must never crash a run


def banner(stage, extra=""):
    log("=" * 78)
    log("[phase-start] stage=%s %s" % (stage, extra))
    log("=" * 78)


def _progress(seq, total=None, desc=None):
    try:
        from tqdm import tqdm
        return tqdm(seq, total=total, desc=desc)
    except Exception:
        return seq


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ======================================================================================
# NaN/Inf guard  (design §7 — RAISE before writing, do not silently sanitize)
# ======================================================================================
def _find_nonfinite(obj, path="", found=None):
    if found is None:
        found = []
    if isinstance(obj, bool):
        return found
    if isinstance(obj, float):
        if not math.isfinite(obj):
            found.append(path or "<root>")
        return found
    if isinstance(obj, dict):
        for k, v in obj.items():
            _find_nonfinite(v, "%s.%s" % (path, k) if path else str(k), found)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _find_nonfinite(v, "%s[%d]" % (path, i), found)
    return found


def assert_finite(obj):
    """Raise ValueError (listing key-paths) if ``obj`` contains any NaN/Inf float. Else None."""
    bad = _find_nonfinite(obj)
    if bad:
        raise ValueError("results contain %d non-finite (NaN/Inf) value(s): %s%s"
                         % (len(bad), bad[:8], " ..." if len(bad) > 8 else ""))


def sanitize_nonfinite(obj):
    """Return a deep copy of ``obj`` with every non-finite float (NaN/Inf) replaced by ``None``,
    plus the count replaced. Phase B gates emit a documented ``(nan, nan)`` CI sentinel for
    empty / degenerate evidence (an empty step pool, a single-class label vector); that is a
    LEGITIMATE 'undefined CI' rather than a bug, so we map it to JSON ``null`` before the NaN/Inf
    write-guard rather than letting the guard hard-fail a valid degenerate result. The count is
    logged so a genuine NaN bug is still visible."""
    n = [0]

    def _rec(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, float):
            if not math.isfinite(x):
                n[0] += 1
                return None
            return x
        if isinstance(x, dict):
            return {k: _rec(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_rec(v) for v in x]
        return x

    return _rec(obj), n[0]


def write_results(path, obj):
    """Guard against NaN/Inf (RAISE), then write pretty JSON. Nothing is written on a violation."""
    assert_finite(obj)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    log("wrote %s" % path)
    return path


# ======================================================================================
# JSONL cache paths + shard merge
# ======================================================================================
def _cache_path(out_dir, stage, shard_i=None):
    if shard_i is None:
        return os.path.join(out_dir, "cache_%s.jsonl" % stage)
    return os.path.join(out_dir, "cache_%s_shard%d.jsonl" % (stage, shard_i))


def load_stage_cache(out_dir, stage):
    """Merged sample cache for a stage: every ``cache_<stage>*.jsonl`` (single + all shards),
    de-duplicated last-write-wins by the 4-tuple key. This is the single analysis input.

    NOTE: the a3 extraction cache lives in ``cache_a3_extract*.jsonl``. Its records are keyed by
    ``story_id`` (NOT the 4-tuple sample key) and would be silently dropped by ``sc.load_cache``,
    so this glob deliberately co-loads them without harm — but a3 reads extractions through
    ``_load_a3_extracts`` (below), never through this sample-cache path."""
    merged = {}
    for fp in sorted(glob.glob(os.path.join(out_dir, "cache_%s*.jsonl" % stage))):
        if "_extract" in os.path.basename(fp):          # a3 story-keyed extraction cache, not samples
            continue
        merged.update(sc.load_cache(fp))
    return merged


def _a3_extract_cache_path(out_dir, shard_i=None):
    """Per-shard story-level extraction cache path (JSONL, keyed by story_id)."""
    if shard_i is None:
        return os.path.join(out_dir, "cache_a3_extract.jsonl")
    return os.path.join(out_dir, "cache_a3_extract_shard%d.jsonl" % shard_i)


def _load_a3_extracts(out_dir):
    """Merged ``{story_id: rec}`` extraction cache across all a3 extraction shard files (last
    write wins). Mirrors ``load_stage_cache`` but on ``selffacts.load_extracts`` (story key)."""
    merged = {}
    for fp in sorted(glob.glob(os.path.join(out_dir, "cache_a3_extract*.jsonl"))):
        merged.update(sf.load_extracts(fp))
    return merged


def merge_shards(out_dir, stage):
    """Concatenate per-shard JSONL for a stage into one cache dict, asserting the
    (subtask,item_id,sample_idx,seed_tag) keys are DISJOINT across shard files (an item must
    belong to exactly one shard). Returns the merged {key: record}. Hard-fails on overlap."""
    shard_files = sorted(glob.glob(os.path.join(out_dir, "cache_%s_shard*.jsonl" % stage)))
    merged = {}
    seen_in = {}
    for fp in shard_files:
        cache = sc.load_cache(fp)
        for k in cache:
            if k in merged:
                raise AssertionError(
                    "merge_shards[%s]: key %r not disjoint across shards (also in %s and %s)"
                    % (stage, k, seen_in.get(k), os.path.basename(fp)))
            seen_in[k] = os.path.basename(fp)
        merged.update(cache)
    log("merge_shards[%s]: %d record(s) from %d shard file(s), keys disjoint"
        % (stage, len(merged), len(shard_files)))
    return merged


# ======================================================================================
# data loading
# ======================================================================================
def load_items(data_dir, subtasks=None):
    """Load normalized items for the given subtasks (default all three) from data/{subtask}.json."""
    subtasks = subtasks or list(dm.SUBTASKS)
    items = []
    for sub in subtasks:
        path = os.path.join(data_dir, "%s.json" % sub)
        if not os.path.exists(path):
            continue
        items.extend(dm.load_local(sub, data_dir=data_dir))
    return items


def _items_by_id(items):
    return {it["id"]: it for it in items}


def _data_sha256_map(out_dir):
    """Read per-subtask data sha256 from data_audit.json (download_meta), for the config block."""
    audit_path = os.path.join(out_dir, "data_audit.json")
    if not os.path.exists(audit_path):
        return {}
    try:
        with open(audit_path) as f:
            rep = json.load(f)
    except Exception:
        return {}
    meta = rep.get("download_meta") or []
    out = {}
    if isinstance(meta, dict):                      # single-subtask or {subtask: meta} shapes
        if "subtask" in meta:
            out[meta["subtask"]] = meta.get("sha256")
        else:
            for k, m in meta.items():
                if isinstance(m, dict):
                    out[k] = m.get("sha256")
    elif isinstance(meta, list):
        for m in meta:
            if isinstance(m, dict) and "subtask" in m:
                out[m["subtask"]] = m.get("sha256")
    return out


# ======================================================================================
# config block (echoed into every results.json)
# ======================================================================================
def build_config(args, stage, extra=None):
    cfg = {
        "stage": stage,
        "base_seed": args.base_seed,
        "seed2_base": _seed2_base(args.base_seed),
        "smoke": bool(args.smoke),
        "limit": args.limit,
        "shard": args.shard,
        "models": {"main": args.main_model, "ladder": list(args.ladder_models)},
        "prompt_template_sha256": _sha256_text(sc.PROMPT_TEMPLATE),
        "facts_template_sha256": _sha256_text(sc.FACTS_TEMPLATE),
        "facts_preamble_sha256": _sha256_text(sc.FACTS_PREAMBLE),
        "prereg": dict(stages.PREREG),
        "data_sha256": _data_sha256_map(args.out_dir),
        "generation": {"temperature": sc.DEFAULT_TEMPERATURE,
                       "max_new_tokens": sc.DEFAULT_MAX_NEW_TOKENS,
                       "top_p": sc.DEFAULT_TOP_P, "top_k": sc.DEFAULT_TOP_K,
                       "attn_implementation": "sdpa"},
        "budgets": {"a0_rung": args.a0_rung, "a1_top": args.a1_top,
                    "a1_deepen": list(args.a1_deepen), "a2_facts": args.a2_facts,
                    "a2_ladder": args.a2_ladder, "a2_seed": args.a2_seed},
    }
    if extra:
        cfg.update(extra)
    return cfg


def _seed2_base(base_seed):
    """Second sampling seed (design §5.4). Distinct, deterministic function of base_seed."""
    return base_seed + 1


# ======================================================================================
# emitter factory (DI) + lazy per-model cache
# ======================================================================================
def default_emitter_factory(args):
    """Build the production factory (``model_name -> HFEmitter``). The SOLE torch/GPU site."""
    def _factory(model_name):
        return sc.HFEmitter(model_name, device=args.device, batch=args.batch)
    return _factory


class _EmitterPool:
    """Lazily builds and caches one Emitter per model; releases prior models to free GPU."""

    def __init__(self, factory, keep=1):
        self._factory = factory
        self._keep = keep
        self._pool = {}
        self._order = []

    def get(self, model_name):
        if model_name not in self._pool:
            # Release-before-load: evict down to keep-1 and free GPU BEFORE building the new
            # model, so we never transiently hold (old + new) weights on the card. Loading first
            # would peak at 2 resident models (fatal at the two-adjacent-7B ladder boundary on a
            # 40GB card). Callers must also drop any external ref (e.g. `main`) they hold, else
            # that model's weights stay alive despite eviction from the pool.
            while len(self._order) >= self._keep:
                old = self._order.pop(0)
                self._pool.pop(old, None)
                self._release()
            log("loading emitter for model=%s" % model_name)
            self._pool[model_name] = self._factory(model_name)
            self._order.append(model_name)
        return self._pool[model_name]

    @staticmethod
    def _release():
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ======================================================================================
# sampling helper (draw + cache + resume + OOM handling + per-item log)
# ======================================================================================
def _sample_item(emitter, item, subtask, n, base_seed, seed_tag, cache_path,
                 cache, *, prompt=None, greedy=False, run_stats=None,
                 max_new_tokens=None, keep_text_chars=0):
    """Draw ``n`` samples for one item (resume-aware), append to the JSONL cache, return the
    item's ``sample_idx``-ordered records for the (subtask,item,seed_tag) regime.

    OOM -> record ``oom_skipped`` and return whatever is already cached (design §7). Unparseable
    replies are already counted wrong inside sc_core (never crash here). A per-item context
    guard (design §7) skips + lists any item whose prompt exceeds the context limit; the data
    audit already hard-errs > 30k, so this only ever fires on a downstream template surprise.

    ``max_new_tokens`` / ``keep_text_chars`` (a3 additive, default-off) are threaded verbatim into
    ``sc.draw_samples`` — with the defaults (``None`` / ``0``) the call is byte-for-byte the prior
    behaviour, so every a0/a1/a2 caller is untouched. a3's S1 arm pins its own truncation cap and
    keeps a ``text_head`` for the scaffold-compliance diagnostic."""
    the_prompt = prompt if prompt is not None else sc.format_prompt(item)
    if _context_overflow(emitter, the_prompt):
        if run_stats is not None:
            run_stats.setdefault("context_offenders", []).append(
                {"item_id": item["id"], "seed_tag": seed_tag})
        log("context-overflow skipped item=%s seed_tag=%s" % (item["id"], seed_tag))
        return sc.cached_records(cache, subtask, item["id"], seed_tag), 0, 0.0
    existing = sc.completed_idxs(cache, subtask, item["id"], seed_tag)
    t0 = time.time()
    n_new = 0
    new_tokens = 0
    try:
        recs = sc.draw_samples(emitter, item, subtask, n, base_seed, seed_tag=seed_tag,
                               greedy=greedy, prompt=prompt, existing_idxs=existing,
                               max_new_tokens=max_new_tokens, keep_text_chars=keep_text_chars)
        for r in recs:
            sc.append_record(cache_path, r)
            cache[sc.cache_key(r)] = r
        n_new = len(recs)
        new_tokens = _sum_tokens(recs)                # count ONLY newly-generated tokens
    except sc.EmitterOOM as exc:
        if run_stats is not None:
            run_stats.setdefault("oom_skipped", []).append({"item_id": item["id"],
                                                            "seed_tag": seed_tag, "error": str(exc)})
        log("OOM-skipped item=%s seed_tag=%s (%s)" % (item["id"], seed_tag, exc))
    dt = time.time() - t0
    if run_stats is not None:
        run_stats["n_generated"] = run_stats.get("n_generated", 0) + n_new
        run_stats["tokens_total"] = run_stats.get("tokens_total", 0) + new_tokens
        run_stats["seconds"] = run_stats.get("seconds", 0.0) + dt
    return sc.cached_records(cache, subtask, item["id"], seed_tag), n_new, dt


def _efficiency(run_stats, n_items):
    """Efficiency section: GPU-h, inst/min, mean tokens/sec per sample (design §7)."""
    secs = run_stats.get("seconds", 0.0)
    n_gen = run_stats.get("n_generated", 0)
    tok = run_stats.get("tokens_total", 0)
    return {
        "gpu_seconds": secs,
        "gpu_hours": secs / 3600.0,
        "n_items": n_items,
        "n_samples_generated": n_gen,
        "items_per_min": (n_items / (secs / 60.0)) if secs > 0 else None,
        "samples_per_min": (n_gen / (secs / 60.0)) if secs > 0 else None,
        "tokens_total": tok,
        "mean_tokens_per_sample": (tok / n_gen) if n_gen > 0 else 0.0,
        "mean_sec_per_sample": (secs / n_gen) if n_gen > 0 else 0.0,
        "oom_skipped": len(run_stats.get("oom_skipped", [])),
    }


def _sec_per_sample(run_stats):
    n_gen = run_stats.get("n_generated", 0)
    return (run_stats.get("seconds", 0.0) / n_gen) if n_gen > 0 else 0.0


def _sum_tokens(records):
    return sum(int(r.get("n_new_tokens", 0)) for r in records)


def _context_overflow(emitter, prompt):
    """True iff ``prompt`` exceeds the context token limit, using the emitter's own tokenizer
    when it exposes one (HFEmitter.tok). A no-op for a tokenizer-less emitter (FakeEmitter),
    so the guard is real on the GPU path and inert in the CPU tests. Never raises."""
    tok = getattr(emitter, "tok", None)
    if tok is None:
        return False
    try:
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    except Exception:
        return False
    return n > dm.CONTEXT_TOKEN_LIMIT


# ======================================================================================
# item selection (triage ids + limit/smoke + shard)
# ======================================================================================
def _shard_slice(seq, shard_spec):
    """Strided shard of a list: ``i/M`` -> ``seq[i::M]`` (deterministic, shard-invariant seeds
    make the union identical to the single process)."""
    if not shard_spec:
        return list(seq)
    i, m = (int(x) for x in str(shard_spec).split("/"))
    return list(seq)[i::m]


def _select_a0_items(items, args):
    """The frozen A0 triage ids per subtask, then apply --limit/--smoke, then shard.

    Returns a flat list of items (sorted by id) restricted to this shard."""
    triage = dm.sample_triage_ids(items, base_seed=args.base_seed, n=args.triage_n)
    by_id = _items_by_id(items)
    picked = []
    per_sub_cap = 3 if args.smoke else args.limit
    for sub in sorted(triage.keys()):
        ids = triage[sub]
        if per_sub_cap is not None:
            ids = ids[:per_sub_cap]
        picked.extend(by_id[i] for i in ids if i in by_id)
    picked.sort(key=lambda it: it["id"])
    return _shard_slice(picked, args.shard)


# ======================================================================================
# STAGE: audit
# ======================================================================================
def stage_audit(args):
    banner("audit", "data_dir=%s out_dir=%s" % (args.data_dir, args.out_dir))
    metas = []
    for sub in dm.SUBTASKS:
        log("downloading %s (HF primary, GitHub fallback) ..." % sub)
        meta = dm.download(sub, data_dir=args.data_dir)
        log("  %s: source=%s tree=%s n=%d sha=%s"
            % (sub, meta["source"], meta["tree_source"], meta["n_items"], meta["sha256"][:12]))
        metas.append(meta)
    items = load_items(args.data_dir)
    out_path = os.path.join(args.out_dir, "data_audit.json")
    report = dm.audit(items, download_meta=metas, out_path=out_path, write=True)
    log("---- audit summary ----")
    for sub in dm.SUBTASKS:
        b = report["subtasks"][sub]
        tl = b["narrative_token_lengths"]
        log("  %-18s n=%d tree=%s(%.2f) chance=%.3f tokens[p95=%s max=%s]"
            % (sub, b["n_items"], b["tree_available"], b["tree_available_rate"],
               b["chance_line_mean"], tl["p95"], tl["max"]))
    return report


# ======================================================================================
# STAGE: a0 (triage)
# ======================================================================================
def stage_a0(args, emitter_factory):
    banner("a0", "smoke=%s limit=%s shard=%s rung=%d" %
           (args.smoke, args.limit, args.shard, args.a0_rung))
    items = load_items(args.data_dir)
    if not items:
        raise RuntimeError("a0: no items under %s (run --stage audit first)" % args.data_dir)
    picked = _select_a0_items(items, args)
    by_id = _items_by_id(items)
    log("a0: %d item(s) this shard (rung N=%d)" % (len(picked), args.a0_rung))

    pool = _EmitterPool(emitter_factory)
    emitter = pool.get(args.main_model)
    shard_i = _shard_index(args.shard)
    cache_path = _cache_path(args.out_dir, "a0", shard_i)
    cache = load_stage_cache(args.out_dir, "a0")
    run_stats = {}

    for it in _progress(picked, total=len(picked), desc="a0"):
        recs, n_new, dt = _sample_item(emitter, it, it["subtask"], args.a0_rung,
                                       args.base_seed, "base", cache_path, cache,
                                       run_stats=run_stats)
        vr = sc.vote_at_rung(recs, args.a0_rung)
        log("  [a0] %-26s rung=%d mode_correct=%s tokens=%d sec=%.2f"
            % (it["id"], args.a0_rung, vr.mode_correct, _sum_tokens(recs[:args.a0_rung]), dt))

    # --- analysis on the MERGED cache (all shards present) ---
    merged = load_stage_cache(args.out_dir, "a0")
    a0_by_subtask, per_sub_items = _assemble_a0(merged, by_id, args.a0_rung)
    # A resume / consolidation pass generates nothing -> keep the FIRST run's measured
    # sec/sample (from the prior results_a0.json) so the top-2 budget decision stays stable.
    sec_per_sample = _sec_per_sample(run_stats) or _prior_sec_per_sample(args.out_dir)
    projection = _a1_budget_projection(sec_per_sample, per_sub_items, by_id, args)
    verdict = stages.triage(a0_by_subtask, budget_projection=projection)

    eff = _efficiency(run_stats, len(picked))
    _print_summary("a0", {"selected": verdict.selected, "qualifying": verdict.qualifying,
                          "early_negative": verdict.early_negative,
                          "budget_proj_gpuh": projection}, eff)

    results = {
        "stage": "a0",
        "config": build_config(args, "a0"),
        "triage": _triage_to_dict(verdict),
        "per_subtask_stats": verdict.per_subtask,
        "selected_arena": verdict.selected,
        "budget": {"sec_per_sample": sec_per_sample,
                   "a1_gpuh_projection": projection,
                   "top2_admitted": verdict.top2_admitted},
        "efficiency": eff,
        "run_stats": _run_stats_public(run_stats),
    }
    write_results(os.path.join(args.out_dir, "results_a0.json"), results)
    return results


def _assemble_a0(cache, by_id, rung):
    """Build stages.triage input {subtask: {"items":[record-lists], "chance_mean": ...}}."""
    # group base-seed records by (subtask, item)
    grouped = {}
    for r in cache.values():
        if r.get("seed_tag") != "base":
            continue
        grouped.setdefault((r["subtask"], r["item_id"]), []).append(r)
    a0_by_subtask = {}
    per_sub_item_ids = {}
    subs = sorted({k[0] for k in grouped})
    for sub in subs:
        item_ids = sorted({iid for (s, iid) in grouped if s == sub})
        rec_lists = [sorted(grouped[(sub, iid)], key=lambda r: r.get("sample_idx", 0))
                     for iid in item_ids]
        chance_items = [by_id[iid] for iid in item_ids if iid in by_id]
        chance_mean = dm.chance_line_subtask_mean(chance_items) if chance_items else 0.5
        a0_by_subtask[sub] = {"items": rec_lists, "chance_mean": chance_mean}
        per_sub_item_ids[sub] = item_ids
    return a0_by_subtask, per_sub_item_ids


def _a1_budget_projection(sec_per_sample, per_sub_item_ids, by_id, args):
    """Projected A1 GPU-h for the top-2 arena rule (design §3): sec/sample x full-item-count x
    256, summed over the two most-populated triage subtasks (proxy for the top-2 arenas). Uses
    the audited full counts when available, else the loaded counts. None if no timing yet."""
    if sec_per_sample <= 0:
        return None
    full_counts = _full_subtask_counts(args.out_dir)
    counts = []
    for sub in per_sub_item_ids:
        n_full = full_counts.get(sub) or dm.EXPECTED_COUNTS.get(sub) or len(per_sub_item_ids[sub])
        counts.append(n_full)
    counts.sort(reverse=True)
    top2 = sum(counts[:2])
    top_rung = args.a1_top
    return sec_per_sample * top2 * top_rung / 3600.0


def _prior_sec_per_sample(out_dir):
    """The sec/sample measured on a prior a0 run (from results_a0.json), or 0.0 if none."""
    path = os.path.join(out_dir, "results_a0.json")
    if not os.path.exists(path):
        return 0.0
    try:
        with open(path) as f:
            return float(json.load(f).get("budget", {}).get("sec_per_sample") or 0.0)
    except Exception:
        return 0.0


def _full_subtask_counts(out_dir):
    audit_path = os.path.join(out_dir, "data_audit.json")
    if not os.path.exists(audit_path):
        return {}
    try:
        with open(audit_path) as f:
            rep = json.load(f)
        return {s: b.get("n_items") for s, b in rep.get("subtasks", {}).items()}
    except Exception:
        return {}


def _triage_to_dict(v):
    return {
        "early_negative": v.early_negative,
        "selected": v.selected,
        "qualifying": v.qualifying,
        "top2_admitted": v.top2_admitted,
        "budget_projection": v.budget_projection,
        "per_subtask": v.per_subtask,
    }


# ======================================================================================
# STAGE: a1 (attrition + pass@N)
# ======================================================================================
def stage_a1(args, emitter_factory):
    arena = _resolve_arena(args)
    banner("a1", "arena=%s top=%d deepen=%s" % (arena, args.a1_top, args.a1_deepen))
    items = load_items(args.data_dir, subtasks=arena)
    if not items:
        raise RuntimeError("a1: no items for arena %s under %s" % (arena, args.data_dir))
    picked = _shard_slice(sorted(items, key=lambda it: it["id"]), args.shard)
    log("a1: %d item(s) this shard" % len(picked))

    pool = _EmitterPool(emitter_factory)
    emitter = pool.get(args.main_model)
    shard_i = _shard_index(args.shard)
    cache_path = _cache_path(args.out_dir, "a1", shard_i)
    cache = load_stage_cache(args.out_dir, "a1")
    run_stats = {}

    # 1) top draw (N=256), prefix gives all lower rungs
    for it in _progress(picked, total=len(picked), desc="a1-top"):
        recs, n_new, dt = _sample_item(emitter, it, it["subtask"], args.a1_top,
                                       args.base_seed, "base", cache_path, cache,
                                       run_stats=run_stats)
        vr = sc.vote_at_rung(recs, args.a1_top)
        log("  [a1-top] %-26s rung=%d mode_correct=%s tokens=%d sec=%.2f"
            % (it["id"], args.a1_top, vr.mode_correct, _sum_tokens(recs), dt))

    # 2) escalation set (N=top mode still wrong) -> deepen to 512/1024
    merged = load_stage_cache(args.out_dir, "a1")
    a1_by_item = _group_by_item(merged, "base")
    esc = stages.a1_escalation_set(a1_by_item)
    esc = [iid for iid in esc if iid in {it["id"] for it in picked}]     # only this shard's
    by_id = _items_by_id(items)
    max_deepen = max(args.a1_deepen) if args.a1_deepen else args.a1_top
    for iid in _progress(esc, total=len(esc), desc="a1-deepen"):
        it = by_id[iid]
        recs, n_new, dt = _sample_item(emitter, it, it["subtask"], max_deepen,
                                       args.base_seed, "base", cache_path, cache,
                                       run_stats=run_stats)
        log("  [a1-deepen] %-26s -> N=%d sec=%.2f" % (iid, max_deepen, dt))

    # --- analysis on merged cache ---
    merged = load_stage_cache(args.out_dir, "a1")
    a1_by_item = _group_by_item(merged, "base")
    attrition, passn, ledger, survivors = _a1_curves(a1_by_item, args, arena)

    eff = _efficiency(run_stats, len(picked))
    _print_summary("a1", {"arena": arena, "n_survivors": len(survivors),
                          "attrition": attrition}, eff)

    results = {
        "stage": "a1",
        "config": build_config(args, "a1", extra={"arena": arena}),
        "arena": arena,
        "attrition_curve": attrition,
        "pass_at_n_curve": passn,
        "token_ledger": ledger,
        "survivors": survivors,
        "n_survivors": len(survivors),
        "efficiency": eff,
        "run_stats": _run_stats_public(run_stats),
    }
    write_results(os.path.join(args.out_dir, "results_a1.json"), results)
    return results


def _a1_curves(a1_by_item, args, arena):
    """Attrition (mode-wrong count per rung), pass@N coverage (>=1 correct sample per rung),
    the token ledger, and the N=1024-equiv survivor set (escalated & still mode-wrong)."""
    rungs = [r for r in A1_PREFIX_RUNGS if r <= args.a1_top]
    deepen = sorted(set(args.a1_deepen))
    all_rungs = rungs + [r for r in deepen if r > args.a1_top]

    attrition = {}
    passn = {}
    for R_ in all_rungs:
        survivors_at_R = 0
        cov = 0
        n_items = 0
        for iid, recs in a1_by_item.items():
            if not recs:
                continue
            n_items += 1
            vr = sc.vote_at_rung(recs, R_)
            if not vr.mode_correct:
                survivors_at_R += 1
            if any(r.get("correct") for r in sorted(recs, key=lambda r: r.get("sample_idx", 0))[:R_]):
                cov += 1
        attrition[str(R_)] = survivors_at_R
        passn[str(R_)] = (cov / n_items) if n_items else 0.0

    # survivors = deepest-rung mode still wrong (== N=1024-equiv still-mode-wrong set)
    deepest = max(all_rungs) if all_rungs else args.a1_top
    survivors = sorted(iid for iid, recs in a1_by_item.items()
                       if recs and not sc.vote_at_rung(recs, deepest).mode_correct)

    ledger = {
        "total_new_tokens": sum(_sum_tokens(recs) for recs in a1_by_item.values()),
        "n_items": len(a1_by_item),
        "deepest_rung": deepest,
        "escalation_rung": stages.PREREG["a1_escalation_rung"],
    }
    return attrition, passn, ledger, survivors


def _resolve_arena(args):
    """Arena for a1/a2: --arena override, else results_a0.json 'selected_arena'."""
    if args.arena:
        return [s.strip() for s in args.arena.split(",") if s.strip()]
    a0_path = os.path.join(args.out_dir, "results_a0.json")
    if os.path.exists(a0_path):
        with open(a0_path) as f:
            a0 = json.load(f)
        sel = a0.get("selected_arena") or []
        if sel:
            return list(sel)
    raise RuntimeError("a1/a2: no arena (pass --arena or run --stage a0 first)")


# ======================================================================================
# STAGE: a2 (four-gate certification)
# ======================================================================================
def stage_a2(args, emitter_factory):
    arena = _resolve_arena(args) if not args.smoke else None
    banner("a2", "smoke=%s arena=%s" % (args.smoke, arena))

    # survivors + their base records + the item dicts
    if args.smoke:
        survivors, base_by_item, items_by_id, arena = _a2_smoke_survivors(args)
        log("a2[smoke]: fabricated %d survivor(s) from the a0 cache" % len(survivors))
    else:
        survivors, base_by_item, items_by_id = _a2_real_survivors(args, arena)
        log("a2: %d survivor(s) from results_a1.json" % len(survivors))

    # ladder/seed subsample cap (design §5): applied to the GLOBAL survivor set BEFORE sharding,
    # so a sharded a2 still honors the 60-item cap (each shard only generates the expensive
    # ladder/seed arms for its slice of the global-60). Computing it AFTER sharding would let
    # every shard's <=60 slice pass through in full, defeating the R1-Distill budget guard.
    global_survivors = sorted(survivors)
    subset = set(stages.ladder_seed_subsample(
        global_survivors, stages.PREREG["ladder_sample_cap"],
        stages.PREREG["subsample_base_seed"]))
    survivors = _shard_slice(global_survivors, args.shard)

    pool = _EmitterPool(emitter_factory)
    main = pool.get(args.main_model)
    shard_i = _shard_index(args.shard)
    cache_path = _cache_path(args.out_dir, "a2", shard_i)
    cache = load_stage_cache(args.out_dir, "a2")
    run_stats = {}

    stats_gates, facts_gates, ladder_gates, seed_gates = {}, {}, {}, {}
    primary_mode = {}
    subsample = [s for s in survivors if s in subset]

    # ---- main-model arms: stats (full set) + facts (full set) + seed2 (subsample) ----
    # main stays resident for the whole sweep (no per-item reload).
    for iid in _progress(survivors, total=len(survivors), desc="a2-main"):
        it = items_by_id.get(iid)
        if it is None:                                     # survivor id not in the loaded arena
            run_stats.setdefault("context_offenders", []).append(
                {"item_id": iid, "reason": "item not found in arena"})
            log("  [a2] %-26s SKIPPED (item dict not found)" % iid)
            continue
        base_recs = base_by_item.get(iid, [])
        primary_mode[iid] = sc.vote_at_rung(base_recs, args.a2_stats_rung).mode_idx

        # gate 1 — stats (on the N=1024-equiv base records; no new generation)
        stats_gates[iid] = stages.a2_gate_stats(base_recs, N=args.a2_stats_rung)

        # gate 2 — facts (facts-fed prompt, SC@8) + a non-driving greedy readout column (design
        # §5.2). The greedy sample is a diagnostic only; a2_gate_facts's decision is the SC@8 mode.
        facts_prompt = sc.format_facts_prompt(it)
        f_recs, _, _ = _sample_item(main, it, it["subtask"], args.a2_facts, args.base_seed,
                                    "facts", cache_path, cache,
                                    prompt=facts_prompt, run_stats=run_stats)
        g_recs, _, _ = _sample_item(main, it, it["subtask"], 1, args.base_seed,
                                    "facts_greedy", cache_path, cache,
                                    prompt=facts_prompt, greedy=True, run_stats=run_stats)
        facts_gates[iid] = stages.a2_gate_facts(f_recs, N=args.a2_facts, greedy_records=g_recs)

        # gate 4 — second seed (subsample only), SAME main model, distinct base seed
        if iid in subset:
            s2_recs, _, _ = _sample_item(main, it, it["subtask"], args.a2_seed,
                                         _seed2_base(args.base_seed), "seed2", cache_path, cache,
                                         run_stats=run_stats)
            seed_gates[iid] = stages.a2_gate_seed(s2_recs, primary_mode.get(iid), N=args.a2_seed)
        log("  [a2] %-26s stats=%s facts=%s(%s)"
            % (iid, stats_gates[iid].passed, facts_gates[iid].passed, facts_gates[iid].kind))

    # ---- gate 3 — ladder (subsample): model-OUTER / item-INNER so each model loads ONCE ----
    # Drop the main-model ref BEFORE the ladder sweep: the main arms are done, and holding `main`
    # live would keep Qwen3-4B's weights on the card while the ladder loads two adjacent 7B models
    # (pool eviction alone can't free a model an external local still references).
    del main
    _EmitterPool._release()
    ladder_recs = {iid: {} for iid in subsample}
    for model in args.ladder_models:
        em = pool.get(model)                               # one load per ladder model
        short = model.split("/")[-1]
        for iid in _progress(subsample, desc="a2-ladder:%s" % short):
            it = items_by_id.get(iid)
            if it is None:
                continue
            recs, _, _ = _sample_item(em, it, it["subtask"], args.a2_ladder, args.base_seed,
                                      "ladder:%s" % short, cache_path, cache, run_stats=run_stats)
            ladder_recs[iid][model] = recs
    for iid in subsample:
        if ladder_recs.get(iid):
            ladder_gates[iid] = stages.a2_gate_ladder(iid, ladder_recs[iid], N=args.a2_ladder)
            log("  [a2] %-26s ladder=%s seed=%s"
                % (iid, ladder_gates[iid].classification,
                   seed_gates.get(iid).passed if iid in seed_gates else None))

    # Only certify survivors that were actually stats+facts gated (a skipped/absent item is a
    # context_offender, not an assembly bug -> excluded here so certify's missing-gate guard
    # keeps firing on GENUINE assembly bugs). In practice every survivor is present.
    processed = [s for s in survivors if s in stats_gates]
    verdict = stages.certify(processed, stats_gates, facts_gates, ladder_gates, seed_gates)
    ledger = _certify_to_dict(verdict)
    ledger["excluded_offenders"] = [s for s in survivors if s not in stats_gates]

    eff = _efficiency(run_stats, len(survivors))
    _print_summary("a2", {"n_certified": verdict.n_certified, "verdict": verdict.verdict}, eff)

    phase_b = {
        "certified_ids": verdict.commitment_certified,
        "record_paths": {
            "a1_base_cache": sorted(glob.glob(os.path.join(args.out_dir, "cache_a1*.jsonl"))),
            "a2_cache": sorted(glob.glob(os.path.join(args.out_dir, "cache_a2*.jsonl"))),
        },
        "arena": arena,
    }
    results = {
        "stage": "a2",
        "config": build_config(args, "a2", extra={"arena": arena}),
        "ledger": ledger,
        "verdict": verdict.verdict,
        "gates": _gate_summaries(stats_gates, facts_gates, ladder_gates, seed_gates),
        "phase_b_interface": phase_b,
        "efficiency": eff,
        "run_stats": _run_stats_public(run_stats),
    }
    write_results(os.path.join(args.out_dir, "results_a2.json"), results)
    return results


def _a2_real_survivors(args, arena):
    a1_path = os.path.join(args.out_dir, "results_a1.json")
    if not os.path.exists(a1_path):
        raise RuntimeError("a2: results_a1.json not found (run --stage a1 first)")
    with open(a1_path) as f:
        a1 = json.load(f)
    survivors = list(a1.get("survivors") or [])
    base_by_item = _group_by_item(load_stage_cache(args.out_dir, "a1"), "base")
    items_by_id = _items_by_id(load_items(args.data_dir, subtasks=arena))
    return survivors, base_by_item, items_by_id


def _a2_smoke_survivors(args):
    """Smoke: fabricate 2 survivors from the a0 cache (design/Subtask-4 spec)."""
    cache = load_stage_cache(args.out_dir, "a0")
    base_by_item = _group_by_item(cache, "base")
    ids = sorted(base_by_item.keys())[:2]
    if not ids:
        raise RuntimeError("a2[smoke]: a0 cache empty (run --stage a0 --smoke first)")
    items_by_id = _items_by_id(load_items(args.data_dir))
    arena = sorted({items_by_id[i]["subtask"] for i in ids if i in items_by_id})
    return ids, base_by_item, items_by_id, arena


def _certify_to_dict(v):
    return {
        "commitment_certified": v.commitment_certified,
        "knowledge_type": v.knowledge_type,
        "parameter_maskable": v.parameter_maskable,
        "luck_type": v.luck_type,
        "seed_unstable": v.seed_unstable,
        "n_certified": v.n_certified,
        "verdict": v.verdict,
        "sampling": v.sampling,
    }


def _gate_summaries(stats_gates, facts_gates, ladder_gates, seed_gates):
    out = {}
    for iid, g in stats_gates.items():
        out.setdefault(iid, {})["stats"] = {"passed": g.passed, "corr_ci": list(g.corr_ci), "n": g.n}
    for iid, g in facts_gates.items():
        out.setdefault(iid, {})["facts"] = {"passed": g.passed, "kind": g.kind,
                                            "mode_correct": g.mode_correct}
    for iid, g in ladder_gates.items():
        out.setdefault(iid, {})["ladder"] = {"passed": g.passed, "classification": g.classification,
                                             "solved_by": g.solved_by, "all_sticky": g.all_sticky}
    for iid, g in seed_gates.items():
        out.setdefault(iid, {})["seed"] = {"passed": g.passed, "seed2_mode_idx": g.seed2_mode_idx,
                                           "primary_mode_idx": g.primary_mode_idx}
    return out


# ======================================================================================
# STAGE: a3 (self-facts control — three arms S1/S2/O + per-story greedy extraction)
# ======================================================================================
# The zero-oracle prompting THREAT test (design 2026-07-06-selffacts-a3.md). Mirrors stage_a2's
# main-resident structure: one main model stays loaded for all three arms; there is NO training,
# NO checkpoint, NO ladder — this is an eval-only SC contrast. Three arms share the frozen
# ``selffacts`` prompt layer:
#   * S1 (seed_tag "selffacts_s1") — one-call scaffold prompt, capped at S1_MAX_NEW_TOKENS, with a
#     400-char ``text_head`` kept for the scaffold-compliance diagnostic.
#   * S2 (seed_tag "selffacts_s2") — one greedy per-story extraction injected into the oracle's
#     FACTS wrapper (default 2048 cap).
#   * O  (seed_tag "facts") — the oracle arm, RESUMED from a2's facts records (0..7) by overlaying
#     them into the in-memory working cache, so O only draws 8..a3_n and reuses the a2 draws.
def stage_a3(args, emitter_factory):
    banner("a3", "smoke=%s shard=%s a3_n=%d" % (args.smoke, args.shard, args.a3_n))

    # --- scope: certified + knowledge (results_a2.json ledger, or a smoke fabrication) ---------
    if args.smoke:
        certified, knowledge, items_by_id = _a3_smoke_scope(args)
        log("a3[smoke]: fabricated %d certified + %d knowledge from the a0 cache"
            % (len(certified), len(knowledge)))
    else:
        certified, knowledge = _a3_real_scope(args)       # asserts the (26,46) set completeness
        items_by_id = _items_by_id(load_items(args.data_dir))
        log("a3: %d certified + %d knowledge from results_a2.json"
            % (len(certified), len(knowledge)))

    certified = sorted(certified)
    knowledge = sorted(knowledge)
    certified_set = set(certified)
    inscope = sorted(certified + knowledge)
    inscope_shard = _shard_slice(inscope, args.shard)     # O arm runs only on this shard's certified

    pool = _EmitterPool(emitter_factory)
    main = pool.get(args.main_model)
    shard_i = _shard_index(args.shard)
    cache_path = _cache_path(args.out_dir, "a3", shard_i)

    # Working cache = a3 records OVERLAID (in-memory only) with a2's seed_tag=="facts" records. The
    # O arm reuses seed_tag "facts" + the IDENTICAL oracle prompt (sc.format_facts_prompt), so the
    # overlay makes ``completed_idxs`` see a2's 0..7 already done -> O only draws 8..a3_n. New O draws
    # append to cache_a3 ALONE (the a2 files are never rewritten); the later vote reads all a3_n via
    # ``sc.cached_records`` over this overlaid cache (0..7 from a2 + 8..a3_n from a3).
    cache = load_stage_cache(args.out_dir, "a3")
    a2_cache = load_stage_cache(args.out_dir, "a2")
    n_overlay = 0
    for k, r in a2_cache.items():
        if r.get("seed_tag") == "facts" and k not in cache:
            cache[k] = r
            n_overlay += 1
    log("a3: overlaid %d a2 'facts' record(s) into the working cache (O-arm resume)" % n_overlay)
    run_stats = {}

    # --- per-story greedy extraction, BEFORE the sampling loop (one extraction per story) -------
    # Question-agnostic (narrative only) + greedy + cached by story_id, so every question of a story
    # reuses the SAME extracted observation list.
    extract_path = _a3_extract_cache_path(args.out_dir, shard_i)
    extract_cache = _load_a3_extracts(args.out_dir)
    stories = sorted({sf.story_id(iid) for iid in inscope_shard})
    extract_map = {}
    for sid in stories:
        if sid in extract_cache:
            extract_map[sid] = extract_cache[sid].get("text", "")
            continue
        rep_iid = min(iid for iid in inscope_shard if sf.story_id(iid) == sid)
        rep_item = items_by_id.get(rep_iid)
        if rep_item is None:                              # story item absent from arena -> record + skip
            run_stats.setdefault("context_offenders", []).append(
                {"item_id": rep_iid, "story_id": sid, "reason": "extraction item not found in arena"})
            extract_map[sid] = ""
            continue
        seed = sc.sample_seed(args.base_seed, rep_item["subtask"], sid, 0)
        # The extraction is a raw greedy generate (not via _sample_item), so guard OOM here so one
        # story cannot abort the shard: on OOM record an offender + treat the extract as empty (S2
        # then injects empty facts for that story — degenerate but non-fatal, already tracked).
        t0 = time.time()
        try:
            outs = main.generate([sf.format_extract_prompt(rep_item)], [seed],
                                 greedy=True, max_new_tokens=sf.EXTRACT_MAX_NEW_TOKENS)
        except sc.EmitterOOM as exc:
            run_stats.setdefault("extraction_oom", []).append({"story_id": sid, "error": str(exc)})
            run_stats["seconds"] = run_stats.get("seconds", 0.0) + (time.time() - t0)
            extract_map[sid] = ""
            log("OOM-skipped extraction story=%s (%s)" % (sid, exc))
            continue
        ntok = int(outs[0].n_new_tokens)
        text = outs[0].text
        rec = {"story_id": sid, "subtask": rep_item["subtask"], "rep_item_id": rep_iid,
               "text": text, "n_new_tokens": ntok}
        sf.append_extract(extract_path, rec)
        extract_cache[sid] = rec
        extract_map[sid] = text
        # Fold the extraction's GPU cost into the SAME counters _sample_item feeds, so _efficiency
        # (gpu_hours / tokens_total / mean_tokens_per_sample) reflects the true cost. Extraction is
        # cache-hit on a rerun, so this does not break the "2nd run generates 0" resume invariant.
        run_stats["seconds"] = run_stats.get("seconds", 0.0) + (time.time() - t0)
        run_stats["n_generated"] = run_stats.get("n_generated", 0) + 1
        run_stats["tokens_total"] = run_stats.get("tokens_total", 0) + ntok
        run_stats["n_extractions"] = run_stats.get("n_extractions", 0) + 1
        run_stats["extraction_tokens"] = run_stats.get("extraction_tokens", 0) + ntok
    run_stats["extraction_degenerate_stories"] = sum(
        1 for t in extract_map.values() if sf.extraction_degenerate(t))

    # --- three-arm sampling loop (main stays resident): S1 + S2 all in-scope; O certified only ---
    for iid in _progress(inscope_shard, total=len(inscope_shard), desc="a3"):
        it = items_by_id.get(iid)
        if it is None:
            run_stats.setdefault("context_offenders", []).append(
                {"item_id": iid, "reason": "item not found in arena"})
            log("  [a3] %-26s SKIPPED (item dict not found)" % iid)
            continue
        extract_text = extract_map.get(sf.story_id(iid), "")
        s1_recs, _, _ = _sample_item(main, it, it["subtask"], args.a3_n, args.base_seed,
                                     "selffacts_s1", cache_path, cache,
                                     prompt=sf.format_s1_prompt(it),
                                     max_new_tokens=sf.S1_MAX_NEW_TOKENS,
                                     keep_text_chars=400, run_stats=run_stats)
        s2_recs, _, _ = _sample_item(main, it, it["subtask"], args.a3_n, args.base_seed,
                                     "selffacts_s2", cache_path, cache,
                                     prompt=sf.format_s2_prompt(it, extract_text),
                                     run_stats=run_stats)
        o_fix = None
        if iid in certified_set:
            o_recs, _, _ = _sample_item(main, it, it["subtask"], args.a3_n, args.base_seed,
                                        "facts", cache_path, cache,
                                        prompt=sc.format_facts_prompt(it), run_stats=run_stats)
            o_fix = stages.a3_stable_fix(o_recs, N=args.a3_n)
        s1_fix = stages.a3_stable_fix(s1_recs, N=args.a3_n)
        s2_fix = stages.a3_stable_fix(s2_recs, N=args.a3_n)
        log("  [a3] %-26s s1_fix=%s s2_fix=%s o_fix=%s" % (iid, s1_fix, s2_fix, o_fix))

    # --- verdict on the merged working cache (canonical in the no-shard consolidation pass) -----
    s1_recs = _a3_gather(cache, items_by_id, inscope, "selffacts_s1")
    s2_recs = _a3_gather(cache, items_by_id, inscope, "selffacts_s2")
    o_recs = _a3_gather(cache, items_by_id, certified, "facts")

    # Scope the verdict to items actually COVERED this pass by the FIX-DETERMINING arms (S1/S2)
    # ONLY. The O arm is NOT a coverage signal: its "facts" records are overlaid from a2 for the
    # WHOLE certified set (0..7, the resume) regardless of shard, so keying coverage on o_recs would
    # mark every certified item covered even though S1/S2 hold just this shard's slice -> a3_verdict
    # would then KeyError on a certified item outside the shard. Keying on [s1_recs, s2_recs] makes
    # cert_scope == the shard's actually-processed certified slice; a3_verdict requires o_recs[cid]
    # only for cid in that slice (o_recs has it; any extra overlaid keys are harmless). A fully-
    # skipped offender is excluded + recorded (not a bug); an item covered on S1 XOR S2 still trips
    # a3_verdict's KeyError guard (a genuine assembly bug). Mirrors a2's `processed` handling.
    def _covered(ids, dicts):
        scoped, excluded = [], []
        for iid in ids:
            (scoped if any(iid in d for d in dicts) else excluded).append(iid)
        return scoped, excluded
    cert_scope, cert_excl = _covered(certified, [s1_recs, s2_recs])
    know_scope, know_excl = _covered(knowledge, [s1_recs, s2_recs])
    verdict = stages.a3_verdict(cert_scope, know_scope, s1_recs, s2_recs, o_recs)

    # --- diagnostics (never gating): S1 scaffold compliance + extraction degeneracy -------------
    s1_all = [r for recs in s1_recs.values() for r in recs]
    n_s1 = len(s1_all)
    n_compliant = sum(1 for r in s1_all if sf.scaffold_present(r.get("text_head")))
    s1_scaffold_compliance = (n_compliant / n_s1) if n_s1 else 0.0
    n_degenerate = run_stats.get("extraction_degenerate_stories", 0)

    eff = _efficiency(run_stats, len(inscope_shard))
    _print_summary("a3", {"n_residual": verdict.n_residual, "verdict": verdict.verdict,
                          "tripwire_fired": verdict.tripwire_fired}, eff)

    ledger = {
        # scope = the set actually PROCESSED + scored this pass. In a --shard run that is the
        # shard's strided slice; the canonical no-shard consolidation pass processes the full set,
        # so its scope == the full certified/knowledge sets (and cert_excl/know_excl are then the
        # genuinely-skipped offenders, not sibling-shard items).
        "scope": {"certified": list(cert_scope), "knowledge": list(know_scope)},
        "fixed_by": verdict.fixed_by,
        "residual": verdict.residual,
        "n_residual": verdict.n_residual,
        "knowledge_fix_rate": verdict.knowledge_fix_rate,
        "fragile_fix": verdict.fragile_fix,
        "tripwire_fired": verdict.tripwire_fired,
        "s1_scaffold_compliance": s1_scaffold_compliance,
        "extraction_degenerate_stories": n_degenerate,
        # extraction compute is FOLDED into efficiency (gpu_hours/tokens_total); this is the
        # legible per-arm breakdown of that fold (a3-only; not in the shared run_stats schema).
        "extraction_compute": {
            "n_extractions": run_stats.get("n_extractions", 0),
            "extraction_tokens": run_stats.get("extraction_tokens", 0),
            "extraction_oom": run_stats.get("extraction_oom", []),
        },
        "excluded": {"certified": cert_excl, "knowledge": know_excl},
    }
    phase_b_patch = {
        "residual_ids": verdict.residual,
        "fixed_ids": {iid: arms for iid, arms in verdict.fixed_by.items() if arms},
        "fragile_fix": verdict.fragile_fix,
        "scaffold_baseline": {
            "s1_scaffold_compliance": s1_scaffold_compliance,
            "n_s1_samples": n_s1,
            "extraction_degenerate_stories": n_degenerate,
            "n_stories": len(stories),
        },
    }
    extra = {
        "arms": {"S1": "selffacts_s1", "S2": "selffacts_s2", "O": "facts"},
        "a3_n": args.a3_n,
        "s1_template_sha256": sf.sha256_text(sf.S1_TEMPLATE),
        "extract_template_sha256": sf.sha256_text(sf.EXTRACT_TEMPLATE),
        "s1_max_new_tokens": sf.S1_MAX_NEW_TOKENS,
        "extract_max_new_tokens": sf.EXTRACT_MAX_NEW_TOKENS,
        "prereg_a3": dict(stages.PREREG_A3),
        "s2_composition": "FACTS_TEMPLATE+FACTS_PREAMBLE verbatim",
    }
    results = {
        "stage": "a3",
        "config": build_config(args, "a3", extra=extra),
        "ledger": ledger,
        "verdict": verdict.verdict,
        "phase_b_patch": phase_b_patch,
        "efficiency": eff,
        "run_stats": _run_stats_public(run_stats),
    }
    assert_finite(results)
    write_results(os.path.join(args.out_dir, "results_a3.json"), results)
    return results


def _a3_smoke_scope(args):
    """Smoke: fabricate 2 certified + 1 knowledge from the a0 cache (mirror ``_a2_smoke_survivors``);
    the (26,46) count check is SKIPPED under --smoke."""
    cache = load_stage_cache(args.out_dir, "a0")
    base_by_item = _group_by_item(cache, "base")
    ids = sorted(base_by_item.keys())
    if len(ids) < 3:
        raise RuntimeError("a3[smoke]: a0 cache has <3 items (run --stage a0 --smoke first)")
    certified = ids[:2]
    knowledge = ids[2:3]
    items_by_id = _items_by_id(load_items(args.data_dir))
    return certified, knowledge, items_by_id


def _a3_real_scope(args):
    """Non-smoke: read the certified + knowledge id sets from results_a2.json's ledger and assert
    their counts == ``PREREG_A3['a3_expected_counts']`` (26, 46). A mismatch is an assembly /
    provenance error (wrong or partial a2 ledger), RAISED — never proceed on a wrong scope."""
    a2_path = os.path.join(args.out_dir, "results_a2.json")
    if not os.path.exists(a2_path):
        raise RuntimeError("a3: results_a2.json not found (run --stage a2 first)")
    with open(a2_path) as f:
        a2 = json.load(f)
    ledger = a2.get("ledger", {})
    certified = list(ledger.get("commitment_certified") or [])
    knowledge = list(ledger.get("knowledge_type") or [])
    expected = tuple(stages.PREREG_A3["a3_expected_counts"])
    got = (len(certified), len(knowledge))
    if got != expected:
        raise RuntimeError(
            "a3: scope count mismatch (certified,knowledge)=%s != a3_expected_counts=%s "
            "(results_a2.json provenance / assembly error)" % (got, expected))
    return certified, knowledge


def _a3_gather(cache, items_by_id, ids, seed_tag):
    """``{item_id: sample_idx-ordered records}`` for ``seed_tag`` over ``ids`` present in ``cache``.
    Items with ZERO records are OMITTED, so ``a3_verdict``'s missing-arm KeyError still fires on a
    genuine assembly bug (an item covered on one arm but missing this one)."""
    out = {}
    for iid in ids:
        it = items_by_id.get(iid)
        if it is None:
            continue
        recs = sc.cached_records(cache, it["subtask"], iid, seed_tag)
        if recs:
            out[iid] = recs
    return out


# ======================================================================================
# ============================  PHASE B  (b0 / b1 / b2)  ================================
# ======================================================================================
# The five-gate funnel assembled on Subtasks 1-5. b0 = answer-level verifier + G5a; b1 =
# structured belief-table pool + preservation hard-kill; b2 = step-level verifier + G1/G2/G3 +
# the SAVI@lambda* vs token-matched SC main readout. Generation reuses ``_sample_item`` (OOM /
# resume / efficiency) with the full text kept on each record (``B_TEXT_KEEP``); the featurizer
# (the sole extra torch site) is dependency-injected exactly like the emitter.

def default_featurizer_factory(args):
    """Production featurizer factory (``model_name -> HiddenStateFeaturizer``). Lazy GPU site."""
    def _factory(model_name):
        return fz.HiddenStateFeaturizer(model_name, device=args.device,
                                        batch=getattr(args, "featurize_batch", 8))
    return _factory


def _b_prompt_free(item):
    return sc.format_prompt(item)                 # Phase A free-CoT prompt (b0 answer verifier)


def _b_prompt_belief(item):
    return bs.format_belief_prompt(item)          # frozen belief-table CoT prompt (b1 structured)


# ---- patient set / survivors / split -----------------------------------------------------------
def _b_patient_ids(args, items_by_id):
    """The residual patient question-ids (design §2): a3's ``phase_b_patch.residual_ids`` (21).
    Under --smoke, fabricate 2 patients from the loaded object_placements items."""
    if args.smoke:
        op = sorted(i for i, it in items_by_id.items() if it["subtask"] == "object_placements")
        if len(op) < 2:
            raise RuntimeError("b[smoke]: need >=2 object_placements items in the data dir")
        return op[:2]
    path = os.path.join(args.out_dir, "results_a3.json")
    if not os.path.exists(path):
        raise RuntimeError("b: results_a3.json not found (run --stage a3 first for patient ids)")
    with open(path) as f:
        a3 = json.load(f)
    ids = list((a3.get("phase_b_patch") or {}).get("residual_ids") or [])
    if not ids:
        raise RuntimeError("b: results_a3.json has no phase_b_patch.residual_ids")
    return sorted(ids)


def _b_survivor_ids(args):
    """A1 still-wrong item-ids (``results_a1.json['survivors']``) — hard items routed to tuning."""
    path = os.path.join(args.out_dir, "results_a1.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return list(json.load(f).get("survivors") or [])
    except Exception:
        return []


def _b_split(args, items, patient_ids):
    """Deterministic story-level split (leakage guard) + smoke item caps (2 each)."""
    survivor_ids = _b_survivor_ids(args)
    split = pbs.build_split(items, patient_ids, base_seed=B_SPLIT_SEED, survivor_ids=survivor_ids)
    if args.smoke:                                 # tiny: 2 train items, 2 calib, 2 tuning
        split["train"] = split["train"][:2]
        split["calib"] = split["calib"][:2]
        split["tuning"] = split["tuning"][:2]
    return split


# ---- generation / pool assembly ----------------------------------------------------------------
def _b_generate_pool(emitter, items_by_id, ids, N, seed_tag, prompt_fn, args, cache_path,
                     cache, gen_stats, desc):
    """Draw N belief/free-CoT samples per id into ``seed_tag`` (full text kept), resume-aware."""
    for iid in _progress(ids, total=len(ids), desc=desc):
        it = items_by_id.get(iid)
        if it is None:
            gen_stats.setdefault("context_offenders", []).append(
                {"item_id": iid, "reason": "item not in loaded arena"})
            continue
        recs, n_new, dt = _sample_item(
            emitter, it, it["subtask"], N, args.base_seed, seed_tag, cache_path, cache,
            prompt=prompt_fn(it), keep_text_chars=B_TEXT_KEEP,
            max_new_tokens=args.b_max_new_tokens, run_stats=gen_stats)
        log("  [%s] %-30s N=%d new=%d sec=%.2f" % (desc, iid, N, n_new, dt))


def _b_pool_recs(cache, items_by_id, ids, seed_tag):
    """``{item_id: sample_idx-ordered records}`` for ``seed_tag`` over ``ids`` present in cache."""
    out = {}
    for iid in ids:
        it = items_by_id.get(iid)
        if it is None:
            continue
        recs = sc.cached_records(cache, it["subtask"], iid, seed_tag)
        if recs:
            out[iid] = recs
    return out


def _rec_text(rec):
    """The (effectively full) generated text stored on a Phase B record."""
    return rec.get("text_head") or rec.get("text") or ""


def _b_collect(pool_recs, items_by_id, prompt_fn):
    """Flatten a pool into parallel (items, prompts, texts, labels, seeds) sample lists."""
    items, prompts, texts, labels, seeds = [], [], [], [], []
    for iid, recs in pool_recs.items():
        it = items_by_id[iid]
        pr = prompt_fn(it)
        for r in recs:
            items.append(it)
            prompts.append(pr)
            texts.append(_rec_text(r))
            labels.append(bool(r.get("correct")))
            seeds.append(int(r.get("seed", 0)))
    return items, prompts, texts, labels, seeds


# ---- efficiency (GENERATION vs VERIFIER-SCORING gpu-seconds, reported separately) --------------
def _score_timed(fn, score_stats, n_scored=0):
    """Run ``fn`` (a featurizer/critic scoring call), folding wall-time into ``score_stats``."""
    t0 = time.time()
    res = fn()
    score_stats["seconds"] = score_stats.get("seconds", 0.0) + (time.time() - t0)
    score_stats["n_scored"] = score_stats.get("n_scored", 0) + int(n_scored)
    return res


def _b_efficiency(gen_stats, score_stats, n_items):
    """Efficiency block with GENERATION and VERIFIER-SCORING GPU-seconds in SEPARATE sub-blocks."""
    gen_secs = gen_stats.get("seconds", 0.0)
    sco_secs = score_stats.get("seconds", 0.0)
    return {
        "n_items": n_items,
        "generation": {
            "gpu_seconds": gen_secs,
            "gpu_hours": gen_secs / 3600.0,
            "n_samples_generated": gen_stats.get("n_generated", 0),
            "tokens_total": gen_stats.get("tokens_total", 0),
            "oom_skipped": len(gen_stats.get("oom_skipped", [])),
        },
        "verifier_scoring": {
            "gpu_seconds": sco_secs,
            "gpu_hours": sco_secs / 3600.0,
            "n_scored": score_stats.get("n_scored", 0),
        },
        "total_gpu_hours": (gen_secs + sco_secs) / 3600.0,
    }


# ---- answer/step verifier fitting + selection (leakage-guarded) --------------------------------
def _fit_probe(featurizer, prompts_tr, texts_tr, y_tr, prompts_ca, texts_ca, y_ca,
               score_stats, save_path=None):
    """Featurize train/calib last-token features and fit a ``ProbeVerifier`` (answer-level).
    Returns ``(probe_or_None, calib_auc_or_None)``; a fit failure (e.g. single-class train)
    degrades to ``(None, None)`` so form-selection falls back to the critic."""
    try:
        X_tr = _score_timed(lambda: featurizer.featurize_last_token(prompts_tr, texts_tr),
                            score_stats, len(prompts_tr))
        X_ca = _score_timed(lambda: featurizer.featurize_last_token(prompts_ca, texts_ca),
                            score_stats, len(prompts_ca))
        probe = vf.ProbeVerifier().fit(X_tr, y_tr, X_ca, y_ca)
        if save_path:
            probe.save(save_path)
        return probe, float(probe.val_auc)
    except Exception as exc:
        log("  probe fit degraded -> critic-only (%s)" % exc)
        return None, None


# ======================================================================================
# STAGE: b0 — answer-level verifier + G5a discrimination info-gate (design §3 / B0)
# ======================================================================================
def stage_b0(args, emitter_factory, featurizer_factory):
    banner("b0", "smoke=%s shard=%s patient_n=%d train_n=%d"
           % (args.smoke, args.shard, args.b0_patient_n, args.b0_train_n))
    items = load_items(args.data_dir, subtasks=PHASE_B_ARENA)
    if not items:
        raise RuntimeError("b0: no object_placements items under %s" % args.data_dir)
    items_by_id = _items_by_id(items)
    patient_ids = _b_patient_ids(args, items_by_id)
    split = _b_split(args, items, patient_ids)
    log("b0: %d patient(s), split train=%d calib=%d tuning=%d (nonpatient stories=%d)"
        % (len(patient_ids), len(split["train"]), len(split["calib"]),
           len(split["tuning"]), split["n_nonpatient_stories"]))

    pool = _EmitterPool(emitter_factory)
    emitter = pool.get(args.main_model)
    featurizer = featurizer_factory(args.main_model)
    shard_i = _shard_index(args.shard)
    cache_path = _cache_path(args.out_dir, "b0", shard_i)
    cache = load_stage_cache(args.out_dir, "b0")
    gen_stats, score_stats = {}, {}

    # 1) generate free-CoT pools (patient N=256; train/calib N=64), full text kept, resumable.
    pat_shard = _shard_slice(sorted(patient_ids), args.shard)
    _b_generate_pool(emitter, items_by_id, pat_shard, args.b0_patient_n, "b0_patient",
                     _b_prompt_free, args, cache_path, cache, gen_stats, "b0-patient")
    _b_generate_pool(emitter, items_by_id, _shard_slice(split["train"], args.shard),
                     args.b0_train_n, "b0_train", _b_prompt_free, args, cache_path, cache,
                     gen_stats, "b0-train")
    _b_generate_pool(emitter, items_by_id, _shard_slice(split["calib"], args.shard),
                     args.b0_train_n, "b0_calib", _b_prompt_free, args, cache_path, cache,
                     gen_stats, "b0-calib")

    # --- analysis on the MERGED cache (all shards present) ---
    merged = load_stage_cache(args.out_dir, "b0")
    train_recs = _b_pool_recs(merged, items_by_id, split["train"], "b0_train")
    calib_recs = _b_pool_recs(merged, items_by_id, split["calib"], "b0_calib")
    pat_recs = _b_pool_recs(merged, items_by_id, patient_ids, "b0_patient")

    # 2) LEAKAGE GUARDS — question-id membership AND story-level disjointness (design §2).
    held_ids = set(patient_ids) | set(split["tuning"])
    vf.assert_no_leakage(list(train_recs), list(calib_recs), held_ids)
    pbs.assert_story_level_disjoint(split)

    # 3) fit the two answer-verifier forms; select on CALIBRATION AUC only.
    tr_items, tr_prompts, tr_texts, tr_y, tr_seeds = _b_collect(train_recs, items_by_id, _b_prompt_free)
    ca_items, ca_prompts, ca_texts, ca_y, ca_seeds = _b_collect(calib_recs, items_by_id, _b_prompt_free)
    probe_path = os.path.join(args.out_dir, "verifier_b0_answer_probe.joblib")
    probe, probe_auc = _fit_probe(featurizer, tr_prompts, tr_texts, tr_y,
                                  ca_prompts, ca_texts, ca_y, score_stats, save_path=probe_path)
    critic = vf.CriticVerifier(emitter)
    crit_ca = _score_timed(lambda: critic.score_chains(ca_items, ca_texts, ca_seeds),
                           score_stats, len(ca_items))
    crit_auc = vf.roc_auc(crit_ca, ca_y)
    candidates = {"critic": {"val_auc": crit_auc}}
    if probe is not None:
        candidates["probe"] = {"val_auc": probe_auc}
    winner, report = vf.select_verifier(candidates, train_ids=list(train_recs),
                                        calib_ids=list(calib_recs), held_ids=held_ids)
    log("b0: answer-verifier winner=%s calib_auc=%.3f (probe=%s critic=%.3f)"
        % (winner, candidates[winner]["val_auc"], probe_auc, crit_auc))

    # 4) score the patient pool ONCE with the selected verifier -> G5a.
    pa_items, pa_prompts, pa_texts, pa_y, pa_seeds, pa_ids = [], [], [], [], [], []
    for iid, recs in pat_recs.items():
        it = items_by_id[iid]
        for r in recs:
            pa_items.append(it); pa_prompts.append(_b_prompt_free(it)); pa_texts.append(_rec_text(r))
            pa_y.append(bool(r.get("correct"))); pa_seeds.append(int(r.get("seed", 0))); pa_ids.append(iid)
    if winner == "probe" and probe is not None:
        pat_scores = _score_timed(
            lambda: probe.score(featurizer.featurize_last_token(pa_prompts, pa_texts)),
            score_stats, len(pa_prompts))
    else:
        pat_scores = _score_timed(lambda: critic.score_chains(pa_items, pa_texts, pa_seeds),
                                  score_stats, len(pa_items))
    sample_rows = [{"item_id": iid, "score": float(s), "correct": bool(y)}
                   for iid, s, y in zip(pa_ids, pat_scores, pa_y)]
    sc_modes = {iid: sc.vote_at_rung(recs, args.b0_patient_n).mode_correct
                for iid, recs in pat_recs.items()}
    g5a = sb.gate_g5a(sample_rows, sc_modes)

    # 5) zero-coverage sub-class (from the a1 patient base cache) + coverage baseline table.
    zero_cov = _b_zero_coverage(args, patient_ids)
    coverage = {iid: {"n": len(recs), "n_correct": sum(1 for r in recs if r.get("correct")),
                      "pass_at_n": any(r.get("correct") for r in recs)}
                for iid, recs in pat_recs.items()}

    eff = _b_efficiency(gen_stats, score_stats, len(patient_ids))
    _b_print_summary("b0", {"g5a": g5a.status, "winner": winner,
                            "calib_auc": candidates[winner]["val_auc"],
                            "zero_cov": zero_cov}, eff)

    results = {
        "stage": "b0",
        "config": _b_config(args, "b0"),
        "split": {k: split[k] for k in ("patients", "patient_stories", "tuning_stories",
                                        "train_stories", "calib_stories", "tuning", "train",
                                        "calib", "counts", "n_nonpatient_stories")},
        "g5a": g5a.to_dict(),
        "selected_answer_verifier": {
            "form": winner, "calib_auc": candidates[winner]["val_auc"], "report": report,
            "probe_path": probe_path if (winner == "probe" and probe is not None) else None,
            "probe_layer": (probe.layer if (winner == "probe" and probe is not None) else None),
            "probe_calib_auc": probe_auc, "critic_calib_auc": crit_auc,
        },
        "zero_coverage_ids": zero_cov,
        "coverage_baseline": coverage,
        "efficiency": eff,
        "run_stats": _run_stats_public(gen_stats),
    }
    results, n_nf = sanitize_nonfinite(results)
    if n_nf:
        log("b0: sanitized %d non-finite CI sentinel(s) -> null before write" % n_nf)
    write_results(os.path.join(args.out_dir, "results_b0.json"), results)
    return results


def _b_zero_coverage(args, patient_ids):
    """The zero-coverage sub-class: patient ids with 0 correct samples in the a1 base cache
    (design §2). Absent a1 cache (smoke) -> [] (recorded; the real 4 ids come from the full run)."""
    a1_cache = load_stage_cache(args.out_dir, "a1")
    pat = set(patient_ids)
    rows = [{"item_id": r["item_id"], "correct": bool(r.get("correct"))}
            for r in a1_cache.values()
            if r.get("seed_tag") == "base" and r.get("item_id") in pat]
    return sb.zero_coverage_ids(rows) if rows else []


# ======================================================================================
# STAGE: b1 — structured belief-table pool + preservation hard-kill (design §3 / B1)
# ======================================================================================
def stage_b1(args, emitter_factory, featurizer_factory=None):
    banner("b1", "smoke=%s shard=%s patient_n=%d tuning_n=%d train_n=%d"
           % (args.smoke, args.shard, args.b1_patient_n, args.b1_tuning_n, args.b1_train_n))
    items = load_items(args.data_dir, subtasks=PHASE_B_ARENA)
    if not items:
        raise RuntimeError("b1: no object_placements items under %s" % args.data_dir)
    items_by_id = _items_by_id(items)
    patient_ids = _b_patient_ids(args, items_by_id)
    split = _b_split(args, items, patient_ids)

    pool = _EmitterPool(emitter_factory)
    emitter = pool.get(args.main_model)
    shard_i = _shard_index(args.shard)
    cache_path = _cache_path(args.out_dir, "b1", shard_i)
    cache = load_stage_cache(args.out_dir, "b1")
    gen_stats = {}

    # 1) patient pool @ N=64 -> preservation pre-check.
    pat_shard = _shard_slice(sorted(patient_ids), args.shard)
    _b_generate_pool(emitter, items_by_id, pat_shard, args.b1_patient_n, "b1_patient",
                     _b_prompt_belief, args, cache_path, cache, gen_stats, "b1-patient")
    merged = load_stage_cache(args.out_dir, "b1")
    pat_recs = _b_pool_recs(merged, items_by_id, patient_ids, "b1_patient")
    pool_rows = [{"item_id": iid,
                  "sc_mode_correct": sc.vote_at_rung(recs, args.b1_patient_n).mode_correct}
                 for iid, recs in pat_recs.items()]
    pres = sb.gate_preservation(pool_rows)
    sticky_ids = pres.detail["sticky_ids"]
    log("b1: preservation status=%s sticky=%d/%d" % (pres.status, len(sticky_ids), len(pool_rows)))

    # 2) tuning + train + calib structured pools (ALWAYS generated — b2 needs them even when
    #    preservation TERMINATEs, so terminating only skips the expensive patient escalation).
    _b_generate_pool(emitter, items_by_id, _shard_slice(split["tuning"], args.shard),
                     args.b1_tuning_n, "b1_tuning", _b_prompt_belief, args, cache_path, cache,
                     gen_stats, "b1-tuning")
    _b_generate_pool(emitter, items_by_id, _shard_slice(split["train"], args.shard),
                     args.b1_train_n, "b1_train", _b_prompt_belief, args, cache_path, cache,
                     gen_stats, "b1-train")
    _b_generate_pool(emitter, items_by_id, _shard_slice(split["calib"], args.shard),
                     args.b1_train_n, "b1_calib", _b_prompt_belief, args, cache_path, cache,
                     gen_stats, "b1-calib")

    # 3) escalate STICKY patients to the N=1024 equivalent — ONLY when not terminated.
    escalated = []
    if pres.status != sb.TERMINATE and args.b1_deepen_n > args.b1_patient_n:
        esc = [iid for iid in sticky_ids if iid in set(pat_shard)]
        for iid in _progress(esc, total=len(esc), desc="b1-deepen"):
            it = items_by_id.get(iid)
            if it is None:
                continue
            _sample_item(emitter, it, it["subtask"], args.b1_deepen_n, args.base_seed,
                         "b1_patient", cache_path, cache, prompt=_b_prompt_belief(it),
                         keep_text_chars=B_TEXT_KEEP, max_new_tokens=args.b_max_new_tokens,
                         run_stats=gen_stats)
            escalated.append(iid)

    # 4) per-pool parse-rate gate (>= 0.80; report, never silently pass on the new template).
    merged = load_stage_cache(args.out_dir, "b1")
    parse_rates = {}
    for tag in ("b1_patient", "b1_tuning", "b1_train", "b1_calib"):
        recs = [r for r in merged.values() if r.get("seed_tag") == tag]
        ok = sum(1 for r in recs if bs.parse_chain(_rec_text(r)).parse_ok)
        parse_rates[tag] = {"parse_rate": (ok / len(recs)) if recs else 0.0,
                            "n": len(recs), "n_parse_ok": ok,
                            "below_floor": (len(recs) > 0 and (ok / len(recs)) < 0.80)}
    low = [t for t, d in parse_rates.items() if d["below_floor"]]
    if low:
        log("b1: WARNING parse_rate < 0.80 for pools %s (belief template concern)" % low)

    # carried patient set for b2: sticky survivors (main set shrinks); fall back to the full
    # residual when nothing is sticky so b2 always has a non-empty pool to read out.
    carried = sorted(sticky_ids) if sticky_ids else sorted(patient_ids)

    eff = _b_efficiency(gen_stats, {}, len(patient_ids))
    _b_print_summary("b1", {"preservation": pres.status, "sticky": len(sticky_ids),
                            "escalated": len(escalated),
                            "parse_rate_patient": parse_rates["b1_patient"]["parse_rate"]}, eff)

    terminated = (pres.status == sb.TERMINATE)
    results = {
        "stage": "b1",
        "config": _b_config(args, "b1"),
        "preservation": pres.to_dict(),
        "sticky_ids": sorted(sticky_ids),
        "carried_patient_ids": carried,
        "escalated_ids": sorted(escalated),
        "parse_rates": parse_rates,
        "parse_rate_below_floor": bool(low),        # ANY pool < 0.80 -> surfaced to the b2 verdict
        "parse_rate_below_floor_pools": sorted(low),
        "pool_sizes": {t: parse_rates[t]["n"] for t in parse_rates},
        "terminated": terminated,
        "verdict": ("preservation_terminated" if terminated else "preservation_ok"),
        "efficiency": eff,
        "run_stats": _run_stats_public(gen_stats),
    }
    results, n_nf = sanitize_nonfinite(results)
    if n_nf:
        log("b1: sanitized %d non-finite value(s) -> null before write" % n_nf)
    write_results(os.path.join(args.out_dir, "results_b1.json"), results)
    if terminated:
        log("b1: PRESERVATION TERMINATE (sticky<%d) -> wrote terminate verdict, clean exit"
            % sb.PREREG_B["preservation_shrink_min"])
    return results


# ======================================================================================
# STAGE: b2 — step verifier + G1/G2/G3 + SAVI@lambda* vs SC main readout (design §3 / B2)
# ======================================================================================
def stage_b2(args, emitter_factory, featurizer_factory):
    banner("b2", "smoke=%s shard=%s" % (args.smoke, args.shard))
    items = load_items(args.data_dir, subtasks=PHASE_B_ARENA)
    if not items:
        raise RuntimeError("b2: no object_placements items under %s" % args.data_dir)
    items_by_id = _items_by_id(items)
    patient_ids = _b_patient_ids(args, items_by_id)
    split = _b_split(args, items, patient_ids)
    pbs.assert_story_level_disjoint(split)          # leakage guard (defense; b0 also asserts it)

    b0 = _b_read_results(args, "b0")
    b1 = _b_read_results(args, "b1")
    carried = list((b1 or {}).get("carried_patient_ids") or patient_ids)
    zero_cov = list((b0 or {}).get("zero_coverage_ids") or [])
    log("b2: %d carried patient(s), %d zero-cov, tuning=%d train=%d calib=%d"
        % (len(carried), len(zero_cov), len(split["tuning"]), len(split["train"]), len(split["calib"])))

    pool = _EmitterPool(emitter_factory)
    emitter = pool.get(args.main_model)
    featurizer = featurizer_factory(args.main_model)
    merged = load_stage_cache(args.out_dir, "b1")
    score_stats = {}

    train_recs = _b_pool_recs(merged, items_by_id, split["train"], "b1_train")
    calib_recs = _b_pool_recs(merged, items_by_id, split["calib"], "b1_calib")
    tuning_recs = _b_pool_recs(merged, items_by_id, split["tuning"], "b1_tuning")
    pat_recs = _b_pool_recs(merged, items_by_id, carried, "b1_patient")

    # 1) STEP verifier: fit on train/calib structured chains (label = step consistency vs gold).
    step_probe, step_probe_auc = _fit_step_probe(featurizer, train_recs, calib_recs,
                                                 items_by_id, score_stats, args)
    step_critic = vf.CriticVerifier(emitter)
    crit_auc = _step_critic_calib_auc(step_critic, calib_recs, items_by_id, score_stats)
    step_candidates = {"critic": {"val_auc": crit_auc}}
    if step_probe is not None:
        step_candidates["probe"] = {"val_auc": step_probe_auc}
    step_winner, step_report = vf.select_verifier(
        step_candidates, train_ids=list(train_recs), calib_ids=list(calib_recs),
        held_ids=set(patient_ids) | set(carried) | set(split["tuning"]))
    log("b2: step-verifier winner=%s calib_auc=%.3f" % (step_winner, step_candidates[step_winner]["val_auc"]))

    def step_score_fn(item, parsed, text, seed):
        """Per-state verifier scores for a parsed chain (probe on step features, else critic)."""
        if step_winner == "probe" and step_probe is not None:
            X = featurizer.featurize_step_tokens(_b_prompt_belief(item), parsed, text)
            return list(step_probe.score(X))
        return step_critic.score_steps(item, parsed, [seed] * len(parsed.states))

    # 2) G5b — step-level discrimination on the patient pool. Score every patient chain ONCE here
    #    and reuse the per-patient canonical-state map (SAVI weights) AND the per-chain mean score
    #    (the BoN whole-chain value under the SAME step verifier) — no re-featurize.
    step_rows, patient_step_scores, patient_chain_scores, patient_parse_ok = _patient_step_pass(
        pat_recs, items_by_id, step_score_fn, score_stats)
    g5b = sb.gate_g5b(step_rows)

    # 3) lambda* on the tuning pool (net_fix of SAVI-soft vs SC across the grid).
    lambda_grid = list(sb.PREREG_B["lambda_grid"])
    lam_sel = _select_lambda_tuning(tuning_recs, items_by_id, step_score_fn, lambda_grid, score_stats)
    lam_star = lam_sel["lambda_star"]

    # 4) main readout per carried patient: SAVI@lambda* / SC / BoN + oracle upper bound. BoN is the
    #    trellis ablation: best whole chain under the SAME selected STEP verifier (mean per-step
    #    score) — isolates the trellis merge/stitch, not a different (OOD answer-level) verifier.
    patient_results, oracle_rows, merge_rows, marginal_rows = [], [], [], []
    for iid in carried:
        it = items_by_id.get(iid)
        recs = pat_recs.get(iid, [])
        if it is None or not recs:
            patient_results.append({"item_id": iid, "savi_correct": False,
                                    "sc_correct": False, "bon_correct": False})
            oracle_rows.append({"item_id": iid, "oracle_fixed": False})
            merge_rows.append(0.0)
            continue
        chains = [bs.parse_chain(_rec_text(r)) for r in recs]
        trellis = dmn.build_trellis(chains)
        step_scores = patient_step_scores.get(iid, {})     # scored once above (no re-featurize)
        savi = dmn.savi_decode(trellis, "soft", lam=(lam_star or 0.0),
                               step_scores=step_scores, item=it)
        oracle = dmn.savi_decode(trellis, "oracle", item=it)
        sc_ans = dmn.sc_answer(chains)
        bon_scores = patient_chain_scores.get(iid, [])     # same step verifier as SAVI (ablation)
        bon_ans = dmn.bon_answer(chains, bon_scores)
        savi_correct = fo.is_goal_answer(savi["answer_idx"], it)
        oracle_fixed = fo.is_goal_answer(oracle["answer_idx"], it)
        patient_results.append({"item_id": iid, "savi_correct": savi_correct,
                                "sc_correct": fo.is_goal_answer(sc_ans, it),
                                "bon_correct": fo.is_goal_answer(bon_ans, it)})
        oracle_rows.append({"item_id": iid, "oracle_fixed": oracle_fixed})
        merge_rows.append(float(trellis.merge_rate))
        if oracle_fixed:
            marginal_rows.append(bool(oracle["marginal_not_in_single_chain"]))

    g1 = sb.gate_g1_oracle(oracle_rows)
    g2 = sb.gate_g2_merge(merge_rows)
    g3 = sb.gate_g3_marginal(marginal_rows)
    # PREREG_B.fragile_ids are stored short-form ("0032-q0"); patient ids are full-form
    # ("object_placements-0032-q0"). Resolve to the full carried ids by suffix match so the
    # fragile with/without dual actually fires on real data (frozen constant left untouched).
    fragile_full = sorted(pid for pid in carried
                          if any(str(pid).endswith(str(f)) for f in sb.PREREG_B["fragile_ids"]))
    readout = sb.readout_hb(patient_results, zero_cov, fragile_full)

    # 6) assemble the overall three-valued verdict (reconstruct G5a / preservation from b0/b1).
    gates = {
        "g5a": _gate_from_json((b0 or {}).get("g5a")),
        "preservation": _gate_from_json((b1 or {}).get("preservation")),
        "g5b": g5b, "g1_oracle": g1, "g2_merge": g2,
        "g3_marginal": g3, "readout": readout,
    }
    verdict = sb.verdict_b(gates)
    # Annotate (do NOT modify the frozen stages_b.verdict_b): a structured pool below the 0.80
    # parse-rate floor is a prior downgrade the VERDICT must surface, not just a b1 log warning.
    parse_below = bool((b1 or {}).get("parse_rate_below_floor"))
    verdict["parse_rate_downgrade"] = parse_below
    if parse_below:
        verdict["parse_rate_note"] = "a b1 structured pool parse_rate < 0.80 (see results_b1.parse_rates)"

    eff = _b_efficiency({}, score_stats, len(carried))
    _b_print_summary("b2", {"verdict": verdict.get("verdict"),
                            "hb": readout["hb_status"],
                            "net_fix_savi_sc": readout["primary"]["net_fix_savi_sc"],
                            "lambda_star": lam_star, "g1": g1.status, "g2": g2.status,
                            "g5b": g5b.status}, eff)

    results = {
        "stage": "b2",
        "config": _b_config(args, "b2"),
        "carried_patient_ids": sorted(carried),
        "zero_coverage_ids": sorted(zero_cov),
        "step_verifier": {"form": step_winner, "calib_auc": step_candidates[step_winner]["val_auc"],
                          "report": step_report, "probe_calib_auc": step_probe_auc,
                          "critic_calib_auc": crit_auc},
        "bon_arm": "same-verifier ablation: BoN scores whole chains by mean per-step score from the "
                   "SELECTED step verifier (isolates trellis structure, not a different verifier)",
        "lambda_selection": lam_sel,
        "gates": {"g5b": g5b.to_dict(), "g1_oracle": g1.to_dict(),
                  "g2_merge": g2.to_dict(), "g3_marginal": g3},
        # SC votes ALL chains (incl. parse-fails); trellis/BoN use parse_ok only — recorded so the
        # asymmetry is auditable per patient (no logic change).
        "patient_parse_ok": patient_parse_ok,
        "parse_rate_below_floor_b1": bool((b1 or {}).get("parse_rate_below_floor")),
        "readout": readout,
        "verdict": verdict,
        "efficiency": eff,
        "run_stats": {"score_seconds": score_stats.get("seconds", 0.0)},
    }
    results, n_nf = sanitize_nonfinite(results)
    if n_nf:
        log("b2: sanitized %d non-finite CI sentinel(s) -> null before write" % n_nf)
    write_results(os.path.join(args.out_dir, "results_b2.json"), results)
    _b_write_figures(args, readout, oracle_rows, merge_rows, step_rows)
    return results


# ---- b2 sub-helpers ----------------------------------------------------------------------------
def _fit_step_probe(featurizer, train_recs, calib_recs, items_by_id, score_stats, args):
    """Fit the step-level probe on train/calib structured chains. Labels = per-step consistency
    with the gold facts tree (``state_consistent(ref='nearest') >= 0.5``). Degrades to None."""
    def build(pool):
        feats, labels = [], []
        for iid, recs in pool.items():
            it = items_by_id[iid]
            for r in recs:
                parsed = bs.parse_chain(_rec_text(r))
                if not parsed.parse_ok:
                    continue
                X = featurizer.featurize_step_tokens(_b_prompt_belief(it), parsed, _rec_text(r))
                for k, st in enumerate(parsed.states):
                    if k >= len(X):
                        break
                    cons = fo.state_consistent(st, it, ref="nearest")
                    if cons != cons:            # NaN (no gold) -> skip
                        continue
                    feats.append(X[k]); labels.append(1 if cons >= 0.5 else 0)
        return feats, labels
    try:
        Ftr, ytr = _score_timed(lambda: build(train_recs), score_stats, len(train_recs))
        Fca, yca = _score_timed(lambda: build(calib_recs), score_stats, len(calib_recs))
        if not Ftr or not Fca:
            return None, None
        import numpy as _np
        probe = vf.ProbeVerifier().fit(_np.stack(Ftr), ytr, _np.stack(Fca), yca)
        return probe, float(probe.val_auc)
    except Exception as exc:
        log("  step-probe fit degraded -> critic-only (%s)" % exc)
        return None, None


def _step_critic_calib_auc(step_critic, calib_recs, items_by_id, score_stats):
    """Calibration AUC of the step critic (per-step consistency scores vs gold labels)."""
    scores, labels = [], []
    for iid, recs in calib_recs.items():
        it = items_by_id[iid]
        for r in recs:
            parsed = bs.parse_chain(_rec_text(r))
            if not parsed.parse_ok:
                continue
            s = _score_timed(lambda: step_critic.score_steps(it, parsed, [int(r.get("seed", 0))]),
                             score_stats, len(parsed.states))
            for k, st in enumerate(parsed.states):
                if k >= len(s):
                    break
                cons = fo.state_consistent(st, it, ref="nearest")
                if cons != cons:
                    continue
                scores.append(s[k]); labels.append(1 if cons >= 0.5 else 0)
    return vf.roc_auc(scores, labels)


def _patient_step_pass(pat_recs, items_by_id, step_score_fn, score_stats):
    """Single featurize/score pass over the patient pool's parsed chains. Featurizes each chain
    EXACTLY ONCE and returns four aligned products from that one pass:
      * ``rows``               — G5b rows ``{score, consistent}`` per gold-labelled step;
      * ``scores_by_iid``      — per-patient ``{canon_state: mean score}`` (SAVI-soft weights);
      * ``chain_scores_by_iid``— per-patient per-CHAIN mean step score (aligned to ``pat_recs[iid]``
                                 order), the BoN whole-chain soft value under the SAME step verifier
                                 SAVI uses; a parse-fail chain gets ``_BON_PARSEFAIL_SCORE`` so it
                                 ranks below any real chain;
      * ``parse_ok_by_iid``    — per-patient ``{n_total, n_parse_ok}`` (SC votes ALL chains incl.
                                 parse-fails; the trellis/BoN use parse_ok only — recorded so the
                                 asymmetry is visible, no logic change)."""
    rows = []
    scores_by_iid, chain_scores_by_iid, parse_ok_by_iid = {}, {}, {}
    for iid, recs in pat_recs.items():
        it = items_by_id[iid]
        accum = {}
        chain_scores = []
        n_ok = 0
        for r in recs:
            parsed = bs.parse_chain(_rec_text(r))
            if not parsed.parse_ok:
                chain_scores.append(_BON_PARSEFAIL_SCORE)     # keep alignment; never BoN-picked
                continue
            n_ok += 1
            s = _score_timed(lambda: step_score_fn(it, parsed, _rec_text(r), int(r.get("seed", 0))),
                             score_stats, len(parsed.states))
            per_step = []
            for k, st in enumerate(parsed.states):
                if k >= len(s):
                    break
                cons = fo.state_consistent(st, it, ref="nearest")
                if cons == cons:                    # not NaN -> a gold-labelled step
                    rows.append({"score": float(s[k]), "consistent": cons >= 0.5})
                accum.setdefault(bs.canon_state(st), []).append(float(s[k]))
                per_step.append(float(s[k]))
            chain_scores.append((sum(per_step) / len(per_step)) if per_step else _BON_PARSEFAIL_SCORE)
        scores_by_iid[iid] = {c: (sum(v) / len(v)) for c, v in accum.items() if v}
        chain_scores_by_iid[iid] = chain_scores
        parse_ok_by_iid[iid] = {"n_total": len(recs), "n_parse_ok": n_ok}
    return rows, scores_by_iid, chain_scores_by_iid, parse_ok_by_iid


def _patient_step_scores(item, recs, step_score_fn, score_stats):
    """``{canon_state: mean verifier score}`` over a patient's chains (for SAVI-soft weighting)."""
    accum = {}
    for r in recs:
        parsed = bs.parse_chain(_rec_text(r))
        if not parsed.parse_ok:
            continue
        s = _score_timed(lambda: step_score_fn(item, parsed, _rec_text(r), int(r.get("seed", 0))),
                         score_stats, len(parsed.states))
        for k, st in enumerate(parsed.states):
            if k >= len(s):
                break
            accum.setdefault(bs.canon_state(st), []).append(float(s[k]))
    return {c: (sum(v) / len(v)) for c, v in accum.items() if v}


def _select_lambda_tuning(tuning_recs, items_by_id, step_score_fn, lambda_grid, score_stats):
    """select_lambda over the tuning pool: net_fix of SAVI-soft(lambda) vs SC per grid value."""
    per_item = []
    for iid, recs in tuning_recs.items():
        it = items_by_id[iid]
        chains = [bs.parse_chain(_rec_text(r)) for r in recs]
        trellis = dmn.build_trellis(chains)
        step_scores = _patient_step_scores(it, recs, step_score_fn, score_stats)
        sc_correct = fo.is_goal_answer(dmn.sc_answer(chains), it)
        per_item.append((it, trellis, step_scores, sc_correct))
    grid = {}
    for lam in lambda_grid:
        net = 0
        for it, trellis, step_scores, sc_correct in per_item:
            savi = dmn.savi_decode(trellis, "soft", lam=lam, step_scores=step_scores, item=it)
            net += int(fo.is_goal_answer(savi["answer_idx"], it)) - int(sc_correct)
        grid[lam] = net
    return sb.select_lambda(grid)


def _gate_from_json(d):
    """Rebuild a ``stages_b.GateResult`` from its json (``{status, detail}``) or None."""
    if not d or "status" not in d:
        return None
    return sb.GateResult(status=d["status"], detail=d.get("detail") or {})


def _b_read_results(args, stage):
    path = os.path.join(args.out_dir, "results_%s.json" % stage)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---- Phase B config block + PREREG_b freeze/round-trip -----------------------------------------
def _b_config(args, stage):
    """Config block for a Phase B results.json (Phase A config + the frozen Phase B additions)."""
    cfg = build_config(args, stage)
    cfg.update({
        "phase": "B",
        "split_seed": B_SPLIT_SEED,
        "prereg_b": dict(sb.PREREG_B),
        "belief_template_sha256": bs.BELIEF_TEMPLATE_SHA256,
        "critic_template_sha256": vf.CRITIC_TEMPLATE_SHA256,
        "step_critic_template_sha256": vf.STEP_CRITIC_TEMPLATE_SHA256,
        "lambda_grid": list(sb.PREREG_B["lambda_grid"]),
        "primary_comparison": "SAVI@lambda* vs token-matched SC (single primary comparison)",
        "b_budgets": {"b0_patient_n": args.b0_patient_n, "b0_train_n": args.b0_train_n,
                      "b1_patient_n": args.b1_patient_n, "b1_deepen_n": args.b1_deepen_n,
                      "b1_tuning_n": args.b1_tuning_n, "b1_train_n": args.b1_train_n,
                      "b_max_new_tokens": args.b_max_new_tokens,
                      "featurize_batch": getattr(args, "featurize_batch", 8),
                      "featurize_max_tokens": fz.FEATURIZE_MAX_TOKENS},
    })
    return cfg


def write_prereg_b(path=PREREG_B_PATH):
    """Freeze the Phase B pre-registration to ``PREREG_b.md`` (a fenced ```json PREREG_B block +
    the template shas / lambda grid / split seed / primary-comparison declaration). Round-tripped
    by ``load_prereg_b`` + ``stages_b.prereg_roundtrip`` (a3 convention: freezing == testing)."""
    payload = json.loads(json.dumps(dict(sb.PREREG_B)))   # tuples -> lists (json-normalized)
    md = [
        "# MuSR-cant Phase B pre-registration (PREREG_b)",
        "",
        "Frozen BEFORE any full Phase B run (smoke may precede). Every threshold below is the "
        "single source of truth in `stages_b.PREREG_B`; this file is round-trip checked against "
        "the code constant by `stages_b.prereg_roundtrip` (a3 convention).",
        "",
        "## Frozen thresholds (`stages_b.PREREG_B`)",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen template sha256",
        "",
        "- belief-table CoT (`belief_schema.BELIEF_TEMPLATE`): `%s`" % bs.BELIEF_TEMPLATE_SHA256,
        "- answer critic (`verifiers.CRITIC_TEMPLATE`): `%s`" % vf.CRITIC_TEMPLATE_SHA256,
        "- step critic (`verifiers.STEP_CRITIC_TEMPLATE`): `%s`" % vf.STEP_CRITIC_TEMPLATE_SHA256,
        "- Phase A free-CoT prompt (`sc_core.PROMPT_TEMPLATE`): `%s`" % _sha256_text(sc.PROMPT_TEMPLATE),
        "",
        "## lambda grid",
        "",
        "`%s` (tuning pool only; lambda* = argmax net_fix, ties -> smallest)."
        % (list(sb.PREREG_B["lambda_grid"]),),
        "",
        "## Split / seeds",
        "",
        "- story-level split seed: `%d` (design §2 防泄漏切分; story-disjoint train/calib/tuning)."
        % B_SPLIT_SEED,
        "- base sampling seed: `%d` (E1 protocol, shard-invariant)." % dm.BASE_SEED,
        "",
        "## Zero-coverage sub-class + fragile ids",
        "",
        "- zero-coverage ids (mechanically frozen from the a1 patient cache in b0; the design's "
        "4 zero-cov sub-class) — reported separately, excluded from the primary readout.",
        "- fragile-fix ids (a3 leftover, reported with/without in the H-B dual): `%s`."
        % (list(sb.PREREG_B["fragile_ids"]),),
        "",
        "## Single primary comparison",
        "",
        "The ONLY primary comparison is **SAVI@lambda\\* vs token-matched SC** "
        "(paired bootstrap CI > %.1f AND net_fix >= %d over the non-zero-coverage primary set). "
        "SAVI-vs-BoN is a pre-registered secondary ablation; every other readout is secondary."
        % (sb.PREREG_B["hb_ci_floor"], sb.PREREG_B["hb_net_fix_min"]),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def load_prereg_b(path=PREREG_B_PATH):
    """Parse the fenced ```json PREREG_B block out of ``PREREG_b.md`` -> dict (for round-trip)."""
    with open(path) as f:
        text = f.read()
    start = text.find("```json")
    if start == -1:
        raise ValueError("load_prereg_b: no ```json block in %s" % path)
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise ValueError("load_prereg_b: unterminated ```json block in %s" % path)
    return json.loads(text[start:end])


def _b_prereg_selfcheck(args):
    """Runtime PREREG guard for a FULL Phase B run: assert the frozen ``PREREG_b.md`` still matches
    ``stages_b.PREREG_B`` (cheap; prevents running against a drifted prereg). Hard-errors on
    mismatch or on a missing frozen file. SKIPPED under --smoke (smoke may precede the freeze)."""
    if args.smoke:
        return
    candidates = [os.path.join(args.out_dir, "PREREG_b.md"), PREREG_B_PATH]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise RuntimeError(
            "Phase B full run requires a frozen PREREG_b.md (run write_prereg_b first); "
            "none found at %s" % " or ".join(candidates))
    sb.prereg_roundtrip(load_prereg_b(path))        # raises AssertionError naming any drifted key
    log("prereg self-check OK: %s matches stages_b.PREREG_B" % path)


# ---- Phase B figures (design §11; matplotlib, analysis-time) -----------------------------------
def _b_write_figures(args, readout, oracle_rows, merge_rows, step_rows):
    """Write the 4 Phase B figures (discrimination ROC, preservation/merge, merge-rate dist, main
    paired readout). Best-effort: a headless/matplotlib-less environment logs + skips, never fatal."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log("b2: matplotlib unavailable, skipping figures (%s)" % exc)
        return
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    try:
        # 1) step-discrimination score distribution by consistency label (ROC substrate).
        fig, ax = plt.subplots(figsize=(4, 3))
        pos = [r["score"] for r in step_rows if r["consistent"]]
        neg = [r["score"] for r in step_rows if not r["consistent"]]
        ax.hist([neg, pos], bins=8, label=["inconsistent", "consistent"], stacked=False)
        ax.set_title("b2 step discrimination"); ax.set_xlabel("verifier score"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "b2_discrimination.png")); plt.close(fig)
        # 2) oracle-fixed bar (G1).
        fig, ax = plt.subplots(figsize=(4, 3))
        n_fix = sum(1 for r in oracle_rows if r["oracle_fixed"])
        ax.bar(["oracle-fixed", "not"], [n_fix, len(oracle_rows) - n_fix])
        ax.set_title("b2 G1 oracle upper bound"); fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "b2_preservation.png")); plt.close(fig)
        # 3) merge-rate distribution (G2).
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(merge_rows or [0.0], bins=8); ax.set_title("b2 merge-rate dist"); ax.set_xlabel("merge_rate")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "b2_merge_rate.png")); plt.close(fig)
        # 4) main paired readout (SAVI vs SC vs BoN net over the primary set).
        fig, ax = plt.subplots(figsize=(4, 3))
        prim = readout["primary"]
        ax.bar(["SAVI-SC", "SAVI-BoN"], [prim["net_fix_savi_sc"], prim["net_fix_savi_bon"]])
        ax.axhline(0, color="k", lw=0.6); ax.set_title("b2 main readout (net fix)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "b2_main_readout.png")); plt.close(fig)
        log("b2: wrote 4 figures to %s" % fig_dir)
    except Exception as exc:
        log("b2: figure generation error (non-fatal): %s" % exc)


def _b_print_summary(stage, headline, eff):
    log("---- %s summary ----" % stage)
    for k, v in headline.items():
        log("  %-20s %s" % (k, v))
    gen = eff.get("generation", {})
    sco = eff.get("verifier_scoring", {})
    log("  efficiency: gen_gpu_h=%.4f gen_samples=%d | scoring_gpu_h=%.4f scored=%d | total_gpu_h=%.4f"
        % (gen.get("gpu_hours", 0.0), gen.get("n_samples_generated", 0),
           sco.get("gpu_hours", 0.0), sco.get("n_scored", 0), eff.get("total_gpu_hours", 0.0)))


# ======================================================================================
# ==========================  EXACT-GSD  (g0 / g1)  =====================================
# ======================================================================================
# The zero-approximation decode pipeline assembled on Subtasks 1-5 (plan 2026-07-07-exact-gsd,
# design §2/§4/§5/§6/§7). g0 = PREREG_g freeze + G-A fidelity kill-gate + backtrace-readout
# ablation; g1 = preflight-gated per-item enumerate->score->two-arm-decode->Δ-ledger + the
# six-readout assembly + figures. All analyzer logic lives in stages_g / gsd_*; this runner
# only wires data, caches, failure handling and observability.

def _g_open_log(args, stage):
    """Open (append) the g-stage log file — outputs/g{0,1}.log, or the shard-named variant."""
    shard_i = _shard_index(args.shard)
    name = "%s.log" % stage if shard_i is None else "%s_shard%d.log" % (stage, shard_i)
    os.makedirs(args.out_dir, exist_ok=True)
    return open(os.path.join(args.out_dir, name), "a")


def _g_sticky_ids(args):
    """The certified patient set: ``results_b1.json['sticky_ids']`` (19 on real data)."""
    path = os.path.join(args.out_dir, "results_b1.json")
    if not os.path.exists(path):
        raise RuntimeError("g: results_b1.json not found under %s (need sticky_ids; "
                           "run the Phase B funnel first)" % args.out_dir)
    with open(path) as f:
        ids = list(json.load(f).get("sticky_ids") or [])
    if not ids:
        raise RuntimeError("g: results_b1.json carries no sticky_ids")
    return sorted(ids)


def _g_tuning_ids(b1_cache):
    """The deferred tuning pool: distinct item ids with seed_tag b1_tuning in cache_b1 (40)."""
    return sorted({r["item_id"] for r in b1_cache.values()
                   if r.get("seed_tag") == "b1_tuning"})


def _g_chains(b1_cache, item, seed_tag):
    """Parsed chains of one item's cached b1 pool (sample_idx order, ``_rec_text`` texts)."""
    recs = sc.cached_records(b1_cache, item["subtask"], item["id"], seed_tag)
    return [bs.parse_chain(_rec_text(r)) for r in recs]


def _g_build_spaces(items_by_id, ids, excluded):
    """``{item_id: GsdSpace}`` over ``ids``; malformed trees / absent items go on the
    ``excluded`` roster (design §7: exclusion is RECORDED, never silent)."""
    spaces = {}
    for iid in sorted(set(ids)):
        it = items_by_id.get(iid)
        if it is None:
            excluded.append({"item_id": iid, "reason": "item not in loaded arena"})
            continue
        space = gsp.build_space(it)
        if space is None:
            excluded.append({"item_id": iid,
                             "reason": "malformed gold tree (build_space sentinel)"})
            continue
        spaces[iid] = space
    return spaces


def _g_prereg_freeze(args, write_if_missing):
    """PREREG_g freeze + round-trip check. g0 writes the file when absent (freeze); g1
    REQUIRES it to exist already. Either way a drifted file hard-raises (stages_g)."""
    path = os.path.join(args.out_dir, "PREREG_g.md")
    if not os.path.exists(path):
        if not write_if_missing:
            raise RuntimeError("g1: PREREG_g.md not found under %s (run --stage g0 first: "
                               "the g1 full run must follow the freeze)" % args.out_dir)
        sg.write_prereg_g(path)
        log("PREREG_g frozen -> %s" % path)
    sg.check_prereg_g_roundtrip(path)                     # AssertionError names drifted keys
    log("PREREG_g round-trip OK: %s" % path)
    return path


def _g_scope(args, sticky, tuning_ids):
    """(ga_patient_ids, ga_tuning_ids, ablation_ids, patient_ids, knowledge_ids) after the
    smoke scoping rule. Smoke = the frozen deep-dive duo (+ the deep-dive knowledge control
    in g1) on the SAME code path; full = the frozen PREREG pools."""
    dd = sg.PREREG_G["deepdive_ids"]
    if args.smoke:
        duo = sorted([dd["nonzero"], dd["zero_cov"]])
        return duo, [], duo, duo, [dd["knowledge"]]
    return (list(sticky), list(tuning_ids), list(sticky), list(sticky),
            sorted(sg.PREREG_G["knowledge_control_ids"]))


class _GaCachingScorer:
    """Content-keyed JSONL cache + GPU-time meter around a TF scorer's ``score_batch`` for
    the g0 G-A pass (``score_transitions`` caches internally in g1; the G-A pass calls
    ``score_batch`` directly, so the runner adds the cache here). Key = (sha256(prefix),
    sha256(target)) — the prefix bytes already carry item / template / layer / s_prev, so
    the key is exact. Records ``{"psha","tsha","lp"}`` append to the shard's
    ``cache_g0_scores*.jsonl`` (atomic, resume-idempotent); load merges ALL shard files.
    A full-hit rerun scores nothing (``n_scored == 0``). ScorerOOM/ValueError propagate
    (per-item handling is the caller's)."""

    def __init__(self, inner, cache_path, out_dir):
        self.inner = inner
        self.cache_path = cache_path
        self.seconds = 0.0
        self.n_scored = 0
        self._cache = {}
        for fp in sorted(glob.glob(os.path.join(out_dir, "cache_g0_scores*.jsonl"))):
            self._cache.update(_load_ga_score_cache(fp))

    def score_batch(self, prefixes, target_lines, context=""):
        keys = [(_sha256_text(p), _sha256_text(t))
                for p, t in zip(prefixes, target_lines)]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            t0 = time.time()
            scores = self.inner.score_batch([prefixes[i] for i in missing],
                                            [target_lines[i] for i in missing],
                                            context=context)
            dt = time.time() - t0
            self.seconds += dt
            self.n_scored += len(missing)
            for i, lp in zip(missing, scores):
                sc.append_record(self.cache_path,
                                 {"psha": keys[i][0], "tsha": keys[i][1], "lp": float(lp)})
                self._cache[keys[i]] = float(lp)
            log("  [g0-score]%s pairs=%d cached=%d scored=%d sec=%.2f"
                % (context, len(keys), len(keys) - len(missing), len(missing), dt))
        return [self._cache[k] for k in keys]


def _load_ga_score_cache(path):
    """``{(psha, tsha): lp}`` from one G-A score cache file; torn/blank/non-finite rows are
    skipped (a skipped row is simply re-scored — deterministic, so last-write-wins is safe)."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                lp = float(rec["lp"])
            except (ValueError, TypeError, KeyError):
                continue
            if math.isfinite(lp):
                out[(rec["psha"], rec["tsha"])] = lp
    return out


def _g_merge_score_cache(scorer, out_dir):
    """Prime a TFScorer's in-memory transition cache with EVERY ``cache_g1_scores*.jsonl``
    present (sibling shards / the un-sharded file), so the consolidation pass re-scores
    nothing. The scorer's own file keeps precedence (scores are deterministic anyway).
    A scorer without the ``_cache`` dict (a bare test stub) is left untouched."""
    cache = getattr(scorer, "_cache", None)
    if not isinstance(cache, dict):
        return
    n0 = len(cache)
    for fp in sorted(glob.glob(os.path.join(out_dir, "cache_g1_scores*.jsonl"))):
        for k, v in gsd_score._load_score_cache(fp).items():
            cache.setdefault(k, v)
    if len(cache) > n0:
        log("g1: merged %d cached transition score(s) from sibling shard files"
            % (len(cache) - n0))


def _g_default_scorer(args, cache_path, stage):
    """The production TF scorer (bf16 / sdpa / offline single card) — the SOLE g-stage GPU
    site, reached only when no scorer is injected."""
    log("%s: loading TF scorer model=%s device=%s batch=%d"
        % (stage, args.main_model, args.device, args.gsd_batch))
    return gsd_score.load_scorer(args.main_model, device=args.device,
                                 cache_path=cache_path, batch=args.gsd_batch, stage=stage)


def _g_efficiency(seconds, n_scored, n_items):
    """g-stage efficiency block: generation is structurally ZERO (eval-only re-scoring of
    cached chains); TF scoring GPU time and forward count are the only costs, reported
    separately per stage (design §7)."""
    return {
        "n_items": n_items,
        "generation": {"gpu_seconds": 0.0, "gpu_hours": 0.0, "n_samples_generated": 0},
        "scoring": {"gpu_seconds": seconds, "gpu_hours": seconds / 3600.0,
                    "n_scored": n_scored},
        "total_gpu_hours": seconds / 3600.0,
    }


def _g_config(args, stage):
    """g-stage config block: Phase A config + the frozen PREREG_G snapshot, the three
    template shas, the scorer batch and the evidence-pool definitions (design §7)."""
    cfg = build_config(args, stage)
    cfg.update({
        "phase": "G",
        "prereg_g": json.loads(json.dumps(sg.PREREG_G)),
        "gsd_template_sha256": gsd_score.GSD_TEMPLATE_SHA256,
        "sensitivity_template_sha256": gsd_score.SENSITIVITY_TEMPLATE_SHA256,
        "gsd_root_template_sha256": gsd_score.GSD_ROOT_TEMPLATE_SHA256,
        "gsd_batch": args.gsd_batch,
        "normalization_main": sg.PREREG_G["normalization_main"],
        "pools": {
            "ga_evidence": json.loads(json.dumps(sg.PREREG_G["ga_evidence_pools"])),
            "ablation": "results_b1.sticky_ids @ seed_tag b1_patient (N=256)",
            "sc_readout": "cache_b1 patient chains, vote_at_rung @ N=%d"
                          % sg.PREREG_G["ga_evidence_pools"]["patient"]["n_per_item"],
        },
    })
    return cfg


# ======================================================================================
# STAGE: g0 — PREREG_g freeze + G-A fidelity kill-gate + backtrace-readout ablation
# ======================================================================================
def stage_g0(args, scorer=None):
    global _LOG_FH
    fh = _g_open_log(args, "g0")
    _LOG_FH = fh
    try:
        return _stage_g0(args, scorer)
    finally:
        _LOG_FH = None
        fh.close()


def _stage_g0(args, scorer):
    banner("g0", "smoke=%s shard=%s" % (args.smoke, args.shard))
    items = load_items(args.data_dir, subtasks=PHASE_B_ARENA)
    if not items:
        raise RuntimeError("g0: no object_placements items under %s" % args.data_dir)
    items_by_id = _items_by_id(items)

    b1_cache = load_stage_cache(args.out_dir, "b1")
    sticky = _g_sticky_ids(args) if not args.smoke else []
    tuning_ids = _g_tuning_ids(b1_cache)
    ga_pat_ids, ga_tun_ids, abl_ids, _pats, _know = _g_scope(args, sticky, tuning_ids)
    smoke_tuning_ids = []
    if args.smoke:
        # L1 fix: the two deep-dive patients alone can yield ZERO countable G-A groups
        # (every (t, s_prev) group under the >= 3-distinct-successors floor), leaving the
        # smoke pass without a single real TF forward (n_scored == 0 -> fail-closed KILL
        # from empty evidence). Extend the smoke feed with a DETERMINISTIC smoke-tuning
        # pick so the scoring path actually moves. This pick serves the smoke scoring
        # path only — smoke results are never canonical / pre-registered inference; the
        # chosen ids are echoed into the results config for the audit trail.
        smoke_tuning_ids = _g0_smoke_tuning_pick(items_by_id, b1_cache, tuning_ids)
        ga_tun_ids = list(smoke_tuning_ids)
        log("g0[smoke]: G-A feed extended with smoke-tuning item(s) %s (first sorted "
            "tuning ids with a countable group; guarantees TF scoring runs)"
            % smoke_tuning_ids)
    pools = sg.PREREG_G["ga_evidence_pools"]
    if not args.smoke and (len(ga_pat_ids) != pools["patient"]["n_items"]
                           or len(ga_tun_ids) != pools["tuning"]["n_items"]):
        log("g0: WARNING pool sizes (patient=%d tuning=%d) differ from PREREG (%d/%d) — "
            "check results_b1 / cache_b1 provenance"
            % (len(ga_pat_ids), len(ga_tun_ids),
               pools["patient"]["n_items"], pools["tuning"]["n_items"]))

    # shard by item within each pool (strided; canonical = the final un-sharded pass)
    ga_pat_ids = _shard_slice(sorted(ga_pat_ids), args.shard)
    ga_tun_ids = _shard_slice(sorted(ga_tun_ids), args.shard)
    abl_ids = _shard_slice(sorted(abl_ids), args.shard)
    log("g0: this pass — G-A patients=%d tuning=%d | ablation=%d"
        % (len(ga_pat_ids), len(ga_tun_ids), len(abl_ids)))

    # ---- 1) PREREG_g freeze (BEFORE any scoring; idempotent, drift hard-raises) -------
    _g_prereg_freeze(args, write_if_missing=True)

    # ---- 2) reachable spaces + exclusion roster ----------------------------------------
    excluded = []
    spaces = _g_build_spaces(items_by_id,
                             list(ga_pat_ids) + list(ga_tun_ids) + list(abl_ids), excluded)
    for e in excluded:
        log("g0: EXCLUDED %s (%s)" % (e["item_id"], e["reason"]))

    # ---- 3) G-A fidelity gate over the two-pool feed (design §4, the only kill-gate) ---
    shard_i = _shard_index(args.shard)
    if scorer is None:
        scorer = _g_default_scorer(args, cache_path=None, stage="g0")
    ga_scorer = _GaCachingScorer(scorer, _cache_path(args.out_dir, "g0_scores", shard_i),
                                 args.out_dir)
    ga = _g0_ga_pass(items_by_id, spaces, b1_cache, ga_pat_ids, ga_tun_ids,
                     excluded, ga_scorer)
    log("g0: G-A verdict=%s median_rho=%s groups=%d"
        % (ga["verdict"], ga["median_rho"], ga["n_groups_counted"]))
    if ga["verdict"] == sg.GA_KILL:
        log("g0: G-A KILL — the likelihood-scoring route is dead; g1 will REFUSE to run. "
            "Write the diagnostic VERDICT_gsd.md from this results_g0 (design §4).")

    # ---- 4) backtrace-readout ablation (zero GPU; readout 5) --------------------------
    abl_feed = []
    for iid in abl_ids:
        it = items_by_id.get(iid)
        chains = _g_chains(b1_cache, it, "b1_patient") if it is not None else []
        abl_feed.append({"item": it, "chains": chains, "item_id": iid})
    ablation = sg.ablation_backtrace_readout(abl_feed)
    for row in ablation["per_item"]:
        log("  [g0-abl] %-30s vote_fixed=%s mech_fixed=%s (vote=%r mech=%r)"
            % (row["item_id"], row["vote_fixed"], row["mech_fixed"],
               row["vote_answer_idx"], row["mech_answer_idx"]))
    log("g0: ablation vote_total=%d mech_total=%d mech-baseline=%+d"
        % (ablation["vote_total"], ablation["mech_total"],
           ablation["mech_minus_baseline"]))

    # ---- 5) results_g0 (NaN/Inf-guarded write) -----------------------------------------
    eff = _g_efficiency(ga_scorer.seconds, ga_scorer.n_scored,
                        len(ga_pat_ids) + len(ga_tun_ids))
    _print_g_summary("g0", {"ga_verdict": ga["verdict"], "median_rho": ga["median_rho"],
                            "n_groups": ga["n_groups_counted"],
                            "abl_mech_total": ablation["mech_total"],
                            "excluded": len(excluded)}, eff)
    cfg = _g_config(args, "g0")
    if args.smoke:
        cfg["smoke_tuning_ids"] = list(smoke_tuning_ids)
    results = {
        "stage": "g0",
        "config": cfg,
        "scope": {"ga_patient_ids": list(ga_pat_ids), "ga_tuning_ids": list(ga_tun_ids),
                  "ablation_ids": list(abl_ids)},
        "ga": ga,
        "ablation": ablation,
        "excluded_items": excluded,
        "efficiency": eff,
        "run_stats": {"score_seconds": ga_scorer.seconds},
    }
    write_results(os.path.join(args.out_dir, "results_g0.json"), results)
    return results


def _g0_smoke_tuning_pick(items_by_id, b1_cache, tuning_ids, k=2):
    """The deterministic smoke-tuning pick: walk ``sorted(tuning_ids)``, pure-CPU
    precheck each item with ``stages_g.ga_collect_groups``, and keep the FIRST ``k``
    items carrying at least one COUNTABLE group (>= ``ga_min_distinct_states`` distinct
    successors). Zero GPU (no scoring here); items without a usable space/pool are
    passed over. May return fewer than ``k`` when nothing qualifies (smoke then falls
    back to the deep-dive-only feed)."""
    min_distinct = sg.PREREG_G["ga_min_distinct_states"]
    picked = []
    for iid in sorted(tuning_ids):
        it = items_by_id.get(iid)
        if it is None:
            continue
        space = gsp.build_space(it)
        if space is None:
            continue
        chains = _g_chains(b1_cache, it, "b1_tuning")
        groups = sg.ga_collect_groups(chains, t_max=space.T - 1)
        if any(len(g["states"]) >= min_distinct for g in groups):
            picked.append(iid)
            if len(picked) >= k:
                break
    return picked


def _g0_ga_pass(items_by_id, spaces, b1_cache, patient_ids, tuning_ids, excluded,
                scorer):
    """The G-A pass with per-item observability and OOM skip-lists: composes the audited
    ``stages_g.ga_item`` / ``ga_gate`` per the ``ga_fidelity`` contract (same rows, same
    diag, same verdict) while letting the runner log per item and skip a ScorerOOM item
    instead of aborting the gate (design §7)."""
    feed = ([(iid, "patient", "b1_patient") for iid in patient_ids]
            + [(iid, "tuning", "b1_tuning") for iid in tuning_ids])
    rows = []
    diag = {"n_groups_seen": 0, "n_groups_lt_min_distinct": 0,
            "n_groups_degenerate": 0, "items_skipped_no_space": [],
            "items_oom_skipped": []}
    for iid, pool, seed_tag in _progress(feed, total=len(feed), desc="g0-ga"):
        it = items_by_id.get(iid)
        if it is None:                                     # already on the exclusion roster
            continue
        space = spaces.get(iid)
        if space is None:                                  # malformed tree -> named skip
            diag["items_skipped_no_space"].append(iid)
            log("  [g0-ga] %-30s SKIPPED (malformed gold tree, no space)" % iid)
            continue
        chains = _g_chains(b1_cache, it, seed_tag)
        t0 = time.time()
        n0 = scorer.n_scored
        try:
            res = sg.ga_item(it, space, chains, scorer, template="main", pool=pool)
        except gsd_score.ScorerOOM as exc:
            diag["items_oom_skipped"].append(iid)
            excluded.append({"item_id": iid, "reason": "ScorerOOM in G-A: %s" % exc})
            log("  [g0-ga] %-30s OOM-SKIPPED (%s)" % (iid, exc))
            continue
        rows.extend(res["rows"])
        for k in ("n_groups_seen", "n_groups_lt_min_distinct", "n_groups_degenerate"):
            diag[k] += res["diag"][k]
        log("  [g0-ga] %-30s pool=%-7s layers=%d chains=%d groups=%d counted=%d "
            "scored=%d sec=%.2f"
            % (iid, pool, space.T, len(chains), res["diag"]["n_groups_seen"],
               len(res["rows"]), scorer.n_scored - n0, time.time() - t0))

    out = sg.ga_gate(rows)
    by_pool = {}
    for r in rows:
        key = r["pool"] if r["pool"] is not None else "unlabelled"
        by_pool[key] = by_pool.get(key, 0) + 1
    diag["n_groups_counted_by_pool"] = by_pool
    diag["median_rho_modal_prev"] = sg._median(
        [r["rho"] for r in rows if r["is_modal_prev"]])
    out["diag"] = diag
    return out


# ======================================================================================
# STAGE: g1 — the six-readout main body (enumerate -> score -> two arms -> Δ-ledger)
# ======================================================================================
def stage_g1(args, scorer=None):
    """The g1 main body (see module docstring). Preflight is FAIL-CLOSED on the G-A gate
    with exactly one controlled exception: under ``--smoke`` a WELL-FORMED KILL verdict
    downgrades to a loud warning-and-continue (config marks ``smoke_ga_kill_bypass``),
    because L1 runtime validation must be able to exercise the g1 pipeline end-to-end
    while the smoke-scope g0 gate is not the canonical decision — the canonical gate is
    enforced unchanged by the non-smoke full run. A missing / None / malformed verdict
    still refuses even under ``--smoke``."""
    global _LOG_FH
    fh = _g_open_log(args, "g1")
    _LOG_FH = fh
    try:
        return _stage_g1(args, scorer)
    finally:
        _LOG_FH = None
        fh.close()


def _stage_g1(args, scorer):
    banner("g1", "smoke=%s shard=%s" % (args.smoke, args.shard))
    items = load_items(args.data_dir, subtasks=PHASE_B_ARENA)
    if not items:
        raise RuntimeError("g1: no object_placements items under %s" % args.data_dir)
    items_by_id = _items_by_id(items)

    # ---- preflight (BEFORE any model load): PREREG round-trip + g0 gate + G-C ----------
    _g_prereg_freeze(args, write_if_missing=False)
    g0_path = os.path.join(args.out_dir, "results_g0.json")
    if not os.path.exists(g0_path):
        raise RuntimeError("g1: results_g0.json not found under %s — run --stage g0 "
                           "first (the G-A gate must precede g1)" % args.out_dir)
    with open(g0_path) as f:
        g0 = json.load(f)
    ga_verdict = (g0.get("ga") or {}).get("verdict")
    smoke_kill_bypass = False
    if ga_verdict == sg.GA_KILL:
        if args.smoke:
            # SMOKE-ONLY controlled degradation (L1 fix): a well-formed KILL from a
            # smoke-scope g0 is not the canonical gate decision, and L1 needs the g1
            # pipeline exercisable end-to-end — so warn loudly and continue, marking
            # the bypass in config. The full (non-smoke) run still fail-closes here.
            smoke_kill_bypass = True
            log("g1: WARNING smoke bypass — G-A verdict=KILL comes from a smoke-scope "
                "g0 (NOT the canonical gate); proceeding for pipeline validation only. "
                "The full run will still fail-closed on KILL.")
        else:
            raise RuntimeError(
                "g1 REFUSED: the G-A fidelity gate verdict is KILL (median_rho=%r) — "
                "the TF-likelihood scoring route is dead and g1 must not run (design "
                "§4). Diagnosis: inspect results_g0.json 'ga' (per_group rhos, diag "
                "pools, items_skipped) and write the diagnostic VERDICT_gsd.md; do not "
                "re-run g1 until the gate is re-evaluated."
                % (g0.get("ga") or {}).get("median_rho"))
    elif ga_verdict not in (sg.GA_PASS, sg.GA_FLAG):
        # FAIL-CLOSED (smoke included): a missing / None / malformed verdict is NOT a
        # pass — only an explicit gate decision (PASS / flag_continue / the smoke KILL
        # bypass above) opens g1.
        raise RuntimeError(
            "g1 REFUSED (fail-closed): results_g0.json carries no valid G-A verdict "
            "(got %r; need %r or %r) — the 'ga' section is missing or malformed. Run a "
            "COMPLETE --stage g0 pass first, then re-run g1." % (ga_verdict, sg.GA_PASS,
                                                                 sg.GA_FLAG))
    g0_shard = (g0.get("config") or {}).get("shard")
    if g0_shard is not None:
        log("g1: WARNING results_g0.json was written by a SHARD pass (config.shard=%r) — "
            "the canonical g0 gate comes from the final un-sharded consolidation pass"
            % g0_shard)
    log("g1: preflight OK (G-A verdict=%s)" % ga_verdict)

    sticky = _g_sticky_ids(args) if not args.smoke else []
    _gp, _gt, _abl, patient_ids, knowledge_ids = _g_scope(args, sticky, [])
    proc_patients = _shard_slice(sorted(patient_ids), args.shard)
    proc_knowledge = _shard_slice(sorted(knowledge_ids), args.shard)
    log("g1: this pass — patients=%d knowledge=%d" % (len(proc_patients),
                                                      len(proc_knowledge)))

    excluded = []
    spaces = _g_build_spaces(items_by_id, proc_patients + proc_knowledge, excluded)
    for e in excluded:
        log("g1: EXCLUDED %s (%s)" % (e["item_id"], e["reason"]))
    # G-C scale assertion (design §4): every enumerated PATIENT space fits the ceiling.
    for iid in proc_patients:
        space = spaces.get(iid)
        if space is not None and space.path_count > G_PATH_COUNT_MAX:
            raise AssertionError("g1 G-C violation: item %s path_count=%d > %d"
                                 % (iid, space.path_count, G_PATH_COUNT_MAX))

    # ---- scorer (injected fake in tests; bf16 single card otherwise) -------------------
    shard_i = _shard_index(args.shard)
    g1_cache_path = _cache_path(args.out_dir, "g1_scores", shard_i)
    if scorer is None:
        scorer = _g_default_scorer(args, cache_path=g1_cache_path, stage="g1")
    _g_merge_score_cache(scorer, args.out_dir)

    b1_cache = load_stage_cache(args.out_dir, "b1")
    sc_n = int(sg.PREREG_G["ga_evidence_pools"]["patient"]["n_per_item"])   # SC@256
    dd = sg.PREREG_G["deepdive_ids"]
    dd_ids = set(dd.values())
    score_stats = {"seconds": 0.0, "n_scored": 0}
    deep = {}                          # role -> {item, space, scores_lse, row} (figures)
    rows_patient, rows_knowledge = [], []

    # ---- per-item assembly: score (dual caliber, cached) -> SC -> two arms + ledgers ---
    for kind, ids, rows in (("patient", proc_patients, rows_patient),
                            ("knowledge", proc_knowledge, rows_knowledge)):
        for iid in _progress(ids, total=len(ids), desc="g1-%s" % kind):
            it = items_by_id.get(iid)
            space = spaces.get(iid)
            if it is None or space is None:
                continue                                   # already on the exclusion roster
            t0 = time.time()
            try:
                tr = scorer.score_transitions(it, space, template="main")
            except gsd_score.ScorerOOM as exc:
                excluded.append({"item_id": iid,
                                 "reason": "ScorerOOM in transition scoring: %s" % exc})
                log("  [g1-%s] %-30s OOM-SKIPPED (%s)" % (kind, iid, exc))
                continue
            dt = time.time() - t0
            score_stats["seconds"] += dt
            score_stats["n_scored"] += tr["n_scored"]
            # SC@256 from the cached b1 patient pool. Knowledge controls have NO b1 pool
            # (design §3), so their record set is empty and the SC slot is null by
            # contract — readout 4's denominator is the PATIENT rows alone (stages_g).
            recs = sc.cached_records(b1_cache, it["subtask"], iid, "b1_patient")
            sc_ans = sg.sc_from_cache(recs, n=sc_n)
            row = sg.g1_item_analysis(it, space, tr["lse"], sc_ans)
            row["ledger_raw"] = gdec.delta_ledger(space, tr["raw"], it)   # dual caliber
            rows.append(row)
            log("  [g1-%s] %-30s layers=%d paths=%d scored=%d sec=%.2f map=%r "
                "oracle_fixed=%s delta=%s"
                % (kind, iid, space.T, space.path_count, tr["n_scored"], dt,
                   row["map"]["answer_idx"], row["oracle"]["fixed"],
                   row["ledger"]["delta_total"]))
            if iid in dd_ids:
                role = next(k for k, v in dd.items() if v == iid)
                deep[role] = {"item": it, "space": space, "scores_lse": tr["lse"],
                              "row": row}

    # ---- deep-dive template sensitivity (design §6: one paraphrase rerun, descriptive) -
    sensitivity = _g1_sensitivity(scorer, deep, excluded, score_stats)

    # ---- six-readout assembly (R5 merged from results_g0) + guarded write --------------
    eff = _g_efficiency(score_stats["seconds"], score_stats["n_scored"],
                        len(rows_patient) + len(rows_knowledge))
    cfg = _g_config(args, "g1")
    cfg["smoke_ga_kill_bypass"] = smoke_kill_bypass       # False on every non-smoke run
    results = sg.assemble_results_g1(rows_patient, rows_knowledge, g0,
                                     config=cfg, efficiency=eff,
                                     sensitivity=sensitivity)
    results["scope"] = {"patient": [r["item_id"] for r in rows_patient],
                        "knowledge": [r["item_id"] for r in rows_knowledge]}
    results["excluded_items"] = excluded
    results["run_stats"] = {"score_seconds": score_stats["seconds"]}
    ro = results["readouts"]
    _print_g_summary("g1", {
        "oracle_online_fixed": "%d/%d (%s)" % (ro["2_oracle_online"]["n_fixed"],
                                               ro["2_oracle_online"]["n_patients"],
                                               ro["2_oracle_online"]["verdict"]),
        "zero_cov_fixed": ro["3_zero_coverage"]["n_fixed"],
        "lambda0_sc_agreement": ro["4_lambda0_vs_sc"]["agreement_rate"],
        "readout_artifact_share": ro["5_backtrace_readout"]["readout_artifact_share"],
        "knowledge_fixed": ro["6_knowledge_control"]["n_fixed"],
        "sensitivity_rho": sensitivity["rho"],
        "excluded": len(excluded)}, eff)
    write_results(os.path.join(args.out_dir, "results_g1.json"), results)

    # ---- figures (best-effort AFTER the results write, the b2 convention) --------------
    _g1_write_figures(args, deep, rows_patient)
    return results


def _g1_sensitivity(scorer, deep, excluded, score_stats):
    """Rescore the deep-dive trio under SENSITIVITY_TEMPLATE -> per-item Δ pairs + the
    Δ-ordering Spearman (descriptive; scores cached under the sensitivity template sha)."""
    per_item = []
    for role in ("nonzero", "zero_cov", "knowledge"):
        d = deep.get(role)
        if d is None:                                      # outside this pass's scope
            continue
        iid = d["item"]["id"]
        t0 = time.time()
        try:
            tr_s = scorer.score_transitions(d["item"], d["space"], template="sensitivity")
        except gsd_score.ScorerOOM as exc:
            excluded.append({"item_id": iid,
                             "reason": "ScorerOOM in sensitivity rescoring: %s" % exc})
            log("  [g1-sens] %-30s OOM-SKIPPED (%s)" % (iid, exc))
            continue
        score_stats["seconds"] += time.time() - t0
        score_stats["n_scored"] += tr_s["n_scored"]
        ledger_s = gdec.delta_ledger(d["space"], tr_s["lse"], d["item"])
        per_item.append({"item_id": iid, "role": role,
                         "delta_main": d["row"]["ledger"]["delta_total"],
                         "delta_sens": ledger_s["delta_total"]})
        log("  [g1-sens] %-30s role=%-9s delta_main=%s delta_sens=%s"
            % (iid, role, per_item[-1]["delta_main"], per_item[-1]["delta_sens"]))
    pairs = [(e["delta_main"], e["delta_sens"]) for e in per_item
             if e["delta_main"] is not None and e["delta_sens"] is not None]
    rho = (sg.spearman_rho([a for a, _b in pairs], [b for _a, b in pairs])
           if len(pairs) >= 2 else None)
    return {"rho": rho, "n_pairs": len(pairs), "per_item": per_item,
            "template_sha256": gsd_score.SENSITIVITY_TEMPLATE_SHA256}


def _g1_write_figures(args, deep, rows_patient):
    """The g1 figure set -> ``<out_dir>/figures/gsd_*.png``. Full run: 3 deep-dive
    trellises + Δ-ledger bars + demand curve; --smoke: EXACTLY the two deep-dive patient
    trellises (the frozen smoke contract). Best-effort (b2 convention): a matplotlib
    failure logs loudly and never voids the already-written results."""
    try:
        fig_dir = os.path.join(args.out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        roles = ("nonzero", "zero_cov") if args.smoke else \
            ("nonzero", "zero_cov", "knowledge")
        written = []
        for role in roles:
            d = deep.get(role)
            if d is None:
                continue
            path = os.path.join(fig_dir, "gsd_deepdive_%s.png" % role)
            written.append(sg.fig_deepdive_trellis(
                d["item"], d["space"], d["scores_lse"], d["row"]["map"],
                d["row"]["oracle"], d["row"]["ledger"], path))
        if not args.smoke and rows_patient:
            written.append(sg.fig_delta_ledger(
                rows_patient, os.path.join(fig_dir, "gsd_delta_ledger.png")))
            written.append(sg.fig_demand_curve(
                rows_patient, os.path.join(fig_dir, "gsd_demand_curve.png")))
        log("g1: wrote %d figure(s) to %s" % (len(written), fig_dir))
    except Exception as exc:
        log("g1: figure generation error (non-fatal): %s" % exc)


def _print_g_summary(stage, headline, eff):
    log("---- %s summary ----" % stage)
    for k, v in headline.items():
        log("  %-24s %s" % (k, v))
    sco = eff["scoring"]
    log("  efficiency: scoring_gpu_h=%.4f n_scored=%d | generation=0 | total_gpu_h=%.4f"
        % (sco["gpu_hours"], sco["n_scored"], eff["total_gpu_hours"]))


# ======================================================================================
# STAGE: g2 — G-A fidelity narrowing (three arms: Arm0 noise ceiling / Arm1 matched-context ρ / Arm2 frequency decode)
# ======================================================================================
# The g2 shell wires the pure Subtask 2-5 analyzers (``stages_g2``) around the ONE GPU station
# (``gsd_sample.sample_node_freqs`` — the per-branching-node conditional next-BELIEF sampler,
# Subtask 1) exactly the way g0/g1 wire the pure ``stages_g`` analyzers around the TF scorer:
# both heavy sites (the sampler ``emitter`` and the TF ``scorer``) are dependency-injected;
# ``None`` builds the real bf16 model, tests inject CPU fakes. Phase order (design §2):
# preflight -> item set -> Arm 0 (zero GPU, split-half over b1 chains) -> the sampling pass
# (the sole GPU cost) -> Arm 1 (matched-context rho over the samples + cached/fresh TF) ->
# Arm 2 (frequency-decode the 13 decode items) -> assemble + combined verdict + figures.
class _LazyEmitter:
    """Defer the (heavy) HFEmitter build until the FIRST actual generate call, so a
    full-cache-hit consolidation pass (which re-draws nothing) never loads the 4B model.
    Honours the ``sc_core.Emitter`` protocol (the sampler only ever calls ``generate``)."""

    def __init__(self, build):
        self._build = build
        self._inner = None

    def generate(self, prompts, seeds, *, greedy=False, max_new_tokens=None):
        if self._inner is None:
            self._inner = self._build()
        return self._inner.generate(prompts, seeds, greedy=greedy,
                                    max_new_tokens=max_new_tokens)


def _g2_sample_cache_path(out_dir, shard_i):
    """The g2 conditional-sample JSONL cache (content-keyed): the un-sharded consolidation
    file, or the shard-named variant during a ``--shard`` pass."""
    if shard_i is None:
        return os.path.join(out_dir, "cache_g2_sample.jsonl")
    return os.path.join(out_dir, "cache_g2_sample_shard%d.jsonl" % shard_i)


def _g2_consolidate_sample_cache(out_dir, dest_path):
    """Merge every ``cache_g2_sample_shard*.jsonl`` into the un-sharded consolidation cache so
    the consolidation pass re-samples nothing (full cache hit). Content keys are disjoint
    across shards (item-sharded), and dedup by the 5-part content key keeps re-consolidation
    idempotent (last-write-wins, deterministic classification). Never raises on a torn line."""
    shard_files = sorted(glob.glob(os.path.join(out_dir, "cache_g2_sample_shard*.jsonl")))
    if not shard_files:
        return
    seen = set(gsd_sample._load_sample_cache(dest_path))    # keys already in the dest file
    appended = 0
    with open(dest_path, "a") as out:
        for fp in shard_files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = gsd_sample._sample_cache_key(rec)
                    except (ValueError, TypeError, KeyError):
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(json.dumps(rec) + "\n")
                    appended += 1
    log("g2: consolidated %d shard sample-cache file(s) (+%d record(s)) -> %s"
        % (len(shard_files), appended, os.path.basename(dest_path)))


def _g2_prime_score_cache(scorer, out_dir):
    """Prime a TFScorer's in-memory transition cache with EVERY ``cache_g1_scores*.jsonl`` (the
    22 patient+knowledge items g1 already scored) AND ``cache_g2_scores*.jsonl`` (tuning items
    a prior g2 pass scored), so this pass re-scores only genuinely-missing edges (tuning items
    lacking a g1 score) and a consolidation rerun scores nothing. A scorer without a ``_cache``
    dict (a bare stub) is left untouched (the g1 ``_g_merge_score_cache`` convention)."""
    cache = getattr(scorer, "_cache", None)
    if not isinstance(cache, dict):
        return
    n0 = len(cache)
    for pattern in ("cache_g1_scores*.jsonl", "cache_g2_scores*.jsonl"):
        for fp in sorted(glob.glob(os.path.join(out_dir, pattern))):
            for k, v in gsd_score._load_score_cache(fp).items():
                cache.setdefault(k, v)
    if len(cache) > n0:
        log("g2: primed %d transition score(s) from g1/g2 score caches"
            % (len(cache) - n0))


def _g2_efficiency(sample_seconds, sample_n_sampled, score_seconds, score_n_scored, n_items):
    """g2 efficiency block: the sampler is the SOLE GPU station (sampling GPU-seconds +
    fresh-draw count), scoring is a near-zero cached re-use (fresh forwards only for tuning
    items lacking a g1 score) — the two are reported SEPARATELY (design §7)."""
    total = sample_seconds + score_seconds
    return {
        "n_items": n_items,
        "sampling": {"gpu_seconds": sample_seconds, "gpu_hours": sample_seconds / 3600.0,
                     "n_sampled": sample_n_sampled},
        "scoring": {"gpu_seconds": score_seconds, "gpu_hours": score_seconds / 3600.0,
                    "n_scored": score_n_scored},
        "total_gpu_hours": total / 3600.0,
    }


def _g2_write_verdict_md(path, results):
    """Write the plain-language ``VERDICT_ga.md`` from the assembled results_g2 (the combined
    verdict + the three arms' headline numbers). Mirrors the stages_g VERDICT convention."""
    v = results["verdict"]
    a0, a1a, a1b, a2 = results["arm0"], results["arm1a"], results["arm1b"], results["arm2"]
    eff = results["efficiency"]

    def _n(x):
        return "n/a" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))

    md = [
        "# MuSR-cant G-A fidelity narrowing — VERDICT (stage g2)",
        "",
        "Decides whether the reproducible λ=0 fixes reflect the model's on-policy belief or a "
        "TF-likelihood-under-GSD scoring artifact, by replacing the ρ=0.400 frequency target "
        "with matched-context on-policy sampling under the SAME frozen GSD template. Design: "
        "`plans/2026-07-08-ga-fidelity.md` + `…-design.md` §3. Frozen: "
        "`outputs/PREREG_g2.md`.",
        "",
        "## Verdict: **%s**" % v["label"],
        "",
        v["reason"],
        "",
        "## Arm 2 — frequency-decode reproduction (the main readout)",
        "",
        "- reproducible-7 reproduced: **%s / %s** (denominator = decoded non-zero-coverage "
        "repro items; frozen total %s)."
        % (_n(a2.get("n_repro")), _n(a2.get("n_repro_denominator")),
           _n(a2.get("n_repro_total"))),
        "- zero-coverage controls reproduced: %d / 3 (FINDING axis — free-gen zero coverage does "
        "NOT imply conditional-sampling impossibility; the GSD template spoon-feeds s_prev + the "
        "mechanical event line)."
        % sum(1 for f in a2.get("zero_cov", {}).values() if f),
        "- knowledge controls reproduced: %d / 3 (SPECIFICITY — reproduction should be LOW here; "
        "if knowledge failures reproduce too, frequency-decode reproduction is non-specific)."
        % sum(1 for f in a2.get("knowledge", {}).values() if f),
        "- fail-closed excluded (parse/coverage below gate, NOT counted as 'not reproduced'): "
        "%d — %s." % (len(a2.get("excluded_inconclusive", ())),
                      [e.get("item_id") for e in a2.get("excluded_inconclusive", ())] or "none"),
        "- ε sensitivity (n_repro at each frozen ε): %s." % (a2.get("eps_sensitivity"),),
        "",
        "## Arm 1 — matched-context fidelity ρ",
        "",
        "- Arm 1a (strict, the 35 G-A counted groups vs the flagged 0.400): median ρ_matched "
        "= %s (raw %s) over %s group(s); context share = %s."
        % (_n(a1a.get("median_rho_matched")), _n(a1a.get("median_rho_matched_raw")),
           _n(a1a.get("n_groups")), _n(a1a.get("context_share"))),
        "- Arm 1b (specificity): patient median ρ = %s (%s node(s)) vs knowledge median ρ = %s "
        "(%s node(s)) — the patient median should NOT sit systematically below knowledge."
        % (_n(a1b.get("median_rho_patient")), _n(a1b.get("n_patient")),
           _n(a1b.get("median_rho_knowledge")), _n(a1b.get("n_knowledge"))),
        "",
        "## Arm 0 — split-half noise ceiling (descriptive)",
        "",
        "- median split-half reliability ρ_cc = %s; disattenuated ρ = %s; observed ρ = %s "
        "(over %s counted group(s))."
        % (_n(a0.get("median_rho_cc")), _n(a0.get("median_rho_disattenuated")),
           _n(a0.get("median_rho_observed")), _n(a0.get("n_groups"))),
        "",
        "## Efficiency",
        "",
        "- sampling (the sole GPU station): %.4f GPU-h over %d fresh draw(s); scoring "
        "re-use: %.4f GPU-h over %d fresh forward(s); total %.4f GPU-h."
        % (eff["sampling"]["gpu_hours"], eff["sampling"]["n_sampled"],
           eff["scoring"]["gpu_hours"], eff["scoring"]["n_scored"], eff["total_gpu_hours"]),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def _g2_write_figures(args, results):
    """The g2 figure set -> ``<out_dir>/figures/gsd_g2_*.png`` (best-effort, the g1/b2
    convention: a matplotlib failure logs loudly and never voids the written results)."""
    try:
        fig_dir = os.path.join(args.out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        prereg = results["prereg"]
        written = [
            sg2.fig_rho_three_caliber(results["arm0"], results["arm1a"], prereg,
                                      os.path.join(fig_dir, "gsd_g2_rho_three_caliber.png")),
            sg2.fig_repro_ledger(results["arm2"], prereg,
                                 os.path.join(fig_dir, "gsd_g2_repro_ledger.png")),
            sg2.fig_share_decomposition(results["arm0"], results["arm1a"], prereg,
                                        os.path.join(fig_dir, "gsd_g2_share_decomposition.png")),
        ]
        log("g2: wrote %d figure(s) to %s"
            % (sum(1 for w in written if w), fig_dir))
    except Exception as exc:
        log("g2: figure generation error (non-fatal): %s" % exc)


def stage_g2(args, scorer=None, emitter=None):
    """The g2 (G-A fidelity narrowing) shell. Both heavy sites are dependency-injected: the
    sampler ``emitter`` (the SOLE GPU station; None -> a lazily-built HFEmitter) and the TF
    ``scorer`` (None -> the real bf16 model; tests inject CPU fakes). Log lines tee to
    ``outputs/g2[_shard<i>].log`` (the g0/g1 convention)."""
    global _LOG_FH
    fh = _g_open_log(args, "g2")
    _LOG_FH = fh
    try:
        return _stage_g2(args, scorer, emitter)
    finally:
        _LOG_FH = None
        fh.close()


def _stage_g2(args, scorer, emitter):
    prereg = sg2.PREREG_G2
    banner("g2", "smoke=%s shard=%s M=%s"
           % (args.smoke, args.shard, args.g2_sample_m))
    items = load_items(args.data_dir, subtasks=PHASE_B_ARENA)
    if not items:
        raise RuntimeError("g2: no object_placements items under %s" % args.data_dir)
    items_by_id = _items_by_id(items)

    # ---- 1) preflight: PREREG_g2 freeze + round-trip; g1 exists; g0 gate != KILL -----------
    prereg_path = os.path.join(args.out_dir, "PREREG_g2.md")
    if not os.path.exists(prereg_path):
        sg2.write_prereg_g2(prereg_path)
        log("PREREG_g2 frozen -> %s" % prereg_path)
    sg2.check_prereg_g2_roundtrip(prereg_path)                 # AssertionError names drift
    log("PREREG_g2 round-trip OK: %s" % prereg_path)

    if not os.path.exists(os.path.join(args.out_dir, "results_g1.json")):
        raise RuntimeError("g2: results_g1.json not found under %s — run --stage g1 first "
                           "(g2 diagnoses the g1/g0 G-A gate)" % args.out_dir)
    g0_path = os.path.join(args.out_dir, "results_g0.json")
    if not os.path.exists(g0_path):
        raise RuntimeError("g2: results_g0.json not found under %s — run --stage g0 first "
                           "(g2 needs its ga.per_group + observed rho map)" % args.out_dir)
    with open(g0_path) as f:
        g0 = json.load(f)
    ga = g0.get("ga") or {}
    ga_verdict = ga.get("verdict")
    if ga_verdict == sg.GA_KILL:
        if args.smoke:
            log("g2: WARNING smoke bypass — G-A verdict=KILL comes from a smoke-scope g0 "
                "(NOT the canonical gate); proceeding for pipeline validation only.")
        else:
            raise RuntimeError(
                "g2 REFUSED: the G-A fidelity gate verdict is KILL (median_rho=%r) — the "
                "TF-likelihood scoring route is dead; there is no fidelity to narrow (design "
                "§4). Re-evaluate the gate before g2." % ga.get("median_rho"))
    ga_per_group = ga.get("per_group") or []
    if not ga_per_group:
        raise RuntimeError("g2: results_g0.json carries no ga.per_group — the G-A group set "
                           "(the Arm 0/1 universe) is empty; run a COMPLETE --stage g0 first")
    log("g2: preflight OK (G-A verdict=%s, %d counted group(s))"
        % (ga_verdict, len(ga_per_group)))

    # ---- 2) item set = the G-A group items ∪ the 13 decode items ----------------------------
    observed_rho_by_group = {}
    ga_item_pool = {}
    for r in ga_per_group:
        key = (r["item_id"], r["t"], r["sprev_sha8"])
        observed_rho_by_group[key] = r.get("rho")
        ga_item_pool.setdefault(r["item_id"], r.get("pool"))
    ga_item_ids = sorted(ga_item_pool)
    repro7 = [str(x) for x in prereg["repro7_ids"]]
    zero_cov = [str(x) for x in prereg["zero_cov_ctrl_ids"]]
    knowledge = [str(x) for x in prereg["knowledge_ctrl_ids"]]
    decode_ids = repro7 + zero_cov + knowledge
    if args.smoke:
        # SMOKE scoping (the g0/g1 deep-dive convention on the SAME code path; L1 <10 min):
        # shrink to a tiny deterministic feed — one G-A group item (Arm 0 / Arm 1a) plus one
        # repro patient + one knowledge control (Arm 1b / Arm 2). Smoke results are never
        # canonical; the full (non-smoke) run enumerates the frozen pools unchanged.
        ga_item_ids = ga_item_ids[:1]
        decode_ids = repro7[:1] + knowledge[:1]
        log("g2[smoke]: scoped to G-A item(s) %s + decode item(s) %s (same code path)"
            % (ga_item_ids, decode_ids))
    all_ids = sorted(set(ga_item_ids) | set(decode_ids))

    excluded = []
    spaces = _g_build_spaces(items_by_id, all_ids, excluded)
    for e in excluded:
        log("g2: EXCLUDED %s (%s)" % (e["item_id"], e["reason"]))

    # G-C scale assertion on the decode PATIENT spaces (design §4 — a construction bug raises).
    for iid in [i for i in (repro7 + zero_cov) if i in spaces]:
        if spaces[iid].path_count > G_PATH_COUNT_MAX:
            raise AssertionError("g2 G-C violation: item %s path_count=%d > %d"
                                 % (iid, spaces[iid].path_count, G_PATH_COUNT_MAX))

    # ---- 3) Arm 0 — split-half noise ceiling (ZERO GPU, recompute over the b1 chains) -------
    banner("g2-Arm0", "split-half noise ceiling (zero GPU)")
    b1_cache = load_stage_cache(args.out_dir, "b1")
    chains_by_item = {}
    for iid in ga_item_ids:
        it = items_by_id.get(iid)
        if it is None:
            continue
        seed_tag = "b1_patient" if ga_item_pool.get(iid) == "patient" else "b1_tuning"
        chains_by_item[iid] = _g_chains(b1_cache, it, seed_tag)
    arm0 = sg2.arm0_noise_ceiling(chains_by_item, prereg, observed_rho_by_group)
    log("g2-Arm0: n_groups=%d median_rho_cc=%s median_rho_disatt=%s"
        % (arm0["n_groups"], arm0["median_rho_cc"], arm0["median_rho_disattenuated"]))

    # ---- re-materialise the G-A counted groups (canon_prev states) from the b1 chains -------
    # results_g0 ga.per_group carries only sprev_sha8; re-run ga_collect_groups to recover each
    # group's canon_prev (the Arm 1a successor-set + the Arm 1a sampling node key).
    ga_groups = []
    ga_group_nodes = {}                                        # item_id -> {(t, canon_prev)}
    for iid in ga_item_ids:
        it, space = items_by_id.get(iid), spaces.get(iid)
        if it is None or space is None:
            continue
        by_key = {(g["t"], sg._sha8(g["s_prev_canon"])): g
                  for g in sg.ga_collect_groups(chains_by_item.get(iid, []),
                                                 t_max=space.T - 1)}
        for r in ga_per_group:
            if r["item_id"] != iid:
                continue
            g = by_key.get((r["t"], r["sprev_sha8"]))
            if g is None:
                # A ga.per_group group that ga_collect_groups no longer reproduces from the b1
                # chains (cache drift / a torn record) is RECORDED on the exclusion roster —
                # never a silent shrink of the 35-group denominator (design §7: exclusions are
                # always named). It cannot be re-materialised, so it also cannot be sampled or
                # entered into Arm 1a.
                excluded.append({"item_id": iid, "t": r["t"], "sprev_sha8": r["sprev_sha8"],
                                 "reason": "G-A group not re-materialisable from b1 chains "
                                           "(no ga_collect_groups match for its sprev_sha8)"})
                log("g2: EXCLUDED G-A group %s t=%d sprev=%s (not re-materialisable from b1 "
                    "chains)" % (iid, r["t"], r["sprev_sha8"]))
                continue
            ga_groups.append({"item_id": iid, "t": g["t"], "s_prev_canon": g["s_prev_canon"],
                              "states": g["states"]})
            ga_group_nodes.setdefault(iid, set()).add((g["t"], g["s_prev_canon"]))

    # ---- 4) sampling pass (the ONLY GPU station) — Arm1a nodes ∪ Arm2 branching nodes -------
    # per item the requested nodes = the G-A group nodes (Arm 1a) ∪ every >=2-successor
    # branching node (Arm 2, for the 13 decode items).
    nodes_req = {}
    for iid, nds in ga_group_nodes.items():
        nodes_req.setdefault(iid, set()).update(nds)
    for iid in decode_ids:
        space = spaces.get(iid)
        if space is None:
            continue
        nodes_req.setdefault(iid, set()).update(
            (t, cp) for (t, cp), edges in space.trans.items()
            if t >= 1 and len(edges) >= 2)

    sample_item_ids = sorted(iid for iid, req in nodes_req.items() if req)
    proc_items = _shard_slice(sample_item_ids, args.shard)
    log("g2: sampling %d of %d item(s) this pass (shard=%s)"
        % (len(proc_items), len(sample_item_ids), args.shard))

    sample_M = args.g2_sample_m if args.g2_sample_m is not None else prereg["sample_M"]
    shard_i = _shard_index(args.shard)
    sample_cache_path = _g2_sample_cache_path(args.out_dir, shard_i)
    if shard_i is None:                                       # consolidation: merge shard caches
        _g2_consolidate_sample_cache(args.out_dir, sample_cache_path)

    if emitter is None:                                       # lazy: no model load if full-hit
        emitter = _LazyEmitter(lambda: sc.HFEmitter(
            args.main_model, device=args.device, batch=args.batch,
            max_new_tokens=prereg["sample_max_new_tokens"]))

    banner("g2-sampling", "GSD-template node-conditional next-BELIEF (M=%d)" % sample_M)
    sample_nodes_by_item = {}
    sample_seconds = 0.0
    sample_n_sampled = 0
    for iid in _progress(proc_items, total=len(proc_items), desc="g2-sample"):
        it, space = items_by_id.get(iid), spaces.get(iid)
        if it is None or space is None:
            continue
        req = sorted(nodes_req[iid])
        t0 = time.time()
        try:
            sn = gsd_sample.sample_node_freqs(
                it, space, emitter, M=sample_M, base_seed=prereg["sample_base_seed"],
                nodes=req, min_succ=2, cache_path=sample_cache_path, stage="g2",
                max_new_tokens=prereg["sample_max_new_tokens"], batch=args.batch)
        except sc.EmitterOOM as exc:                          # HFEmitter already halved once
            excluded.append({"item_id": iid, "reason": "EmitterOOM in sampling: %s" % exc})
            log("  [g2-sample] %-30s OOM-SKIPPED (%s)" % (iid, exc))
            continue
        dt = time.time() - t0
        sample_seconds += dt
        sample_n_sampled += sn["n_scored"]
        sample_nodes_by_item[iid] = sn
        n_on = sum(sum(nd["freqs"].values()) for nd in sn["nodes"].values())
        n_off = sum(nd["off_manifold"] for nd in sn["nodes"].values())
        n_tot = sum(nd["n_total"] for nd in sn["nodes"].values())
        log("  [g2-sample] %-30s nodes=%d draws=%d scored=%d parse_ok=%.3f off=%.3f "
            "on_manifold=%d sec=%.2f"
            % (iid, len(sn["nodes"]), n_tot, sn["n_scored"], sn["item_parse_ok_rate"],
               (n_off / n_tot) if n_tot else 0.0, n_on, dt))

    # ---- 5) Arm 1 — matched-context rho (TF from cache; tuning items scored + cached now) ---
    banner("g2-Arm1", "matched-context fidelity ρ (1a strict + 1b specificity)")
    # The score-cache key INCLUDES the stage (gsd_score 6-part key); g1 wrote its 22
    # patient+knowledge transition scores under stage="g1", so the g2 scorer MUST also carry
    # stage="g1" for those lookups to hit (a g2-stage scorer would silently re-score every
    # patient+knowledge edge — the "primed N scores" would then be a no-op). Fresh tuning
    # scores persist under stage="g1" into cache_g2_scores* and consolidate cleanly. NOTE:
    # this is the SCORE cache stage only; the SAMPLE cache stays stage="g2" (a separate cache).
    score_cache_path = _cache_path(args.out_dir, "g2_scores", shard_i)
    if scorer is None:
        scorer = _g_default_scorer(args, cache_path=score_cache_path, stage="g1")
    _g2_prime_score_cache(scorer, args.out_dir)               # g1 + prior-g2 cached edges
    tf_scores_by_item = {}
    score_seconds = 0.0
    score_n_scored = 0
    for iid in sorted(sample_nodes_by_item):                  # only the items we have samples for
        it, space = items_by_id.get(iid), spaces.get(iid)
        if it is None or space is None:
            continue
        t0 = time.time()
        try:
            tr = scorer.score_transitions(it, space, template="main")
        except gsd_score.ScorerOOM as exc:
            excluded.append({"item_id": iid,
                             "reason": "ScorerOOM in transition scoring: %s" % exc})
            log("  [g2-score] %-30s OOM-SKIPPED (%s)" % (iid, exc))
            continue
        score_seconds += time.time() - t0
        score_n_scored += tr["n_scored"]
        tf_scores_by_item[iid] = {"lse": tr["lse"], "raw": tr["raw"]}

    arm1a = sg2.arm1a_strict(ga_groups, sample_nodes_by_item, tf_scores_by_item, prereg)
    log("g2-Arm1a: median_rho_matched=%s over %d group(s) (%d excluded); context_share=%s"
        % (arm1a["median_rho_matched"], arm1a["n_groups"], len(arm1a["excluded"]),
           arm1a["context_share"]))
    # PRE-FILTER Arm 1b input: drop decode items below the parse-ok gate (fail-closed at the
    # boundary — Arm1b itself is descriptive and does not gate).
    filtered_samples = {iid: sn for iid, sn in sample_nodes_by_item.items()
                        if sn.get("item_parse_ok_rate", 0.0) >= prereg["parse_ok_min"]}
    arm1b = sg2.arm1b_specificity(filtered_samples, tf_scores_by_item, prereg,
                                  patient_ids=repro7 + zero_cov, knowledge_ids=knowledge)
    log("g2-Arm1b: patient median_rho=%s (%d node) vs knowledge median_rho=%s (%d node)"
        % (arm1b["median_rho_patient"], arm1b["n_patient"],
           arm1b["median_rho_knowledge"], arm1b["n_knowledge"]))

    # ---- 6) Arm 2 — frequency-decode the 13 decode items ------------------------------------
    banner("g2-Arm2", "frequency-decode reproduction (A_freq -> gsd_decode map)")
    items_spaces_samples = []
    for iid in decode_ids:
        it, space = items_by_id.get(iid), spaces.get(iid)
        sn = sample_nodes_by_item.get(iid)
        if it is None or space is None or sn is None:
            continue                                          # absent / malformed / not sampled
        items_spaces_samples.append({"item_id": iid, "item": it, "space": space,
                                     "sample_nodes": sn})
    arm2 = sg2.arm2_all(items_spaces_samples, prereg)
    log("g2-Arm2: n_repro=%d/%s (total %s); zero_cov_reproduced=%d; excluded_inconclusive=%d"
        % (arm2["n_repro"], arm2["n_repro_denominator"], arm2["n_repro_total"],
           sum(1 for f in arm2["zero_cov"].values() if f),
           len(arm2["excluded_inconclusive"])))

    # ---- 7) assemble + combined verdict + figures + VERDICT_ga.md ---------------------------
    banner("g2-assemble", "combined verdict + results_g2.json")
    scope = {
        "smoke": bool(args.smoke),
        "shard": args.shard,
        "ga_item_ids": ga_item_ids,
        "decode_ids": decode_ids,
        "decode_in_arena": sorted(i for i in decode_ids if i in items_by_id),
        "sampled_item_ids": sorted(sample_nodes_by_item),
        "n_arm1a_nodes": sum(len(v) for v in ga_group_nodes.values()),
        "n_arm2_decode_nodes": sum(len(nodes_req.get(i, ())) for i in decode_ids),
        "dropped_items": [e["item_id"] for e in excluded],
        "sample_M": sample_M,
    }
    eff = _g2_efficiency(sample_seconds, sample_n_sampled, score_seconds, score_n_scored,
                         len(sample_nodes_by_item))
    results = sg2.assemble_results_g2(arm0, arm1a, arm1b, arm2, prereg, scope, excluded, eff)

    # fail-closed override: if OVER HALF of the in-arena decode set could not be decoded
    # (malformed tree / fail-closed low parse|coverage), the verdict is inconclusive — the
    # fidelity share cannot be adjudicated (design §2.4 fail-closed). Decode items ENTIRELY
    # absent from the arena are a provenance issue (named on the exclusion roster), not part of
    # the adjudicable set, so they do not enter this ratio.
    decode_in_arena = [i for i in decode_ids if i in items_by_id]
    decoded_ok = {str(pi["item_id"]) for pi in arm2["per_item"] if not pi["excluded"]}
    n_present = len(decode_in_arena)
    n_undecoded = sum(1 for i in decode_in_arena if i not in decoded_ok)
    if n_present == 0 or n_undecoded * 2 > n_present:
        reason = ("inconclusive: %d of %d in-arena decode item(s) could not be decoded "
                  "(malformed tree / fail-closed parse|coverage) — over half of the "
                  "adjudicable set is missing, so the fidelity share cannot be decided."
                  % (n_undecoded, n_present))
        log("g2: OVER-HALF decode items undecoded (%d/%d) -> verdict forced INCONCLUSIVE"
            % (n_undecoded, n_present))
        results["verdict"] = {"label": sg2.VERDICT_INCONCLUSIVE, "reason": reason}
        results["run_stats"]["verdict_label"] = sg2.VERDICT_INCONCLUSIVE

    results["run_stats"]["sample_n_scored"] = sample_n_sampled
    results["run_stats"]["score_n_scored"] = score_n_scored

    _print_g_summary("g2", {
        "verdict": results["verdict"]["label"],
        "arm2_n_repro": "%d/%s" % (arm2["n_repro"], arm2["n_repro_denominator"]),
        "arm1a_median_rho": arm1a["median_rho_matched"],
        "arm0_median_rho_cc": arm0["median_rho_cc"],
        "sampled_items": len(sample_nodes_by_item),
        "excluded": len(excluded)},
        {"scoring": {"gpu_hours": eff["scoring"]["gpu_hours"],
                     "n_scored": eff["scoring"]["n_scored"]},
         "total_gpu_hours": eff["total_gpu_hours"]})
    log("g2: efficiency — sampling %.4f GPU-h (%d draws) | scoring %.4f GPU-h (%d fresh) "
        "| total %.4f GPU-h" % (eff["sampling"]["gpu_hours"], eff["sampling"]["n_sampled"],
                                eff["scoring"]["gpu_hours"], eff["scoring"]["n_scored"],
                                eff["total_gpu_hours"]))

    write_results(os.path.join(args.out_dir, "results_g2.json"), results)
    _g2_write_verdict_md(os.path.join(args.out_dir, "VERDICT_ga.md"), results)
    log("g2: wrote VERDICT_ga.md (verdict=%s)" % results["verdict"]["label"])
    _g2_write_figures(args, results)
    return results


# ======================================================================================
# STAGE: g3 — event-line ablation (two stations, three arms: Station A TF-scoring audit / Station B sampling probe + continuous readout)
# ======================================================================================
# The g3 shell wires the pure Subtask 3-4 analyzers (``stages_g3``) around the SAME two
# heavy sites g2 wires (the TF ``scorer`` for station A, the sampler ``emitter`` for
# station B), both dependency-injected; the only new lever is the TEMPLATE arm
# (gsd_score "anchor"/"noevent", Subtask 1) threaded through ``score_transitions`` and
# ``sample_node_freqs`` (Subtask 2). E-full is NEVER re-run: station A's baseline is
# asserted from results_g1, station B's from results_g2, and the continuous E-full
# baseline is RECOMPUTED from cache_g2_sample.jsonl behind a raising-stub emitter (any
# fresh draw = a frozen-baseline violation, fail-loud).
def _g3_sample_cache_path(out_dir, arm, shard_i):
    """The g3 per-ARM conditional-sample JSONL cache. The 5-part sample-cache key does
    NOT carry the template (gsd_sample contract), so each arm gets BOTH a unique stage
    string (``g3_<arm>``) and its own file — the plan's double guard against collisions."""
    if shard_i is None:
        return os.path.join(out_dir, "cache_g3_sample_%s.jsonl" % arm)
    return os.path.join(out_dir, "cache_g3_sample_%s_shard%d.jsonl" % (arm, shard_i))


def _g3_consolidate_sample_cache(out_dir, arm, dest_path):
    """Arm-aware generalization of ``_g2_consolidate_sample_cache``: merge every
    ``cache_g3_sample_<arm>_shard*.jsonl`` into the arm's un-sharded consolidation cache
    so the consolidation pass re-samples nothing (full cache hit). Dedup by the 5-part
    content key keeps re-consolidation idempotent; torn lines never raise."""
    shard_files = sorted(glob.glob(
        os.path.join(out_dir, "cache_g3_sample_%s_shard*.jsonl" % arm)))
    if not shard_files:
        return
    seen = set(gsd_sample._load_sample_cache(dest_path))
    appended = 0
    with open(dest_path, "a") as out:
        for fp in shard_files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = gsd_sample._sample_cache_key(rec)
                    except (ValueError, TypeError, KeyError):
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(json.dumps(rec) + "\n")
                    appended += 1
    log("g3: consolidated %d shard sample-cache file(s) for arm=%s (+%d record(s)) -> %s"
        % (len(shard_files), arm, appended, os.path.basename(dest_path)))


def _g3_prime_score_cache(scorer, out_dir):
    """Prime a TFScorer's in-memory transition cache with EVERY ``cache_g3_scores*.jsonl``
    (the un-sharded file + all shard siblings) so a resume / consolidation pass re-scores
    nothing. g1/g2 caches are NOT primed: their records live under the 'main' template sha
    and can never hit a g3a_<arm> key. A scorer without a ``_cache`` dict (a bare stub)
    is left untouched (the ``_g_merge_score_cache`` convention)."""
    cache = getattr(scorer, "_cache", None)
    if not isinstance(cache, dict):
        return
    n0 = len(cache)
    for fp in sorted(glob.glob(os.path.join(out_dir, "cache_g3_scores*.jsonl"))):
        for k, v in gsd_score._load_score_cache(fp).items():
            cache.setdefault(k, v)
    if len(cache) > n0:
        log("g3: primed %d transition score(s) from g3 score caches" % (len(cache) - n0))


class _G3RaisingEmitter:
    """The E-full continuous-baseline recompute must be a FULL cache hit over
    ``cache_g2_sample.jsonl`` — any attempted fresh draw silently re-estimating the
    frozen g2 baseline would be a provenance violation, so it fails loudly instead."""

    def generate(self, prompts, seeds, *, greedy=False, max_new_tokens=None):
        raise AssertionError(
            "g3 E-full baseline recompute attempted a FRESH draw — cache_g2_sample.jsonl "
            "must fully cover the decode items' branching nodes at M=%d (stage='g2', "
            "seed_tag='g2sample'); refusing to re-sample the frozen baseline"
            % sg2.PREREG_G2["sample_M"])


def _g3_stationB_baseline_check(results_g2, prereg):
    """Preflight: assert the E-full station-B baseline recomputed from a loaded
    results_g2 dict equals ``prereg['baseline_stationB']`` byte-for-byte (anti-drift):
    ``n_repro`` from arm2, the zero-cov / knowledge reproduced counts from their fixed-bit
    dicts. Returns the observed dict on a match; raises RuntimeError naming every
    mismatched count otherwise."""
    baseline = prereg["baseline_stationB"]
    arm2 = (results_g2 or {}).get("arm2") or {}
    observed = {
        "n_repro": arm2.get("n_repro"),
        "zero_cov": sum(1 for f in (arm2.get("zero_cov") or {}).values() if f),
        "knowledge": sum(1 for f in (arm2.get("knowledge") or {}).values() if f),
    }
    mismatches = {k: {"prereg": baseline[k], "results_g2": observed[k]}
                  for k in sorted(baseline) if baseline[k] != observed[k]}
    if mismatches:
        raise RuntimeError(
            "g3 preflight: results_g2 arm2 E-full baseline drifted from PREREG_G3 "
            "baseline_stationB: %s" % json.dumps(mismatches, sort_keys=True))
    return observed


def _g3_artifact_paths(args):
    """``(results_dir, fig_dir)`` for this pass. SMOKE-CLOBBER GUARD (the g-line
    backup_smoke dir convention, automated): a ``--smoke`` pass that finds a REAL
    (non-smoke) results_g3.json already in out_dir redirects ALL its g3 artifacts
    (results / VERDICT / figures) into ``<out_dir>/backup_smoke_g3/`` so a canonical
    full-run result is never overwritten by a smoke rerun. Caches are content-keyed
    (per-arm files + stage strings) and shared safely; PREREG_g3.md is byte-idempotent."""
    res_dir = args.out_dir
    if args.smoke:
        path = os.path.join(args.out_dir, "results_g3.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    prior_smoke = bool((json.load(f).get("config") or {}).get("smoke"))
            except Exception:
                prior_smoke = False              # unreadable -> treat as canonical (safe)
            if not prior_smoke:
                res_dir = os.path.join(args.out_dir, "backup_smoke_g3")
                os.makedirs(res_dir, exist_ok=True)
                log("g3[smoke]: existing results_g3.json is a FULL-run artifact — smoke "
                    "results/VERDICT/figures redirected to %s (canonical results never "
                    "clobbered)" % res_dir)
    return res_dir, os.path.join(res_dir, "figures")


def stage_g3(args, scorer=None, emitter=None):
    """The g3 (event-line ablation) shell. Both heavy sites are dependency-injected
    exactly like g2: the station-A TF ``scorer`` (None -> the real bf16 model, built only
    when station A runs) and the station-B sampler ``emitter`` (None -> a ``_LazyEmitter``
    that never loads the model on a full cache hit). Log lines tee to
    ``outputs/g3[_shard<i>].log`` (the g-line convention)."""
    global _LOG_FH
    fh = _g_open_log(args, "g3")
    _LOG_FH = fh
    try:
        return _stage_g3(args, scorer, emitter)
    finally:
        _LOG_FH = None
        fh.close()


def _stage_g3(args, scorer, emitter):
    prereg = sg3.PREREG_G3
    arms = list(prereg["arms"])
    station = args.g3_station
    run_A = station in ("A", "all")
    run_B = station in ("B", "all")
    gen_only = args.shard is not None                     # --shard = gen-only pass (plan §5)
    banner("g3", "smoke=%s shard=%s station=%s M=%s"
           % (args.smoke, args.shard, station, args.g3_sample_m))
    items = load_items(args.data_dir, subtasks=PHASE_B_ARENA)
    if not items:
        raise RuntimeError("g3: no object_placements items under %s" % args.data_dir)
    items_by_id = _items_by_id(items)

    # ---- 1) preflight (BEFORE any model load): freeze + upstream files + E-full baselines --
    prereg_path = os.path.join(args.out_dir, "PREREG_g3.md")
    if not os.path.exists(prereg_path):
        sg3.write_prereg_g3(prereg_path)
        log("PREREG_g3 frozen -> %s" % prereg_path)
    sg3.check_prereg_g3_roundtrip(prereg_path)            # AssertionError names drift
    log("PREREG_g3 round-trip OK: %s" % prereg_path)

    for name, why in (("PREREG_g2.md", "the frozen 13-id source"),
                      ("results_g1.json", "the station-A E-full baseline"),
                      ("results_g2.json", "the station-B E-full baseline"),
                      # existence != coverage: an incomplete cache still fails only at the
                      # E-full raising-stub recompute, which stays the true guard.
                      ("cache_g2_sample.jsonl", "the frozen E-full continuous-baseline draws")):
        if not os.path.exists(os.path.join(args.out_dir, name)):
            raise RuntimeError("g3: %s not found under %s — %s must precede g3"
                               % (name, args.out_dir, why))
    with open(os.path.join(args.out_dir, "results_g1.json")) as f:
        results_g1 = json.load(f)
    with open(os.path.join(args.out_dir, "results_g2.json")) as f:
        results_g2 = json.load(f)
    # both baselines run in EVERY mode (smoke included) — they judge the frozen full-run
    # files, never the smoke scope; a drifted baseline raises before any GPU work.
    sg3.stationA_baseline_check(results_g1, prereg)       # ValueError names each mismatch
    b_baseline = _g3_stationB_baseline_check(results_g2, prereg)
    log("g3: preflight OK (E-full baselines: station A %s | station B %s)"
        % (json.dumps(prereg["baseline_stationA"], sort_keys=True),
           json.dumps(b_baseline, sort_keys=True)))

    # ---- 2) item sets: station A = g1's 22 items (ids + SC from results_g1, never
    # recomputed); station B = the 13 frozen decode items ---------------------------------
    g1_rows = (results_g1.get("per_item") or {})
    sc_idx_by_id = {str(r["item_id"]): r.get("sc_answer_idx")
                    for r in list(g1_rows.get("patient") or [])
                    + list(g1_rows.get("knowledge") or [])}
    stationA_ids = sorted(sc_idx_by_id)
    repro7 = [str(x) for x in prereg["repro7_ids"]]
    zero_cov = [str(x) for x in prereg["zero_cov_ctrl_ids"]]
    knowledge = [str(x) for x in prereg["knowledge_ctrl_ids"]]
    decode_ids = repro7 + zero_cov + knowledge
    if args.smoke:
        # SMOKE scoping (same code path; L1 <20 min): one repro patient + one knowledge
        # control feed BOTH stations. Smoke results are never canonical.
        smoke_duo = [repro7[0], knowledge[0]]
        stationA_ids = [i for i in smoke_duo if i in sc_idx_by_id]
        decode_ids = list(smoke_duo)
        log("g3[smoke]: scoped to %s (both stations, same code path)" % smoke_duo)

    proc_A = _shard_slice(stationA_ids, args.shard) if run_A else []
    proc_B = _shard_slice(decode_ids, args.shard) if run_B else []
    log("g3: this pass — station A items=%d | station B items=%d (shard=%s)"
        % (len(proc_A), len(proc_B), args.shard))

    excluded = []
    spaces = _g_build_spaces(items_by_id, list(proc_A) + list(proc_B), excluded)
    for e in excluded:
        log("g3: EXCLUDED %s (%s)" % (e["item_id"], e["reason"]))
    # G-C scale assertion on the decode PATIENT spaces (the g2 preflight convention).
    for iid in [i for i in (repro7 + zero_cov) if i in spaces]:
        if spaces[iid].path_count > G_PATH_COUNT_MAX:
            raise AssertionError("g3 G-C violation: item %s path_count=%d > %d"
                                 % (iid, spaces[iid].path_count, G_PATH_COUNT_MAX))

    sample_M = args.g3_sample_m if args.g3_sample_m is not None else prereg["sample_M"]
    shard_i = _shard_index(args.shard)

    # ---- 3) station A — TF re-scoring under the ablated templates (cheap, first) ---------
    score_seconds = 0.0
    score_n_scored = 0
    rows_by_arm = {arm: [] for arm in arms}
    if run_A and proc_A:
        banner("g3-A", "TF re-scoring under ablated templates (%d item(s) x arms %s)"
               % (len(proc_A), arms))
        score_cache_path = _cache_path(args.out_dir, "g3_scores", shard_i)
        if scorer is None:
            scorer = _g_default_scorer(args, cache_path=score_cache_path, stage="g3")
        _g3_prime_score_cache(scorer, args.out_dir)       # resume: prior g3 caches full-hit
        for iid in _progress(proc_A, total=len(proc_A), desc="g3-A"):
            it, space = items_by_id.get(iid), spaces.get(iid)
            if it is None or space is None:
                continue                                  # already on the exclusion roster
            for arm in arms:
                t0 = time.time()
                try:
                    tr = scorer.score_transitions(it, space, template=arm,
                                                  stage="g3a_" + arm)
                except gsd_score.ScorerOOM as exc:
                    excluded.append({"item_id": iid, "arm": arm,
                                     "reason": "ScorerOOM in station-A scoring: %s" % exc})
                    log("  [g3-A] %-30s arm=%-8s OOM-SKIPPED (%s)" % (iid, arm, exc))
                    continue
                dt = time.time() - t0
                score_seconds += dt
                score_n_scored += tr["n_scored"]
                if gen_only:                              # shard pass: cache only, no analysis
                    log("  [g3-A] %-30s arm=%-8s scored=%d sec=%.2f (gen-only)"
                        % (iid, arm, tr["n_scored"], dt))
                    continue
                row = sg.g1_item_analysis(it, space, tr["lse"], sc_idx_by_id.get(iid))
                rows_by_arm[arm].append(row)
                log("  [g3-A] %-30s arm=%-8s scored=%d sec=%.2f map=%r map_fixed=%s "
                    "oracle_fixed=%s"
                    % (iid, arm, tr["n_scored"], dt, row["map"]["answer_idx"],
                       row["map"]["fixed"], row["oracle"]["fixed"]))

    # ---- 4) station B — per-arm node-conditional sampling (the ONLY GPU-heavy station) ----
    sample_seconds = 0.0
    sample_n_sampled = 0
    samples_by_arm = {arm: {} for arm in arms}
    if run_B and proc_B:
        banner("g3-B", "GSD node-conditional sampling under ablated templates (M=%d)"
               % sample_M)
        if shard_i is None:                               # consolidation: merge shard caches
            for arm in arms:
                _g3_consolidate_sample_cache(
                    args.out_dir, arm, _g3_sample_cache_path(args.out_dir, arm, None))
        if emitter is None:                               # lazy: no model load if full-hit
            emitter = _LazyEmitter(lambda: sc.HFEmitter(
                args.main_model, device=args.device, batch=args.batch,
                max_new_tokens=prereg["sample_max_new_tokens"]))
        for iid in _progress(proc_B, total=len(proc_B), desc="g3-sample"):
            it, space = items_by_id.get(iid), spaces.get(iid)
            if it is None or space is None:
                continue
            for arm in arms:
                t0 = time.time()
                try:
                    sn = gsd_sample.sample_node_freqs(
                        it, space, emitter, M=sample_M,
                        base_seed=prereg["sample_base_seed"], min_succ=2,
                        cache_path=_g3_sample_cache_path(args.out_dir, arm, shard_i),
                        stage="g3_" + arm,
                        max_new_tokens=prereg["sample_max_new_tokens"], batch=args.batch,
                        template=arm, seed_tag=prereg["seed_tag"])
                except sc.EmitterOOM as exc:              # HFEmitter already halved once
                    excluded.append({"item_id": iid, "arm": arm,
                                     "reason": "EmitterOOM in station-B sampling: %s" % exc})
                    log("  [g3-B] %-30s arm=%-8s OOM-SKIPPED (%s)" % (iid, arm, exc))
                    continue
                dt = time.time() - t0
                sample_seconds += dt
                sample_n_sampled += sn["n_scored"]
                samples_by_arm[arm][iid] = sn
                n_on = sum(sum(nd["freqs"].values()) for nd in sn["nodes"].values())
                n_tot = sum(nd["n_total"] for nd in sn["nodes"].values())
                log("  [g3-B] %-30s arm=%-8s nodes=%d draws=%d sampled=%d parse_ok=%.3f "
                    "on_manifold=%d sec=%.2f"
                    % (iid, arm, len(sn["nodes"]), n_tot, sn["n_scored"],
                       sn["item_parse_ok_rate"], n_on, dt))

    # ---- 5) gen-only shard pass ends here: NO analysis / results / figures ---------------
    if gen_only:
        banner("g3-shard", "gen-only pass complete (analysis deferred to consolidation)")
        log("g3[shard %s]: station A fresh scores=%d | station B fresh draws=%d — no "
            "analysis, no results write (the un-sharded pass consolidates + assembles)"
            % (args.shard, score_n_scored, sample_n_sampled))
        return {"stage": "g3", "shard_pass": True, "shard": args.shard,
                "station": station,
                "run_stats": {"sample_n_scored": sample_n_sampled,
                              "score_n_scored": score_n_scored}}

    # ---- 6) station A analyzers (survival verdict; decision arm = noevent) ---------------
    stationA = {"skipped": not run_A, "per_arm": {}, "verdict": None}
    if run_A:
        stationA["per_arm"] = {arm: sg3.stationA_arm(rows_by_arm[arm], prereg)
                               for arm in arms}
        stationA["verdict"] = sg3.stationA_verdict(stationA["per_arm"], prereg)
        va = stationA["verdict"]
        log("g3-A: survival=%d/10 label=%s survived=%s lost=%s anomaly=%s"
            % (va["survival"], va["label"], va["survived_ids"], va["lost_ids"],
               va["knowledge_anomaly_flag"]))

    # ---- 7) station B analyzers: per-arm arm2 + ordered verdict + continuous readout -----
    stationB = {"skipped": not run_B, "per_arm": {}, "verdict": None}
    continuous = {"skipped": not run_B, "per_arm": {}, "full_baseline": None}
    efull_n_scored = 0
    if run_B:
        banner("g3-B-decode", "frequency-decode reproduction per arm + continuous readout")
        knowledge_set = set(knowledge)

        def pool_of(iid):
            return "knowledge" if str(iid) in knowledge_set else "patient"

        arm2_by_arm = {}
        for arm in arms:
            iss = []
            for iid in decode_ids:
                it, space = items_by_id.get(iid), spaces.get(iid)
                sn = samples_by_arm[arm].get(iid)
                if it is None or space is None or sn is None:
                    continue                              # absent / malformed / not sampled
                iss.append({"item_id": iid, "item": it, "space": space,
                            "sample_nodes": sn})
            arm2 = sg3.stationB_arm(iss, prereg)
            arm2_by_arm[arm] = arm2
            log("g3-B[%s]: n_repro=%d/%s zero_cov=%d knowledge=%d excluded=%d"
                % (arm, arm2["n_repro"], arm2["n_repro_denominator"],
                   sum(1 for f in arm2["zero_cov"].values() if f),
                   sum(1 for f in arm2["knowledge"].values() if f),
                   len(arm2["excluded_inconclusive"])))
        stationB["per_arm"] = {arm: sg2._json_safe_arm2(a2)
                               for arm, a2 in arm2_by_arm.items()}
        stationB["verdict"] = sg3.stationB_verdict(arm2_by_arm, prereg)
        log("g3-B: label=%s (%s)" % (stationB["verdict"]["label"],
                                     stationB["verdict"]["reason"]))

        for arm in arms:
            continuous["per_arm"][arm] = sg3.continuous_readout(
                {iid: sn["nodes"] for iid, sn in samples_by_arm[arm].items()},
                spaces, pool_of)

        # E-full continuous baseline: RECOMPUTED from cache_g2_sample.jsonl at the frozen
        # g2 coordinates (M=128 / stage='g2' / seed_tag='g2sample' / template='main') — a
        # full cache hit by construction; the raising stub + the n_scored hard-assert make
        # any fresh draw fail loudly instead of silently re-estimating the baseline.
        g2_cache_path = os.path.join(args.out_dir, "cache_g2_sample.jsonl")
        stub = _G3RaisingEmitter()
        bl_nodes_by_item = {}
        for iid in decode_ids:
            it, space = items_by_id.get(iid), spaces.get(iid)
            if it is None or space is None:
                continue
            bl = gsd_sample.sample_node_freqs(
                it, space, stub, M=int(sg2.PREREG_G2["sample_M"]),
                base_seed=sg2.PREREG_G2["sample_base_seed"], min_succ=2,
                cache_path=g2_cache_path, stage="g2",
                max_new_tokens=sg2.PREREG_G2["sample_max_new_tokens"],
                template="main", seed_tag="g2sample")
            if bl["n_scored"] != 0:                       # unreachable past the stub — belt+braces
                raise AssertionError("g3 E-full baseline drew %d fresh sample(s) for %s"
                                     % (bl["n_scored"], iid))
            efull_n_scored += bl["n_scored"]
            bl_nodes_by_item[iid] = bl["nodes"]
        continuous["full_baseline"] = sg3.continuous_readout(bl_nodes_by_item, spaces,
                                                             pool_of)
        log("g3-B: E-full continuous baseline recomputed from cache_g2_sample.jsonl "
            "(%d node row(s), 0 fresh draws)"
            % len(continuous["full_baseline"]["per_node"]))

    # ---- 8) assemble results_g3 + VERDICT_g3.md + figures (NaN/Inf-guarded write) --------
    banner("g3-assemble", "two-station verdicts + results_g3.json")
    scope = {
        "smoke": bool(args.smoke),
        "shard": args.shard,
        "station": station,
        "stationA_ids": list(proc_A),
        "decode_ids": list(decode_ids),
        "decode_in_arena": sorted(i for i in decode_ids if i in items_by_id),
        "sampled_item_ids": {arm: sorted(samples_by_arm[arm]) for arm in arms},
        "dropped_items": sorted({e["item_id"] for e in excluded}),
        "sample_M": sample_M,
        "baseline_sample_M": int(sg2.PREREG_G2["sample_M"]),
    }
    eff = _g2_efficiency(sample_seconds, sample_n_sampled, score_seconds, score_n_scored,
                         len(set(proc_A) | set(proc_B)))
    config = {
        "stage": "g3",
        "smoke": bool(args.smoke),
        "shard": args.shard,
        "station": station,
        "arms": arms,
        "sample_M": sample_M,
        "sample_max_new_tokens": prereg["sample_max_new_tokens"],
        "sample_base_seed": prereg["sample_base_seed"],
        "seed_tag": prereg["seed_tag"],
        "epsilon_backoff": prereg["epsilon_backoff"],
        "epsilon_sensitivity": list(prereg["epsilon_sensitivity"]),
        "parse_ok_min": prereg["parse_ok_min"],
        "node_coverage_min": prereg["node_coverage_min"],
        "template_shas": dict(prereg["template_shas"]),
        "models": {"main": args.main_model},
        "gsd_batch": args.gsd_batch,
        "emitter_batch": args.batch,
        # the preflight-asserted E-full numbers (fig_fix_matrix's baseline row)
        "efull_baseline": {
            "map_fixed": prereg["baseline_stationA"]["map_fixed"],
            "oracle_fixed": prereg["baseline_stationA"]["oracle_fixed"],
            "n_repro": b_baseline["n_repro"],
            "zero_cov": b_baseline["zero_cov"],
            "knowledge": b_baseline["knowledge"],
        },
    }
    run_stats = {
        "sample_n_scored": sample_n_sampled,
        "score_n_scored": score_n_scored,
        "efull_baseline_n_scored": efull_n_scored,        # hard-asserted 0 above
        "stationA_label": stationA["verdict"]["label"] if stationA["verdict"] else None,
        "stationB_label": stationB["verdict"]["label"] if stationB["verdict"] else None,
    }
    results = {
        "stage": "g3",
        "config": config,
        "prereg": json.loads(json.dumps(prereg)),         # frozen snapshot, json-normalized
        "stationA": stationA,
        "stationB": stationB,
        "continuous": continuous,
        "scope": scope,
        "excluded": excluded,
        "efficiency": eff,
        "run_stats": run_stats,
    }

    log("---- g3 summary ----")
    log("  %-24s %s" % ("stationA_label", run_stats["stationA_label"]))
    log("  %-24s %s" % ("stationB_label", run_stats["stationB_label"]))
    log("  %-24s %s" % ("excluded", len(excluded)))
    log("g3: efficiency — scoring %.4f GPU-h (%d fresh forward(s)) | sampling %.4f GPU-h "
        "(%d fresh draw(s)) | total %.4f GPU-h"
        % (eff["scoring"]["gpu_hours"], eff["scoring"]["n_scored"],
           eff["sampling"]["gpu_hours"], eff["sampling"]["n_sampled"],
           eff["total_gpu_hours"]))

    res_dir, fig_dir = _g3_artifact_paths(args)
    write_results(os.path.join(res_dir, "results_g3.json"), results)
    _g3_write_verdict_md(os.path.join(res_dir, "VERDICT_g3.md"), results)
    log("g3: wrote VERDICT_g3.md (station A=%s | station B=%s)"
        % (run_stats["stationA_label"], run_stats["stationB_label"]))
    _g3_write_figures(fig_dir, results)
    return results


def _g3_write_verdict_md(path, results):
    """Write the plain-language ``VERDICT_g3.md`` from the assembled results_g3: BOTH
    station verdicts side by side + the arm tables + the continuous one-hot readout + the
    frozen interpretation boundary (design §1) + the efficiency split. Mirrors the
    ``_g2_write_verdict_md`` structure; a skipped station renders n/a."""
    a, b, cont = results["stationA"], results["stationB"], results["continuous"]
    prereg = results["prereg"]
    base = results["config"]["efull_baseline"]
    eff = results["efficiency"]

    def _n(x):
        return "n/a" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))

    md = [
        "# MuSR-cant event-line ablation — VERDICT (stage g3)",
        "",
        "Attributes the EXACT-GSD fixes to the belief scaffold (`BELIEF[t-1]` + global "
        "MAP) vs the per-step event-line fact feed, by deleting/neutralising the event "
        "line and re-reading two stations (A: λ=0 TF-MAP headline audit; B: the g2 "
        "sampling probe). The two stations answer two DIFFERENT questions and are never "
        "merged into one label. Design: `plans/2026-07-09-eventline-ablation.md` + "
        "`…-design.md` §3. Frozen: `outputs/PREREG_g3.md`.",
        "",
        "## Station A — headline attribution (decision arm = noevent): **%s**"
        % (a["verdict"]["label"] if a["verdict"] else "n/a (station skipped)"),
        "",
    ]
    if a["verdict"]:
        va = a["verdict"]
        md += [
            "- survival = |map-fixed(noevent) ∩ original 10 λ=0 fixes| = **%d/10** "
            "(scaffold_carries >= %d / facts_carry <= %d / else mixed)."
            % (va["survival"], prereg["stationA_scaffold_min"],
               prereg["stationA_facts_max"]),
            "- survived: %s." % (va["survived_ids"] or "none"),
            "- lost: %s." % (va["lost_ids"] or "none"),
            "- oracle ceiling by arm (of 19 patients; E-full was %d): %s."
            % (base["oracle_fixed"], va["oracle_ceiling_by_arm"]),
            "- knowledge anomaly flag: %s (map-fixed knowledge ids by arm: %s; expected "
            "0/3 everywhere, the flag never changes the label)."
            % (va["knowledge_anomaly_flag"], va["knowledge_anomaly_ids"]),
            "- per-arm patient counts (E-full baseline map %d/19, oracle %d/19):"
            % (base["map_fixed"], base["oracle_fixed"]),
        ]
        for arm, res_arm in sorted(a["per_arm"].items()):
            md.append("  - `%s`: map-fixed %d/19, oracle-fixed %d/19."
                      % (arm, res_arm["n_map_fixed"], res_arm["n_oracle_fixed"]))
    md += [
        "",
        "## Station B — probe rescue (decision arm = noevent, ordered gates): **%s**"
        % (b["verdict"]["label"] if b["verdict"] else "n/a (station skipped)"),
        "",
    ]
    if b["verdict"]:
        md += [b["verdict"]["reason"], "",
               "- E-full baseline (results_g2 arm2, asserted at preflight): n_repro %d/7, "
               "zero_cov %d/3, knowledge %d/3."
               % (base["n_repro"], base["zero_cov"], base["knowledge"])]
        for arm, ha in sorted(b["verdict"]["per_arm"].items()):
            md.append("- `%s`: n_repro %s/%s, zero_cov %s/3, knowledge %s/3, fail-closed "
                      "excluded %s (%s); ε sensitivity %s."
                      % (arm, _n(ha["n_repro"]), _n(ha["n_repro_denominator"]),
                         _n(ha["n_zero_cov_reproduced"]), _n(ha["knowledge_repro"]),
                         _n(ha["n_excluded"]), ha["excluded_ids"] or "none",
                         b["per_arm"][arm].get("eps_sensitivity")))
        md.append("- zero_cov reproduction is a REPORTED finding, never an assertion "
                  "(the g2 correction).")
    md += [
        "",
        "## Continuous readout — strict one-hot share (pool × layer kind; E-full "
        "recomputed from cache_g2_sample.jsonl at ZERO GPU)",
        "",
    ]
    if cont.get("per_arm") or cont.get("full_baseline"):
        summaries = {arm: c["summary"] for arm, c in (cont.get("per_arm") or {}).items()}
        if cont.get("full_baseline"):
            summaries["full"] = cont["full_baseline"]["summary"]
        cells = sorted({(pool, kind) for s in summaries.values()
                        for pool, kinds in s.items() for kind in kinds})
        for pool, kind in cells:
            parts = []
            for arm in sorted(summaries):
                cell = summaries[arm].get(pool, {}).get(kind)
                parts.append("%s=%s (n=%s)"
                             % (arm, _n(cell["onehot_share"]) if cell else "n/a",
                                _n(cell["n_nodes"]) if cell else "n/a"))
            md.append("- %s / %s: %s." % (pool, kind, "; ".join(parts)))
    else:
        md.append("- n/a (station B skipped).")
    md += [
        "",
        "## Interpretation boundary (frozen; design §1)",
        "",
        "- `s_prev` itself carries the CONSEQUENCES of every earlier step's facts (the "
        "enumerated node IS a state); the ablation strips ONLY the current step's fact "
        "supply. The localisation prediction: a knowledge control's sampling quality "
        "should diverge AT the layer of its missing fact (deep-dive figure), not "
        "uniformly.",
        "- Smoke-scope results are never canonical (`config.smoke` marks them).",
        "",
        "## Efficiency",
        "",
        "- station A scoring: %.4f GPU-h over %d fresh forward(s); station B sampling: "
        "%.4f GPU-h over %d fresh draw(s); total %.4f GPU-h."
        % (eff["scoring"]["gpu_hours"], eff["scoring"]["n_scored"],
           eff["sampling"]["gpu_hours"], eff["sampling"]["n_sampled"],
           eff["total_gpu_hours"]),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def _g3_write_figures(fig_dir, results):
    """The g3 figure set -> ``<fig_dir>/g3_*.png`` (best-effort, the g1/g2 convention: a
    matplotlib failure logs loudly and never voids the written results). The fix-matrix
    E-full row comes from the preflight-asserted baselines; skipped stations render n/a
    cells and drop their dedicated figures."""
    try:
        os.makedirs(fig_dir, exist_ok=True)
        prereg = results["prereg"]
        base = results["config"]["efull_baseline"]
        matrix_rows = [("E-full (g1/g2)", dict(base))]
        for arm in prereg["arms"]:
            vals = {}
            pa = results["stationA"]["per_arm"].get(arm)
            vals["map_fixed"] = pa["n_map_fixed"] if pa else None
            vals["oracle_fixed"] = pa["n_oracle_fixed"] if pa else None
            pb = results["stationB"]["per_arm"].get(arm)
            vals["n_repro"] = pb["n_repro"] if pb else None
            vals["zero_cov"] = (sum(1 for f in pb["zero_cov"].values() if f)
                                if pb else None)
            vals["knowledge"] = (sum(1 for f in pb["knowledge"].values() if f)
                                 if pb else None)
            matrix_rows.append(("E-%s" % arm, vals))
        written = [sg3.fig_fix_matrix(matrix_rows,
                                      os.path.join(fig_dir, "g3_fix_matrix.png"))]
        cont = results["continuous"]
        if cont.get("per_arm"):
            summary_by_arm = {arm: c["summary"] for arm, c in cont["per_arm"].items()}
            rows_by_arm = {arm: c["per_node"] for arm, c in cont["per_arm"].items()}
            if cont.get("full_baseline"):
                summary_by_arm["full"] = cont["full_baseline"]["summary"]
                rows_by_arm["full"] = cont["full_baseline"]["per_node"]
            written.append(sg3.fig_onehot_share(
                summary_by_arm, os.path.join(fig_dir, "g3_onehot_share.png")))
            dd_ids = [prereg["repro7_ids"][0], prereg["knowledge_ctrl_ids"][0]]
            written.append(sg3.fig_entropy_deepdive(
                rows_by_arm, dd_ids, os.path.join(fig_dir, "g3_entropy_deepdive.png")))
        if results["stationA"].get("verdict"):
            written.append(sg3.fig_survival_waterfall(
                results["stationA"]["verdict"],
                os.path.join(fig_dir, "g3_survival_waterfall.png")))
        log("g3: wrote %d figure(s) to %s" % (sum(1 for w in written if w), fig_dir))
    except Exception as exc:
        log("g3: figure generation error (non-fatal): %s" % exc)


# ======================================================================================
# STAGE: g4 — step accuracy + self-parsed event lines (two stations: Station C gold-prefix accuracy via pure cache replay /
# Station D zero-oracle extraction -> injected scoring)
# ======================================================================================
# The g4 shell wires the pure Subtask 1-3 components (``gsd_extract`` + ``stages_g4``)
# around the same two heavy sites as g2/g3, both dependency-injected: the TF ``scorer``
# (station D scoring) and the ``emitter`` (station D extraction — the ONLY generation).
# Station C is deliberately GPU-free: it re-reads the frozen g2/g3 sample caches at their
# exact frozen coordinates behind a raising stub; any fresh draw is a provenance violation
# and fails loudly.

# Station D's noevent-planned layers are scored at the g3 station-A noevent coordinates
# (template='noevent', stage='g3a_noevent' — see _stage_g3's ``stage="g3a_" + arm``) ON
# PURPOSE: the noevent template ignores ``space.moves``, so the real-space prefix g3 built
# is byte-identical to what g4 would build, and every such edge is a FULL cache hit from
# cache_g3_scores.jsonl once the scorer is primed (zero fresh forwards for those layers).
G4_NOEVENT_STAGE = "g3a_noevent"


class _G4RaisingEmitter:
    """Station C is a PURE REPLAY of one frozen sample cache — any attempted fresh draw
    would silently re-estimate a frozen readout at the wrong provenance, so it fails
    loudly instead (the ``_G3RaisingEmitter`` idiom, arm-aware message)."""

    def __init__(self, arm, coords):
        self.arm = arm
        self.coords = dict(coords)

    def generate(self, prompts, seeds, *, greedy=False, max_new_tokens=None):
        raise AssertionError(
            "g4 station C attempted a FRESH draw on arm %r — %s must fully cover the "
            "decode items' branching nodes at M=%d (stage=%r, seed_tag=%r, template=%r); "
            "refusing to re-sample a frozen readout"
            % (self.arm, self.coords["cache"], self.coords["M"], self.coords["stage"],
               self.coords["seed_tag"], self.coords["template"]))


def _g4_extract_sha_preflight():
    """Reviewer-mandated hard preflight, INDEPENDENT of the stages_g4 import-time assert:
    the frozen ``PREREG_G4['extract_template_sha']`` must exist and equal the sha256 of
    the LIVE ``gsd_extract.EXTRACT_TEMPLATE``, recomputed HERE from the template bytes
    (never trusting the exported constant) — a template edit or a stale freeze can never
    reach the extraction pass."""
    if "extract_template_sha" not in sg4.PREREG_G4:
        raise AssertionError(
            "g4 preflight: PREREG_G4 lacks 'extract_template_sha' — the Subtask-3 freeze "
            "of the gsd_extract template never landed; refusing to run station D against "
            "an unfrozen extraction template")
    live = _sha256_text(gsd_extract.EXTRACT_TEMPLATE)
    if sg4.PREREG_G4["extract_template_sha"] != live:
        raise AssertionError(
            "g4 preflight: extract_template_sha drift — frozen %s != sha256(live "
            "gsd_extract.EXTRACT_TEMPLATE) %s"
            % (sg4.PREREG_G4["extract_template_sha"], live))


def _g4_prereg_freeze(args):
    """PREREG_g4 freeze + round-trip (the g3 write-if-missing pattern) with ONE sanctioned
    in-place rewrite: an earlier partial/smoke pass may have frozen a STALE PREREG_g4.md
    that predates the documented Subtask-3 ``extract_template_sha`` key addition. When the
    ONLY diff is exactly that missing key, the file is rewritten through the same
    machinery and the rewrite is logged — the freeze point of record is the FULL run, and
    this file has only ever guarded partial passes. ANY other drift still hard-raises
    (the rewrite is never a general escape hatch)."""
    path = os.path.join(args.out_dir, "PREREG_g4.md")
    if not os.path.exists(path):
        sg4.write_prereg_g4(path)
        log("PREREG_g4 frozen -> %s" % path)
    else:
        try:
            sg4.check_prereg_g4_roundtrip(path)
        except AssertionError:
            loaded = sg4.load_prereg_g4(path)
            code = json.loads(json.dumps(sg4.PREREG_G4))
            diff = sg4._prereg_g4_diff(loaded, code)
            if (set(diff) == {"extract_template_sha"}
                    and diff["extract_template_sha"]["loaded"] == "<MISSING>"):
                sg4.write_prereg_g4(path)
                log("PREREG_g4 was STALE (pre-Subtask-3 key set: extract_template_sha "
                    "missing, everything else byte-equal) — rewritten in place -> %s "
                    "(the freeze point of record is the full run)" % path)
            else:
                raise
    sg4.check_prereg_g4_roundtrip(path)            # AssertionError names drift
    log("PREREG_g4 round-trip OK: %s" % path)
    return path


def _g4_g3_noevent_baseline_check(results_g3, prereg):
    """Preflight: assert the g3 station-A NOEVENT baseline recomputed from a loaded
    results_g3 dict equals ``prereg['baseline_g3_noevent']`` byte-for-byte — the verdict
    survival AND the map-fixed patient id set (anti-drift: station D's survival readout
    only means something against the frozen g3 numbers). Returns the observed dict on a
    match; raises RuntimeError naming every mismatched field otherwise."""
    baseline = prereg["baseline_g3_noevent"]
    station_a = (results_g3 or {}).get("stationA") or {}
    verdict = station_a.get("verdict") or {}
    noevent = (station_a.get("per_arm") or {}).get("noevent") or {}
    observed = {
        "survival": verdict.get("survival"),
        "map_fixed_ids": sorted(str(p["item_id"]) for p in (noevent.get("per_item") or ())
                                if p.get("bucket") == "patient" and p.get("map_fixed")),
    }
    mismatches = {k: {"prereg": baseline[k], "results_g3": observed[k]}
                  for k in sorted(baseline) if baseline[k] != observed[k]}
    if mismatches:
        raise RuntimeError(
            "g4 preflight: results_g3 noevent baseline drifted from PREREG_G4 "
            "baseline_g3_noevent: %s" % json.dumps(mismatches, sort_keys=True))
    return observed


def _g4_prime_score_cache(scorer, out_dir):
    """Prime a TFScorer's in-memory transition cache with EVERY ``cache_g4_scores*.jsonl``
    (resume: a prior g4 pass re-scores nothing) AND ``cache_g3_scores*.jsonl`` (the
    station-D noevent-layer reuse: g3 already scored every 22-item edge at the
    'g3a_noevent' coordinates, so call N never forwards). g1/g2 caches are NOT primed:
    their records live under other stage strings and can never hit a g4 key. A scorer
    without a ``_cache`` dict (a bare stub) is left untouched (the g-line convention)."""
    cache = getattr(scorer, "_cache", None)
    if not isinstance(cache, dict):
        return
    n0 = len(cache)
    for pattern in ("cache_g4_scores*.jsonl", "cache_g3_scores*.jsonl"):
        for fp in sorted(glob.glob(os.path.join(out_dir, pattern))):
            for k, v in gsd_score._load_score_cache(fp).items():
                cache.setdefault(k, v)
    if len(cache) > n0:
        log("g4: primed %d transition score(s) from g4/g3 score caches" % (len(cache) - n0))


def _g4_merge_lse(plan, lse_main, lse_noevent):
    """Assemble station D's per-layer-template edge dict (the ``stationD_prefix_plan``
    wiring note made concrete): the root row rides from whichever call ran (the root
    prefix/cache key are template-independent — a path constant), every layer-t edge
    comes from the 'main' call iff ``plan[t] == 'main'`` else from the 'noevent' call.

    SOUNDNESS: an lse group is the log-softmax over one (t, canon_prev) candidate set
    WITHIN one call; both calls share IDENTICAL candidate sets (``_SpaceView`` delegates
    ``trans``/``layers`` to the real space), so splicing whole layers between calls never
    mixes a normalization group. A layer planned for a call that never ran raises
    (fail-loud, never a silent zero)."""
    by_template = {"main": lse_main, "noevent": lse_noevent}
    base = lse_main if lse_main is not None else lse_noevent
    if base is None:
        raise ValueError("_g4_merge_lse: neither template call was scored")
    merged = {}
    for key, lp in base.items():
        t = key[0]
        if t == 0:                                    # the root path constant
            merged[key] = lp
            continue
        src = by_template.get(plan.get(t))
        if src is None:
            raise ValueError(
                "_g4_merge_lse: layer %r planned %r but that call is missing (plan keys "
                "%s)" % (t, plan.get(t), sorted(plan)))
        merged[key] = src[key]
    return merged


def _g4_efficiency(extract_seconds, extract_n, score_seconds, score_n, n_items):
    """g4 efficiency block: the ONLY GPU work is station D, split extraction (fresh
    greedy generations) vs TF scoring (fresh forwards) — station C is a zero-GPU cache
    replay by construction (hard-asserted upstream). Reported separately (design §7)."""
    total = extract_seconds + score_seconds
    return {
        "n_items": n_items,
        "extraction": {"gpu_seconds": extract_seconds,
                       "gpu_hours": extract_seconds / 3600.0,
                       "n_generated": extract_n},
        "scoring": {"gpu_seconds": score_seconds, "gpu_hours": score_seconds / 3600.0,
                    "n_scored": score_n},
        "total_gpu_hours": total / 3600.0,
    }


def _g4_artifact_paths(args):
    """``(results_dir, fig_dir)`` for this pass — the g3 SMOKE-CLOBBER GUARD verbatim:
    a ``--smoke`` pass that finds a REAL (non-smoke) results_g4.json already in out_dir
    redirects ALL its g4 artifacts (results / VERDICT / figures) into
    ``<out_dir>/backup_smoke_g4/``. Caches are content-keyed and shared safely;
    PREREG_g4.md is byte-idempotent."""
    res_dir = args.out_dir
    if args.smoke:
        path = os.path.join(args.out_dir, "results_g4.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    prior_smoke = bool((json.load(f).get("config") or {}).get("smoke"))
            except Exception:
                prior_smoke = False              # unreadable -> treat as canonical (safe)
            if not prior_smoke:
                res_dir = os.path.join(args.out_dir, "backup_smoke_g4")
                os.makedirs(res_dir, exist_ok=True)
                log("g4[smoke]: existing results_g4.json is a FULL-run artifact — smoke "
                    "results/VERDICT/figures redirected to %s (canonical results never "
                    "clobbered)" % res_dir)
    return res_dir, os.path.join(res_dir, "figures")


def stage_g4(args, scorer=None, emitter=None):
    """The g4 (step accuracy + self-parsed event lines) shell. Both heavy sites are
    dependency-injected exactly like g2/g3: the station-D TF ``scorer`` (None -> the real
    bf16 model, built only when station D runs) and the station-D EXTRACTION ``emitter``
    (None -> a ``_LazyEmitter`` that never loads the model on a full cache hit). Station
    C needs neither. Log lines tee to ``outputs/g4.log`` (the g-line convention)."""
    global _LOG_FH
    fh = _g_open_log(args, "g4")
    _LOG_FH = fh
    try:
        return _stage_g4(args, scorer, emitter)
    finally:
        _LOG_FH = None
        fh.close()


def _stage_g4(args, scorer, emitter):
    prereg = sg4.PREREG_G4
    if args.shard is not None:
        raise RuntimeError("g4 takes no --shard (<= 0.5 GPU-h on a single card; the "
                           "station-C replay and the 22-item station D never shard)")
    station = args.g4_station
    run_C = station in ("C", "all")
    run_D = station in ("D", "all")
    banner("g4", "smoke=%s station=%s" % (args.smoke, station))
    items = load_items(args.data_dir, subtasks=PHASE_B_ARENA)
    if not items:
        raise RuntimeError("g4: no object_placements items under %s" % args.data_dir)
    items_by_id = _items_by_id(items)

    # ---- 1) preflight (BEFORE any model load): sha hard-assert + freeze + files + baselines --
    _g4_extract_sha_preflight()
    _g4_prereg_freeze(args)

    for name, why in (
            ("results_g1.json", "the E-full baseline + the SC readout source"),
            ("results_g3.json", "the g3 noevent baseline (station-D comparison point)"),
            # existence != coverage: an incomplete sample cache still fails only at the
            # station-C raising-stub replay, which stays the true guard.
            ("cache_g2_sample.jsonl", "the frozen E-full draws (station-C arm 'full')"),
            ("cache_g3_sample_anchor.jsonl", "the frozen g3 anchor draws (station C)"),
            ("cache_g3_sample_noevent.jsonl", "the frozen g3 noevent draws (station C)"),
            ("cache_g3_scores.jsonl", "the g3a_noevent TF scores station D reuses")):
        if not os.path.exists(os.path.join(args.out_dir, name)):
            raise RuntimeError("g4: %s not found under %s — %s must precede g4"
                               % (name, args.out_dir, why))
    with open(os.path.join(args.out_dir, "results_g1.json")) as f:
        results_g1 = json.load(f)
    with open(os.path.join(args.out_dir, "results_g3.json")) as f:
        results_g3 = json.load(f)
    # score-cache keys carry no model identity, and station D's noevent call is DESIGNED
    # as a 100% hit from cache_g3_scores.jsonl — a different --main-model would silently
    # splice old-model lse into the merged dict. Tolerant form: absent config skips.
    g3_model = ((results_g3.get("config") or {}).get("models") or {}).get("main")
    if g3_model not in (None, args.main_model):
        raise RuntimeError("g4: --main-model %r != results_g3 model %r — the g3a_noevent "
                           "score-cache reuse (station D call N) requires the same model"
                           % (args.main_model, g3_model))
    # both baselines run in EVERY mode (smoke included) — they judge the frozen full-run
    # files, never the smoke scope; a drifted baseline raises before any GPU work.
    sg3.stationA_baseline_check(results_g1, {"baseline_stationA": prereg["baseline_g1"]})
    g3_baseline = _g4_g3_noevent_baseline_check(results_g3, prereg)
    log("g4: preflight OK (g1 baseline %s | g3 noevent baseline %s)"
        % (json.dumps(prereg["baseline_g1"], sort_keys=True),
           json.dumps(g3_baseline, sort_keys=True)))

    # ---- 2) item sets: station C = the 13 frozen decode items; station D = g1's 22 items
    # (ids + SC from results_g1, never recomputed) ------------------------------------------
    g1_rows = (results_g1.get("per_item") or {})
    sc_idx_by_id = {str(r["item_id"]): r.get("sc_answer_idx")
                    for r in list(g1_rows.get("patient") or [])
                    + list(g1_rows.get("knowledge") or [])}
    knowledge3 = [str(x) for x in prereg["knowledge_ctrl_ids"]]
    decode_ids = [str(x) for x in prereg["decode13_ids"]]
    stationD_ids = [str(x) for x in prereg["patients19_ids"]] + knowledge3
    if args.smoke:
        # SMOKE scoping (same code path; L1 <15 min): one repro patient + one knowledge
        # control feed BOTH stations — station C stays replay-only at the frozen M/coords,
        # just restricted to the duo's nodes. Smoke results are never canonical.
        smoke_duo = [decode_ids[0], knowledge3[0]]    # repro7[0] + knowledge3[0]
        decode_ids = list(smoke_duo)
        stationD_ids = list(smoke_duo)
        log("g4[smoke]: scoped to %s (both stations, same code path)" % smoke_duo)

    proc_C = list(decode_ids) if run_C else []
    proc_D = list(stationD_ids) if run_D else []
    log("g4: this pass — station C items=%d | station D items=%d"
        % (len(proc_C), len(proc_D)))

    excluded = []
    spaces = _g_build_spaces(items_by_id, list(proc_C) + list(proc_D), excluded)
    for e in excluded:
        log("g4: EXCLUDED %s (%s)" % (e["item_id"], e["reason"]))
    # G-C scale assertion on the PATIENT spaces (the g2/g3 preflight convention).
    knowledge_set = set(knowledge3)
    for iid in sorted(spaces):
        if iid in knowledge_set:
            continue
        if spaces[iid].path_count > G_PATH_COUNT_MAX:
            raise AssertionError("g4 G-C violation: item %s path_count=%d > %d"
                                 % (iid, spaces[iid].path_count, G_PATH_COUNT_MAX))

    def pool_of(iid):
        return "knowledge" if str(iid) in knowledge_set else "patient"

    # ---- 3) station C — gold-prefix step accuracy (ZERO GPU: pure frozen-cache replay) ----
    stationC = {"skipped": not run_C, "per_arm": {}, "verdict": None}
    stationC_n_scored = 0
    if run_C and proc_C:
        banner("g4-C", "gold-prefix step accuracy — pure cache replay, 0 fresh draws "
                       "(arms %s)" % (prereg["stationC_arms"],))
        arm_results = {}
        for arm in prereg["stationC_arms"]:
            coords = prereg["stationC_cache_coords"][arm]
            # the frozen sampler coordinates the caches were drawn at (inert behind the
            # stub — every key must hit — but passed faithfully all the same)
            if arm == "full":
                base_seed = int(sg2.PREREG_G2["sample_base_seed"])
                max_new = int(sg2.PREREG_G2["sample_max_new_tokens"])
            else:
                base_seed = int(sg3.PREREG_G3["sample_base_seed"])
                max_new = int(sg3.PREREG_G3["sample_max_new_tokens"])
            stub = _G4RaisingEmitter(arm, coords)
            nodes_by_item, arm_spaces, arm_items = {}, {}, {}
            for iid in proc_C:
                it, space = items_by_id.get(iid), spaces.get(iid)
                if it is None or space is None:
                    continue                          # already on the exclusion roster
                sn = gsd_sample.sample_node_freqs(
                    it, space, stub, M=int(coords["M"]), base_seed=base_seed,
                    min_succ=2,
                    cache_path=os.path.join(args.out_dir, coords["cache"]),
                    stage=coords["stage"], max_new_tokens=max_new,
                    template=coords["template"], seed_tag=coords["seed_tag"])
                if sn["n_scored"] != 0:               # unreachable past the stub — belt+braces
                    raise AssertionError(
                        "g4-C: arm=%s item=%s drew %d fresh sample(s) — station C must "
                        "be a pure replay" % (arm, iid, sn["n_scored"]))
                stationC_n_scored += sn["n_scored"]
                nodes_by_item[iid] = sn["nodes"]
                arm_spaces[iid] = space
                arm_items[iid] = it
            res_arm = sg4.stationC_arm(nodes_by_item, arm_spaces, arm_items, pool_of)
            arm_results[arm] = res_arm
            log("  [g4-C] arm=%-8s items=%d gold_prefix_nodes=%d excluded=%d per_pool=%s"
                % (arm, len(nodes_by_item), len(res_arm["per_node"]),
                   len(res_arm["excluded"]),
                   json.dumps(res_arm["per_pool"], sort_keys=True)))
        stationC["per_arm"] = arm_results
        stationC["verdict"] = sg4.stationC_verdict(arm_results, prereg)
        vc = stationC["verdict"]
        log("g4-C: delta_acc=%.4f label=%s (0 fresh draws, hard-asserted)"
            % (vc["delta_acc"], vc["label"]))

    # ---- 4) station D — zero-oracle extraction (the ONLY generation), then injection
    # scoring under the per-layer template plan -------------------------------------------
    stationD = {"skipped": not run_D, "arm": None, "verdict": None,
                "quality_table": None, "per_item_quality": {}, "per_item_plan": {}}
    extract_seconds = 0.0
    extract_n = 0
    score_seconds = 0.0
    score_n = 0
    if run_D and proc_D:
        banner("g4-D", "zero-oracle event extraction (greedy, narrative-only, %d item(s))"
               % len(proc_D))
        extract_cache_path = os.path.join(args.out_dir, "cache_g4_extract.jsonl")
        if emitter is None:                           # lazy: no model load if full-hit
            emitter = _LazyEmitter(lambda: sc.HFEmitter(
                args.main_model, device=args.device, batch=args.batch,
                max_new_tokens=prereg["extract_max_new_tokens"]))
        ext_by_item = {}
        for iid in _progress(proc_D, total=len(proc_D), desc="g4-extract"):
            it, space = items_by_id.get(iid), spaces.get(iid)
            if it is None or space is None:
                continue                              # already on the exclusion roster
            t0 = time.time()
            try:
                ext = gsd_extract.extract_events(
                    it, emitter, seed=prereg["extract_seed"],
                    max_new_tokens=prereg["extract_max_new_tokens"],
                    cache_path=extract_cache_path)
            except sc.EmitterOOM as exc:              # HFEmitter already halved once
                excluded.append({"item_id": iid,
                                 "reason": "EmitterOOM in station-D extraction: %s" % exc})
                log("  [g4-extract] %-30s OOM-SKIPPED (%s)" % (iid, exc))
                continue
            dt = time.time() - t0
            extract_seconds += dt
            extract_n += ext["n_scored"]
            quality = gsd_extract.extraction_quality(ext["moves"], space)
            aligned = gsd_extract.align_moves(ext["moves"], space.T)
            plan = sg4.stationD_prefix_plan(aligned, space.T)
            ext_by_item[iid] = {"moves": ext["moves"], "aligned": aligned, "plan": plan,
                                "quality": quality}
            log("  [g4-extract] %-30s parsed=%d layer_match_rate=%.3f all_exact=%s "
                "n_scored=%d sec=%.2f"
                % (iid, len(ext["moves"]), quality["layer_match_rate"],
                   quality["all_exact"], ext["n_scored"], dt))

        banner("g4-D", "scoring under the injected event lines (main stage=%s | noevent "
                       "reuse stage=%s)" % (prereg["stationD_stage"], G4_NOEVENT_STAGE))
        score_cache_path = _cache_path(args.out_dir, "g4_scores")
        if scorer is None:
            scorer = _g_default_scorer(args, cache_path=score_cache_path, stage="g4")
        _g4_prime_score_cache(scorer, args.out_dir)   # resume + the g3 noevent reuse
        rows = []
        for iid in _progress(proc_D, total=len(proc_D), desc="g4-score"):
            info = ext_by_item.get(iid)
            it, space = items_by_id.get(iid), spaces.get(iid)
            if info is None or it is None or space is None:
                continue                              # extraction OOM / exclusion roster
            plan = info["plan"]
            injected = [mv for t in sorted(info["aligned"]) if info["aligned"][t]
                        for mv in info["aligned"][t]]
            view = sg4._SpaceView(space, injected)
            n_main = sum(1 for v in plan.values() if v == "main")
            n_noevent = len(plan) - n_main
            lse_main = None
            lse_noevent = None
            n_scored_item = 0
            t0 = time.time()
            try:
                # AT MOST two calls per item; each scores every layer, the MERGE picks
                # per-layer sources (module comment above). All-'main' plans skip call N
                # entirely; an all-'noevent' plan skips call M (the root rides on N).
                if n_main > 0 or n_noevent == 0:
                    tr_main = scorer.score_transitions(
                        it, view, template="main", stage=prereg["stationD_stage"])
                    lse_main = tr_main["lse"]
                    n_scored_item += tr_main["n_scored"]
                if n_noevent > 0:
                    tr_noevent = scorer.score_transitions(
                        it, space, template="noevent", stage=G4_NOEVENT_STAGE)
                    lse_noevent = tr_noevent["lse"]
                    n_scored_item += tr_noevent["n_scored"]
            except gsd_score.ScorerOOM as exc:
                excluded.append({"item_id": iid,
                                 "reason": "ScorerOOM in station-D scoring: %s" % exc})
                log("  [g4-score] %-30s OOM-SKIPPED (%s)" % (iid, exc))
                continue
            dt = time.time() - t0
            score_seconds += dt
            score_n += n_scored_item
            merged = _g4_merge_lse(plan, lse_main, lse_noevent)
            # NOTE: the decode runs on the REAL space — the parsed moves changed only
            # what the model was TOLD (the prefixes), never the state space.
            row = sg.g1_item_analysis(it, space, merged, sc_idx_by_id.get(iid))
            rows.append(row)
            stationD["per_item_plan"][iid] = {"n_main": n_main, "n_noevent": n_noevent,
                                              "n_parsed_moves": len(info["moves"])}
            log("  [g4-score] %-30s plan=%dmain/%dnoevent scored=%d sec=%.2f map=%r "
                "map_fixed=%s oracle_fixed=%s"
                % (iid, n_main, n_noevent, n_scored_item, dt, row["map"]["answer_idx"],
                   row["map"]["fixed"], row["oracle"]["fixed"]))

        arm = sg3.stationA_arm(rows, prereg)          # the g3 station-A machinery, reused
        stationD["arm"] = arm
        stationD["verdict"] = sg4.stationD_verdict(arm, prereg)
        vd = stationD["verdict"]
        log("g4-D: survival=%d/10 label=%s survived=%s lost=%s gained=%s anomaly=%s"
            % (vd["survival"], vd["label"], vd["survived_ids"], vd["lost_ids"],
               vd["gained_ids"], vd["knowledge_anomaly_flag"]))

        # quality x recovery over the PATIENT rows actually assembled (19 at full scope)
        patient_fixed = {p["item_id"]: p["map_fixed"] for p in arm["per_item"]
                         if p["bucket"] == "patient"}
        per_item_quality = {iid: ext_by_item[iid]["quality"] for iid in patient_fixed
                            if iid in ext_by_item}
        stationD["quality_table"] = sg4.quality_recovery_table(
            per_item_quality, {iid: patient_fixed[iid] for iid in per_item_quality},
            prereg)
        stationD["per_item_quality"] = {iid: info["quality"]
                                        for iid, info in sorted(ext_by_item.items())}
        qt = stationD["quality_table"]
        log("g4-D: quality x recovery — exact %d/%d recovered | inexact %d/%d recovered "
            "| spearman(rate, fixed)=%s"
            % (qt["exact"]["n_recovered"], qt["exact"]["n_items"],
               qt["inexact"]["n_recovered"], qt["inexact"]["n_items"],
               qt["spearman_quality_vs_fixed"]))

    # ---- 5) assemble results_g4 + VERDICT_g4.md + figures (NaN/Inf-guarded write) --------
    banner("g4-assemble", "two-station verdicts + results_g4.json")
    scope = {
        "smoke": bool(args.smoke),
        "station": station,
        "stationC_ids": list(proc_C),
        "stationD_ids": list(proc_D),
        "in_arena": sorted(i for i in set(proc_C) | set(proc_D) if i in items_by_id),
        "dropped_items": sorted({e["item_id"] for e in excluded}),
    }
    eff = _g4_efficiency(extract_seconds, extract_n, score_seconds, score_n,
                         len(set(proc_C) | set(proc_D)))
    config = {
        "stage": "g4",
        "smoke": bool(args.smoke),
        "station": station,
        "models": {"main": args.main_model},
        "gsd_batch": args.gsd_batch,
        "emitter_batch": args.batch,
        "extract": {"seed": prereg["extract_seed"], "greedy": prereg["extract_greedy"],
                    "max_new_tokens": prereg["extract_max_new_tokens"],
                    "template_sha": prereg["extract_template_sha"]},
        "stationD_stage": prereg["stationD_stage"],
        "noevent_reuse_stage": G4_NOEVENT_STAGE,
        "template_shas": {"main": gsd_score.GSD_TEMPLATE_SHA256,
                          "noevent": gsd_score.E_NONE_TEMPLATE_SHA256,
                          "root": gsd_score.GSD_ROOT_TEMPLATE_SHA256},
        # the preflight-asserted baselines (station D's comparison points)
        "baseline_g1": dict(prereg["baseline_g1"]),
        "baseline_g3_noevent": json.loads(json.dumps(prereg["baseline_g3_noevent"])),
    }
    run_stats = {
        "extract_n_scored": extract_n,
        "score_n_scored": score_n,
        "stationC_n_scored": stationC_n_scored,       # hard-asserted 0 above
        "stationC_label": stationC["verdict"]["label"] if stationC["verdict"] else None,
        "stationD_label": stationD["verdict"]["label"] if stationD["verdict"] else None,
    }
    results = {
        "stage": "g4",
        "config": config,
        "prereg": json.loads(json.dumps(prereg)),     # frozen snapshot, json-normalized
        "stationC": stationC,
        "stationD": stationD,
        "scope": scope,
        "excluded": excluded,
        "efficiency": eff,
        "run_stats": run_stats,
    }

    log("---- g4 summary ----")
    log("  %-24s %s" % ("stationC_label", run_stats["stationC_label"]))
    log("  %-24s %s" % ("stationD_label", run_stats["stationD_label"]))
    log("  %-24s %s" % ("excluded", len(excluded)))
    log("g4: efficiency — extraction %.4f GPU-h (%d fresh generation(s)) | scoring "
        "%.4f GPU-h (%d fresh forward(s)) | station C 0 GPU (replay) | total %.4f GPU-h"
        % (eff["extraction"]["gpu_hours"], eff["extraction"]["n_generated"],
           eff["scoring"]["gpu_hours"], eff["scoring"]["n_scored"],
           eff["total_gpu_hours"]))

    res_dir, fig_dir = _g4_artifact_paths(args)
    write_results(os.path.join(res_dir, "results_g4.json"), results)
    _g4_write_verdict_md(os.path.join(res_dir, "VERDICT_g4.md"), results)
    log("g4: wrote VERDICT_g4.md (station C=%s | station D=%s)"
        % (run_stats["stationC_label"], run_stats["stationD_label"]))
    _g4_write_figures(fig_dir, results)
    return results


def _g4_write_verdict_md(path, results):
    """Write the plain-language ``VERDICT_g4.md`` from the assembled results_g4: BOTH
    station verdicts side by side, the per-arm accuracy table, the quality x recovery
    cross table, the frozen interpretation boundary (design §2) and the efficiency
    split. Mirrors the ``_g3_write_verdict_md`` structure; a skipped station renders
    n/a."""
    c, d = results["stationC"], results["stationD"]
    prereg = results["prereg"]
    eff = results["efficiency"]

    def _n(x):
        return "n/a" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))

    md = [
        "# MuSR-cant step accuracy + self-parsed event lines — VERDICT (stage g4)",
        "",
        "The two g3 follow-ups, one stage: station C upgrades the one-hot differential "
        "to an ACCURACY claim (on gold-prefix branching nodes, does the modal sampled "
        "successor equal the gold successor — patient vs knowledge, per template arm, "
        "all replayed from the frozen g2/g3 sample caches at zero GPU); station D "
        "quantifies the deployable share of the λ=0 fixes when the model must parse its "
        "OWN event lines from the narrative (zero oracle, misparse-is-the-treatment). "
        "The two stations answer two DIFFERENT questions and are never merged into one "
        "label. Design: `plans/2026-07-09-g4-step-parsed.md` + `…-design.md` §2. "
        "Frozen: `outputs/PREREG_g4.md`.",
        "",
        "## Station C — gold-prefix step accuracy (decision arm = noevent): **%s**"
        % (c["verdict"]["label"] if c["verdict"] else "n/a (station skipped)"),
        "",
    ]
    if c["verdict"]:
        vc = c["verdict"]
        md += [
            "- delta_acc = modal_acc(patient) − modal_acc(knowledge) on noevent = "
            "**%.4f** (step_specific >= %s / no_step_signal <= %s / else mixed)."
            % (vc["delta_acc"], prereg["stationC_delta_min"],
               prereg["stationC_delta_null"]),
            "- %s" % vc["reason"],
            "- per arm x pool (modal_acc | mass_on_gold | n_nodes):",
        ]
        for arm, pools in sorted(vc["per_arm_pools"].items()):
            for pool, cell in sorted(pools.items()):
                md.append("  - `%s` / %s: %s | %s | n=%d"
                          % (arm, pool, _n(cell["modal_acc"]),
                             _n(cell["mass_on_gold"]), cell["n_nodes"]))
        n_excluded = sum(len(a["excluded"]) for a in c["per_arm"].values())
        md.append("- named exclusions across arms (no_gold_trajectory / "
                  "gold_prefix_not_sampled — neither is an error): %d." % n_excluded)
    md += [
        "",
        "## Station D — self-parsed event lines (survival of the λ=0 fixes): **%s**"
        % (d["verdict"]["label"] if d["verdict"] else "n/a (station skipped)"),
        "",
    ]
    if d["verdict"]:
        vd = d["verdict"]
        md += [
            "- survival = |map-fixed(parsed) ∩ original 10 λ=0 fixes| = **%d/10** "
            "(parse_deployable >= %d / parse_insufficient <= %d / else mixed)."
            % (vd["survival"], prereg["stationD_deploy_min"],
               prereg["stationD_wall_max"]),
            "- survived: %s." % (vd["survived_ids"] or "none"),
            "- lost: %s." % (vd["lost_ids"] or "none"),
            "- gained (map-fixed patients OUTSIDE the original 10 — churn, never counted "
            "toward survival): %s." % (vd["gained_ids"] or "none"),
            "- knowledge anomaly flag: %s (map-fixed knowledge ids: %s; expected 0/3, "
            "the flag never changes the label)."
            % (vd["knowledge_anomaly_flag"], vd["knowledge_anomaly_ids"] or "none"),
            "- patient counts under the parsed event lines (E-full was map %d/19, "
            "oracle %d/19): map-fixed %d, oracle-fixed %d."
            % (results["config"]["baseline_g1"]["map_fixed"],
               results["config"]["baseline_g1"]["oracle_fixed"],
               d["arm"]["n_map_fixed"], d["arm"]["n_oracle_fixed"]),
        ]
        qt = d["quality_table"]
        if qt:
            md += [
                "- quality x recovery (WHICH explanation carries — recovery concentrated "
                "in the all-exact bucket says fact-feed QUALITY is the bottleneck):",
                "  - all-exact extractions: recovered %d of %d item(s) (%s; lost %s)."
                % (qt["exact"]["n_recovered"], qt["exact"]["n_items"],
                   qt["exact"]["ids_recovered"] or "none",
                   qt["exact"]["ids_lost"] or "none"),
                "  - inexact extractions: recovered %d of %d item(s) (%s; lost %s)."
                % (qt["inexact"]["n_recovered"], qt["inexact"]["n_items"],
                   qt["inexact"]["ids_recovered"] or "none",
                   qt["inexact"]["ids_lost"] or "none"),
                "  - spearman(layer_match_rate, map_fixed) = %s."
                % _n(qt["spearman_quality_vs_fixed"]),
            ]
    md += [
        "",
        "## Interpretation boundary (frozen; design §2)",
        "",
        "- The gold belief trajectory (`facts_oracle.gold_beliefs`) is an EVALUATION "
        "readout only — it never enters any model input (station C judges sampled "
        "successors against it post hoc; the sample caches were drawn before g4 "
        "existed).",
        "- Station C conditions on the CORRECT history: `s_prev` itself carries the "
        "consequences of every earlier step's facts, so the readout is \"given the "
        "correct history, is the current single step correct\" — error accumulation is "
        "NOT covered.",
        "- Station D never excludes an item for bad parsing — misparse IS the treatment "
        "(an unparsed layer degrades to the E-none rendering, the deployment-honest "
        "path).",
        "- Smoke-scope results are never canonical (`config.smoke` marks them).",
        "",
        "## Efficiency",
        "",
        "- station D extraction: %.4f GPU-h over %d fresh generation(s); station D "
        "scoring: %.4f GPU-h over %d fresh forward(s); station C: 0 GPU (pure replay, "
        "hard-asserted); total %.4f GPU-h."
        % (eff["extraction"]["gpu_hours"], eff["extraction"]["n_generated"],
           eff["scoring"]["gpu_hours"], eff["scoring"]["n_scored"],
           eff["total_gpu_hours"]),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def _g4_write_figures(fig_dir, results):
    """The g4 figure pair -> ``<fig_dir>/g4_*.png`` (best-effort, the g-line convention:
    a matplotlib failure logs loudly and never voids the written results). A skipped
    station simply drops its figure."""
    try:
        os.makedirs(fig_dir, exist_ok=True)
        written = []
        c = results["stationC"]
        if c.get("per_arm"):
            per_pool_by_arm = {arm: res["per_pool"] for arm, res in c["per_arm"].items()}
            written.append(sg4.fig_step_accuracy(
                per_pool_by_arm, os.path.join(fig_dir, "g4_step_accuracy.png")))
        d = results["stationD"]
        if d.get("arm") and d.get("quality_table"):
            original10 = {str(x) for x in results["prereg"]["original10_ids"]}
            quality = d.get("per_item_quality") or {}
            rows = [{"item_id": p["item_id"],
                     "layer_match_rate": quality[p["item_id"]]["layer_match_rate"],
                     "map_fixed": p["map_fixed"],
                     "in_original10": p["item_id"] in original10}
                    for p in d["arm"]["per_item"]
                    if p["bucket"] == "patient" and p["item_id"] in quality]
            written.append(sg4.fig_parse_recovery(
                d["quality_table"], rows, os.path.join(fig_dir, "g4_parse_recovery.png")))
        log("g4: wrote %d figure(s) to %s" % (sum(1 for w in written if w), fig_dir))
    except Exception as exc:
        log("g4: figure generation error (non-fatal): %s" % exc)


# ======================================================================================
# shared helpers
# ======================================================================================
def _group_by_item(cache, seed_tag):
    """{item_id: sample_idx-ordered records} for one seed_tag regime."""
    grouped = {}
    for r in cache.values():
        if r.get("seed_tag") != seed_tag:
            continue
        grouped.setdefault(r["item_id"], []).append(r)
    for iid in grouped:
        grouped[iid].sort(key=lambda r: r.get("sample_idx", 0))
    return grouped


def _shard_index(shard_spec):
    if not shard_spec:
        return None
    return int(str(shard_spec).split("/")[0])


def _run_stats_public(run_stats):
    return {
        "n_generated": run_stats.get("n_generated", 0),
        "seconds": run_stats.get("seconds", 0.0),
        "oom_skipped": run_stats.get("oom_skipped", []),
        "context_offenders": run_stats.get("context_offenders", []),
    }


def _print_summary(stage, headline, eff):
    log("---- %s summary ----" % stage)
    for k, v in headline.items():
        log("  %-18s %s" % (k, v))
    log("  efficiency: gpu_h=%.4f inst/min=%s gen=%d mean_tok/sample=%.1f mean_sec/sample=%.4f"
        % (eff["gpu_hours"], eff["items_per_min"], eff["n_samples_generated"],
           eff["mean_tokens_per_sample"], eff["mean_sec_per_sample"]))


# ======================================================================================
# CLI
# ======================================================================================
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="MuSR-cant funnel (audit/a0/a1/a2) + a3 self-facts control.")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny same-code-path run (3 items/subtask, N=4)")
    ap.add_argument("--limit", type=int, default=None, help="cap triage ids per subtask")
    ap.add_argument("--shard", default=None, help="i/M strided shard (seed is shard-invariant)")
    ap.add_argument("--base-seed", dest="base_seed", type=int, default=dm.BASE_SEED)
    ap.add_argument("--triage-n", dest="triage_n", type=int, default=dm.TRIAGE_N,
                    help="frozen triage ids per subtask (design §3: 80)")
    ap.add_argument("--arena", default=None, help="override selected arena(s), comma-separated")
    # models
    ap.add_argument("--main-model", dest="main_model", default=MAIN_MODEL)
    ap.add_argument("--models", dest="ladder_models_spec", default=None,
                    help="comma-separated ladder models (a2); default = frozen roster")
    # per-stage sample budgets
    ap.add_argument("--a0-rung", dest="a0_rung", type=int, default=DEFAULT_A0_RUNG)
    ap.add_argument("--a1-top", dest="a1_top", type=int, default=DEFAULT_A1_TOP)
    ap.add_argument("--a1-deepen", dest="a1_deepen_spec", default=None,
                    help="comma-separated deepen rungs (default 512,1024)")
    ap.add_argument("--a2-facts", dest="a2_facts", type=int, default=DEFAULT_A2_FACTS)
    ap.add_argument("--a2-ladder", dest="a2_ladder", type=int, default=DEFAULT_A2_LADDER)
    ap.add_argument("--a2-seed", dest="a2_seed", type=int, default=DEFAULT_A2_SEED)
    ap.add_argument("--a2-stats-rung", dest="a2_stats_rung", type=int,
                    default=stages.PREREG["a2_stats_rung"], help="N=1024-equiv stats rung")
    ap.add_argument("--a3-n", dest="a3_n", type=int, default=DEFAULT_A3_N,
                    help="a3 per-arm SC rung (S1/S2/O all vote at N; design a3_fix_rung=64)")
    # Phase B sample budgets (design §3/§5)
    ap.add_argument("--b0-patient-n", dest="b0_patient_n", type=int, default=DEFAULT_B0_PATIENT_N)
    ap.add_argument("--b0-train-n", dest="b0_train_n", type=int, default=DEFAULT_B0_TRAIN_N)
    ap.add_argument("--b1-patient-n", dest="b1_patient_n", type=int, default=DEFAULT_B1_PATIENT_N)
    ap.add_argument("--b1-deepen-n", dest="b1_deepen_n", type=int, default=DEFAULT_B1_DEEPEN_N)
    ap.add_argument("--b1-tuning-n", dest="b1_tuning_n", type=int, default=DEFAULT_B1_TUNING_N)
    ap.add_argument("--b1-train-n", dest="b1_train_n", type=int, default=DEFAULT_B1_TRAIN_N)
    ap.add_argument("--b-max-new-tokens", dest="b_max_new_tokens", type=int,
                    default=DEFAULT_B_MAX_NEW_TOKENS, help="Phase B generation cap (frozen 2048)")
    ap.add_argument("--featurize-batch", dest="featurize_batch", type=int, default=8,
                    help="hidden-state featurizer forward micro-batch (memory lever; halves on OOM)")
    # EXACT-GSD (g0/g1)
    ap.add_argument("--gsd-batch", dest="gsd_batch", type=int, default=DEFAULT_GSD_BATCH,
                    help="TF scorer forward batch (g0/g1; gsd_score halves once on OOM)")
    # G-A fidelity narrowing (g2): per-branching-node conditional-sampling draws M (None ->
    # frozen PREREG_G2.sample_M=256; --smoke -> 16; L1 extrapolates the full-run GPU-h).
    ap.add_argument("--g2-sample-m", dest="g2_sample_m", type=int, default=None,
                    help="g2 draws per branching node (default PREREG_G2.sample_M; smoke=16)")
    # Event-line ablation (g3): per-node draws M (None -> frozen PREREG_G3.sample_M=64;
    # --smoke -> 8) + the station selector (full-run driver: A un-sharded first, then B
    # sharded gen-only, then one un-sharded 'all' consolidation pass).
    ap.add_argument("--g3-sample-m", dest="g3_sample_m", type=int, default=None,
                    help="g3 draws per branching node (default PREREG_G3.sample_M; smoke=8)")
    ap.add_argument("--g3-station", dest="g3_station", choices=("A", "B", "all"),
                    default="all",
                    help="g3 stations to run (A=TF scoring audit, B=sampling probe)")
    # Step accuracy + self-parsed event lines (g4): the station selector. No --shard and
    # no sample-M lever: station C is a pure replay of the frozen g2/g3 caches at their
    # frozen M, station D is 22 items on a single card (<= 0.5 GPU-h).
    ap.add_argument("--g4-station", dest="g4_station", choices=("C", "D", "all"),
                    default="all",
                    help="g4 stations to run (C=gold-prefix accuracy replay, "
                         "D=self-parsed event-line rerun)")
    # io / device
    ap.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--data-dir", dest="data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16, help="HFEmitter generation batch size")
    return ap


def parse_args(argv=None):
    args = build_arg_parser().parse_args(argv)
    # resolve list-valued args + smoke overrides
    args.ladder_models = ([s.strip() for s in args.ladder_models_spec.split(",") if s.strip()]
                          if args.ladder_models_spec else list(LADDER_MODELS))
    args.a1_deepen = ([int(x) for x in args.a1_deepen_spec.split(",") if str(x).strip()]
                      if args.a1_deepen_spec else list(DEFAULT_A1_DEEPEN))
    if args.smoke:
        _apply_smoke(args)
    return args


def _apply_smoke(args):
    args.a0_rung = 4
    args.a1_top = 8
    args.a1_deepen = [16]
    args.a2_facts = 4
    args.a2_ladder = 4
    args.a2_seed = 4
    args.a2_stats_rung = 8
    args.a3_n = 4
    # Phase B smoke: 2 patients + 2 train items x N=8 (same code path, tiny sizes).
    args.b0_patient_n = 8
    args.b0_train_n = 8
    args.b1_patient_n = 8
    args.b1_deepen_n = 8
    args.b1_tuning_n = 8
    args.b1_train_n = 8
    args.b_max_new_tokens = 512
    # g2 smoke: 16 draws per branching node (L1 <10 min; full run extrapolates to M=256),
    # unless the caller explicitly pinned --g2-sample-m.
    if args.g2_sample_m is None:
        args.g2_sample_m = 16
    # g3 smoke: 8 draws per branching node (plan: --smoke -> 8), unless explicitly pinned.
    if args.g3_sample_m is None:
        args.g3_sample_m = 8
    if args.limit is None:
        args.limit = 3
    # L1 budget guard: exercise the ladder PATH with a single small model (the full 5-model
    # ladder — incl. two 7B — only runs at the real full a2, not in the <10min smoke), unless
    # the caller explicitly pinned --models.
    if args.ladder_models_spec is None:
        args.ladder_models = [SMOKE_LADDER_MODEL]


def run_stage(args, emitter_factory=None, featurizer_factory=None, scorer=None):
    """Dispatch a stage. Loads the real (GPU) emitter/featurizer factories only if none injected
    and the stage needs them. ``audit`` needs no emitter; b0/b2 additionally need a featurizer;
    g0/g1 need neither — their sole GPU site is the TF ``scorer`` (injected in tests, else the
    real bf16 model via ``gsd_score.load_scorer``)."""
    os.makedirs(args.out_dir, exist_ok=True)
    if args.stage == "audit":
        return stage_audit(args)
    if args.stage == "g0":
        return stage_g0(args, scorer=scorer)
    if args.stage == "g1":
        return stage_g1(args, scorer=scorer)
    if args.stage == "g2":
        # g2's two heavy sites are BOTH dependency-injected: the TF ``scorer`` and the sampler
        # ``emitter`` (built lazily inside when None — the sole GPU station is the sampler).
        return stage_g2(args, scorer=scorer)
    if args.stage == "g3":
        # g3 mirrors g2's injection contract: the station-A TF ``scorer`` (built only when
        # station A runs) and the station-B ``emitter`` (a _LazyEmitter when None).
        return stage_g3(args, scorer=scorer)
    if args.stage == "g4":
        # g4 mirrors g3's injection contract: the station-D TF ``scorer`` (built only when
        # station D runs) and the station-D extraction ``emitter`` (a _LazyEmitter when
        # None); station C is a zero-GPU cache replay and needs neither.
        return stage_g4(args, scorer=scorer)
    if emitter_factory is None:
        emitter_factory = default_emitter_factory(args)
    if args.stage == "a0":
        return stage_a0(args, emitter_factory)
    if args.stage == "a1":
        return stage_a1(args, emitter_factory)
    if args.stage == "a2":
        return stage_a2(args, emitter_factory)
    if args.stage == "a3":
        return stage_a3(args, emitter_factory)
    if args.stage in ("b0", "b1", "b2"):
        if featurizer_factory is None:
            featurizer_factory = default_featurizer_factory(args)
        _b_prereg_selfcheck(args)                   # full run only: guard against a drifted PREREG_b
        if args.stage == "b0":
            return stage_b0(args, emitter_factory, featurizer_factory)
        if args.stage == "b1":
            return stage_b1(args, emitter_factory, featurizer_factory)
        return stage_b2(args, emitter_factory, featurizer_factory)
    raise ValueError("unknown stage %r" % args.stage)


def main(argv=None):
    args = parse_args(argv)
    log("run_musr_cant | stage=%s smoke=%s out_dir=%s" % (args.stage, args.smoke, args.out_dir))
    res = run_stage(args)
    if args.stage == "a0" and res.get("triage", {}).get("early_negative"):
        log("a0 EARLY-NEGATIVE: zero subtasks qualify -> MuSR line closes (see VERDICT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
