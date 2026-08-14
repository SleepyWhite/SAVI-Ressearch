"""gsd_sample — GSD-template node-conditional next-BELIEF sampler (EXACT-GSD stage g2).

The ONLY GPU station of the G-A fidelity unit (stage g2). Given ``(item, space, emitter, M,
base_seed)`` it draws, for each ENUMERATED BRANCHING node ``(t, canon_prev)`` of the reachable
belief-state space, ``M`` next-BELIEF-line continuations UNDER THE FROZEN GSD TEMPLATE (the same
byte-identical ``gsd_score.build_prefix`` context the teacher-forcing scorer uses), parses each,
and classifies it — by EXACT canonical-string match — to one of the node's enumerated successor
canons or to an off-manifold bucket. The per-node frequency tables are the Monte-Carlo
``P(s_t | s_{t-1} = enumerated state)`` estimates consumed by Arm 1 (matched-context rho) and
Arm 2 (frequency-decode). The ONLY difference from the TF scorer is point-estimate (TF) vs
finite-sample frequency (this) — exactly the estimator axis the unit measures.

Single-step, node-independent: each node is sampled INDEPENDENTLY conditioned on its canonical
enumerated ``s_prev`` (never the model's own previous free-generation), so there is no multi-step
self-generation drift. The prefix carries no witness gold / gold_idx / locset (it is the GSD
BELIEF-tracking template, identical to the EXACT-GSD lambda=0 arm), so nothing here leaks the
task answer; classification uses ONLY the enumerated graph structure.

==========================================================================================
FROZEN CONVENTIONS
==========================================================================================
* Branching node = ``(t, canon_prev)`` with ``t >= 1`` and
  ``len(space.trans[(t, canon_prev)]) >= min_succ`` (default 2). The t=0 root is never sampled
  (its transition is a unique path constant); single-successor layers are forced edges (no
  frequency to estimate). When ``nodes`` is given explicitly the sampler draws EXACTLY those
  ``(t, canon_prev)`` keys that exist in ``space.trans`` (t >= 1) — trusting the caller's request
  (Arm 1a's 35 G-A group nodes vs Arm 2's every-branching-node union) rather than re-filtering.
* Prefix = ``gsd_score.build_prefix(item, space, t, by_canon[t-1][canon_prev], template)`` where
  ``template`` is the caller-selected registered GSD template name (default ``"main"`` — the
  frozen g2 path) and ``by_canon[t-1]`` maps layer-(t-1) canon -> its state dict (built from
  ``space.layers`` via ``belief_schema.canon_state`` — the same map
  ``gsd_score.score_transitions`` builds). This is byte-identical to the TF scoring prefix
  under the SAME template name, so frequency and likelihood are estimated in the SAME context.
* Draw seed = ``sc_core.sample_seed(base_seed, seed_tag, item_id, sample_idx)`` (``seed_tag``
  defaults to ``"g2sample"`` — the frozen g2 derivation; stage g3 passes ``"g3sample"``) for
  ``sample_idx`` in ``0..M-1`` — a pure function of its args, hence shard-invariant and resumable
  at sample granularity. (Seeds are shared across an item's nodes for a fixed sample_idx: benign
  common-random-numbers — the real emitter conditions on the distinct per-node prefix, so the
  draws differ; per-node frequencies are unaffected. FakeEmitter honours the contract purely.)
* Classification of one continuation ``text``:
    - ``pc = belief_schema.parse_chain(text)``; take ``pc.states[0]`` (the FIRST parsed BELIEF
      line = the sampled next state). NO parsed state -> a parse fail: it counts toward NEITHER
      ``freqs`` NOR ``n_parse_ok`` (but toward ``n_total``). [A single-line BELIEF continuation
      carries no ``ANSWER:`` line, so "parse ok" here means a BELIEF STATE was recovered, not the
      full ANSWER-bearing ``ParsedChain.parse_ok`` — see the unit test's n_parse_ok assertion.]
    - ``cn = belief_schema.canon_state(pc.states[0])``. EXACT string membership in the node's
      enumerated successor canon set: hit -> ``freqs[cn] += 1``; miss -> ``off_manifold += 1``.
      Both a hit and an off-manifold miss are parse-ok (``n_parse_ok += 1``). NEVER approximate-
      match: an off-path canon must fall in the off-manifold bucket, not be snapped to a neighbour.
* JSONL content-key cache (``sc_core.append_record`` atomic append; ``gsd_score`` idiom): key =
  ``(item_id, stage, t, sha256(canon_prev), sample_idx)``. The TEMPLATE does NOT enter this
  5-part key (unlike the 6-part scoring-cache key of ``gsd_score``), so two template arms
  sharing a stage string would silently collide on cached text: CALLERS MUST USE A UNIQUE
  ``stage`` STRING PER TEMPLATE ARM (g3 convention: ``"g3_anchor"`` / ``"g3_noevent"``, plus
  per-arm cache files as a second guard); the record stores the raw generated
  ``text``. A rerun reloads the text and RE-CLASSIFIES (classification is cheap + deterministic,
  so a full cache hit reproduces byte-identical freqs and scores nothing: ``n_scored == 0``).
  ``n_scored`` = fresh draws (cache misses) generated THIS call.
* ``item_parse_ok_rate`` = sum(n_parse_ok) / sum(n_total) across the sampled nodes
  (0.0 when nothing was sampled — fail-closed for the downstream coverage gate).

CODE SEPARATION: imports stdlib + ``belief_schema`` (parse/canon) + ``gsd_score`` (build_prefix)
+ ``sc_core`` (seed / atomic append / GenOut Emitter contract). ``gsd_score``/``sc_core`` import
their heavy deps lazily, so this module imports with ZERO torch/transformers. It NEVER imports
test code; the emitter is dependency-injected via the ``sc_core.Emitter`` protocol (real
``HFEmitter`` at L1, a scripted CPU emitter in the unit tests).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import belief_schema
import gsd_score
import sc_core as sc


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ======================================================================================
# Sample cache (JSONL; sc_core atomic-append idiom, 5-part content key)
# ======================================================================================
def _sample_cache_key(rec: dict) -> tuple:
    return (rec["item_id"], rec["stage"], rec["layer"], rec["sprev_sha"], rec["sample_idx"])


def _load_sample_cache(path: Optional[str]) -> dict:
    """``{5-part key: generated_text}`` with last-write-wins dedup; blank/torn lines skipped."""
    out: dict = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[_sample_cache_key(rec)] = rec["text"]
            except (ValueError, TypeError, KeyError):
                continue
    return out


# ======================================================================================
# Classification — EXACT canon match against the node's enumerated successor set
# ======================================================================================
def _classify(text, succ_canons):
    """Classify one continuation into ('on', canon) | ('off', None) | ('parsefail', None).

    ``pc.states[0]`` is the first parsed BELIEF state (the sampled next state); no parsed state
    is a parse fail. ``belief_schema.canon_state`` + EXACT set membership decides on/off-manifold
    (never an approximate/nearest-neighbour snap)."""
    pc = belief_schema.parse_chain(text)
    if not pc.states:
        return ("parsefail", None)
    cn = belief_schema.canon_state(pc.states[0])
    if cn in succ_canons:
        return ("on", cn)
    return ("off", None)


# ======================================================================================
# The sampler
# ======================================================================================
def sample_node_freqs(item, space, emitter, M, base_seed, nodes=None, min_succ=2,
                      cache_path=None, stage="g2", max_new_tokens=256, batch=64,
                      template="main", seed_tag="g2sample"):
    """Sample ``M`` next-BELIEF continuations per enumerated branching node and tally the
    per-node successor-canon frequencies. See the module docstring for the full contract.

    ``template`` selects the registered GSD prefix template (``gsd_score._TEMPLATES``);
    ``seed_tag`` selects the ``sc_core.sample_seed`` derivation tag. The defaults reproduce
    the frozen g2 behaviour byte-for-byte. The template does NOT enter the sample cache key —
    callers must pass a unique ``stage`` per template arm (module docstring).

    Returns::

        {"nodes": {(t, canon_prev): {"freqs": {canon_next: int},
                                     "off_manifold": int,
                                     "n_parse_ok": int,
                                     "n_total": int}, ...},
         "item_parse_ok_rate": float,   # sum(n_parse_ok) / sum(n_total), 0.0 if none sampled
         "n_scored": int}               # fresh draws generated this call (0 == full cache hit)
    """
    item_id = item.get("id") if isinstance(item, dict) else None
    by_canon = [{belief_schema.canon_state(s): s for s in layer} for layer in space.layers]

    # ---- select the nodes to sample ------------------------------------------------------
    if nodes is None:
        sel = [(t, cp) for (t, cp) in space.trans
               if t >= 1 and len(space.trans[(t, cp)]) >= min_succ]
    else:
        # explicit request: EXACTLY the given keys that are real (t >= 1) transitions.
        sel = [(t, cp) for (t, cp) in nodes if t >= 1 and (t, cp) in space.trans]
    sel = sorted(set(sel))

    cache = _load_sample_cache(cache_path)
    out_nodes: dict = {}
    n_scored = 0

    for (t, cp) in sel:
        sprev_sha = _sha256(cp)
        succ_canons = {cn for cn, _mask in space.trans[(t, cp)]}
        prefix = gsd_score.build_prefix(item, space, t, by_canon[t - 1][cp], template)

        # resolve each of the M draws from cache or the emitter ----------------------------
        texts = [None] * M
        todo = [i for i in range(M)
                if (item_id, stage, t, sprev_sha, i) not in cache]
        for i in range(M):
            if i not in todo:
                texts[i] = cache[(item_id, stage, t, sprev_sha, i)]

        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            seeds = [sc.sample_seed(base_seed, seed_tag, item_id, i) for i in chunk]
            outs = emitter.generate([prefix] * len(chunk), seeds,
                                    max_new_tokens=max_new_tokens)
            assert len(outs) == len(chunk), (          # loud, not a silent parse-rate deflation
                "emitter returned %d of %d requested draws" % (len(outs), len(chunk)))
            for i, out in zip(chunk, outs):
                texts[i] = out.text
                rec = {"item_id": item_id, "stage": stage, "layer": t,
                       "sprev_sha": sprev_sha, "sample_idx": i,
                       "text": out.text, "n_new_tokens": int(out.n_new_tokens)}
                if cache_path:
                    sc.append_record(cache_path, rec)
                cache[_sample_cache_key(rec)] = out.text
            n_scored += len(chunk)

        # classify all M draws (fresh + cached) --------------------------------------------
        freqs: dict = {}
        off_manifold = 0
        n_parse_ok = 0
        for i in range(M):
            kind, cn = _classify(texts[i], succ_canons)
            if kind == "on":
                n_parse_ok += 1
                freqs[cn] = freqs.get(cn, 0) + 1
            elif kind == "off":
                n_parse_ok += 1
                off_manifold += 1
            # "parsefail": counts toward n_total only
        out_nodes[(t, cp)] = {"freqs": freqs, "off_manifold": off_manifold,
                              "n_parse_ok": n_parse_ok, "n_total": M}

    tot = sum(nd["n_total"] for nd in out_nodes.values())
    pok = sum(nd["n_parse_ok"] for nd in out_nodes.values())
    return {"nodes": out_nodes,
            "item_parse_ok_rate": (pok / tot) if tot > 0 else 0.0,
            "n_scored": n_scored}
