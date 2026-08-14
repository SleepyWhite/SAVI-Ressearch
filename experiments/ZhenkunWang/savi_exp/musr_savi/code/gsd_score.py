"""gsd_score — frozen GSD scoring template + teacher-forcing likelihood scorer (EXACT-GSD, R2 fix).

Turns any candidate belief-state transition of a ``gsd_space.GsdSpace`` into an EXACT model
log-likelihood: batched, cached, deterministic. These scores are the edge weights the exact DP
decoder (gsd_decode) and the Δ-ledger consume, and the same template feeds the g0 G-A fidelity
gate (stages_g scores sampled b1 states with ``build_prefix`` + ``score_batch``).

==========================================================================================
LEAK BOUNDARY — READ BEFORE TOUCHING THIS MODULE
==========================================================================================
This module is ALLOWED to read exactly:

  * ``item["narrative"] / item["question"] / item["choices"]`` — the TASK INPUTS the model
    itself reads (the witness clues live in the narrative, where solving them is the task);
  * ``space.moves / space.T / space.root / space.layers / space.trans`` — products of the
    declared STRUCTURAL oracle (gsd_space, which has its own audited leak boundary).

ANY read of the item's gold tree (the ``tree_raw`` field) — in particular its per-event NL
groups or gold belief tables — is leakage, full stop: on real object_placements data those
NL groups contain
explicit witness sentences ("X saw / did not see the move"), i.e. THE TASK ANSWER, and an
earlier draft of this module consumed them (orchestrator-ruled leak, removed). The event
anchor line shown to the model is therefore MECHANICALLY RENDERED from ``space.moves`` only
(``render_event_line`` below) — it names WHAT moved WHERE (structural-oracle scope) and
never who saw it. A source-grep unit test pins the absence of tree reads.

==========================================================================================
FROZEN TEMPLATES (sha256 constants exported; snapshot-tested — freezing == testing)
==========================================================================================
* ``GSD_TEMPLATE``          — the MAIN transition template (t >= 1).
* ``SENSITIVITY_TEMPLATE``  — same structure/fields, ONLY the final instruction sentence
                              paraphrased (the design's template-sensitivity probe).
* ``E_ANCHOR_TEMPLATE``     — g3 event-line ablation arm "anchor": byte-identical to
                              GSD_TEMPLATE except ``{event_line}`` becomes the CONSTANT
                              line "Event {t}: see the story for what happens in this
                              step." — step index only, never move content, and
                              deliberately NOT distinguishing move from no-move layers
                              ("nothing is moved" is fact content too).
* ``E_NONE_TEMPLATE``       — g3 arm "noevent": byte-identical to GSD_TEMPLATE except
                              the ``"{event_line}\n\n"`` substring is deleted outright
                              (one blank line remains between the BELIEF[t-1] line and
                              the instruction sentence).
* ``GSD_ROOT_TEMPLATE``     — the t = 0 (root) variant: NO BELIEF[t-1] line exists yet, the
                              event-0 line is the frozen constant
                              "Event 0: the story begins; objects are at their initial
                              locations.", and a format spec replaces the "same format"
                              reference. The root state is UNIQUE per item, so its score is
                              a PATH CONSTANT — it cannot change any argmax — and the root
                              template is deliberately SHARED by every transition-template
                              pass (root rows are cached under its own sha).
All five are filled with ``str.format``; substituted values are never re-scanned for braces
(only the template string is), so narratives containing ``{``/``}`` are safe — the
``belief_schema.BELIEF_TEMPLATE`` convention. The only literal braces (root format spec) are
escaped ``{{``/``}}``. ``str.format`` ignores unconsumed kwargs, so ``build_prefix`` passes
``t`` and ``event_line`` to every transition template and each consumes what it declares.

==========================================================================================
FROZEN CONVENTIONS
==========================================================================================
* Prefix (t >= 1) = full narrative + question + 1-based numbered choices
  (``sc_core.render_numbered_choices``) + the SINGLE line ``BELIEF[t-1]: {json(s_prev)}``
  (never any earlier layer — the first-order Markov property holds BY CONSTRUCTION of the
  scoring context, not by assumption) + the mechanical event anchor line + instruction.
  The s_prev JSON is ``json.dumps(s_prev, ensure_ascii=False, sort_keys=True)`` — byte-equal
  to the ``belief_schema.render_belief_line`` payload, so the prefix line round-trips through
  ``belief_schema.parse_chain``.
* Event anchor line (``render_event_line``, frozen wording):
    - layer with moves:    ``Event {t}: the {obj} is moved to the {loc}.`` — several moves in
      one layer become parallel sentences on the same line, objects folded-sorted
      (deterministic); object/location strings verbatim from ``space.moves``.
    - layer without moves: ``Event {t}: nothing is moved in this step.``
    - root (inside GSD_ROOT_TEMPLATE): ``Event 0: the story begins; objects are at their
      initial locations.``
* Target = the FULL ``render_belief_line(t, s_next)`` line: scored from the head
  ``"BELIEF[t]: "`` to the end of the line (the prefix ends with ``"\n"``; the target is
  appended directly, no separator).
* Tokenization: ``ids = tok.encode(prefix, add_special_tokens=False) +
  tok.encode(target, add_special_tokens=False)``. Encoding the target STANDALONE (instead of
  slicing a joint encoding) makes the target token identity independent of batching and
  identical across all candidates sharing a prefix; the boundary is a ``"\n"`` so the
  convention is also the natural BPE split. Score = sum of per-token log-softmax logprobs of
  the target ids only. The per-row TARGET WINDOW of the logits is sliced on the model-dtype
  tensor BEFORE the float32 cast + softmax (the featurizer 6201422 slice-before-cast
  convention): softmax is independent per position, so this is positionwise equivalent while
  keeping the fp32 transient at [target_len, V] per row instead of [B, max_len, V]
  (~15-18 GB at V≈152k). Model dtype free.
* ``score_transitions`` -> ``{"raw": A_raw, "lse": A_lse, "n_scored": int}`` with
  ``A_raw[(t, canon_prev, canon_next)]`` = raw target logprob sum and ``A_lse`` = per-
  (t, canon_prev) log-softmax over the candidate set (the MAIN normalization; both reported).
  Root key = ``(0, None, canon_root)``; a singleton candidate set has lse exactly 0.0.
* JSONL cache: one record per scored line, appended atomically (``sc_core.append_record``),
  key = (item_id, stage, template_sha, layer, sha256(canon_prev), sha256(canon_next)) with
  sprev_sha = "" on the root row and template_sha = the sha of the template that BUILT the
  prefix (root rows -> GSD_ROOT_TEMPLATE_SHA256, hence shared across main/sensitivity).
  Last-write-wins on load; torn/blank lines skipped; a full-hit rerun scores nothing
  (``n_scored == 0``). Records are appended per scored CHUNK, so a killed run resumes at
  chunk granularity.
* Determinism: ``model.eval()`` at construction, ``torch.no_grad()`` around every forward,
  no sampling anywhere; right-padding (causal attention -> real-token logits unaffected).
  Two identical calls are bitwise equal; single-vs-batched agrees to fp32 tolerance (the
  padded batched matmul reorders float ops — atol 1e-5 unit-tested on the fp32 tiny model;
  the bf16 production path is NOT bit-tested, by spec).
* Failure policy: non-finite logprob -> hard ``ValueError`` naming the item (nothing is
  cached for the failing call); CUDA OOM -> ONE halved-batch retry, then ``ScorerOOM``
  (the runner records the item as skipped, design §7).

CODE SEPARATION: imports stdlib + ``belief_schema`` + ``sc_core``; torch is imported lazily
inside ``TFScorer`` and transformers inside ``load_scorer`` only — the module imports with
zero heavy deps. It NEVER imports test code. The unit tests inject a tiny fp32 model + a
self-made byte tokenizer; ``load_scorer`` (bf16 / sdpa / offline, the sc_core HF convention)
is exercised at L1 only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Optional, Sequence

import belief_schema
import sc_core as sc

# ======================================================================================
# Frozen templates (sha256 snapshot-tested in tests/test_gsd_score.py)
# ======================================================================================
GSD_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    "Track what every character believes after each event. The belief state after "
    "event {t_prev} is:\n"
    "BELIEF[{t_prev}]: {sprev_json}\n\n"
    "{event_line}\n\n"
    "Write exactly one line recording what every character now believes about where "
    "every tracked object is, in the same format.\n"
)

# Same structure and fields; ONLY the final instruction sentence is paraphrased.
SENSITIVITY_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    "Track what every character believes after each event. The belief state after "
    "event {t_prev} is:\n"
    "BELIEF[{t_prev}]: {sprev_json}\n\n"
    "{event_line}\n\n"
    "State, in one line of the same format, where every character now believes each "
    "tracked object to be.\n"
)

# g3 event-line ablation arm "anchor": GSD_TEMPLATE byte-for-byte except {event_line} ->
# the CONSTANT anchor line below. Only the step index {t} is passed through — never what
# moved where — and move / no-move layers are deliberately NOT distinguished ("nothing is
# moved in this step" is fact content too and is stripped with the rest).
E_ANCHOR_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    "Track what every character believes after each event. The belief state after "
    "event {t_prev} is:\n"
    "BELIEF[{t_prev}]: {sprev_json}\n\n"
    "Event {t}: see the story for what happens in this step.\n\n"
    "Write exactly one line recording what every character now believes about where "
    "every tracked object is, in the same format.\n"
)

# g3 arm "noevent": GSD_TEMPLATE byte-for-byte except the "{event_line}\n\n" substring is
# deleted outright — exactly one blank line remains between the BELIEF[t-1] line and the
# instruction sentence. This is the ablation's DECIDING arm.
E_NONE_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    "Track what every character believes after each event. The belief state after "
    "event {t_prev} is:\n"
    "BELIEF[{t_prev}]: {sprev_json}\n\n"
    "Write exactly one line recording what every character now believes about where "
    "every tracked object is, in the same format.\n"
)

# t = 0: no previous BELIEF line exists; the event-0 line is a frozen constant (NEVER the
# gold tree's opening NL, which narrates who saw the initial placements) and a format spec
# replaces the "same format" reference. Root scores are path constants.
GSD_ROOT_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    "Track what every character believes after each event.\n\n"
    "Event 0: the story begins; objects are at their initial locations.\n\n"
    "Write exactly one line recording what every character now believes about where "
    "every tracked object is, in this exact format:\n"
    "BELIEF[t]: {{\"Character\": {{\"object\": \"location\"}}, ...}}\n"
    "where t is the 0-based event number.\n"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


GSD_TEMPLATE_SHA256 = _sha256(GSD_TEMPLATE)
SENSITIVITY_TEMPLATE_SHA256 = _sha256(SENSITIVITY_TEMPLATE)
E_ANCHOR_TEMPLATE_SHA256 = _sha256(E_ANCHOR_TEMPLATE)
E_NONE_TEMPLATE_SHA256 = _sha256(E_NONE_TEMPLATE)
GSD_ROOT_TEMPLATE_SHA256 = _sha256(GSD_ROOT_TEMPLATE)

_TEMPLATES = {
    "main": (GSD_TEMPLATE, GSD_TEMPLATE_SHA256),
    "sensitivity": (SENSITIVITY_TEMPLATE, SENSITIVITY_TEMPLATE_SHA256),
    "anchor": (E_ANCHOR_TEMPLATE, E_ANCHOR_TEMPLATE_SHA256),
    "noevent": (E_NONE_TEMPLATE, E_NONE_TEMPLATE_SHA256),
}


class ScorerOOM(RuntimeError):
    """Raised when a scoring batch OOMs even after the single halved-batch retry. The
    caller (runner) records the item as oom-skipped and moves on (design §7)."""


# ======================================================================================
# Prefix / target construction (pure; exported so stages_g reuses them for G-A)
# ======================================================================================
def render_event_line(space, t: int) -> str:
    """The mechanical event anchor line for layer ``t`` (1 <= t < space.T), rendered from
    ``space.moves`` ONLY (structural-oracle scope: what moved where — never who saw it).

    Frozen wording: ``Event {t}: the {obj} is moved to the {loc}.`` per move (parallel
    sentences on one line, objects folded-sorted); a no-move layer renders
    ``Event {t}: nothing is moved in this step.``
    """
    if not (1 <= t < space.T):
        raise ValueError("event line is defined for 1 <= t < T (got t=%r, T=%r)"
                         % (t, space.T))
    mvs = sorted((mv for mv in space.moves if mv[0] == t),
                 key=lambda mv: str(mv[1]).strip().lower())
    if not mvs:
        return "Event %d: nothing is moved in this step." % t
    sentences = " ".join("the %s is moved to the %s." % (obj, loc)
                         for _t, obj, loc in mvs)
    return "Event %d: %s" % (t, sentences)


def build_prefix(item: dict, space, t: int, s_prev: Optional[dict],
                 template: str = "main") -> str:
    """The frozen GSD scoring prefix for layer ``t``.

    ``t == 0``: the root prefix (``GSD_ROOT_TEMPLATE``; ``s_prev`` must be None).
    ``t >= 1``: the registered transition template (main/sensitivity/anchor/noevent) with
    the SINGLE ``BELIEF[t-1]`` line rendering ``s_prev`` (first-order Markov by
    construction); main/sensitivity carry the mechanical event anchor line from
    ``space.moves``, anchor the index-only constant line, noevent no event line at all
    (``str.format`` ignores the unconsumed kwargs, so every template gets both ``t`` and
    ``event_line``). Reads ONLY narrative/question/choices from ``item`` (task inputs)
    — never the gold tree. The template string is the only thing ``.format`` scans, so
    braces inside narrative/state JSON are safe.
    """
    if template not in _TEMPLATES:
        raise ValueError(
            "unknown template %r (want 'main', 'sensitivity', 'anchor' or 'noevent')"
            % (template,))
    numbered = sc.render_numbered_choices(item["choices"])
    if t == 0:
        if s_prev is not None:
            raise ValueError("t=0 is the root layer; it takes no s_prev")
        return GSD_ROOT_TEMPLATE.format(
            narrative=item["narrative"], question=item["question"],
            numbered_choices=numbered)
    if s_prev is None:
        raise ValueError("t>=1 requires the previous belief state s_prev")
    sprev_json = json.dumps(s_prev, ensure_ascii=False, sort_keys=True)
    return _TEMPLATES[template][0].format(
        narrative=item["narrative"], question=item["question"],
        numbered_choices=numbered, t_prev=t - 1, t=t, sprev_json=sprev_json,
        event_line=render_event_line(space, t))


def build_target(t: int, state: dict) -> str:
    """The scored line: the FULL ``BELIEF[t]: {json}`` rendering of ``state`` (head to end
    of line) — byte-identical to ``belief_schema.render_belief_line``."""
    return belief_schema.render_belief_line(t, state)


# ======================================================================================
# Score cache (JSONL; sc_core atomic-append idiom, 6-part key)
# ======================================================================================
def _score_cache_key(rec: dict) -> tuple:
    return (rec["item_id"], rec["stage"], rec["template_sha"], rec["layer"],
            rec["sprev_sha"], rec["snext_sha"])


def _load_score_cache(path: Optional[str]) -> dict:
    """``{6-part key: lp}`` with last-write-wins dedup; blank/torn lines skipped."""
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
                out[_score_cache_key(rec)] = float(rec["lp"])
            except (ValueError, TypeError, KeyError):
                continue
    return out


# ======================================================================================
# TFScorer
# ======================================================================================
class TFScorer:
    """Teacher-forcing likelihood scorer over an injected (model, tokenizer) pair.

    The tokenizer only needs ``encode(text, add_special_tokens=False) -> list[int]`` and a
    ``pad_token_id`` attribute; the model only needs ``.eval()`` and
    ``model(input_ids=, attention_mask=) -> output.logits`` — both the HF classes and the
    unit-test fakes satisfy this. Production construction goes through ``load_scorer``.
    """

    def __init__(self, model, tokenizer, device=None, cache_path: Optional[str] = None,
                 batch: int = 8, stage: str = "g1"):
        import torch  # lazy: keep the module importable with zero heavy deps
        self._torch = torch
        self.model = model.eval()
        self.tok = tokenizer
        if device is None:
            try:
                device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                device = "cpu"
        self.device = device
        self.cache_path = cache_path
        self.batch = max(1, int(batch))
        self.stage = stage
        self.total_scored = 0  # lifetime forwards (efficiency ledger)
        self._cache = _load_score_cache(cache_path)

    # ---- low-level batched TF scoring ----------------------------------------------
    def _score_chunk(self, pairs: list) -> list:
        """One padded forward over ``pairs`` = [(prefix_ids, target_ids), ...] ->
        per-pair sum of target-token logprobs.

        Slice-FIRST fp32 policy (featurizer.py 6201422 convention): each row's target
        window is sliced from the model-dtype logits BEFORE ``.float()`` + log_softmax, so
        the fp32 transient is ``[target_len, V]`` per row — casting the whole
        ``[B, max_len, V]`` tensor first would peak at ~15-18 GB with a 152k vocab.
        log_softmax is computed independently per position, so slicing first is
        positionwise equivalent (the determinism/batch-invariance tests pin this)."""
        torch = self._torch
        pad_id = getattr(self.tok, "pad_token_id", None)
        pad_id = int(pad_id) if pad_id is not None else 0
        max_len = max(len(p) + len(t) for p, t in pairs)
        ids = torch.full((len(pairs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(pairs), max_len), dtype=torch.long)
        for row, (p, t) in enumerate(pairs):
            seq = p + t
            ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, :len(seq)] = 1
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        scores = []
        for row, (p, t) in enumerate(pairs):
            lp, lt = len(p), len(t)
            tgt = ids[row, lp:lp + lt]
            # logits at position k predict token k+1 -> target token j sits at logits row
            # lp - 1 + j (lp >= 1 is guaranteed by the non-empty templates). Slice the
            # target window on the model-dtype tensor, THEN cast fp32 (see docstring).
            window = logits[row, lp - 1:lp + lt - 1, :].float()
            tok_lps = torch.log_softmax(window, dim=-1).gather(
                -1, tgt.unsqueeze(-1)).squeeze(-1)
            scores.append(float(tok_lps.sum().item()))
        return scores

    def _is_oom(self, exc) -> bool:
        torch = self._torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def score_batch(self, prefixes: Sequence[str], target_lines: Sequence[str],
                    context: str = "") -> list:
        """Teacher-forcing scores for parallel (prefix, target) pairs: the sum of the
        per-token logprobs of the TARGET segment only, one forward per pair (batched).

        OOM: one halved-batch retry across the call, then ``ScorerOOM``. A non-finite
        score raises ``ValueError`` immediately (``context`` names the item)."""
        assert len(prefixes) == len(target_lines), "prefixes/targets must be parallel"
        pairs = []
        for prefix, target in zip(prefixes, target_lines):
            p = self.tok.encode(prefix, add_special_tokens=False)
            t = self.tok.encode(target, add_special_tokens=False)
            if not p or not t:
                raise ValueError("empty prefix/target after tokenization%s" % context)
            pairs.append((p, t))

        out: list = []
        cur_batch = self.batch
        retried = False
        i = 0
        while i < len(pairs):
            chunk = pairs[i:i + cur_batch]
            try:
                scores = self._score_chunk(chunk)
            except Exception as exc:
                if not self._is_oom(exc):
                    raise
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
                if retried or cur_batch <= 1:
                    raise ScorerOOM(
                        "OOM after halved-batch retry at batch=%d%s (%d pairs)"
                        % (cur_batch, context, len(chunk)))
                cur_batch = max(1, cur_batch // 2)  # one halving retry, then give up
                retried = True
                continue
            for j, s in enumerate(scores):
                if not math.isfinite(s):
                    raise ValueError(
                        "non-finite logprob%s (pair %d): %r" % (context, i + j, s))
            out.extend(scores)
            self.total_scored += len(chunk)
            i += len(chunk)
        return out

    # ---- transition scoring over a GsdSpace -----------------------------------------
    def score_transitions(self, item: dict, space, template: str = "main",
                          stage: Optional[str] = None) -> dict:
        """Score EVERY transition of ``space`` (plus the unique root line) under the frozen
        GSD template and return the dual-normalization edge weights:

            {"raw":  {(t, canon_prev, canon_next): raw logprob sum, ...,
                      (0, None, canon_root): root path constant},
             "lse":  same keys, log-softmax over each fixed (t, canon_prev) candidate set
                     (the MAIN normalization; singleton sets — root, identity layers — are
                     exactly 0.0),
             "n_scored": forwards actually run this call (0 == full cache hit)}

        Deterministic job order (t ascending, canon_prev sorted, edges canon-sorted);
        cache-hit rows are never re-scored; fresh rows are appended per chunk.
        """
        if template not in _TEMPLATES:
            raise ValueError("unknown template %r" % (template,))
        stage = self.stage if stage is None else stage
        item_id = item.get("id") if isinstance(item, dict) else None
        context = " (item %r)" % (item_id,)
        tmpl_sha = _TEMPLATES[template][1]

        by_canon = [{belief_schema.canon_state(s): s for s in layer}
                    for layer in space.layers]
        root_canon = belief_schema.canon_state(space.root)

        # ---- enumerate jobs: (cache_key, A_key, prefix, target) ----------------------
        jobs = [(
            (item_id, stage, GSD_ROOT_TEMPLATE_SHA256, 0, "", _sha256(root_canon)),
            (0, None, root_canon),
            build_prefix(item, space, 0, None, template),
            build_target(0, space.root),
        )]
        for t in range(1, space.T):
            for canon_prev in sorted(by_canon[t - 1]):
                prefix = build_prefix(item, space, t, by_canon[t - 1][canon_prev], template)
                for canon_next, _mask in space.trans[(t, canon_prev)]:
                    jobs.append((
                        (item_id, stage, tmpl_sha, t,
                         _sha256(canon_prev), _sha256(canon_next)),
                        (t, canon_prev, canon_next),
                        prefix,
                        build_target(t, by_canon[t][canon_next]),
                    ))

        # ---- score the cache misses, appending per chunk (resume granularity) -------
        pending = [job for job in jobs if job[0] not in self._cache]
        for start in range(0, len(pending), self.batch):
            chunk = pending[start:start + self.batch]
            scores = self.score_batch([j[2] for j in chunk], [j[3] for j in chunk],
                                      context=context)
            for (key, _akey, _p, _t), lp in zip(chunk, scores):
                rec = {"item_id": key[0], "stage": key[1], "template_sha": key[2],
                       "layer": key[3], "sprev_sha": key[4], "snext_sha": key[5],
                       "lp": lp}
                if self.cache_path:
                    sc.append_record(self.cache_path, rec)
                self._cache[key] = lp

        # ---- assemble raw + lse ------------------------------------------------------
        A_raw: dict = {}
        for key, akey, _p, _t in jobs:
            lp = self._cache[key]
            if not math.isfinite(lp):  # a poisoned cache row must not decode silently
                raise ValueError("non-finite cached logprob%s at %r" % (context, akey))
            A_raw[akey] = lp

        groups: dict = {}
        for (t, cp, cn), lp in A_raw.items():
            groups.setdefault((t, cp), []).append((cn, lp))
        A_lse: dict = {}
        for (t, cp), members in groups.items():
            m = max(lp for _cn, lp in members)
            denom = m + math.log(sum(math.exp(lp - m) for _cn, lp in members))
            for cn, lp in members:
                A_lse[(t, cp, cn)] = lp - denom
        return {"raw": A_raw, "lse": A_lse, "n_scored": len(pending)}


# ======================================================================================
# Production factory (offline HF load; sc_core convention). NOT unit-tested — L1 only.
# ======================================================================================
def load_scorer(model_name: str, device: str = "cuda", cache_path: Optional[str] = None,
                batch: int = 8, stage: str = "g1", dtype=None) -> TFScorer:
    """Build a production ``TFScorer`` over a locally-cached HF checkpoint (bf16, sdpa,
    offline — the exact ``sc_core.HFEmitter`` loading convention, including the refs/main
    snapshot-symlink gotcha being an operational fix, not a code path)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype if dtype is not None else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, local_files_only=True,
            attn_implementation="sdpa").to(device).eval()
    except (ValueError, ImportError):  # backbone without sdpa support -> default attn
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, local_files_only=True).to(device).eval()
    return TFScorer(model, tok, device=device, cache_path=cache_path,
                    batch=batch, stage=stage)
