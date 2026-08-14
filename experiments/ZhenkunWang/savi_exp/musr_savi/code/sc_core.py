"""sc_core — MuSR-cant Phase A sampling / voting / statistics core.

The single home for every generation and statistical primitive of the falsification funnel:
the frozen prompt, the Emitter abstraction, the shard-invariant seed protocol, prefix-rung
majority voting, the Wilson score interval, the sticky criterion, the token ledger, and the
resumable JSONL sample cache. Phase A does NOT test any decode method — this module only
produces the raw sampling evidence that ``stages.py`` (Subtask 3) turns into triage /
attrition / certification verdicts.

CODE SEPARATION: this module NEVER imports test code. ``torch`` / ``transformers`` are
imported lazily inside ``HFEmitter`` only, so the whole module (and every statistical
primitive) imports and runs with ZERO GPU and no heavy deps. The test-side ``FakeEmitter``
lives in ``tests/test_sc_core.py``.


==========================================================================================
RECORD SCHEMA  (consumed by stages.py — field names/types are a contract)
==========================================================================================

**Sample record** — the atomic unit produced by ``draw_samples``, stored one-per-line in the
JSONL cache, and fed (as an ordered list) to ``vote_at_rung`` / ``sticky``. A plain ``dict``:

    {
      "subtask":      str,        # subtask name, e.g. "murder_mystery"     | cache-key part
      "item_id":      str,        # MuSR item id                            | cache-key part
      "sample_idx":   int,        # 0-based PREFIX index within (item,seed_tag); cache-key part
      "seed_tag":     str,        # sampling regime: "base" | "seed2" | "facts" | "greedy" | ...
      "seed":         int,        # the actual RNG seed used (== sample_seed(base,subtask,id,idx))
      "answer_idx":   int | None, # parsed 0-based option index; None = parse-fail/out-of-range
      "correct":      bool,       # answer_idx == gold_idx (None -> False)
      "gold_idx":     int,        # gold 0-based option index (echoed for self-containment)
      "n_options":    int,        # number of options for this item (binary iff == 2)
      "n_new_tokens": int,        # generated token-id count (HF tokenizer) — token-ledger unit
    }

  * ``sample_idx`` gives the PREFIX order: ``vote_at_rung(records, N)`` sorts by ``sample_idx``
    then takes the first N, so any prefix is a pure function of the lowest-N sample_idxs
    (independent of siblings drawn later / of shard grouping — see ``sample_seed``).
  * The cache key is the 4-tuple ``(subtask, item_id, sample_idx, seed_tag)``.

**VoteResult** — dataclass returned by ``vote_at_rung`` (headline 4 fields first, then the
supporting counts ``sticky`` / stages read):

    VoteResult(
      mode_idx:     int | None,   # majority option (None-bucket may win); tie -> lowest option
      mode_correct: bool,         # mode_idx == gold_idx AND not a tie (tie => wrong, convention)
      p_hat:        float,        # sample CORRECTNESS rate = k_correct / n  (design's p̂)
      margin:       float,        # mode vote margin = (mode_count - runner_up_count) / n
      n:            int,          # number of samples actually voted (== min(N, len(records)))
      k_correct:    int,          # count of correct samples in the first n
      mode_count:   int,          # votes for mode_idx
      tie:          bool,         # True iff >=2 distinct options share the top count
    )


==========================================================================================
FROZEN CONVENTIONS
==========================================================================================
* Option numbering shown to the model is 1-BASED (options rendered "1. ...", "2. ...").
  ``parse_answer`` maps the parsed number k back to the 0-based index k-1; valid iff
  1 <= k <= n_options. MuSR ``choices`` is a list, indexed 0-based internally.
* PROMPT_TEMPLATE is frozen; its sha256 is snapshot-tested (freezing = testing).
* Seed = sha256(f"{base_seed}|{subtask}|{item_id}|{batch_idx}") truncated to 63 bits — a pure
  function of (base_seed, subtask, item, batch_idx), hence SHARD-INVARIANT.
* Temperature T=1.0, max_new_tokens=512 (design §1 / Subtask 2).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

# ======================================================================================
# Frozen prompt (English — MuSR is English). sha256 snapshot-tested in the test file.
# ======================================================================================
PROMPT_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    'Think step by step, then finish your reply with a line "ANSWER: <option number>".'
)

# Generation defaults (design §1 / Subtask 2).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_NEW_TOKENS = 2048
# Pure-temperature control: top_p/top_k pinned so sampling is identical across the model ladder
# (defeats per-checkpoint generation_config.json truncation). Frozen; echoed in results config.
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0


def render_numbered_choices(choices: Sequence[str]) -> str:
    """Render ``choices`` as a 1-based numbered block ("1. ..\n2. .."). The model sees 1-based
    numbers; ``parse_answer`` maps them back to 0-based indices."""
    return "\n".join(f"{i}. {c}" for i, c in enumerate(choices, start=1))


def format_prompt(item: dict) -> str:
    """Fill PROMPT_TEMPLATE from a normalized item (keys: narrative/question/choices)."""
    return PROMPT_TEMPLATE.format(
        narrative=item["narrative"],
        question=item["question"],
        numbered_choices=render_numbered_choices(item["choices"]),
    )


# ======================================================================================
# Gold-facts block (A2 knowledge-vs-commitment separator, design §5.2). FROZEN + snapshot-
# tested. Rendered from the CLEAN structured facts in ``tree_raw['intermediate_data']`` —
# NOT the raw nested tree JSON dump. Because ``intermediate_data`` is heterogeneous across the
# three subtasks (design §5.2 allows a per-subtask format as long as each is frozen), one
# renderer per subtask; every renderer is deterministic (source-list order is stable; dict
# iteration is sorted). The block is the gold REASONING INPUTS (ground clues / observation log
# / skill matrix) and deliberately WITHHOLDS the derived answer node (the murderer verdict,
# each person's belief, the optimal pairing) so the facts arm stays a genuine
# knowledge-vs-commitment probe rather than a trivial answer hand-off.
#
# Probed against the real 2026-07-05 download (data/{subtask}.json, tree availability 100%):
#   murder_mystery   -> intermediate_data[0]['suspect_info']  (per-suspect reasoning trees)
#   object_placements-> intermediate_data[0]['events']        (chronological observation log)
#   team_allocation  -> intermediate_data[0]['tasks'/'matrix'](per-person skill/cooperation)
# Each renderer also tolerates the trimmed fixture shapes (tests/fixtures/musr_tiny.json):
# a suspect may carry name+has_means/has_motive/has_opportunity flags instead of a tree;
# ``events`` may be a flat list of strings; ``matrix`` may omit ``cooperation``.
# tree_raw absent / empty / an unknown subtask -> FACTS_NONE (deterministic, never crashes).
FACTS_PREAMBLE = (
    "Below are the established gold facts for this scenario, drawn from its reference "
    "solution. Treat every statement as ground truth, then answer the question."
)
FACTS_NONE = "No gold facts are available for this item."
# Wrapper that composes the facts block with the standard question prompt. Frozen; its sha256
# is echoed in every results.json config block (config's ``facts_template_sha256``).
FACTS_TEMPLATE = (
    "{facts_preamble}\n\n"
    "=== GOLD FACTS ===\n"
    "{facts}\n"
    "=== END GOLD FACTS ===\n\n"
    "{base_prompt}"
)


def _intermediate_data_head(item: dict):
    """Return the first ``intermediate_data`` fact-dict of ``item['tree_raw']`` or None.

    Accepts both the real shape (``intermediate_data`` = list, take element 0) and a dict.
    """
    tr = item.get("tree_raw")
    if not isinstance(tr, dict):
        return None
    idata = tr.get("intermediate_data")
    if isinstance(idata, list):
        return idata[0] if idata else None
    if isinstance(idata, dict):
        return idata
    return None


def _tree_leaf_values(tree: dict) -> list:
    """Ordered, de-duplicated leaf ``value`` strings of a reasoning tree (ground clues).

    A leaf = a node with no ``children``; its ``value`` is an atomic narrative/commonsense
    fact. Intermediate nodes (deductive claims like "X has a means") and the root verdict are
    NOT emitted, so the block carries ground facts, not the derived conclusion.
    """
    out = []

    def _rec(node):
        if not isinstance(node, dict):
            return
        children = node.get("children") or []
        if not children:
            v = (node.get("value") or "").strip()
            if v:
                out.append(v)
            return
        for c in children:
            _rec(c)

    for rs in (tree.get("root_structure") or []):
        _rec(rs)
    seen, deduped = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _facts_murder_mystery(idata: dict) -> Optional[str]:
    """Per-suspect ground clues (design's ``suspect_info``). Frozen format."""
    suspects = idata.get("suspect_info")
    if not isinstance(suspects, list) or not suspects:
        return None
    blocks = []
    for s in suspects:
        if not isinstance(s, dict):
            continue
        tree = s.get("used_tree") or s.get("tree")   # narrative-realized tree preferred
        if isinstance(tree, dict):
            root = ((tree.get("root_structure") or [{}])[0] or {}).get("value", "") or ""
            root = root.strip()
            name = root.split(" is the murderer")[0].strip() if " is the murderer" in root else root
            facts = _tree_leaf_values(tree)
        elif "name" in s:                            # trimmed fixture shape
            name = str(s["name"])
            facts = [k.replace("has_", "has ").replace("_", " ")
                     for k in ("has_means", "has_motive", "has_opportunity") if s.get(k)]
        else:
            continue
        lines = ["Regarding %s:" % (name or "this suspect")]
        lines += ["- " + f for f in facts]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else None


def _facts_object_placements(idata: dict) -> Optional[str]:
    """Chronological observation log (design's ``events``). Frozen format."""
    events = idata.get("events")
    if not isinstance(events, list) or not events:
        return None
    flat = []
    for e in events:
        if isinstance(e, list):
            flat.extend(str(x) for x in e)
        else:
            flat.append(str(e))
    if not flat:
        return None
    lines = ["Observations, in the order they happened:"]
    lines += ["%d. %s" % (i + 1, s) for i, s in enumerate(flat)]
    return "\n".join(lines)


def _facts_team_allocation(idata: dict) -> Optional[str]:
    """Tasks + per-person skill/cooperation ratings (design's ``matrix``). Frozen format."""
    tasks = idata.get("tasks") or []
    matrix = idata.get("matrix") or {}
    if not isinstance(matrix, dict) or not matrix:
        return None
    lines = []
    if tasks:
        lines.append("Tasks: " + ", ".join(str(t) for t in tasks))
    lines.append("Per-person ratings for the tasks (higher is a better fit):")
    for person in sorted(matrix.keys()):                 # sorted dict iteration -> deterministic
        d = matrix[person] or {}
        parts = ["%s=%s" % (k, d[k]) for k in sorted(d.keys())] if isinstance(d, dict) else [str(d)]
        lines.append("- %s: %s" % (person, ", ".join(parts)))
    return "\n".join(lines)


_FACTS_RENDERERS = {
    "murder_mystery": _facts_murder_mystery,
    "object_placements": _facts_object_placements,
    "team_allocation": _facts_team_allocation,
}


def facts_block(item: dict) -> str:
    """Gold-facts block for the A2 knowledge/commitment arm (design §5.2). FROZEN.

    Renders the clean structured facts of ``item['tree_raw']['intermediate_data']`` per
    subtask (see the section header). Deterministic. Any missing/heterogeneous/empty structure
    degrades to ``FACTS_NONE`` instead of raising, so the pipeline never crashes on a
    tree-absent item (real tree availability is 100%, but the smoke/fabricated path and the
    one tree-null fixture item exercise this fallback).
    """
    idata = _intermediate_data_head(item)
    if not isinstance(idata, dict):
        return FACTS_NONE
    renderer = _FACTS_RENDERERS.get(item.get("subtask"))
    if renderer is None:
        return FACTS_NONE
    try:
        block = renderer(idata)
    except Exception:                                    # never let a malformed tree crash a run
        block = None
    return block if block else FACTS_NONE


def format_facts_prompt(item: dict) -> str:
    """Compose the A2 facts-fed prompt: gold-facts preamble + block, then the original
    question (via the frozen ``PROMPT_TEMPLATE``). This is the exact string the facts arm
    samples over (seed_tag="facts"). Frozen via ``FACTS_TEMPLATE`` (sha in the config block)."""
    return FACTS_TEMPLATE.format(
        facts_preamble=FACTS_PREAMBLE,
        facts=facts_block(item),
        base_prompt=format_prompt(item),
    )


# ======================================================================================
# Answer parsing
# ======================================================================================
# Case-insensitive "ANSWER", optional space, ASCII ':' or full-width '：', optional space,
# then the digits. Full-width digits are normalized to ASCII before matching (robustness).
_ANSWER_RE = re.compile(r"ANSWER\s*[:：]\s*(\d+)", re.IGNORECASE)
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def parse_answer(text: Optional[str], n_options: Optional[int] = None) -> Optional[int]:
    """Return the LAST ``ANSWER: <k>`` match mapped to a 0-based option index, else None.

    Convention (frozen): the prompt shows 1-BASED numbers, so the parsed k maps to index k-1.
    Missing marker, k < 1, or (when ``n_options`` is given) k > n_options -> None (counts as
    wrong / charged to budget). Tolerant of case and ASCII/full-width colon; full-width digits
    are normalized. Repeated ANSWER lines: the LAST wins.
    """
    if not text:
        return None
    norm = text.translate(_FULLWIDTH_DIGITS)
    hits = _ANSWER_RE.findall(norm)
    if not hits:
        return None
    k = int(hits[-1])            # last match wins
    if k < 1:                    # 0 / negative invalid under 1-based numbering
        return None
    idx = k - 1
    if n_options is not None and idx >= n_options:
        return None              # out of range
    return idx


# ======================================================================================
# Voting  (prefix-rung majority)
# ======================================================================================
@dataclass
class VoteResult:
    """Majority-vote read-out over a sample-record prefix. See module docstring RECORD SCHEMA."""
    mode_idx: Optional[int]
    mode_correct: bool
    p_hat: float
    margin: float
    n: int
    k_correct: int
    mode_count: int
    tie: bool


def _ordered_prefix(records: Sequence[dict], N: int) -> list:
    """First ``N`` records in ``sample_idx`` order (stable for records lacking the field), so a
    rung is a pure function of the lowest-N sample_idxs regardless of input ordering."""
    ordered = sorted(records, key=lambda r: r.get("sample_idx", 0))
    return ordered[:N]


def vote_at_rung(records: Sequence[dict], N: int) -> VoteResult:
    """Majority vote over the first ``N`` samples (PREFIX property). Tie -> wrong (convention).

    The None-bucket (parse failures) is a votable option: if parse-fails are the plurality the
    mode is ``None`` and ``mode_correct`` is False. ``p_hat`` is the sample CORRECTNESS rate
    (design's p̂), NOT the mode share; ``mode_count`` (the mode's share numerator) is returned
    separately for the sticky criterion.
    """
    prefix = _ordered_prefix(records, N)
    n = len(prefix)
    if n == 0:
        return VoteResult(None, False, 0.0, 0.0, 0, 0, 0, False)

    gold = prefix[0].get("gold_idx")
    k_correct = sum(1 for r in prefix if r.get("correct"))
    # Counter over answer_idx; None is a real bucket (a de-facto "failed to commit" vote).
    counts = Counter(r.get("answer_idx") for r in prefix)
    top = max(counts.values())
    winners = [opt for opt, c in counts.items() if c == top]
    tie = len(winners) > 1

    # Deterministic mode_idx: prefer a real option over None on a tie, then lowest index.
    def _key(opt):
        return (opt is None, opt if opt is not None else 0)
    mode_idx = sorted(winners, key=_key)[0]
    mode_count = counts[mode_idx]

    second = 0
    if len(counts) > 1:
        second = sorted(counts.values(), reverse=True)[1]
    margin = (mode_count - second) / n

    mode_correct = (not tie) and (mode_idx is not None) and (mode_idx == gold)
    return VoteResult(
        mode_idx=mode_idx, mode_correct=mode_correct,
        p_hat=k_correct / n, margin=margin,
        n=n, k_correct=k_correct, mode_count=mode_count, tie=tie,
    )


# ======================================================================================
# Wilson score interval
# ======================================================================================
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI (default z=1.96) for ``k`` successes in ``n`` trials, clamped [0,1].

    n == 0 -> (0.0, 1.0) (no information -> widest interval). Verified numerically against the
    known value: wilson_ci(10, 32) == (0.17952, 0.48567) (the plan's [0.180, 0.467] quotes a
    correct lower bound but an upper that belongs to Wilson(11,36); we use the true value).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


# ======================================================================================
# Sticky criterion  (design §3 / Subtask 2)
# ======================================================================================
def sticky(item_records: Sequence[dict], N: int) -> bool:
    """Is the item STABLY stuck on a WRONG answer over its first ``N`` samples?

    * mode correct  -> not sticky (nothing stuck-wrong).
    * binary item (n_options == 2): correctness Wilson UPPER bound < 0.5 (confidently sub-half).
    * multi-choice: correctness-rate Wilson UPPER < mode-wrong-rate Wilson LOWER (the correct
      answer's rate is confidently below the dominant wrong answer's rate).
    """
    vr = vote_at_rung(item_records, N)
    if vr.n == 0 or vr.mode_correct:
        return False
    n_options = _ordered_prefix(item_records, N)[0].get("n_options", 0)

    corr_lo, corr_hi = wilson_ci(vr.k_correct, vr.n)
    if n_options == 2:
        return corr_hi < 0.5
    mode_lo, mode_hi = wilson_ci(vr.mode_count, vr.n)
    return corr_hi < mode_lo


# ======================================================================================
# Seed protocol  (shard-invariant)
# ======================================================================================
def sample_seed(base_seed: int, subtask: str, item_id: str, batch_idx: int) -> int:
    """Deterministic per-sample RNG seed = sha256(f"{base_seed}|{subtask}|{item_id}|{batch_idx}").

    A pure function of its four arguments -> SHARD-INVARIANT: sample ``batch_idx`` of an item
    gets the same seed no matter how items are split across shards / processes, so resume and
    shard-merge reproduce the identical sample set. Returns a non-negative 63-bit int (first 8
    digest bytes, big-endian) — the capability-envelope ``batch_seed`` idiom, extended with the
    subtask field (Phase A has three subtasks with potentially colliding item ids).
    """
    payload = f"{base_seed}|{subtask}|{item_id}|{batch_idx}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") >> 1


# ======================================================================================
# JSONL sample cache  (atomic append / dedup last-write-wins / resume)
# ======================================================================================
def cache_key(rec: dict) -> tuple:
    """The 4-tuple identity of a sample record: (subtask, item_id, sample_idx, seed_tag)."""
    return (rec["subtask"], rec["item_id"], rec["sample_idx"], rec["seed_tag"])


def _ensure_trailing_newline(path: str) -> None:
    """Guard the append path against a torn final line from a previously killed process."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
    if last != b"\n":
        with open(path, "a") as f:
            f.write("\n")


def append_record(path: str, rec: dict) -> None:
    """Atomically append one record as a JSON line (creates parent dirs; heals torn tails)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    _ensure_trailing_newline(path)
    line = json.dumps(rec) + "\n"           # single write < PIPE_BUF is atomic on POSIX append
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def load_cache(path: str) -> dict:
    """Load the JSONL cache into ``{cache_key: record}`` with LAST-write-wins dedup.

    Blank and malformed (torn) lines are skipped, never fatal. Missing file -> empty dict.
    """
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
            except json.JSONDecodeError:
                continue
            try:
                out[cache_key(rec)] = rec       # last write wins
            except (KeyError, TypeError):
                continue
    return out


def completed_idxs(cache: dict, subtask: str, item_id: str, seed_tag: str) -> set:
    """Set of ``sample_idx`` already present for (subtask, item, seed_tag) — the resume skip-set."""
    return {
        k[2] for k, r in cache.items()
        if r.get("subtask") == subtask and r.get("item_id") == item_id
        and r.get("seed_tag") == seed_tag
    }


def cached_records(cache: dict, subtask: str, item_id: str, seed_tag: str) -> list:
    """The (subtask, item, seed_tag) records, ``sample_idx``-ordered — ready for vote_at_rung."""
    recs = [
        r for r in cache.values()
        if r.get("subtask") == subtask and r.get("item_id") == item_id
        and r.get("seed_tag") == seed_tag
    ]
    return sorted(recs, key=lambda r: r.get("sample_idx", 0))


# ======================================================================================
# Emitter protocol + generation primitive
# ======================================================================================
@dataclass
class GenOut:
    """One generation: the decoded ``text`` and its generated token-id count (ledger unit)."""
    text: str
    n_new_tokens: int


@runtime_checkable
class Emitter(Protocol):
    """Generation surface consumed by ``draw_samples`` (dependency-injected).

    Contract: ``generate`` returns one ``GenOut`` per input position, with ``GenOut[i]`` a
    function of ``(prompts[i], seeds[i])``. ``FakeEmitter`` (tests) honours this purely;
    ``HFEmitter`` approximates it (GPU batching shares the global RNG within a mini-batch — the
    exact tokens are not bit-independent across groupings, but the seed still shapes the draw
    and the answer STATISTICS are what Phase A measures). ``greedy=True`` disables sampling
    (deterministic readout for the A2 facts arm).
    """
    def generate(self, prompts: Sequence[str], seeds: Sequence[int], *,
                 greedy: bool = False, max_new_tokens: Optional[int] = None) -> list: ...


class EmitterOOM(RuntimeError):
    """Raised by HFEmitter when a batch OOMs even after one halved-batch retry. The caller
    (runner) records ``oom_skipped`` for the offending item and moves on (design §7)."""


def draw_samples(emitter, item: dict, subtask: str, n: int, base_seed: int, *,
                 seed_tag: str = "base", batch: int = 64, greedy: bool = False,
                 prompt: Optional[str] = None, existing_idxs=None,
                 max_new_tokens: Optional[int] = None, keep_text_chars: int = 0) -> list:
    """Draw ``n`` samples for ``item`` through ``emitter`` and return sample records (RECORD
    SCHEMA above). Pure of I/O — the caller appends the returned records to the JSONL cache.

    * Seed of sample ``i`` = ``sample_seed(base_seed, subtask, item['id'], i)`` (each sample is
      its own seed-"batch" -> shard-invariant + resumable at sample granularity).
    * ``existing_idxs`` (resume): those sample_idxs are skipped; only the missing ones are drawn.
    * ``prompt`` overrides the standard ``format_prompt(item)`` (e.g. the facts-arm prompt).
    * ``batch`` bounds how many prompts are handed to ``emitter.generate`` per call.
    * ``max_new_tokens`` (a3 additive, DEFAULT ``None``): per-call generation cap threaded to
      ``emitter.generate`` (the S1 arm pins its own truncation-guard cap). ``None`` reproduces the
      prior behaviour exactly — the emitter falls back to its own ``max_new_tokens`` default.
    * ``keep_text_chars`` (a3 additive, DEFAULT ``0``): when > 0, each record gains a
      ``text_head`` = ``out.text[:keep_text_chars]`` (S1 scaffold-compliance diagnostic). When 0
      (default) NO ``text_head`` key is added, so the RECORD SCHEMA is byte-for-byte unchanged.
      Never part of the ``cache_key`` (4-tuple), so the JSONL cache stays compatible either way.
    """
    existing = set(existing_idxs or ())
    todo = [i for i in range(n) if i not in existing]
    if not todo:
        return []

    the_prompt = prompt if prompt is not None else format_prompt(item)
    item_id = item["id"]
    gold_idx = item["gold_idx"]
    n_options = item.get("n_options", len(item["choices"]))

    recs = []
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        seeds = [sample_seed(base_seed, subtask, item_id, i) for i in chunk]
        prompts = [the_prompt] * len(chunk)
        outs = emitter.generate(prompts, seeds, greedy=greedy, max_new_tokens=max_new_tokens)
        for i, s, out in zip(chunk, seeds, outs):
            ans = parse_answer(out.text, n_options)
            rec = {
                "subtask": subtask,
                "item_id": item_id,
                "sample_idx": i,
                "seed_tag": seed_tag,
                "seed": s,
                "answer_idx": ans,
                "correct": (ans is not None and ans == gold_idx),
                "gold_idx": gold_idx,
                "n_options": n_options,
                "n_new_tokens": int(out.n_new_tokens),
            }
            if keep_text_chars > 0:
                rec["text_head"] = out.text[:keep_text_chars]
            recs.append(rec)
    recs.sort(key=lambda r: r["sample_idx"])
    return recs


# ======================================================================================
# HFEmitter — the real (GPU) generation surface. Lazy torch/transformers import.
# ======================================================================================
class HFEmitter:
    """Real-model Emitter over a locally-cached HF checkpoint (offline). NOT exercised by the
    unit tests (FakeEmitter substitutes on CPU); it is validated at L1 on the real GPU.

    Loads with ``HF_HUB_OFFLINE=1`` + ``local_files_only=True`` (design §8 offline convention;
    if a refs/main-hash load fails, apply the known snapshot-symlink fix operationally). Prompts
    arrive as rendered PROMPT_TEMPLATE strings; each is wrapped as a single user turn via the
    chat template (``enable_thinking=False`` for Qwen3, tolerated-absent elsewhere). Sampling is
    T=1.0 (``greedy=True`` -> do_sample=False). ``n_new_tokens`` = generated token-id length up
    to and including the first EOS. OOM: one halved-batch retry, then ``EmitterOOM`` (design §7).
    """

    def __init__(self, model_name: str, device: str = "cuda",
                 temperature: float = DEFAULT_TEMPERATURE, batch: int = 16,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, dtype=None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch                                    # lazy: keep module CPU/GPU-free at import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.batch = batch
        self.max_new_tokens = max_new_tokens
        dtype = dtype if dtype is not None else torch.bfloat16

        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.tok.padding_side = "left"                  # left-pad for batched decode
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, local_files_only=True,
                attn_implementation="sdpa").to(device).eval()
        except (ValueError, ImportError):               # backbone without sdpa support -> default
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, local_files_only=True).to(device).eval()
        self.eos_id = self.tok.eos_token_id

    # --- prompt -> chat-templated input ids ------------------------------------------
    def _encode(self, prompts):
        rendered = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            try:
                s = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:                            # model family without enable_thinking
                s = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            rendered.append(s)
        enc = self.tok(rendered, return_tensors="pt", padding=True).to(self.device)
        return enc

    def _gen_len(self, gen_ids) -> int:
        """Generated token count up to and INCLUDING the first EOS (full length if no EOS);
        robust to right-padding after EOS in a batched decode."""
        ids = gen_ids.tolist()
        if self.eos_id is not None:
            for j, t in enumerate(ids):
                if int(t) == int(self.eos_id):
                    return j + 1
        return len(ids)

    def _generate_chunk(self, prompts, seeds, greedy, max_new_tokens):
        torch = self._torch
        enc = self._encode(prompts)
        prompt_len = int(enc["input_ids"].shape[1])
        # One shared manual_seed per chunk (global RNG); mixed from the chunk's seed list so the
        # (prompts, seeds) -> outputs mapping is reproducible for a fixed grouping.
        mix = 0
        for s in seeds:
            mix = (mix * 1000003 + int(s)) & 0x7FFFFFFFFFFFFFFF
        torch.manual_seed(mix)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(not greedy),
            num_return_sequences=1,
            pad_token_id=(self.eos_id if self.eos_id is not None else self.tok.pad_token_id),
        )
        if not greedy:                                   # temperature only meaningful when sampling
            gen_kwargs["temperature"] = self.temperature
            # Pin top_p/top_k so sampling is PURE temperature and identical across the A2 scale
            # ladder. Without this, HF generate() merges each checkpoint's generation_config.json
            # (Qwen ships top_p=0.8, top_k=20; DeepSeek differs), silently truncating the
            # distribution per-model -> a confound for the ladder gate (design §5). Frozen in PREREG.
            gen_kwargs["top_p"] = DEFAULT_TOP_P
            gen_kwargs["top_k"] = DEFAULT_TOP_K
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        results = []
        for i in range(int(out.shape[0])):
            gen = out[i, prompt_len:]
            n_tok = self._gen_len(gen)
            text = self.tok.decode(gen[:n_tok], skip_special_tokens=True)
            results.append(GenOut(text=text, n_new_tokens=int(n_tok)))
        return results

    def generate(self, prompts, seeds, *, greedy=False, max_new_tokens=None):
        assert len(prompts) == len(seeds), "prompts/seeds must be parallel"
        mnt = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        prompts = list(prompts)
        seeds = list(seeds)
        results = []
        cur_batch = self.batch
        i = 0
        retried = False
        while i < len(prompts):
            chunk_p = prompts[i:i + cur_batch]
            chunk_s = seeds[i:i + cur_batch]
            try:
                results.extend(self._generate_chunk(chunk_p, chunk_s, greedy, mnt))
                i += len(chunk_p)
            except (self._torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # Some CUDA OOMs surface as a generic RuntimeError("... out of memory ...")
                # rather than the typed error; catch both. Re-raise non-OOM RuntimeErrors.
                if not isinstance(exc, self._torch.cuda.OutOfMemoryError) \
                        and "out of memory" not in str(exc).lower():
                    raise
                self._torch.cuda.empty_cache()
                if retried or cur_batch <= 1:
                    raise EmitterOOM(
                        f"OOM after halved-batch retry at batch={cur_batch} "
                        f"(model={self.model_name}, {len(chunk_p)} prompts)")
                cur_batch = max(1, cur_batch // 2)       # one halving retry, then give up
                retried = True
        return results
