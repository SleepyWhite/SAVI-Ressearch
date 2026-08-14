"""MuSR data acquisition + audit module — Phase A, Subtask 1.

Loads the three MuSR subtasks into a single normalized item schema, produces an audit
report, and freezes the A0 triage set (fixed item ids + per-item chance line).

CORE MODULE. It must NEVER import from tests/ or any validation code. Tests observe it
externally (dependency-injected token counter; the fixture is passed in, never read here).

--------------------------------------------------------------------------------
REAL schema discovered by direct probing on 2026-07-05 (both download paths):

GitHub raw  (fallback path)
    https://raw.githubusercontent.com/Zayne-sprague/MuSR/main/datasets/{name}.json
    A JSON LIST of *examples*, each:
        {"context": <narrative str>,
         "questions": [ {"question": str,
                         "answer": int,            # gold CHOICE INDEX
                         "choices": [str, ...],
                         "intermediate_trees": [ ... ],   # gold reasoning tree
                         "intermediate_data": [ ... ]}, ... ]}
    NESTED. Carries the gold reasoning trees. Flattening (example, question) pairs gives:
        murder_mystery   250 examples x 1 q  = 250 items   (choices: 2, binary)
        object_placements 64 examples x 4 q  = 256 items   (choices: 2..5, vary per item)
        team_allocation  250 examples x 1 q  = 250 items   (choices: 3)
    Tree availability = 100% on this path.

HuggingFace  (primary path)  datasets.load_dataset("TAUR-Lab/MuSR", <config>)
    CSV-backed (murder_mystery.csv / object_placements.csv / team_allocation.csv),
    ALREADY FLATTENED to one row per question, columns:
        narrative (str), question (str), choices (STR repr of a python list),
        answer_index (int/str gold index), answer_choice (str)
    IMPORTANT: the HF CSV export has NO intermediate_trees / intermediate_data columns,
    i.e. the gold reasoning tree is ABSENT on the HF path. The A2 facts arm needs the
    tree, so download() enriches HF rows with trees from GitHub (verified by index +
    question/choices match); if enrichment can't be verified, tree_raw stays None and
    audit() flags tree_available=false so PREREG can decide the facts-arm fallback.

Normalized item schema (unit of everything downstream = one (context, question) pair):
    {"id": str,               # f"{subtask}-{example_idx:04d}-q{question_idx}"  (sortable)
     "subtask": str,          # one of SUBTASKS
     "narrative": str,
     "question": str,
     "choices": list[str],
     "gold_idx": int,         # 0 <= gold_idx < len(choices)
     "tree_raw": dict|None}   # {"intermediate_trees":..., "intermediate_data":...} or None
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import ast
import datetime as _dt
import hashlib
import json
import os
import random
import urllib.request

# ---- frozen constants ---------------------------------------------------------

SUBTASKS = ("murder_mystery", "object_placements", "team_allocation")
EXPECTED_COUNTS = {"murder_mystery": 250, "object_placements": 256, "team_allocation": 250}
# Number of questions per example on the (already-flattened) HF CSV path, used only to
# reconstruct (example_idx, question_idx) so HF ids match the GitHub ids. Verified constant.
HF_QUESTIONS_PER_EXAMPLE = {"murder_mystery": 1, "object_placements": 4, "team_allocation": 1}

GITHUB_RAW = "https://raw.githubusercontent.com/Zayne-sprague/MuSR/main/datasets/{name}.json"
HF_DATASET = "TAUR-Lab/MuSR"

QWEN3_4B_MODEL = "Qwen/Qwen3-4B"
CONTEXT_TOKEN_LIMIT = 30000       # narratives above this -> hard error (context overflow)
BASE_SEED = 20260705
TRIAGE_N = 80
MAJORITY_WARN = 0.6               # majority answer-choice share above this -> warning
TREE_AVAIL_MIN = 0.95             # per-subtask tree availability below this -> tree_available=false

_HTTP_HEADERS = {"User-Agent": "musr-cant-data/1.0"}


# ---- normalization ------------------------------------------------------------

def _parse_choices(raw_choices):
    """Return list[str]. Accepts a real list, or a str repr of a list (HF CSV)."""
    if isinstance(raw_choices, str):
        try:
            parsed = ast.literal_eval(raw_choices)
        except (ValueError, SyntaxError) as e:
            raise ValueError("could not parse choices string repr: %r (%s)" % (raw_choices, e))
    else:
        parsed = raw_choices
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("choices did not resolve to a list: %r" % (parsed,))
    return [str(c) for c in parsed]


def _build_tree_raw(raw):
    """Bundle gold reasoning tree + intermediate data, or None if absent/empty."""
    trees = raw.get("intermediate_trees")
    data = raw.get("intermediate_data")
    if not trees:  # None or empty list -> no usable gold tree
        return None
    return {"intermediate_trees": trees, "intermediate_data": data if data else []}


def normalize_item(raw, subtask):
    """Map ONE flattened raw question dict (GitHub- or HF-style fields) to the unified schema.

    Required raw fields (either source's name is accepted where they differ):
        narrative  <- raw['narrative'] or raw['context']
        question   <- raw['question']
        choices    <- raw['choices']       (list or str-repr of list)
        gold_idx   <- raw['answer_index'] or raw['answer']   (int, coerced from str)
        id parts   <- raw['example_idx'], raw['question_idx'] (ints; attached by the loader)
    Missing required field or unknown subtask -> hard error (raises). Never fabricates.
    """
    if subtask not in SUBTASKS:
        raise ValueError("unknown subtask %r (expected one of %r)" % (subtask, SUBTASKS))

    if "narrative" in raw and raw["narrative"] is not None:
        narrative = raw["narrative"]
    elif "context" in raw and raw["context"] is not None:
        narrative = raw["context"]
    else:
        raise KeyError("normalize_item[%s]: missing narrative/context" % subtask)

    if "question" not in raw or raw["question"] is None:
        raise KeyError("normalize_item[%s]: missing question" % subtask)
    question = str(raw["question"])

    if "choices" not in raw or raw["choices"] is None:
        raise KeyError("normalize_item[%s]: missing choices" % subtask)
    choices = _parse_choices(raw["choices"])

    if "answer_index" in raw and raw["answer_index"] is not None:
        gold_src = raw["answer_index"]
    elif "answer" in raw and raw["answer"] is not None:
        gold_src = raw["answer"]
    else:
        raise KeyError("normalize_item[%s]: missing answer_index/answer" % subtask)
    try:
        gold_idx = int(gold_src)
    except (TypeError, ValueError):
        raise ValueError("normalize_item[%s]: gold index not an int: %r" % (subtask, gold_src))

    for key in ("example_idx", "question_idx"):
        if key not in raw or raw[key] is None:
            raise KeyError("normalize_item[%s]: missing %s (loader must attach it)" % (subtask, key))
    example_idx = int(raw["example_idx"])
    question_idx = int(raw["question_idx"])

    # Gold-integrity tripwire: the HF row carries answer_choice (the gold option STRING) alongside
    # its integer index. If both are present, verify choices[gold_idx] == answer_choice so an
    # off-by-one or choice reordering on the HF path fails loudly instead of silently mislabeling
    # the gold. Only checked when gold_idx is in range (validate_item enforces range separately).
    ans_choice = raw.get("answer_choice")
    if ans_choice is not None and 0 <= gold_idx < len(choices):
        if str(choices[gold_idx]).strip() != str(ans_choice).strip():
            raise ValueError(
                "normalize_item[%s]: gold mismatch — choices[%d]=%r != answer_choice=%r"
                % (subtask, gold_idx, choices[gold_idx], ans_choice))

    item = {
        "id": "%s-%04d-q%d" % (subtask, example_idx, question_idx),
        "subtask": subtask,
        "narrative": str(narrative),
        "question": question,
        "choices": choices,
        "gold_idx": gold_idx,
        "tree_raw": _build_tree_raw(raw),
    }
    validate_item(item)  # single validation path; catches out-of-range gold from bad raw
    return item


def validate_item(item):
    """Hard schema validator. Raises ValueError/KeyError on any violation. Returns None."""
    required = ("id", "subtask", "narrative", "question", "choices", "gold_idx")
    for key in required:
        if key not in item:
            raise KeyError("validate_item: missing field %r" % key)
    if item["subtask"] not in SUBTASKS:
        raise ValueError("validate_item: bad subtask %r" % (item["subtask"],))
    if not isinstance(item["narrative"], str) or not item["narrative"]:
        raise ValueError("validate_item[%s]: narrative must be a non-empty str" % item["id"])
    if not isinstance(item["question"], str) or not item["question"]:
        raise ValueError("validate_item[%s]: question must be a non-empty str" % item["id"])
    choices = item["choices"]
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError("validate_item[%s]: choices must be a non-empty list" % item["id"])
    if not all(isinstance(c, str) for c in choices):
        raise ValueError("validate_item[%s]: every choice must be a str" % item["id"])
    gold = item["gold_idx"]
    if not isinstance(gold, int) or isinstance(gold, bool):
        raise ValueError("validate_item[%s]: gold_idx must be an int" % item["id"])
    if gold < 0 or gold >= len(choices):
        raise ValueError("validate_item[%s]: gold_idx %d out of range [0,%d)"
                         % (item["id"], gold, len(choices)))
    if "tree_raw" in item and item["tree_raw"] is not None and not isinstance(item["tree_raw"], dict):
        raise ValueError("validate_item[%s]: tree_raw must be dict or None" % item["id"])


# ---- chance line --------------------------------------------------------------

def chance_line(item):
    """Random-guess baseline for one item. murder_mystery = 0.5; else 1/len(choices)."""
    if item["subtask"] == "murder_mystery":
        return 0.5
    n = len(item["choices"])
    if n == 0:
        raise ValueError("chance_line[%s]: empty choices" % item.get("id"))
    return 1.0 / n


def chance_line_subtask_mean(items):
    """Mean chance line over the given items (caller passes a single subtask's items)."""
    if not items:
        raise ValueError("chance_line_subtask_mean: empty item list")
    return sum(chance_line(it) for it in items) / len(items)


# ---- deterministic triage sampling -------------------------------------------

def _derived_seed(base_seed, subtask):
    """Stable per-subtask seed independent of process hash randomization."""
    h = hashlib.sha256(("%d:%s" % (base_seed, subtask)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def sample_triage_ids(items, base_seed=BASE_SEED, n=TRIAGE_N):
    """Deterministic per-subtask triage sampling.

    Sorts each subtask's ids first, so the result is INVARIANT to input ordering, then
    draws min(n, available) without replacement using a subtask-derived RNG. Returns
    {subtask: sorted(list[id])}. This freezes the A0 triage set.
    """
    by_sub = {s: [] for s in SUBTASKS}
    for it in items:
        s = it["subtask"]
        if s not in by_sub:
            by_sub.setdefault(s, [])
        by_sub[s].append(it["id"])
    out = {}
    for subtask, ids in by_sub.items():
        ids_sorted = sorted(set(ids))
        k = min(n, len(ids_sorted))
        rng = random.Random(_derived_seed(base_seed, subtask))
        picked = rng.sample(ids_sorted, k)
        out[subtask] = sorted(picked)
    return out


# ---- token counting (Qwen3-4B, offline) --------------------------------------

_QWEN_TOKENIZER = None


def _qwen_snapshot_symlink_fix():
    """Known-box gotcha: if refs/main hash != any snapshot dir name, local_files_only load
    fails. Symlink the real hash -> the '0000...' snapshot. No-op when they already agree."""
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    root = os.path.join(hub, "models--Qwen--Qwen3-4B")
    refs_main = os.path.join(root, "refs", "main")
    snaps = os.path.join(root, "snapshots")
    if not (os.path.isfile(refs_main) and os.path.isdir(snaps)):
        return
    real_hash = open(refs_main).read().strip()
    if not real_hash:
        return
    target = os.path.join(snaps, real_hash)
    if os.path.exists(target):
        return  # snapshot for the real hash already present
    existing = [d for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d))]
    if existing:
        os.symlink(existing[0], target)  # point real hash at the available snapshot


def load_qwen_token_counter(model=QWEN3_4B_MODEL):
    """Return a callable text->n_tokens using the local Qwen3-4B tokenizer (offline).

    Lazily loaded and cached. Used as the default token counter in audit(); unit tests
    inject their own counter so this never runs without the model present.
    """
    global _QWEN_TOKENIZER
    if _QWEN_TOKENIZER is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        except Exception:
            _qwen_snapshot_symlink_fix()
            tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        _QWEN_TOKENIZER = tok

    def _count(text):
        return len(_QWEN_TOKENIZER(text, add_special_tokens=False)["input_ids"])

    return _count


# ---- percentiles (no numpy dependency at call sites) --------------------------

def _percentile(sorted_vals, q):
    """Linear-interpolation percentile (q in [0,100]) over an ASCENDING-sorted list."""
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


# ---- audit --------------------------------------------------------------------

def audit(items, download_meta=None, token_counter=None, out_path=None,
          expected_counts=EXPECTED_COUNTS, base_seed=BASE_SEED, triage_n=TRIAGE_N,
          write=True):
    """Build (and optionally write) the data audit report.

    Computes per subtask: item count, answer-choice distribution (+ majority-class warn),
    narrative token-length p50/p95/max (via token_counter), gold-tree availability rate/flag,
    per-item chance line, and the frozen triage id set.

    Hard errors:
      * item count != expected_counts[subtask]  (skipped when expected_counts is None)
      * any narrative token length > CONTEXT_TOKEN_LIMIT, or p95 > CONTEXT_TOKEN_LIMIT
        (message lists the offending ids)

    token_counter: text -> int. Defaults to the local Qwen3-4B tokenizer.
    Returns the report dict. Writes JSON to out_path when write=True (out_path defaults to
    outputs/data_audit.json relative to this module).
    """
    if token_counter is None:
        token_counter = load_qwen_token_counter()

    # validate every item up front (hard schema guarantee for downstream stages)
    for it in items:
        validate_item(it)

    by_sub = {}
    for it in items:
        by_sub.setdefault(it["subtask"], []).append(it)

    # count check
    if expected_counts is not None:
        mismatches = []
        for subtask, exp in expected_counts.items():
            got = len(by_sub.get(subtask, []))
            if got != exp:
                mismatches.append("%s: got %d expected %d" % (subtask, got, exp))
        if mismatches:
            raise ValueError("audit: item count mismatch -> " + "; ".join(mismatches))

    report = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "base_seed": base_seed,
        "triage_n": triage_n,
        "context_token_limit": CONTEXT_TOKEN_LIMIT,
        "expected_counts": expected_counts,
        "download_meta": download_meta,
        "subtasks": {},
    }

    triage = sample_triage_ids(items, base_seed=base_seed, n=triage_n)
    token_offenders = []  # (id, n_tokens) across all subtasks

    for subtask in sorted(by_sub.keys()):
        sub_items = by_sub[subtask]
        n = len(sub_items)

        # answer-choice distribution over gold_idx
        dist = {}
        for it in sub_items:
            dist[it["gold_idx"]] = dist.get(it["gold_idx"], 0) + 1
        majority = max(dist.values()) if dist else 0
        majority_frac = majority / n if n else 0.0

        # narrative token lengths
        lengths = []
        for it in sub_items:
            ln = int(token_counter(it["narrative"]))
            lengths.append((it["id"], ln))
            if ln > CONTEXT_TOKEN_LIMIT:
                token_offenders.append((it["id"], ln))
        vals = sorted(ln for _, ln in lengths)
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        mx = vals[-1] if vals else 0
        if p95 > CONTEXT_TOKEN_LIMIT:
            token_offenders.append(("%s:p95" % subtask, p95))

        # tree availability
        n_tree = sum(1 for it in sub_items if it.get("tree_raw") is not None)
        tree_rate = n_tree / n if n else 0.0
        tree_available = tree_rate >= TREE_AVAIL_MIN

        report["subtasks"][subtask] = {
            "n_items": n,
            "answer_distribution": {str(k): dist[k] for k in sorted(dist)},
            "answer_distribution_frac": {str(k): dist[k] / n for k in sorted(dist)},
            "majority_class_frac": majority_frac,
            "majority_class_warn": majority_frac > MAJORITY_WARN,
            "narrative_token_lengths": {"p50": p50, "p95": p95, "max": mx, "n": n},
            "tree_available_rate": tree_rate,
            "tree_available": bool(tree_available),
            "chance_line_mean": chance_line_subtask_mean(sub_items),
            "triage_ids": triage.get(subtask, []),
            "per_item_chance_line": {it["id"]: chance_line(it) for it in sub_items},
        }

    if token_offenders:
        listed = ", ".join("%s=%s" % (i, v) for i, v in token_offenders[:50])
        raise ValueError("audit: narrative token length exceeds %d for: %s"
                         % (CONTEXT_TOKEN_LIMIT, listed))

    if write:
        if out_path is None:
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "outputs", "data_audit.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        report["_written_to"] = out_path

    return report


# ---- download -----------------------------------------------------------------

def _http_get_json(url, timeout=120):
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _load_github(subtask):
    """Fetch + flatten the GitHub-raw JSON into normalized items (WITH gold trees)."""
    if subtask not in SUBTASKS:
        raise ValueError("unknown subtask %r" % subtask)
    data = _http_get_json(GITHUB_RAW.format(name=subtask))
    if not isinstance(data, list):
        raise ValueError("github raw for %s: expected a list of examples" % subtask)
    items = []
    for ei, ex in enumerate(data):
        questions = ex.get("questions")
        if not isinstance(questions, list):
            raise ValueError("github raw for %s: example %d missing 'questions' list" % (subtask, ei))
        for qi, q in enumerate(questions):
            raw = dict(q)
            raw["context"] = ex.get("context")
            raw["example_idx"] = ei
            raw["question_idx"] = qi
            items.append(normalize_item(raw, subtask))
    return items


def _load_hf(subtask):
    """Load the HF dataset for a subtask into normalized items (WITHOUT trees).

    Best-effort: the HF CSV export has no tree columns and its config/split naming has
    varied across releases, so several access forms are attempted; any failure propagates
    to download()'s GitHub fallback. (example_idx, question_idx) are reconstructed from the
    already-flattened row order so ids match the GitHub path.
    """
    if subtask not in SUBTASKS:
        raise ValueError("unknown subtask %r" % subtask)
    from datasets import load_dataset
    ds = None
    errs = []
    for kwargs in ({"name": subtask}, {"split": subtask},
                   {"name": subtask, "split": "train"}):
        try:
            ds = load_dataset(HF_DATASET, **kwargs)
            # a DatasetDict -> pick the subtask split if present
            if hasattr(ds, "keys") and not hasattr(ds, "features"):
                ds = ds[subtask] if subtask in ds else ds[list(ds.keys())[0]]
            break
        except Exception as e:  # noqa: BLE001 - fall through to next access form
            errs.append("%s: %s" % (kwargs, e))
            ds = None
    if ds is None:
        raise RuntimeError("HF load failed for %s: %s" % (subtask, " | ".join(errs)))

    qpe = HF_QUESTIONS_PER_EXAMPLE[subtask]
    items = []
    for i, row in enumerate(ds):
        raw = dict(row)
        raw["example_idx"] = i // qpe
        raw["question_idx"] = i % qpe
        items.append(normalize_item(raw, subtask))
    return items


def _enrich_trees_from_github(items, subtask):
    """Attach gold trees to HF items using the GitHub raw source.

    Alignment is by index AND verified (same narrative + question text + same choices) before
    attaching; on any count/content mismatch, returns items UNCHANGED (trees stay None) rather
    than risk attaching the wrong tree. Returns (items, enriched: bool).

    The narrative is part of the key on purpose: murder_mystery's question is identical across
    all 250 items and team_allocation's is template-constant, so a question+choices-only key
    would degenerate to choices-only there and could mis-attach a tree from another item under
    a row-order divergence. tree_raw is the knowledge-vs-commitment separator (A2 facts arm),
    so the key must be discriminative.
    """
    try:
        gh = _load_github(subtask)
    except Exception:
        return items, False
    if len(gh) != len(items):
        return items, False
    for a, b in zip(items, gh):
        if (a["narrative"] != b["narrative"] or a["question"] != b["question"]
                or a["choices"] != b["choices"]):
            return items, False  # order/content mismatch -> abort, do not fabricate
    for a, b in zip(items, gh):
        a["tree_raw"] = b["tree_raw"]
    return items, True


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(subtask, data_dir=None, prefer="hf", enrich_hf_trees=True):
    """Download one MuSR subtask, normalize, persist to data/{subtask}.json.

    Primary path = HF (datasets.load_dataset); fallback = GitHub raw JSON. Because the HF
    export lacks gold trees, HF-sourced items are enriched with trees from GitHub (verified
    alignment) when enrich_hf_trees=True. Returns a meta dict:
        {subtask, source, tree_source, n_items, path, sha256}
    which the caller records into outputs/data_audit.json (via audit(download_meta=...)).
    """
    if subtask not in SUBTASKS:
        raise ValueError("unknown subtask %r" % subtask)
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    source = None
    tree_source = "none"
    items = None
    if prefer == "hf":
        try:
            items = _load_hf(subtask)
            source = "huggingface"
            if enrich_hf_trees:
                items, enriched = _enrich_trees_from_github(items, subtask)
                if enriched:
                    tree_source = "github_raw"
            else:
                tree_source = "none"
        except Exception:
            items = None
    if items is None:
        items = _load_github(subtask)
        source = "github_raw"
        tree_source = "github_raw"

    if len(items) != EXPECTED_COUNTS[subtask]:
        raise ValueError("download[%s]: got %d items, expected %d (schema/flatten mismatch)"
                         % (subtask, len(items), EXPECTED_COUNTS[subtask]))

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "%s.json" % subtask)
    with open(path, "w") as f:
        json.dump(items, f)
    n_tree = sum(1 for it in items if it.get("tree_raw") is not None)
    return {
        "subtask": subtask,
        "source": source,
        "tree_source": tree_source,
        "n_items": len(items),
        "tree_available_rate": n_tree / len(items) if items else 0.0,
        "path": path,
        "sha256": _sha256_file(path),
    }


def load_local(subtask, data_dir=None):
    """Read back the persisted normalized items for a subtask."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    path = os.path.join(data_dir, "%s.json" % subtask)
    with open(path) as f:
        items = json.load(f)
    for it in items:
        validate_item(it)
    return items
