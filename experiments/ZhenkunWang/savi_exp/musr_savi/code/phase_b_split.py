"""phase_b_split — leakage-safe STORY-LEVEL split for MuSR-cant Phase B (Subtask 6 helper).

Phase B trains a hidden-state probe / text critic that READS the model's internal states while
it processes a narrative. MuSR packs ~4 questions per story over ONE narrative, so two questions
of the same story share the same story text. A question-id split (``verifiers.assert_no_leakage``)
is therefore NECESSARY BUT NOT SUFFICIENT — a sibling question of a patient's story would leak the
patient's narrative into training even though the question ids differ. This module builds a
STORY-DISJOINT three-way split (train / calib / tuning) that excludes every patient story, and
exposes ``assert_story_level_disjoint`` as the hard leakage guard (raises on any story overlap).

Deterministic: given the same ``items``, ``patient_ids`` and ``base_seed`` the split is
byte-for-byte reproducible (a seeded sha256 order over the non-patient stories; A1 survivor
stories are preferentially routed into ``tuning`` so it holds hard items for lambda selection).

CODE SEPARATION: imports ONLY stdlib (``hashlib``) + ``selffacts`` (for ``story_id``). No numpy,
no torch, no test code; pure, CPU-only, import-safe. Every public function is TOTAL — an empty
item set or an empty candidate pool yields empty splits rather than raising (only a genuine
story-level LEAKAGE raises, which is the whole point of the guard).
"""
from __future__ import annotations

import hashlib

import selffacts as sf

# Default three-way proportions (design §2 leak-proof split): ~10 of 45 non-patient stories -> tuning
# (10/45 ≈ 0.222); of the remaining ~35, ~20% -> calib, ~80% -> train. Clamped so that, whenever
# there is enough material, every split gets at least one story.
DEFAULT_TUNING_FRAC = 10.0 / 45.0
DEFAULT_CALIB_FRAC = 0.20


def _story_order_key(story_id: str, base_seed: int) -> str:
    """Deterministic per-story sort key = sha256(f"{base_seed}|{story_id}") — a seeded shuffle."""
    return hashlib.sha256(("%d|%s" % (base_seed, story_id)).encode("utf-8")).hexdigest()


def _stories_of(item_ids):
    """Sorted unique story ids for a collection of item ids (via ``selffacts.story_id``)."""
    return sorted({sf.story_id(i) for i in item_ids})


def _items_in_stories(items_by_story, stories):
    """Sorted item-ids belonging to any story in ``stories`` (order-stable)."""
    out = []
    for s in sorted(stories):
        out.extend(items_by_story.get(s, []))
    return sorted(out)


def build_split(items, patient_ids, base_seed=20260706, survivor_ids=None,
                tuning_frac=DEFAULT_TUNING_FRAC, calib_frac=DEFAULT_CALIB_FRAC) -> dict:
    """Story-disjoint three-way split over the NON-patient stories.

    ``items``        : normalized item dicts (each with an ``id``); the universe of stories.
    ``patient_ids``  : the residual patient question-ids (their stories are held out entirely).
    ``survivor_ids`` : (optional) A1 still-wrong item-ids; their non-patient stories are routed
                       into ``tuning`` first so lambda selection sees hard items.

    Returns a dict with per-split STORY lists and ITEM-ID lists:
        patients / patient_stories,
        tuning / tuning_stories, train / train_stories, calib / calib_stories,
        plus counts, base_seed and the survivor-in-tuning bookkeeping.

    Guarantees (asserted before return via ``assert_story_level_disjoint``): the patient stories
    are disjoint from tuning ∪ train ∪ calib, and tuning / train / calib are mutually disjoint —
    a violation RAISES ``AssertionError`` (the leakage guard). Never silently returns a leaky split.
    """
    items_by_story: dict = {}
    for it in items:
        items_by_story.setdefault(sf.story_id(it["id"]), []).append(it["id"])
    for s in items_by_story:
        items_by_story[s] = sorted(items_by_story[s])

    patient_stories = set(_stories_of(patient_ids))
    all_stories = set(items_by_story.keys())
    candidate_stories = sorted(all_stories - patient_stories)

    survivor_stories = set(_stories_of(survivor_ids or [])) & set(candidate_stories)

    # Deterministic order: survivor-containing stories first (hard items for tuning), then the
    # rest; within each group a seeded sha256 order (a reproducible shuffle).
    ordered = sorted(
        candidate_stories,
        key=lambda s: (0 if s in survivor_stories else 1, _story_order_key(s, base_seed)))

    n = len(ordered)
    # tuning size: proportional, clamped so >=2 stories remain for train+calib when possible.
    if n <= 0:
        n_tuning = 0
    elif n <= 2:
        n_tuning = 0                      # too few stories -> no tuning; keep material for train/calib
    else:
        n_tuning = max(1, int(round(n * tuning_frac)))
        n_tuning = min(n_tuning, n - 2)   # leave >=2 for a train + calib split
    tuning_stories = ordered[:n_tuning]
    rest = ordered[n_tuning:]

    # calib size: proportional of the rest, clamped so train keeps >=1 story when possible.
    m = len(rest)
    if m <= 0:
        n_calib = 0
    elif m == 1:
        n_calib = 0                       # a single story -> train only
    else:
        n_calib = max(1, int(round(m * calib_frac)))
        n_calib = min(n_calib, m - 1)     # leave >=1 for train
    calib_stories = rest[:n_calib]
    train_stories = rest[n_calib:]

    split = {
        "base_seed": base_seed,
        "patients": sorted(patient_ids),
        "patient_stories": sorted(patient_stories),
        "tuning_stories": sorted(tuning_stories),
        "train_stories": sorted(train_stories),
        "calib_stories": sorted(calib_stories),
        "tuning": _items_in_stories(items_by_story, tuning_stories),
        "train": _items_in_stories(items_by_story, train_stories),
        "calib": _items_in_stories(items_by_story, calib_stories),
        "survivor_stories_in_tuning": sorted(survivor_stories & set(tuning_stories)),
        "n_nonpatient_stories": n,
    }
    split["counts"] = {
        "patients": len(split["patients"]),
        "patient_stories": len(split["patient_stories"]),
        "tuning_stories": len(tuning_stories),
        "train_stories": len(train_stories),
        "calib_stories": len(calib_stories),
        "tuning_items": len(split["tuning"]),
        "train_items": len(split["train"]),
        "calib_items": len(split["calib"]),
    }
    assert_story_level_disjoint(split)     # the leakage guard — raises on any story overlap
    return split


def assert_story_level_disjoint(split: dict) -> None:
    """Raise ``AssertionError`` if any two of {patient, tuning, train, calib} share a story.

    This is the STORY-LEVEL leakage guard the question-id guard (``verifiers.assert_no_leakage``)
    cannot provide. ``split`` carries ``*_stories`` lists (as produced by ``build_split`` or a
    hand-built dict). The message names the offending stories so a contamination is diagnosable.
    """
    groups = {
        "patient": set(split.get("patient_stories") or []),
        "tuning": set(split.get("tuning_stories") or []),
        "train": set(split.get("train_stories") or []),
        "calib": set(split.get("calib_stories") or []),
    }
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = groups[a] & groups[b]
            if overlap:
                offending = ", ".join(sorted(overlap))
                raise AssertionError(
                    "story-level leakage: %s and %s share %d story(ies): %s"
                    % (a, b, len(overlap), offending))
