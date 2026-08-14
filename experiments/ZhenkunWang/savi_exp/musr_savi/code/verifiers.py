"""verifiers — MuSR-cant Phase B soft-verifier instruments (Subtask 3).

The verifier is the LOAD-BEARING instrument of Phase B: the whole experiment tests whether a
learned soft verifier can discriminate correct reasoning from wrong on the certified can't
patients. This module builds the two candidate verifier FORMS, the ROC-AUC used to compare
them, and the leakage-guarded form selection.

==========================================================================================
DESIGN — FEATURIZATION IS SEPARATED FROM SCORING (so the whole module is CPU-testable)
==========================================================================================
A hidden-state probe needs a model to turn a chain into features; a text critic needs a model
to emit a judgement. Neither dependency lives here:

  * ``ProbeVerifier`` operates on numpy FEATURES that the caller (Subtask 6) extracts from a
    real model. Its ``fit``/``score`` are pure sklearn — no torch, no transformers. The feature
    contract is ``X`` shaped ``[N, L+1, H]`` (per-layer hidden states, layer 0 = embedding) OR
    an already-sliced ``[N, H]``; ``y`` is sample correctness (1 = correct). Subtask 6 is
    responsible for producing X (e.g. the last-token hidden state at every layer for the chain,
    or the step-end token for a step) and joining the cached ``correct`` label.
  * ``CriticVerifier`` takes an INJECTED ``sc_core.Emitter`` (``generate(prompts, seeds, *,
    greedy, max_new_tokens) -> list[GenOut]``). In production the runner passes an ``HFEmitter``;
    in tests a pure ``FakeCriticEmitter`` drives it. The critic itself only builds frozen prompts
    and parses a ``SCORE: <0-100>`` verdict — pure string work.

Both forms are compared on the CALIBRATION set only; ``assert_no_leakage`` guarantees the
patient/deferred (held-out) ids never enter the train/calibration sets that fit or select the
verifier.

LEAKAGE GUARD IS QUESTION-ID-ONLY — READ THIS (Phase B's single most dangerous confound):
``assert_no_leakage`` checks QUESTION-ID set membership only. It does NOT detect story-level
aliasing. MuSR packs ~4 questions per story, so a SIBLING question of a patient's story shares
the same narrative; feeding that sibling to a hidden-state probe would leak the patient's story
into training even though the two question ids differ. This guard is NECESSARY BUT NOT
SUFFICIENT: story-level isolation MUST be enforced UPSTREAM at split time (Subtask 6 groups by
``selffacts.story_id`` before assigning splits). Do not treat a green ``assert_no_leakage`` as
proof the probe is uncontaminated.

CODE SEPARATION: imports ONLY stdlib + numpy + scikit-learn + joblib + sc_core (all torch-free
at import — sc_core lazy-imports torch inside HFEmitter). ``score_steps`` duck-types
``parsed.states`` rather than importing ``belief_schema``. NEVER imports test code and pulls in
no torch/transformers at module top.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sc_core as sc  # noqa: F401  (GenOut/Emitter protocol source of truth; used by the runner)

# ======================================================================================
# Token cap — CANDIDATE, pinned by the L1 smoke in the integration subtask (mirrors selffacts'
# cap convention). Re-pinning does NOT change the frozen template shas below.
# ======================================================================================
CRITIC_MAX_NEW_TOKENS = 512


# ======================================================================================
# roc_auc — the layer-selection + form-comparison metric
# ======================================================================================
def roc_auc(scores, labels) -> float:
    """ROC-AUC of ``scores`` against binary ``labels`` (1 = correct).

    Returns 0.5 when the labels are all one class (AUC undefined) or empty — the neutral,
    no-discrimination value. Non-finite scores are neutralised to 0.5 as a crash guard.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    return float(roc_auc_score(labels, scores))


# ======================================================================================
# ProbeVerifier — hidden-state probe over INJECTED numpy features (model-free)
# ======================================================================================
class ProbeVerifier:
    """Per-layer logistic probe of P(correct); the probe LAYER is selected on validation AUC.

    Mirrors the structure of ``belief-arc/stage0_freegen_gate/core/probe.py`` (one
    ``LogisticRegression`` per layer, layer chosen on a held-out split, non-finite rows dropped
    and counted) but targets binary correctness and selects on ROC-AUC (a ranking metric — the
    verifier's job is to RANK correct above wrong), not accuracy.

    Fitted attributes: ``layer`` (selected), ``clf`` (trained at that layer), ``val_auc``,
    ``per_layer_val_auc``, ``n_dropped_train``/``n_dropped_val``, ``n_train``/``n_val``,
    ``n_layers``.
    """

    def __init__(self):
        self.layer: Optional[int] = None
        self.clf: Optional[LogisticRegression] = None
        self.val_auc: Optional[float] = None
        self.per_layer_val_auc: Optional[list] = None
        self.n_dropped_train: int = 0
        self.n_dropped_val: int = 0
        self.n_train: int = 0
        self.n_val: int = 0
        self.n_layers: Optional[int] = None

    # -- shape helpers -----------------------------------------------------------------
    @staticmethod
    def _as_3d(X):
        """Normalise features to ``[N, L+1, H]``. An ``[N, H]`` block becomes a single layer."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            return X
        if X.ndim == 2:
            return X[:, None, :]
        raise ValueError(f"features must be [N, L+1, H] or [N, H]; got shape {X.shape}")

    @staticmethod
    def _finite_item_mask(X3d):
        """Boolean mask [N]: True for rows whose entire [L+1, H] block is finite."""
        flat = X3d.reshape(X3d.shape[0], -1)
        return np.isfinite(flat).all(axis=1)

    def _layer_matrix(self, X):
        """Slice X to the selected layer's [N, H] design matrix (accepts [N, L+1, H] or [N, H])."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            return X[:, self.layer, :]
        if X.ndim == 2:
            return X
        raise ValueError(f"features must be [N, L+1, H] or [N, H]; got shape {X.shape}")

    @staticmethod
    def _p_correct(clf, X2d):
        """P(correct) column of ``clf.predict_proba`` (the column of class label 1)."""
        proba = clf.predict_proba(X2d)
        classes = list(clf.classes_)
        j = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, j]

    @staticmethod
    def _new_clf():
        return LogisticRegression(class_weight="balanced", max_iter=2000)

    # -- fit / score -------------------------------------------------------------------
    def fit(self, X_tr, y_tr, X_val, y_val):
        """Fit one logistic probe per layer on train; select the layer by validation ROC-AUC.

        X_* : ``[N, L+1, H]`` (per-layer states) or ``[N, H]`` (single already-sliced layer).
        y_* : sample correctness (1 = correct). Non-finite feature rows are dropped and counted
        (the same clean item set is used at every layer so the layer comparison is fair).
        Returns ``self``.
        """
        Xtr = self._as_3d(X_tr)
        Xval = self._as_3d(X_val)
        ytr = np.asarray(y_tr)
        yval = np.asarray(y_val)

        m_tr = self._finite_item_mask(Xtr)
        m_val = self._finite_item_mask(Xval)
        self.n_dropped_train = int((~m_tr).sum())
        self.n_dropped_val = int((~m_val).sum())
        Xtr, ytr = Xtr[m_tr], ytr[m_tr]
        Xval, yval = Xval[m_val], yval[m_val]
        if len(Xtr) == 0 or len(Xval) == 0:
            raise ValueError("no finite rows left in train or val after non-finite rejection")
        if np.unique(ytr).size < 2:
            raise ValueError("train labels must contain both correct and incorrect samples")

        n_layers = Xtr.shape[1]
        per_layer_val_auc, clfs = [], []
        for layer in range(n_layers):
            clf = self._new_clf()
            clf.fit(Xtr[:, layer, :], ytr)
            p_val = self._p_correct(clf, Xval[:, layer, :])
            per_layer_val_auc.append(roc_auc(p_val, yval))
            clfs.append(clf)

        best = int(np.argmax(per_layer_val_auc))  # first max on ties (deterministic)
        self.layer = best
        self.clf = clfs[best]
        self.val_auc = float(per_layer_val_auc[best])
        self.per_layer_val_auc = [float(a) for a in per_layer_val_auc]
        self.n_layers = int(n_layers)
        self.n_train = int(len(Xtr))
        self.n_val = int(len(Xval))
        return self

    def score(self, X) -> np.ndarray:
        """P(correct) for each row (selected layer). Non-finite rows -> 0.5 (neutral)."""
        if self.clf is None or self.layer is None:
            raise RuntimeError("ProbeVerifier.score called before fit/load")
        X2d = self._layer_matrix(X)
        out = np.full(X2d.shape[0], 0.5, dtype=float)
        finite = np.isfinite(X2d).all(axis=1)
        if finite.any():
            out[finite] = self._p_correct(self.clf, X2d[finite])
        return out

    # -- persistence -------------------------------------------------------------------
    def save(self, path):
        """Persist the fitted probe (numpy + sklearn state) via joblib. Round-trips ``score``."""
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path):
        """Load a ProbeVerifier saved by ``save``."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not hold a ProbeVerifier (got {type(obj)})")
        return obj


# ======================================================================================
# CriticVerifier — prompted text-consistency judge (INJECTED emitter)
# ======================================================================================
# FROZEN prompt templates (English — MuSR is English). Each presents the item and a candidate
# (a full reasoning chain, or a single belief step) and asks for a parseable verdict line
# ``SCORE: <0-100>`` (0 = inconsistent, 100 = fully consistent). shas snapshot-tested.
CRITIC_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "Pick one option:\n"
    "{numbered_choices}\n\n"
    "A solver produced the following step-by-step reasoning for this problem:\n"
    "--- BEGIN REASONING ---\n"
    "{candidate}\n"
    "--- END REASONING ---\n\n"
    "Judge how consistent this reasoning is with the story and how correctly it tracks the "
    "facts. Do NOT judge whether the final option is the one you would pick; judge only the "
    "reasoning's internal and factual consistency with the story. Finish your reply with a "
    'single line "SCORE: <0-100>", where 0 means completely inconsistent and 100 means fully '
    "consistent."
)

STEP_CRITIC_TEMPLATE = (
    "{narrative}\n\n"
    "{question}\n\n"
    "While solving this, a solver recorded the following belief state — a map from each "
    "character to where that character currently believes each object is:\n"
    "{state}\n\n"
    "Judge whether this belief state is consistent with what the story has established up to "
    "this point. Finish your reply with a single line \"SCORE: <0-100>\", where 0 means "
    "completely inconsistent with the story and 100 means fully consistent."
)

CRITIC_TEMPLATE_SHA256 = hashlib.sha256(CRITIC_TEMPLATE.encode("utf-8")).hexdigest()
STEP_CRITIC_TEMPLATE_SHA256 = hashlib.sha256(STEP_CRITIC_TEMPLATE.encode("utf-8")).hexdigest()

# Verdict parser: last "SCORE: <n>" wins; a "CONSISTENT: yes/no" line is a tolerated fallback.
_SCORE_RE = re.compile(r"SCORE\s*[:：]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_CONSISTENT_RE = re.compile(r"CONSISTENT\s*[:：]\s*(yes|no)", re.IGNORECASE)
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def parse_verdict(text: Optional[str]) -> Optional[float]:
    """Map a critic reply to a consistency score in [0,1], or None if no verdict is present.

    Primary contract: the LAST ``SCORE: <0-100>`` line, clamped to [0,100] then divided by 100.
    Fallback: a ``CONSISTENT: yes/no`` line (yes -> 1.0, no -> 0.0). Tolerant of case and
    ASCII/full-width colon; full-width digits normalised. None/empty/markerless -> None.
    """
    if not text:
        return None
    norm = text.translate(_FULLWIDTH_DIGITS)
    hits = _SCORE_RE.findall(norm)
    if hits:
        v = float(hits[-1])
        v = max(0.0, min(100.0, v))
        return v / 100.0
    yn = _CONSISTENT_RE.findall(norm)
    if yn:
        return 1.0 if yn[-1].lower() == "yes" else 0.0
    return None


def format_critic_prompt(item: dict, chain_text: str) -> str:
    """Build the answer-level critic prompt (frozen ``CRITIC_TEMPLATE``): the item narrative /
    question / 1-based choices + the candidate reasoning ``chain_text``."""
    return CRITIC_TEMPLATE.format(
        narrative=item["narrative"],
        question=item["question"],
        numbered_choices=sc.render_numbered_choices(item["choices"]),
        candidate=("" if chain_text is None else str(chain_text)),
    )


def format_step_critic_prompt(item: dict, step_state) -> str:
    """Build the step-level critic prompt (frozen ``STEP_CRITIC_TEMPLATE``): the item narrative /
    question + a single belief state ``{character: {object: location}}`` rendered as JSON."""
    import json
    state_text = json.dumps(step_state, ensure_ascii=False, sort_keys=True)
    return STEP_CRITIC_TEMPLATE.format(
        narrative=item["narrative"],
        question=item["question"],
        state=state_text,
    )


class CriticVerifier:
    """Zero-training prompted judge: asks an injected model to score consistency.

    ``emitter`` honours the ``sc_core.Emitter`` protocol. ``greedy=True`` (default) makes the
    verdict a deterministic, reproducible readout (seeds are still threaded for API symmetry and
    the non-greedy option). ``n_unparsed`` is a CUMULATIVE ledger of replies with no parseable
    verdict (each scored as the neutral 0.5), across all ``score_*`` calls on this instance.
    """

    def __init__(self, emitter, max_new_tokens: int = CRITIC_MAX_NEW_TOKENS, greedy: bool = True):
        self.emitter = emitter
        self.max_new_tokens = max_new_tokens
        self.greedy = greedy
        self.n_unparsed = 0

    def _score_prompts(self, prompts, seeds) -> np.ndarray:
        """Generate for ``prompts`` and parse each verdict -> [0,1]; unparseable -> 0.5 + count."""
        if len(prompts) == 0:
            return np.asarray([], dtype=float)
        outs = self.emitter.generate(
            list(prompts), list(seeds), greedy=self.greedy, max_new_tokens=self.max_new_tokens)
        scores = []
        for o in outs:
            v = parse_verdict(getattr(o, "text", None))
            if v is None:
                v = 0.5
                self.n_unparsed += 1
            scores.append(v)
        return np.asarray(scores, dtype=float)

    def score_chains(self, items: Sequence[dict], texts: Sequence[str],
                     seeds: Sequence[int]) -> np.ndarray:
        """Answer-level consistency score in [0,1] for each (item, candidate chain text)."""
        if not (len(items) == len(texts) == len(seeds)):
            raise ValueError("items/texts/seeds must be parallel")
        prompts = [format_critic_prompt(it, tx) for it, tx in zip(items, texts)]
        return self._score_prompts(prompts, seeds)

    def score_steps(self, item: dict, parsed, seeds: Sequence[int]) -> list:
        """Per-step consistency score in [0,1], one per state in ``parsed.states``.

        ``seeds`` is aligned to the states; if fewer seeds are given it is padded (with the last
        seed, or 0) so an under-specified seed list never crashes the step scan. Empty chain -> []."""
        states = list(getattr(parsed, "states", []) or [])
        if not states:
            return []
        seeds = list(seeds)
        if len(seeds) < len(states):
            fill = seeds[-1] if seeds else 0
            seeds = seeds + [fill] * (len(states) - len(seeds))
        seeds = seeds[:len(states)]
        prompts = [format_step_critic_prompt(item, st) for st in states]
        return self._score_prompts(prompts, seeds).tolist()


# ======================================================================================
# Leakage guard + form selection
# ======================================================================================
def assert_no_leakage(train_ids, calib_ids, held_ids) -> None:
    """Raise ``AssertionError`` if any held-out id appears in the train or calibration set.

    ``held_ids`` are the patient ∪ deferred ids that must NEVER be used to fit or select the
    verifier. The message names the offending ids so a violation is diagnosable.

    SCOPE — QUESTION-ID MEMBERSHIP ONLY (necessary, not sufficient): this checks that no held-out
    QUESTION id is in train/calib. It does NOT detect STORY-LEVEL aliasing — MuSR packs ~4
    questions per story, so a sibling question of a patient's story carries the SAME narrative and
    would leak into a hidden-state probe even though its question id differs. Story-level
    isolation must be enforced UPSTREAM at split time (group by ``selffacts.story_id`` before
    assigning splits — Subtask 6). A green result here is not proof the probe is uncontaminated.
    """
    train = set(train_ids or [])
    calib = set(calib_ids or [])
    held = set(held_ids or [])
    overlap = (train | calib) & held
    if overlap:
        offending = ", ".join(sorted(str(x) for x in overlap))
        raise AssertionError(
            f"verifier leakage: {len(overlap)} held-out id(s) appear in train/calib: {offending}")


def select_verifier(candidates: dict, *, train_ids=None, calib_ids=None, held_ids=None):
    """Select the verifier FORM with the highest CALIBRATION val AUC.

    ``candidates`` = ``{name: {"val_auc": float}}`` (val AUC precomputed on the CALIBRATION set by
    the caller — selection MUST use only calibration data). Returns ``(winner_name, report)``.
    Tie on AUC -> the lexicographically smallest name (deterministic).

    If all of ``train_ids``/``calib_ids``/``held_ids`` are provided, ``assert_no_leakage`` is run
    first so a mis-scoped selection fails loudly rather than silently leaking.
    """
    if train_ids is not None and calib_ids is not None and held_ids is not None:
        assert_no_leakage(train_ids, calib_ids, held_ids)
    if not candidates:
        raise ValueError("select_verifier: no candidates given")

    def auc_of(name):
        return float(candidates[name]["val_auc"])

    # max AUC, tie -> smallest name.
    winner = min(candidates, key=lambda n: (-auc_of(n), str(n)))
    ranking = sorted(((str(n), auc_of(n)) for n in candidates), key=lambda t: (-t[1], t[0]))
    report = {
        "winner": winner,
        "val_auc": auc_of(winner),
        "ranking": [list(r) for r in ranking],
        "candidates": {str(n): auc_of(n) for n in candidates},
        "tie_break": "max val_auc; ties broken by lexicographically smallest name",
    }
    return winner, report
