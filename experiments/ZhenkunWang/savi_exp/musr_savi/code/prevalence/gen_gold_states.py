#!/usr/bin/env python
"""P3 — correct-state generator for E5 (base-rate) + 100% consistency gate.

Self-contained, CPU-only, re-runnable. Produces, for every cache_b1 item
(MuSR object_placements), the per-layer gold structural/belief data:

  * moves          — [(t, obj, new_loc), ...] diffed INDEPENDENTLY from
                     tree_raw.intermediate_data[0]["actual_locs"] (this module does
                     NOT call LE build_space to derive its own moves);
  * event_lines    — one string per layer 0..T-1, rendered by THIS module's own
                     wording (NOT LE render_event_line);
  * gold_belief_canons        — the WITNESS-structured gold answer key per layer,
                     canon(gold_beliefs[t]); this is what the g4 evaluation
                     mechanism (stages_g4.goldprefix_nodes) uses as the correct
                     successor and what E5 §7 must reuse. Read by THIS module's own
                     tree reader, then canonicalised via belief_schema.canon_state
                     (the required equality literal).
  * gold_belief_canons_leakfree — the LEAK-FREE reconstruction canon per layer:
                     root = gold_beliefs[0], every move applied to EVERY character
                     (the only witness-agnostic update). Diagnostic only.

INDEPENDENCE: move/event derivation is hand-written here. LE functions are imported
ONLY as comparison baselines:
  * gsd_space.build_space + gsd_score.render_event_line  -> the 100% EVENT-LINE gate;
  * facts_oracle.gold_beliefs                            -> the belief anchor.
belief_schema.canon_state is imported because the task fixes it as the state-equality
literal for E5 (canon form must be dumped).

Usage:
    python gen_gold_states.py            # writes artefacts next to this script
"""
from __future__ import annotations

import glob
import json
import os
import platform
import sys
import time

# Release path resolution (original ran against internal copies of the musr-cant modules and
# a frozen b1 shard dir; both are now env-var overridable with in-repo fallbacks).
HERE = os.path.dirname(os.path.abspath(__file__))
LE_MUSR = os.path.abspath(os.path.join(HERE, ".."))         # flat musr-cant modules live here
WORKDIR = os.environ.get("MUSR_SAVI_B1_DIR", os.path.join(LE_MUSR, "outputs"))  # cache_b1_shard*.jsonl (run_musr_cant.py --stage b1)
DATA_JSON = os.environ.get("MUSR_SAVI_DATA", os.path.join(LE_MUSR, "data", "object_placements.json"))

# LE modules used ONLY as baselines / required literal (read-only import).
sys.path.insert(0, LE_MUSR)
import belief_schema          # canon_state — required equality literal
import gsd_space              # build_space — event-line gate baseline
import gsd_score              # render_event_line — event-line gate baseline
import facts_oracle           # gold_beliefs — belief anchor baseline

# 19 certified + 3 knowledge control = 22 overlap items with existing g3/g4.
CERT = ["0000-q2", "0003-q3", "0006-q2", "0006-q3", "0011-q2", "0013-q1", "0014-q0",
        "0018-q0", "0021-q0", "0032-q0", "0042-q0", "0042-q1", "0043-q2", "0045-q0",
        "0046-q0", "0047-q2", "0050-q1", "0051-q2", "0055-q3"]
KNOW = ["0020-q1", "0034-q3", "0035-q0"]
OVERLAP = ["object_placements-" + x for x in CERT + KNOW]

# frozen root event-0 wording (hand-copied; independence from LE template).
ROOT_EVENT_LINE = "Event 0: the story begins; objects are at their initial locations."


# ---------------------------------------------------------------------------
# hand-written helpers (no LE import path)
# ---------------------------------------------------------------------------
def _norm(x) -> str:
    return str(x).strip().lower()


def _ci_get(d, key):
    nk = _norm(key)
    for k, v in d.items():
        if _norm(k) == nk:
            return v
    return None


def _ci_key(d, key):
    nk = _norm(key)
    for k in d:
        if _norm(k) == nk:
            return k
    return key


def _read_tree0(item):
    """item["tree_raw"]["intermediate_data"][0] or None (own reader, no facts_oracle)."""
    if not isinstance(item, dict):
        return None
    tr = item.get("tree_raw")
    if not isinstance(tr, dict):
        return None
    idata = tr.get("intermediate_data")
    if not idata or not isinstance(idata, (list, tuple)):
        return None
    d0 = idata[0]
    return d0 if isinstance(d0, dict) else None


def derive_moves(actual_locs):
    """INDEPENDENT per-cell diff of actual_locs -> [(t, obj, new_loc), ...].

    Layer 0 is the initial placement (no diff). For t=1..T-1 compare
    actual_locs[t-1] vs actual_locs[t] case/whitespace-insensitively; obj and
    new_loc kept verbatim; objects folded-sorted within a layer.
    """
    T = len(actual_locs)
    moves = []
    for t in range(1, T):
        prev, cur = actual_locs[t - 1], actual_locs[t]
        for obj in sorted(cur, key=_norm):
            old = _ci_get(prev, obj)
            if old is None or _norm(old) != _norm(cur[obj]):
                moves.append((t, obj, cur[obj]))
    return moves


def render_event_lines(moves, T):
    """INDEPENDENT event-line rendering, one per layer 0..T-1."""
    by_t = {}
    for (t, obj, loc) in moves:
        by_t.setdefault(t, []).append((obj, loc))
    lines = [ROOT_EVENT_LINE]
    for t in range(1, T):
        mvs = sorted(by_t.get(t, []), key=lambda ol: _norm(ol[0]))
        if not mvs:
            lines.append("Event %d: nothing is moved in this step." % t)
        else:
            sents = " ".join("the %s is moved to the %s." % (obj, loc) for obj, loc in mvs)
            lines.append("Event %d: %s" % (t, sents))
    return lines


def leakfree_traj(root, moves, T):
    """LEAK-FREE reconstruction: root replicated, every move applied to EVERY char."""
    chars = sorted(root, key=_norm)
    by_t = {}
    for (t, obj, loc) in moves:
        by_t.setdefault(t, []).append((obj, loc))
    states = [{c: dict(tbl) for c, tbl in root.items()}]
    cur = {c: dict(tbl) for c, tbl in root.items()}
    for t in range(1, T):
        nxt = {c: dict(tbl) for c, tbl in cur.items()}
        for (obj, loc) in by_t.get(t, []):
            for c in chars:
                nxt[c][_ci_key(nxt[c], obj)] = loc
        states.append(nxt)
        cur = nxt
    return states


# ---------------------------------------------------------------------------
# item id inventory from cache_b1 shards (VD workdir is authoritative)
# ---------------------------------------------------------------------------
def load_b1_ids():
    ids = set()
    shards = sorted(glob.glob(os.path.join(WORKDIR, "cache_b1_shard*.jsonl")))
    for p in shards:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                iid = d.get("item_id")
                if iid:
                    ids.add(iid)
    return sorted(ids), [os.path.basename(p) for p in shards]


def main():
    t_start = time.time()
    b1_ids, shard_names = load_b1_ids()
    data = json.load(open(DATA_JSON))
    by_id = {it["id"]: it for it in data}

    missing = [i for i in b1_ids if i not in by_id]

    rows = []
    anomalies = []
    for iid in b1_ids:
        item = by_id.get(iid)
        if item is None:
            anomalies.append({"item_id": iid, "reason": "not_in_data"})
            continue
        d0 = _read_tree0(item)
        if d0 is None:
            anomalies.append({"item_id": iid, "reason": "no_tree0"})
            continue
        actual_locs = d0.get("actual_locs")
        beliefs = d0.get("beliefs")
        if not isinstance(actual_locs, list) or not actual_locs:
            anomalies.append({"item_id": iid, "reason": "bad_actual_locs"})
            continue
        if not isinstance(beliefs, list) or not beliefs:
            anomalies.append({"item_id": iid, "reason": "bad_beliefs"})
            continue
        T = len(actual_locs)
        root = beliefs[0]
        moves = derive_moves(actual_locs)
        event_lines = render_event_lines(moves, T)

        # witness-structured gold answer key per layer (own reader of gold_beliefs).
        witness_canons = []
        for t in range(T):
            g = beliefs[t] if t < len(beliefs) else None
            witness_canons.append(belief_schema.canon_state(g) if isinstance(g, dict) else None)

        # leak-free reconstruction canon per layer (diagnostic).
        leakfree_states = leakfree_traj(root, moves, T)
        leakfree_canons = [belief_schema.canon_state(s) for s in leakfree_states]

        rows.append({
            "item_id": iid,
            "n_layers": T,
            "moves": [[t, obj, loc] for (t, obj, loc) in moves],
            "event_lines": event_lines,
            "gold_belief_canons": witness_canons,              # E5 correct state (witness gold)
            "gold_belief_canons_leakfree": leakfree_canons,    # diagnostic (root+moves-to-all)
        })

    # ---------------- GATE: event lines vs LE render_event_line (22 overlap) --------------
    mismatches = []
    n_lines_compared = 0
    n_overlap_items = 0
    root_line_checks = 0
    for iid in OVERLAP:
        item = by_id.get(iid)
        if item is None:
            mismatches.append({"item_id": iid, "t": None, "reason": "overlap_not_in_data"})
            continue
        space = gsd_space.build_space(item)
        if space is None:
            mismatches.append({"item_id": iid, "t": None, "reason": "build_space_none"})
            continue
        n_overlap_items += 1
        # my event lines for this item
        row = next((r for r in rows if r["item_id"] == iid), None)
        if row is None:
            mismatches.append({"item_id": iid, "t": None, "reason": "no_generated_row"})
            continue
        mine = row["event_lines"]
        # t = 0: compare to the frozen root constant (render_event_line undefined at 0)
        root_line_checks += 1
        n_lines_compared += 1
        if mine[0] != ROOT_EVENT_LINE:
            mismatches.append({"item_id": iid, "t": 0, "mine": mine[0], "ref": ROOT_EVENT_LINE})
        # t = 1..T-1: byte-exact vs render_event_line(space, t)
        for t in range(1, int(space.T)):
            ref = gsd_score.render_event_line(space, t)
            got = mine[t] if t < len(mine) else None
            n_lines_compared += 1
            if got != ref:
                mismatches.append({"item_id": iid, "t": t, "mine": got, "ref": ref})

    # ---------------- ANCHORS: belief trajectory vs facts_oracle.gold_beliefs (201) -------
    # anchor-1: leak-free reconstruction vs witness gold (expected to diverge structurally)
    # anchor-2: our witness reader vs facts_oracle.gold_beliefs (validates the own reader)
    a1_items_full = 0
    a1_layer_total = 0
    a1_layer_match = 0
    a2_items_full = 0
    a2_layer_total = 0
    a2_layer_match = 0
    a2_mismatch_items = []
    for r in rows:
        iid = r["item_id"]
        item = by_id[iid]
        fo = facts_oracle.gold_beliefs(item)
        fo_canons = [belief_schema.canon_state(fo[t]) if (fo and t < len(fo) and isinstance(fo[t], dict)) else None
                     for t in range(r["n_layers"])]
        # anchor-1
        lf = r["gold_belief_canons_leakfree"]
        ok1 = True
        for t in range(r["n_layers"]):
            a1_layer_total += 1
            if lf[t] == fo_canons[t]:
                a1_layer_match += 1
            else:
                ok1 = False
        if ok1:
            a1_items_full += 1
        # anchor-2
        wt = r["gold_belief_canons"]
        ok2 = True
        for t in range(r["n_layers"]):
            a2_layer_total += 1
            if wt[t] == fo_canons[t]:
                a2_layer_match += 1
            else:
                ok2 = False
        if ok2:
            a2_items_full += 1
        else:
            a2_mismatch_items.append(iid)

    # ---------------- EXTENDED (non-gate) event-line cross-check on ALL 201 items ---------
    ext_layers = 0
    ext_mismatches = []
    for r in rows:
        item = by_id[r["item_id"]]
        space = gsd_space.build_space(item)
        if space is None:
            ext_mismatches.append({"item_id": r["item_id"], "t": None, "reason": "build_space_none"})
            continue
        for t in range(1, int(space.T)):
            ref = gsd_score.render_event_line(space, t)
            got = r["event_lines"][t] if t < len(r["event_lines"]) else None
            ext_layers += 1
            if got != ref:
                ext_mismatches.append({"item_id": r["item_id"], "t": t, "mine": got, "ref": ref})

    # per-event-layer move-multiplicity census (documents which render branches fire)
    mult_census = {}
    for r in rows:
        by_t = {}
        for (t, _o, _l) in r["moves"]:
            by_t.setdefault(t, 0)
            by_t[t] += 1
        for t in range(1, r["n_layers"]):
            k = by_t.get(t, 0)
            mult_census[k] = mult_census.get(k, 0) + 1

    gate = "PASS" if not mismatches else "FAIL"

    # ---------------- write artefacts -----------------------------------------------------
    out_jsonl = os.path.join(HERE, "gold_states_b1.jsonl")
    with open(out_jsonl, "w") as f:
        for r in sorted(rows, key=lambda x: x["item_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "gate": gate,
        "n_b1_items_total": len(b1_ids),
        "n_rows_written": len(rows),
        "n_anomalies": len(anomalies),
        "anomalies": anomalies,
        "n_missing_from_data": len(missing),
        "missing_from_data": missing,
        "n_overlap_items": n_overlap_items,
        "n_lines_compared": n_lines_compared,
        "n_root_line_checks": root_line_checks,
        "mismatches": mismatches,
        "extended_all201_event_line_check": {
            "desc": "non-gate: my event lines vs render_event_line(build_space(item),t) on ALL 201",
            "render_layers_compared": ext_layers,
            "mismatches": ext_mismatches,
        },
        "moves_per_event_layer_census": mult_census,
        "anchor_belief_check": {
            "anchor1_leakfree_vs_witness_gold": {
                "desc": "leak-free root+moves-to-all reconstruction vs facts_oracle.gold_beliefs",
                "items_full_match": a1_items_full,
                "items_total": len(rows),
                "layers_match": a1_layer_match,
                "layers_total": a1_layer_total,
            },
            "anchor2_own_witness_reader_vs_facts_oracle": {
                "desc": "our own gold_beliefs[t] reader (canon) vs facts_oracle.gold_beliefs (canon)",
                "items_full_match": a2_items_full,
                "items_total": len(rows),
                "layers_match": a2_layer_match,
                "layers_total": a2_layer_total,
                "mismatch_items": a2_mismatch_items,
            },
        },
        "shards_used": shard_names,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "belief_schema": os.path.join(LE_MUSR, "belief_schema.py"),
            "gsd_space": os.path.join(LE_MUSR, "gsd_space.py"),
            "gsd_score": os.path.join(LE_MUSR, "gsd_score.py"),
            "facts_oracle": os.path.join(LE_MUSR, "facts_oracle.py"),
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t_start, 3),
    }
    with open(os.path.join(HERE, "p3_gate_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "gate": gate,
        "n_b1_items": len(b1_ids),
        "n_rows": len(rows),
        "n_overlap_items": n_overlap_items,
        "n_lines_compared": n_lines_compared,
        "n_event_line_mismatches": len(mismatches),
        "anchor1_leakfree_full": a1_items_full,
        "anchor2_witness_full": a2_items_full,
        "anchor2_mismatch_items": a2_mismatch_items[:5],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
