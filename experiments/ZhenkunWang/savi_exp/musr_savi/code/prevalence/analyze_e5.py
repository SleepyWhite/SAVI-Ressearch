#!/usr/bin/env python3
"""E5 base-rate stage 1 (PREREG_batch1 §7 + REVISION_v1.1 Revision 1). Zero GPU, pure CPU.

Denominator = non-outcome-constructed labels (b1_tuning / b1_train / b1_calib), reported by
label + by chain-count layer; b1_patient reported separately (constructed by voting failure).

CONSENSUS MECHANISM (convention, declared line-by-line in the report — chosen as the caliber
closest to the frozen existing machinery theta_sweep.b1_margins / p4_cache_check.b1_base_margins):

  * Trellis built over RAW BELIEF layer indices with canon-level state equality
    (`_canon_json` = json.dumps(sort_keys), the reanalysis/theta caliber). Adjacent-repeated
    states fold into self-loops (this is the "adjacent repeated states within a chain fold into one; identity at canon level" folding;
    validated separately by reproducing INVENTORY anchor 130/201 folded-mode-length==1).
  * baseline path `base` = stepwise-local-mode: base[0]=layer0 mode; base[t+1]=top1 successor
    of base[t] at raw transition t->t+1. meta[t]={gap,share,top1,tot}.
  * folded-step <-> raw BELIEF index correspondence = IDENTITY: folded/event layer t maps to
    raw BELIEF index t maps to gold layer t (event layers t=1..T-1, T=n_layers=4).
  * consensus successor at layer t = base[t] = top1 successor at the BASELINE PARENT base[t-1]
    (margin index t-1). This pins down "at which parent the consensus top1 is taken" = the baseline parent.
  * consensus SHARE at layer t = meta[t-1]["share"] = top1_freq/tot (NOT the gap; PREREG E5
    defines this quantity as the share). primary consensus threshold: share==1.0; sensitivity band: share>=0.9.
  * available chains at layer t = meta[t-1]["tot"] (= E4's N_parent). eligible iff >=64
    (three-rule upper bound 3/64=4.69%<=4.7%). ineligible layer => undecidable, excluded from denom.

  * CORRECT STATE (REVISION_v1.1 Revision 1) = witness gold canon_state(gold_beliefs[t]) =
    gold_states_b1.gold_belief_canons[t], aligned by the successor's raw BELIEF index t.
  * WRONG at layer t iff bs.canon_state(base[t]) != gold_belief_canons[t] (canon_state literal).
  * a layer is a CONSENSUS-WRONG layer iff (eligible) AND (share meets threshold) AND (wrong).

Products (e5_run/): RESULTS_e5.json, e5_layers.jsonl, manifest.json, E5_report.md, this script.
"""
from __future__ import annotations
import os, sys, json, glob, re, collections, hashlib, random, statistics, platform, time

# Release path resolution (original ran against a frozen internal copy of the b1 shards;
# the module path / cache dir are now env-var overridable with in-repo fallbacks).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, ".."))          # flat musr-cant modules live here
WORKDIR = os.environ.get("MUSR_SAVI_B1_DIR", os.path.join(_CODE, "outputs"))  # cache_b1_shard*.jsonl (run_musr_cant.py --stage b1)
GOLD = os.environ.get("MUSR_SAVI_GOLD_STATES", os.path.join(_HERE, "gold_states_b1.jsonl"))  # from gen_gold_states.py
OUT = _HERE
sys.path.insert(0, _CODE)
import belief_schema as bs
import sc_core as sc

BEL = re.compile(r'^BELIEF\[(\d+)\]:\s*(\{.*\})\s*$', re.M)
ELIGIBLE_MIN = 64
BOOT_B = 10000
BOOT_SEED = 20260718
NON_PATIENT_TAGS = ["b1_tuning", "b1_train", "b1_calib"]

# cross-anchor: the 6 known blind-spot layers (gap=1.0 consensus AND wrong) from the
# b1_margins share=1.0 caliber (E4_report §1 / p4_cache_check ANCHORS).
KNOWN_BLIND = {
    ("object_placements-0006-q3", 2), ("object_placements-0011-q2", 1),
    ("object_placements-0014-q0", 2), ("object_placements-0042-q1", 1),
    ("object_placements-0046-q0", 1), ("object_placements-0050-q1", 1),
}
# b1 gap anchors (edge t gated by margin index t-1) from p4_cache_check ANCHORS (validation gate)
GAP_ANCHORS = {
    "object_placements-0006-q3": {1: 0.875, 2: 1.0, 3: 1.0},
    "object_placements-0011-q2": {1: 1.0, 2: 1.0, 3: 1.0},
    "object_placements-0014-q0": {2: 1.0, 3: 1.0},
    "object_placements-0042-q1": {1: 1.0, 2: 1.0, 3: 0.919643},
    "object_placements-0046-q0": {1: 1.0, 2: 1.0, 3: 1.0},
    "object_placements-0050-q1": {1: 1.0, 2: 1.0, 3: 1.0},
    "object_placements-0047-q2": {2: 0.933333, 3: 0.25},
    "object_placements-0051-q2": {1: 0.989071, 2: 0.989011, 3: 0.906077},
}


def _canon_json(s):
    try:
        return json.dumps(json.loads(s), sort_keys=True)
    except Exception:
        return None


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest() if s is not None else None


def load_b1():
    """(item_id) -> {tag, chains:[{steps,answer,gold,correct,sample_idx}], records:[raw for vote]}."""
    by_item = {}
    n_fail = 0
    for f in sorted(glob.glob(os.path.join(WORKDIR, "cache_b1_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            iid = r["item_id"]
            d = by_item.setdefault(iid, {"tag": r["seed_tag"], "chains": [], "records": []})
            d["records"].append({"answer_idx": r.get("answer_idx"),
                                 "gold_idx": r.get("gold_idx"),
                                 "correct": r.get("correct"),
                                 "sample_idx": r.get("sample_idx")})
            steps, ok = [], True
            for m in BEL.finditer(r["text_head"]):
                c = _canon_json(m.group(2))
                if c is None:
                    ok = False
                    break
                steps.append((int(m.group(1)), c))
            if not ok or not steps:
                n_fail += 1
                continue
            d["chains"].append({"steps": steps, "answer": r.get("answer_idx"),
                                "gold": r.get("gold_idx"), "sample_idx": r.get("sample_idx")})
    return by_item, n_fail


def b1_base_margins(chains):
    """theta_sweep.b1_margins / p4_cache_check.b1_base_margins caliber (verbatim)."""
    trans = collections.defaultdict(collections.Counter)
    layer0 = collections.Counter()
    maxidx = collections.Counter()
    for ch in chains:
        st = ch["steps"]
        idx = {i: s for i, s in st}
        if 0 in idx:
            layer0[idx[0]] += 1
        mi = max(i for i, _ in st)
        maxidx[mi] += 1
        for i, s in st:
            if i + 1 in idx:
                trans[(i, s)][idx[i + 1]] += 1
    if not layer0:
        return [], [], {}
    L = maxidx.most_common(1)[0][0]
    s0 = layer0.most_common(1)[0][0]
    base = [s0]
    margins = []
    meta = {}
    for t in range(0, L):
        d = trans.get((t, base[-1]), collections.Counter())
        tot = sum(d.values())
        if tot == 0:
            break
        mc = d.most_common(2)
        top1, top1f = mc[0]
        gap = (top1f - (mc[1][1] if len(mc) > 1 else 0)) / tot
        margins.append((t, gap))
        meta[t] = {"gap": gap, "share": top1f / tot, "top1": top1, "tot": tot}
        base.append(top1)
    return margins, base, meta


def folded_mode_len(chains):
    flens = []
    for ch in chains:
        seq = [s for _, s in ch["steps"]]
        folded = []
        for s in seq:
            if not folded or folded[-1] != s:
                folded.append(s)
        flens.append(len(folded))
    return collections.Counter(flens).most_common(1)[0][0] if flens else None


def _pairs(canon_str):
    """set of (char, obj) keys from a canon_state string (lowercased already)."""
    o = json.loads(canon_str)
    return frozenset((c, k) for c, d in o.items() for k in d)


def wrong_category(cs_canon, gold_c, root_canon):
    """Diagnostic decomposition of a canon mismatch (cs != gold)."""
    if cs_canon == gold_c:
        return "correct"
    pc, pg = _pairs(cs_canon), _pairs(gold_c)
    if pc != pg:
        extra = bool(pc - pg)
        missing = bool(pg - pc)
        if extra and missing:
            return "objset_both"
        return "objset_extra" if extra else "objset_missing"
    if cs_canon == root_canon:
        return "stuck_at_root"
    return "samescheme_loc_diff"


def directional_consistent(cs_canon, gold_c):
    """facts_oracle caliber: does the consensus state reproduce EVERY gold (char,obj)->loc pair?
    Extra model objects are ignored (directional). Isolates genuine tracking errors from
    object-set verbosity. Returns True iff every gold pair present+equal in cs."""
    cs = json.loads(cs_canon)
    gd = json.loads(gold_c)
    for c, d in gd.items():
        cc = cs.get(c)
        if not isinstance(cc, dict):
            return False
        for k, loc in d.items():
            if cc.get(k) != loc:
                return False
    return True


def main():
    t_start = time.time()
    by_item, n_fail = load_b1()

    gold = {}
    for line in open(GOLD):
        r = json.loads(line)
        gold[r["item_id"]] = r
    assert all(iid in gold for iid in by_item), "missing gold for some item"

    # ---- ANCHOR GATE 1: folding 130/201 ------------------------------------------------
    n_mode1 = sum(1 for iid, d in by_item.items()
                  if folded_mode_len(d["chains"]) == 1)
    folding_gate = {"n_items": len(by_item), "n_folded_mode_len_1": n_mode1,
                    "expected": 130, "pass": n_mode1 == 130}
    if not folding_gate["pass"]:
        print("FOLDING ANCHOR FAILED: %d != 130" % n_mode1)
        json.dump({"folding_gate": folding_gate}, open(os.path.join(OUT, "e5_ANCHOR_FAIL.json"), "w"), indent=2)
        sys.exit(1)
    print("[e5] folding anchor PASS: %d/201 folded-mode-length==1" % n_mode1)

    # ---- per-item baseline path + meta -------------------------------------------------
    base_by, meta_by = {}, {}
    for iid, d in by_item.items():
        chains = d["chains"]
        if len(chains) >= 4:
            _, base, meta = b1_base_margins(chains)
        else:
            base, meta = [], {}
        base_by[iid] = base
        meta_by[iid] = meta

    # ---- ANCHOR GATE 2: b1 gap anchors (8 patient items) -------------------------------
    gap_gate = {"pass": True, "rows": {}}
    for iid, want in GAP_ANCHORS.items():
        meta = meta_by[iid]
        got, ok = {}, True
        for t, w in want.items():
            g = round(meta.get(t - 1, {}).get("gap", 1.0), 6)
            got[t] = g
            if abs(g - round(w, 6)) > 1e-9:
                ok = False
        gap_gate["rows"][iid] = {"want": want, "got": got, "match": ok}
        gap_gate["pass"] = gap_gate["pass"] and ok
    if not gap_gate["pass"]:
        print("GAP ANCHOR FAILED")
        json.dump({"gap_gate": gap_gate}, open(os.path.join(OUT, "e5_ANCHOR_FAIL.json"), "w"), indent=2)
        sys.exit(1)
    print("[e5] b1 gap anchor PASS (%d items)" % len(GAP_ANCHORS))

    # ---- per-layer table ---------------------------------------------------------------
    layer_rows = []           # every candidate event layer (t=1..T-1) for every item
    item_summ = {}            # iid -> dict
    for iid, d in by_item.items():
        tag = d["tag"]
        chains = d["chains"]
        base = base_by[iid]
        meta = meta_by[iid]
        g = gold[iid]
        T = g["n_layers"]                                  # =4 for all
        gcanon = g["gold_belief_canons"]
        # voting over ALL chains (N = all raw records)
        recs = d["records"]
        vr = sc.vote_at_rung(recs, len(recs))
        voting_wrong = (not vr.mode_correct)

        # canon_state set along the WHOLE baseline path (slow-indexing diagnostic)
        base_canon_all = set()
        root_canon = None
        if base:
            for i, bcj in enumerate(base):
                cc = bs.canon_state(json.loads(bcj))
                base_canon_all.add(cc)
                if i == 0:
                    root_canon = cc

        rows_this = []
        for t in range(1, T):
            mi = t - 1
            m = meta.get(mi)
            gold_c = gcanon[t] if t < len(gcanon) else None
            gold_sha = _sha(gold_c)
            if m is None:
                # baseline path did not reach this layer -> no consensus successor
                row = {"item_id": iid, "seed_tag": tag, "layer": t,
                       "n_avail_chains": (m["tot"] if m else 0),
                       "eligible": False, "consensus_share": None,
                       "consensus_gap": None,
                       "consensus_succ_canon_sha": None, "gold_canon_sha": gold_sha,
                       "is_consensus_share1": None, "is_consensus_share09": None,
                       "is_wrong": None,
                       "consensus_wrong_share1": False, "consensus_wrong_share09": False,
                       "undecidable": True, "undecidable_reason": "baseline_path_short",
                       "fold_note": "no consensus successor (baseline ended before layer)"}
                rows_this.append(row)
                continue
            tot = m["tot"]
            share = m["share"]
            gap = m["gap"]
            cj = base[t] if t < len(base) else None
            cs_canon = bs.canon_state(json.loads(cj)) if cj is not None else None
            cs_sha = _sha(cs_canon)
            eligible = tot >= ELIGIBLE_MIN
            is_wrong = (cs_canon != gold_c) if (cs_canon is not None and gold_c is not None) else None
            is_cons1 = (share == 1.0)
            is_cons09 = (share >= 0.9)
            cw1 = bool(eligible and is_cons1 and (is_wrong is True))
            cw09 = bool(eligible and is_cons09 and (is_wrong is True))
            # diagnostics (do not change the primary judgment)
            wcat = wrong_category(cs_canon, gold_c, root_canon) if (cs_canon and gold_c) else None
            dir_ok = directional_consistent(cs_canon, gold_c) if (cs_canon and gold_c) else None
            dir_wrong = (dir_ok is False)                         # facts_oracle-caliber wrong
            gold_reached = (gold_c in base_canon_all) if gold_c else None  # slow-indexing check
            cw1_dir = bool(eligible and is_cons1 and dir_wrong)
            row = {"item_id": iid, "seed_tag": tag, "layer": t,
                   "n_avail_chains": tot, "eligible": eligible,
                   "consensus_share": share, "consensus_gap": gap,
                   "consensus_succ_canon_sha": cs_sha, "gold_canon_sha": gold_sha,
                   "is_consensus_share1": is_cons1, "is_consensus_share09": is_cons09,
                   "is_wrong": is_wrong,
                   "consensus_wrong_share1": cw1, "consensus_wrong_share09": cw09,
                   "wrong_category": wcat,
                   "directional_consistent": dir_ok,
                   "consensus_wrong_share1_directional": cw1_dir,
                   "gold_reached_elsewhere_on_baseline": gold_reached,
                   "undecidable": (not eligible),
                   "undecidable_reason": (None if eligible else "avail_chains_lt_64"),
                   "fold_note": "base[t]=top1 succ at baseline parent base[t-1]; raw idx t == gold layer t"}
            rows_this.append(row)
        layer_rows.extend(rows_this)

        decidable = [r for r in rows_this if r["eligible"]]
        n_dec = len(decidable)
        item_summ[iid] = {
            "seed_tag": tag, "n_chains": len(chains), "n_records": len(recs),
            "n_event_layers": T - 1,
            "n_decidable_layers": n_dec,
            "no_decidable_layer": (n_dec == 0),
            "n_consensus_wrong_share1": sum(1 for r in decidable if r["consensus_wrong_share1"]),
            "n_consensus_wrong_share09": sum(1 for r in decidable if r["consensus_wrong_share09"]),
            "has_consensus_wrong_share1": any(r["consensus_wrong_share1"] for r in decidable),
            "has_consensus_wrong_share09": any(r["consensus_wrong_share09"] for r in decidable),
            "voting_wrong": voting_wrong,
            "voting_mode_idx": vr.mode_idx, "voting_mode_correct": vr.mode_correct,
            "gold_idx": recs[0]["gold_idx"] if recs else None,
        }

    # ---- write per-layer JSONL ---------------------------------------------------------
    with open(os.path.join(OUT, "e5_layers.jsonl"), "w") as f:
        for r in layer_rows:
            f.write(json.dumps(r) + "\n")

    # ---- aggregate by label ------------------------------------------------------------
    def agg_label(iids, band):
        """band in {'share1','share09'}. Returns dict of item/layer stats over given iids."""
        key_item = "has_consensus_wrong_%s" % band
        key_layer = "consensus_wrong_%s" % band
        items = [item_summ[i] for i in iids]
        decidable_items = [i for i in iids if not item_summ[i]["no_decidable_layer"]]
        no_dec = [i for i in iids if item_summ[i]["no_decidable_layer"]]
        # item-level base rate over decidable items
        n_dec_items = len(decidable_items)
        n_pos = sum(1 for i in decidable_items if item_summ[i][key_item])
        rate = (n_pos / n_dec_items) if n_dec_items else None
        # layer-level
        rows = [r for r in layer_rows if r["item_id"] in set(iids)]
        dec_layers = [r for r in rows if r["eligible"]]
        cw_layers = [r for r in dec_layers if r[key_layer]]
        undec_layers = [r for r in rows if not r["eligible"]]
        return {
            "n_items": len(iids),
            "n_items_decidable": n_dec_items,
            "n_items_no_decidable_layer": len(no_dec),
            "ids_no_decidable_layer": sorted(no_dec),
            "n_items_with_consensus_wrong": n_pos,
            "item_base_rate_over_decidable_items": rate,
            "n_candidate_layers": len(rows),
            "n_decidable_layers": len(dec_layers),
            "n_undecidable_layers": len(undec_layers),
            "undecidable_layer_fraction": (len(undec_layers) / len(rows)) if rows else None,
            "n_consensus_wrong_layers": len(cw_layers),
            "layer_fraction_consensus_wrong_over_decidable": (len(cw_layers) / len(dec_layers)) if dec_layers else None,
        }

    def bootstrap_item_rate(iids, band, seed):
        key_item = "has_consensus_wrong_%s" % band
        decidable_items = [i for i in iids if not item_summ[i]["no_decidable_layer"]]
        if not decidable_items:
            return None
        vals = [1 if item_summ[i][key_item] else 0 for i in decidable_items]
        n = len(vals)
        rng = random.Random(seed)
        boots = []
        for _ in range(BOOT_B):
            s = sum(vals[rng.randrange(n)] for _ in range(n))
            boots.append(s / n)
        boots.sort()
        lo = boots[int(0.025 * BOOT_B)]
        hi = boots[int(0.975 * BOOT_B)]
        return {"point": statistics.mean(vals), "ci95": [lo, hi], "n_decidable_items": n, "B": BOOT_B, "seed": seed}

    # by-label + by-chain-count stratification
    items_by_tag = collections.defaultdict(list)
    for iid, d in by_item.items():
        items_by_tag[d["tag"]].append(iid)

    # chain-count strata (per item n_chains) for the stratified report
    def chain_stratum(n):
        if n >= 200:
            return ">=200"
        if n >= 64:
            return "64-199"
        return "<64"

    results = {"per_label": {}, "per_label_sensitivity_share09": {},
               "per_chaincount_stratum": {}}
    for tag, iids in items_by_tag.items():
        results["per_label"][tag] = agg_label(iids, "share1")
        results["per_label"][tag]["bootstrap_item_base_rate"] = bootstrap_item_rate(iids, "share1", BOOT_SEED)
        results["per_label_sensitivity_share09"][tag] = agg_label(iids, "share09")
        results["per_label_sensitivity_share09"][tag]["bootstrap_item_base_rate"] = bootstrap_item_rate(iids, "share09", BOOT_SEED)

    # chain-count stratum crossed with label
    strata = collections.defaultdict(list)
    for iid, d in by_item.items():
        strata[(d["tag"], chain_stratum(len(d["chains"])))].append(iid)
    for (tag, stratum), iids in sorted(strata.items()):
        results["per_chaincount_stratum"]["%s|%s" % (tag, stratum)] = {
            "n_items": len(iids),
            "median_n_chains": statistics.median(len(by_item[i]["chains"]) for i in iids),
            **agg_label(iids, "share1"),
        }

    # non-patient pooled denominator (the reportable base-rate denominator)
    nonpat = [i for t in NON_PATIENT_TAGS for i in items_by_tag.get(t, [])]
    results["nonpatient_pooled"] = agg_label(nonpat, "share1")
    results["nonpatient_pooled"]["bootstrap_item_base_rate"] = bootstrap_item_rate(nonpat, "share1", BOOT_SEED)
    results["nonpatient_pooled_sensitivity_share09"] = agg_label(nonpat, "share09")
    results["nonpatient_pooled_sensitivity_share09"]["bootstrap_item_base_rate"] = bootstrap_item_rate(nonpat, "share09", BOOT_SEED)

    # ---- overlap with voting error (per label; among items with >=1 consensus-wrong layer) --
    def overlap_voting(iids, band):
        key_item = "has_consensus_wrong_%s" % band
        pos = [i for i in iids if (not item_summ[i]["no_decidable_layer"]) and item_summ[i][key_item]]
        if not pos:
            return {"n_items_with_consensus_wrong": 0, "n_voting_wrong": 0, "overlap_fraction": None}
        nvw = sum(1 for i in pos if item_summ[i]["voting_wrong"])
        return {"n_items_with_consensus_wrong": len(pos), "n_voting_wrong": nvw,
                "overlap_fraction": nvw / len(pos),
                "ids": sorted(pos),
                "voting_wrong_ids": sorted(i for i in pos if item_summ[i]["voting_wrong"])}
    results["voting_overlap"] = {tag: overlap_voting(iids, "share1")
                                 for tag, iids in items_by_tag.items()}
    results["voting_overlap"]["nonpatient_pooled"] = overlap_voting(nonpat, "share1")
    results["voting_overlap_sensitivity_share09"] = {
        tag: overlap_voting(iids, "share09") for tag, iids in items_by_tag.items()}
    results["voting_overlap_sensitivity_share09"]["nonpatient_pooled"] = overlap_voting(nonpat, "share09")

    # ---- cross-anchor: 6 blind-spot layers on b1_patient -------------------------------
    my_cw_patient = set()
    for r in layer_rows:
        if by_item[r["item_id"]]["tag"] == "b1_patient" and r["consensus_wrong_share1"]:
            my_cw_patient.add((r["item_id"], r["layer"]))
    # per-blind-layer detail
    blind_detail = []
    layer_index = {(r["item_id"], r["layer"]): r for r in layer_rows}
    for (iid, t) in sorted(KNOWN_BLIND):
        r = layer_index.get((iid, t))
        blind_detail.append({
            "item_id": iid, "layer": t,
            "share": r["consensus_share"] if r else None,
            "gap": r["consensus_gap"] if r else None,
            "eligible": r["eligible"] if r else None,
            "is_wrong": r["is_wrong"] if r else None,
            "my_consensus_wrong_share1": (iid, t) in my_cw_patient,
        })
    extra_cw = sorted(my_cw_patient - KNOWN_BLIND)
    extra_detail = []
    for (iid, t) in extra_cw:
        r = layer_index[(iid, t)]
        extra_detail.append({"item_id": iid, "layer": t, "seed_tag": by_item[iid]["tag"],
                             "share": r["consensus_share"], "gap": r["consensus_gap"],
                             "eligible": r["eligible"], "is_wrong": r["is_wrong"]})
    anchor = {
        "known_blind_layers": sorted("%s|t%d" % (i, t) for i, t in KNOWN_BLIND),
        "my_consensus_wrong_share1_on_patient": sorted("%s|t%d" % (i, t) for i, t in my_cw_patient),
        "blind_reproduced": sorted("%s|t%d" % (i, t) for i, t in (my_cw_patient & KNOWN_BLIND)),
        "blind_missed": sorted("%s|t%d" % (i, t) for i, t in (KNOWN_BLIND - my_cw_patient)),
        "extra_consensus_wrong_not_in_blind": ["%s|t%d (tag=%s)" % (d["item_id"], d["layer"], d["seed_tag"]) for d in extra_detail],
        "blind_detail": blind_detail,
        "extra_detail": extra_detail,
        "all_6_reproduced": (my_cw_patient & KNOWN_BLIND) == KNOWN_BLIND,
    }

    # ---- DIAGNOSTICS (do NOT change the primary judgment; disclose drivers) ------------
    def diagnostics(iids):
        rows = [r for r in layer_rows if r["item_id"] in set(iids) and r["eligible"]]
        cw = [r for r in rows if r["consensus_wrong_share1"]]
        decomp = collections.Counter(r["wrong_category"] for r in cw)
        # slow-indexing: of consensus-wrong layers, how many have gold[t] reached elsewhere
        reached_late = sum(1 for r in cw if r["gold_reached_elsewhere_on_baseline"])
        never_reached = sum(1 for r in cw if r["gold_reached_elsewhere_on_baseline"] is False)
        # directional caliber (facts_oracle: ignore extra objects)
        cw_dir = [r for r in rows if r["consensus_wrong_share1_directional"]]
        return {
            "n_consensus_wrong_layers_share1": len(cw),
            "wrong_category_decomposition": dict(decomp),
            "slow_indexing_gold_reached_late": reached_late,
            "slow_indexing_gold_never_reached": never_reached,
            "n_consensus_wrong_layers_directional": len(cw_dir),
            "note": ("stuck_at_root/objset_* are canon_state-literal wrongs; directional caliber "
                     "(facts_oracle) counts a layer wrong only if it fails to reproduce some gold "
                     "(char,obj)->loc pair (extra model objects ignored). gold_reached_late = the "
                     "gold layer state IS hit by the baseline path at some OTHER raw index "
                     "(model updates but on a slower event index than gold's)."),
        }

    def dir_agg(iids):
        rows = [r for r in layer_rows if r["item_id"] in set(iids)]
        dec = [r for r in rows if r["eligible"]]
        cwd = [r for r in dec if r["consensus_wrong_share1_directional"]]
        by_it = collections.defaultdict(list)
        for r in dec:
            by_it[r["item_id"]].append(r["consensus_wrong_share1_directional"])
        pos_items = [i for i, v in by_it.items() if any(v)]
        n_dec_items = len(by_it)
        return {
            "n_decidable_items": n_dec_items,
            "n_items_with_directional_consensus_wrong": len(pos_items),
            "item_base_rate_directional": (len(pos_items) / n_dec_items) if n_dec_items else None,
            "n_decidable_layers": len(dec),
            "n_consensus_wrong_layers_directional": len(cwd),
            "layer_fraction_directional": (len(cwd) / len(dec)) if dec else None,
        }

    results["diagnostics_by_label"] = {tag: diagnostics(iids) for tag, iids in items_by_tag.items()}
    results["diagnostics_by_label"]["nonpatient_pooled"] = diagnostics(nonpat)
    results["directional_caliber_by_label"] = {tag: dir_agg(iids) for tag, iids in items_by_tag.items()}
    results["directional_caliber_by_label"]["nonpatient_pooled"] = dir_agg(nonpat)

    # ---- interpretation-table docket (executor only records which band the number falls in) --
    def band_of(rate):
        if rate is None:
            return "undefined"
        if rate >= 0.10:
            return ">=10%: 头条强化"
        if rate >= 0.02:
            return "2%-10%: 存在且不罕见"
        return "<2%: 罕见但有原理刻画"
    docket = {}
    for tag in NON_PATIENT_TAGS:
        rate = results["per_label"][tag]["item_base_rate_over_decidable_items"]
        docket[tag] = {"item_base_rate": rate, "band": band_of(rate)}
    docket["nonpatient_pooled"] = {
        "item_base_rate": results["nonpatient_pooled"]["item_base_rate_over_decidable_items"],
        "band": band_of(results["nonpatient_pooled"]["item_base_rate_over_decidable_items"])}
    docket["nonpatient_pooled_layer_fraction"] = {
        "layer_fraction": results["nonpatient_pooled"]["layer_fraction_consensus_wrong_over_decidable"],
        "band": band_of(results["nonpatient_pooled"]["layer_fraction_consensus_wrong_over_decidable"])}

    out = {
        "meta": {
            "task": "E5 base-rate stage 1",
            "prereg": "PREREG_batch1.md §7 + REVISION_v1.1 修订一 (witness gold)",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu_used": False, "n_forwards": 0,
            "interpreter": sys.executable,
            "eligible_min_chains": ELIGIBLE_MIN,
            "bootstrap_B": BOOT_B, "bootstrap_seed": BOOT_SEED,
            "n_b1_parse_fail": n_fail,
            "consensus_caliber": "theta_sweep.b1_margins baseline path; consensus succ=base[t] "
                                 "(top1 at baseline parent base[t-1]); share=meta[t-1].share; "
                                 "eligible iff meta[t-1].tot>=64; wrong iff canon_state(base[t]) "
                                 "!= gold_belief_canons[t] (witness gold, raw idx t==gold layer t)",
        },
        "folding_gate": folding_gate,
        "gap_anchor_gate": gap_gate,
        "totals": {
            "n_items": len(by_item),
            "items_per_tag": {t: len(iids) for t, iids in items_by_tag.items()},
            "chains_per_tag": {t: sum(len(by_item[i]["chains"]) for i in iids)
                               for t, iids in items_by_tag.items()},
            "n_candidate_layers_total": len(layer_rows),
        },
        "per_label": results["per_label"],
        "per_label_sensitivity_share09": results["per_label_sensitivity_share09"],
        "nonpatient_pooled": results["nonpatient_pooled"],
        "nonpatient_pooled_sensitivity_share09": results["nonpatient_pooled_sensitivity_share09"],
        "per_chaincount_stratum": results["per_chaincount_stratum"],
        "voting_overlap": results["voting_overlap"],
        "voting_overlap_sensitivity_share09": results["voting_overlap_sensitivity_share09"],
        "diagnostics_by_label": results["diagnostics_by_label"],
        "directional_caliber_by_label": results["directional_caliber_by_label"],
        "cross_anchor_blind6": anchor,
        "interpretation_docket": docket,
        "wall_seconds": round(time.time() - t_start, 2),
    }
    json.dump(out, open(os.path.join(OUT, "RESULTS_e5.json"), "w"), indent=2, default=str)

    # ---- manifest ----------------------------------------------------------------------
    manifest = {
        "experiment": "E5 base-rate stage 1",
        "cpu_only": True, "gpu_used": False, "n_forwards": 0, "model_loaded": False,
        "interpreter": sys.executable,
        "python": platform.python_version(), "platform": platform.platform(),
        "inputs_readonly": {
            "b1_shards": sorted(os.path.basename(p) for p in glob.glob(os.path.join(WORKDIR, "cache_b1_shard*.jsonl"))),
            "gold_states": GOLD,
        },
        "imports": {"belief_schema": bs.__file__, "sc_core": sc.__file__},
        "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "outputs": ["RESULTS_e5.json", "e5_layers.jsonl", "manifest.json", "E5_report.md", "analyze_e5.py"],
        "seeds": {"bootstrap": BOOT_SEED},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)

    # ---- console summary ---------------------------------------------------------------
    print("[e5] n_items=%d layers=%d parse_fail=%d wall=%.1fs" %
          (len(by_item), len(layer_rows), n_fail, out["wall_seconds"]))
    for tag in ["b1_tuning", "b1_train", "b1_calib", "b1_patient"]:
        a = results["per_label"][tag]
        br = a["bootstrap_item_base_rate"]
        print("  %-11s items=%3d dec_items=%3d nodec=%3d rate=%s CI=%s | dec_layers=%d cw=%d layerfrac=%s undec_layers=%d" % (
            tag, a["n_items"], a["n_items_decidable"], a["n_items_no_decidable_layer"],
            ("%.4f" % a["item_base_rate_over_decidable_items"]) if a["item_base_rate_over_decidable_items"] is not None else "NA",
            ("[%.4f,%.4f]" % tuple(br["ci95"])) if br else "NA",
            a["n_decidable_layers"], a["n_consensus_wrong_layers"],
            ("%.4f" % a["layer_fraction_consensus_wrong_over_decidable"]) if a["layer_fraction_consensus_wrong_over_decidable"] is not None else "NA",
            a["n_undecidable_layers"]))
    print("[e5] blind6 reproduced=%s missed=%s extra=%d" % (
        anchor["all_6_reproduced"], anchor["blind_missed"], len(anchor["extra_detail"])))
    return out


if __name__ == "__main__":
    main()
