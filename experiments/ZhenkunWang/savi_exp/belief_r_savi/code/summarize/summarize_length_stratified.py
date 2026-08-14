#!/usr/bin/env python
"""Length-stratified reanalysis: how much of L1/L4/L5's 0.545–0.588 is explainable by γ3 sentence length?

Pure CPU reanalysis, no model runs. Reads the readouts already on disk in
outputs/{probe_relation,probe_nli,incontext_choice}; only the slicing changes.

============================================================================
Why this is being done
============================================================================
L6's G2 by-product (`summarize_pairwise_2afc.length_confound`) found the gold label is
collinear with γ3 sentence length: on lv=5, γ3 character count → gold-REQ AUC 0.762,
**γ3−γ1 character difference 0.800**, while γ1 character count (the sentence shared within
a scenario and carrying no manipulation) is a clean null control at 0.497.
The best results of the three acquisition routes — L1 0.588 / L4 0.560 / L5 0.545 — are all
below this character-counting rule, so we must ask: **are they measuring three projections
of the same surface cue.**

Either outcome is a result: signal goes to zero after controlling for length → the three
routes are not independent evidence, "acquisition side closed" is reinforced; residual
remains → there is a weak semantic signal beyond length, and "no semantic signal at all"
must be downgraded one notch.

============================================================================
Pre-registration (frozen before running numbers, see the assignment doc; this file is its executable copy)
============================================================================
[Primary readout] AUC within length strata.
  Stratification variable = γ3 character count − γ1 character count (the strongest of the
  three confound features, AUC 0.800 on lv=5).
  Binning   = quartiles of that variable **within the current subset** (cuts computed by
  the subset itself, never borrowed across subsets).
  Reporting condition = REQ and ALT **each ≥15 scenarios** within a bin; otherwise marked
  `n too small` and excluded from adjudication.
  Aggregation = **sample-size-weighted average** over reportable bins, with per-bin detail
  printed alongside.
[Secondary readout] Increment with length as a covariate. Logistic regression
  `gold REQ ~ z(length) + z(model score)`; report the model-score coefficient with 95% CI,
  plus ΔAUC = AUC(length+score) − AUC(length only).
  If the two estimators point in different directions, **report both side by side**, don't
  pick one.
  ΔAUC is **in-sample** — adding a variable can only not-decrease — so a **permutation
  null line with shuffled scores** (200 draws) is attached to show how much comes "for
  free"; without it the in-sample ΔAUC cannot be read.
[Controls]
  Positive control: before stratifying, reproduce the three routes' registered headline
          numbers (L1 relation@7B 0.588 / L4 bart 0.560 / L5 7B 0.545), computed with the
          **existing scripts' own functions**; stop immediately if not reproduced.
  Null control: rerun with γ1 character count as the stratification variable. γ1 carries
          no manipulation, so stratifying on it should not change conclusions; if it also
          kills the signal, the binning itself is producing artifacts.
[Scope] Statistical unit = scenario (dataset_id), twins enter/leave together; primary
  readout set = `agreement_lv == 5` exact match (the 4 lv=6 rows are a data anomaly,
  always excluded); both full 872 and lv=5 391 are reported.
[CI] Scenario-clustered stratified bootstrap, 5000 draws, seed=0, same method as
  `summarize_generative_ci.py`: REQ and ALT each resample scenarios within their own
  stratum (after binning this is resampling within "bin × class"; bin sizes are fixed by
  the covariate, not drawn, so they are not resampled).
[Criteria] Based on the **best post-stratification result of each route** on lv=5:
  ≤0.55 → the original signal is explainable by length | ≥0.60 and all three routes agree
  in direction → residual signal beyond length |
  0.55–0.60 or directions disagree → report honestly as uncertain, don't force a bin.
  "Direction agreement" uses the **strict reading**: all cells of a route (L1's three
  wordings / L4's three NLI models / L5's six models) fall on the same side of 0.5, and the
  three routes are on the same side. The lenient reading (only each route's best cell) is
  printed as well. The strict reading is used because the core reason L1 was judged
  negative was precisely that its three wordings contradict each other in direction;
  picking one cell after the fact would overturn that criterion.

============================================================================
Things not allowed (hard rules of this line)
============================================================================
- **Report all** three wordings / three NLI models / six models; don't pick the best cell
  to tell a story.
- After seeing results, don't change bins, criteria, or subsets.
- If "instrument broke" cannot be distinguished from "really no signal", report instrument
  broke.

Usage
  python scripts/summarize_length_stratified.py > outputs/ci/length_stratified.txt
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The positive control must be computed with the existing scripts' own functions;
# otherwise "reproduction" only reproduces my newly written bug.
from summarize_probe_relation import auc as auc_ref            # noqa: E402
from summarize_incontext_choice import bal_acc as balacc_ref   # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# Belief-R dataset CSV. The data is not distributed with this repo (see the README's data section);
# point env var BELIEF_R_CSV at a local copy; falls back to <repo>/data/queries_time_t1.csv if unset.
CSV = os.environ.get('BELIEF_R_CSV') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'queries_time_t1.csv')
D_REL = os.path.join(ROOT, 'outputs', 'probe_relation')
D_NLI = os.path.join(ROOT, 'outputs', 'probe_nli')
D_IC = os.path.join(ROOT, 'outputs', 'incontext_choice')

ITERS, SEED = 5000, 0
PERM = 200                 # permutation null draws; only marks the "for free" magnitude, not adjudicated
MIN_PER_CLASS = 15         # reporting condition: REQ / ALT each >= this many scenarios per bin
TEMPLATES = ('sufficiency', 'relation', 'certainty')
NAME = {'Qwen_Qwen2.5_0.5B_Instruct': '0.5B', 'Qwen_Qwen2.5_1.5B_Instruct': '1.5B',
        'Qwen_Qwen2.5_3B_Instruct': '3B', 'Qwen_Qwen2.5_7B_Instruct': '7B',
        'Qwen_Qwen3_4B': 'Qwen3-4B', 'meta_llama_Llama_3.1_8B_Instruct': 'Llama-8B'}
MODELS = ('0.5B', '1.5B', '3B', '7B', 'Qwen3-4B', 'Llama-8B')
NLI_SHORT = {'facebook/bart-large-mnli': 'bart-large-mnli',
             'cross-encoder/nli-distilroberta-base': 'distilroberta',
             'typeform/distilbert-base-uncased-mnli': 'distilbert'}
# Previously registered headline numbers (STATUS §6.10 / PRED-S / VERDICT_incontext_choice),
# anchors for the positive control
ANCHORS = {'L1 relation@7B (全量 1744 行合并)': 0.588,
           'L4 bart-large-mnli (lv=5)': 0.560,
           'L5 7B 宽口径 (lv=5, 两排法均值)': 0.545}


# --------------------------------------------------------------------------- data
def scenario_table():
    """Scenario-level table: gold class / agreement_lv / two length features. Three precondition self-checks attached.

    The stratification variable is a **scenario-level** attribute, which presupposes twins
    share γ1/γ3; stratifying without checking amounts to assuming it.
    """
    df = pd.read_csv(CSV)
    lines = df['questions'].str.split('\n')
    df['g1'], df['g3'] = lines.str[0], lines.str[2]
    df['req'] = df['ground_truth'] == 'c'
    g = df.groupby('dataset_id')
    for col in ('g1', 'g3', 'agreement_lv', 'req'):
        bad = int((g[col].nunique() != 1).sum())
        if bad:
            raise AssertionError(f'V1 失败：{bad} 个场景的孪生 {col} 不一致，场景级分层前提不成立')
    pon = df[df.modus == 'ponens'].drop_duplicates('dataset_id').set_index('dataset_id')
    return pd.DataFrame({'req': pon['req'], 'lv': pon['agreement_lv'].astype(int),
                         'd_len': pon['g3'].str.len() - pon['g1'].str.len(),
                         'g1_len': pon['g1'].str.len(),
                         'g3_len': pon['g3'].str.len()}).sort_index()


def load_routes():
    """→ {(route, cell name): (kind, {sid: payload})}. The payload shape depends on kind:

      'auc'    k scores per scenario (L1 k=2, one per twin; L4 k=1); metric = AUC on the
               pooled scores. L1's headline 0.588 is the "merged" cell, i.e. the rows of
               both modi are pooled into one AUC, so that scope is copied verbatim here —
               not changed to per-modus AUC then averaged (that is 0.596, not the headline).
      'balacc' per scenario × two orderings: (usable rows, rows picking REQ); metric = mean
               balanced accuracy over the two orderings. For a binary scorer balanced
               accuracy is identically the AUC, hence directly comparable with the above.
    """
    routes = {}
    for f in sorted(glob.glob(os.path.join(D_REL, '*.jsonl'))):
        tag = NAME[os.path.basename(f)[len('time_t1_'):-len('.jsonl')]]
        recs = [json.loads(l) for l in open(f)]
        for t in TEMPLATES:
            by = {}
            for r in recs:                       # one row each for ponens/tollens, order-independent
                by.setdefault(r['dataset_id'], []).append(r[f'{t}_delta_mean'])
            routes[('L1', f'{tag}/{t}')] = ('auc', {k: np.array(v) for k, v in by.items()})
    for f in sorted(glob.glob(os.path.join(D_NLI, '*.jsonl'))):
        recs = [json.loads(l) for l in open(f)]
        tag = NLI_SHORT.get(recs[0]['model'], recs[0]['model'])
        routes[('L4', tag)] = ('auc', {r['dataset_id']: np.array([r['delta']]) for r in recs})
    per = {}
    for f in sorted(glob.glob(os.path.join(D_IC, '*.jsonl'))):
        stem = os.path.basename(f)[len('time_t1_'):-len('.jsonl')]
        tag, order = NAME[stem.rsplit('_', 2)[0]], '_'.join(stem.rsplit('_', 2)[1:])
        per.setdefault(tag, {})[order] = [json.loads(l) for l in open(f)]
    for tag, d in per.items():
        by = {}
        for j, order in enumerate(('req_first', 'alt_first')):
            for r in d[order]:
                cell = by.setdefault(r['dataset_id'], np.zeros((2, 2)))
                if r['picked_req'] is not None:   # lenient scope: rows where the contract wasn't honored stay out of the denominator
                    cell[j, 0] += 1
                    cell[j, 1] += r['picked_req']
        routes[('L5', tag)] = ('balacc', by)
    return routes


# --------------------------------------------------------------------------- metrics
def auc_fast(pos, neg):
    """Mann-Whitney AUC with average ranks for ties. Numerically identical to
    summarize_probe_relation.auc, but not O(n²), since it runs hundreds of thousands
    of times inside the bootstrap."""
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate((pos, neg)))
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def metric(kind, mat, ireq, ialt):
    """Compute the route's original score-axis readout on the given scenario index sets."""
    if kind == 'auc':
        return auc_fast(mat[ireq].ravel(), mat[ialt].ravel())
    nv_r, np_r = mat[ireq, :, 0].sum(0), mat[ireq, :, 1].sum(0)   # one column per ordering
    nv_a, np_a = mat[ialt, :, 0].sum(0), mat[ialt, :, 1].sum(0)
    if (nv_r == 0).any() or (nv_a == 0).any():
        return float('nan')
    return float(np.mean((np_r / nv_r + (1 - np_a / nv_a)) / 2))


def to_matrix(kind, payload, sids):
    a = np.array([payload[s] for s in sids])
    return a if kind == 'auc' else a.reshape(len(sids), 2, 2)


def boot_metric(kind, mat, ireq, ialt, rng):
    """Scenario-clustered stratified bootstrap: REQ and ALT each resample scenarios within their stratum.

    Clustering is automatic — scenario indices are resampled, and all rows of that scenario
    (twins / two orderings) move along as a whole.
    """
    out = []
    for _ in range(ITERS):
        out.append(metric(kind, mat,
                          ireq[rng.randint(0, len(ireq), len(ireq))],
                          ialt[rng.randint(0, len(ialt), len(ialt))]))
    out = np.array(out, float)
    ok = ~np.isnan(out)
    if ok.sum() < ITERS * 0.9:
        return float('nan'), float('nan'), int((~ok).sum())
    lo, hi = np.percentile(out[ok], [2.5, 97.5])
    return float(lo), float(hi), int((~ok).sum())


def boot_weighted(kind, mat, bins, rng):
    """CI of the weighted average over reportable bins: in each resample all bins are
    resampled together, then combined with fixed weights."""
    w = np.array([len(ir) + len(ia) for ir, ia in bins], float)
    w /= w.sum()
    out = []
    for _ in range(ITERS):
        vals = [metric(kind, mat, ir[rng.randint(0, len(ir), len(ir))],
                       ia[rng.randint(0, len(ia), len(ia))]) for ir, ia in bins]
        out.append(np.dot(w, vals))
    out = np.array(out, float)
    ok = ~np.isnan(out)
    if ok.sum() < ITERS * 0.9:
        return float('nan'), float('nan')
    return tuple(float(x) for x in np.percentile(out[ok], [2.5, 97.5]))


# --------------------------------------------------------------------------- logistic regression
def irls(X, y, ridge=1e-6, steps=60):
    """Newton-Raphson logistic regression for two or three covariates. Hand-written because
    it runs over a hundred thousand times inside the bootstrap; agreement with
    sklearn(penalty=None) is asserted in main — not "trust me"."""
    b = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        g = X.T @ (y - p) - ridge * b
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return b, False
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            return b, True
    return b, False


def scenario_score(kind, payload, sids):
    """Compress the route readout into one number per scenario, as the covariate for logistic regression.

    L1/L4 take the twin mean (γ3 is scenario-level; twins are just two renderings of the
    same scenario); L5 takes the fraction of REQ picks over the four rows (2 orderings ×
    2 modi), lenient scope. All rows unhonored → NaN.
    """
    if kind == 'auc':
        return np.array([np.mean(payload[s]) for s in sids])
    out = []
    for s in sids:
        c = payload[s]
        n = c[:, 0].sum()
        out.append(c[:, 1].sum() / n if n else np.nan)
    return np.array(out)


def z(v):
    sd = np.std(v)
    return (v - np.mean(v)) / sd if sd > 0 else v * 0.0


# --------------------------------------------------------------------------- report
def bin_index(vals, ireq_mask):
    """Cut into quartile bins; returns [(REQ indices in bin, ALT indices in bin, range string)], always length 4."""
    cuts = np.quantile(vals, [.25, .5, .75])
    b = np.digitize(vals, cuts, right=False)
    out = []
    for k in range(4):
        m = b == k
        rng_s = f'[{vals[m].min():g},{vals[m].max():g}]' if m.any() else '空'
        out.append((np.where(m & ireq_mask)[0], np.where(m & ~ireq_mask)[0], rng_s))
    return out


def stratified_block(title, routes, tab, sids, feat, order):
    """Print a full stratification table for one (subset × stratification variable);
    returns {cell name: (weighted average, lo, hi, n reportable bins)}."""
    req = tab.loc[sids, 'req'].to_numpy()
    bins = bin_index(feat, req)
    print(f'\n{"=" * 108}\n{title}\n{"=" * 108}')
    print(f'  四分位切箱（在本子集内算）：'
          + '  '.join(f'箱{k + 1} n={len(ir) + len(ia)} REQ={len(ir)} ALT={len(ia)} '
                      f'{rs}' for k, (ir, ia, rs) in enumerate(bins)))
    keep = [k for k, (ir, ia, _) in enumerate(bins)
            if len(ir) >= MIN_PER_CLASS and len(ia) >= MIN_PER_CLASS]
    for k, (ir, ia, _) in enumerate(bins):
        if k not in keep:
            print(f'  ⚠ 箱{k + 1} REQ={len(ir)} / ALT={len(ia)} 未达 ≥{MIN_PER_CLASS}/'
                  f'{MIN_PER_CLASS} → **n too small**，不参与判定')
    print(f'  可报箱：{[k + 1 for k in keep]}（共 {len(keep)}/4）\n')
    hdr = (f"  {'路线':<4}{'格':<22}{'分层前':>8}{'分层前 95%CI':>18}"
           + ''.join(f'{"箱" + str(k + 1):>9}' for k in range(4))
           + f"{'加权平均':>10}{'加权 95%CI':>18}{'Δ':>8}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    res = {}
    for key in order:
        kind, payload = routes[key]
        mat = to_matrix(kind, payload, sids)
        allr, alla = np.where(req)[0], np.where(~req)[0]
        rng = np.random.RandomState(SEED)
        pre = metric(kind, mat, allr, alla)
        plo, phi, _ = boot_metric(kind, mat, allr, alla, rng)
        cells = []
        for k, (ir, ia, _) in enumerate(bins):
            cells.append(metric(kind, mat, ir, ia) if k in keep else float('nan'))
        rng = np.random.RandomState(SEED)
        wavg = float(np.average([cells[k] for k in keep],
                                weights=[len(bins[k][0]) + len(bins[k][1]) for k in keep]))
        wlo, whi = boot_weighted(kind, mat, [(bins[k][0], bins[k][1]) for k in keep], rng)
        res[key] = (pre, wavg, wlo, whi, len(keep))
        print(f"  {key[0]:<4}{key[1]:<22}{pre:>8.3f}   [{plo:.3f},{phi:.3f}]"
              + ''.join(f'{c:>9.3f}' if not np.isnan(c) else f'{"—":>9}' for c in cells)
              + f'{wavg:>10.3f}   [{wlo:.3f},{whi:.3f}]{wavg - pre:>+8.3f}')
    return res


def covariate_block(routes, tab, sids, feat, order, featname):
    """Secondary readout: logistic regression gold REQ ~ z(length) + z(score).
    Coefficient CI reported together with ΔAUC."""
    y = tab.loc[sids, 'req'].to_numpy().astype(float)
    print(f'\n{"=" * 108}\n次读数：逻辑回归 金标REQ ~ z({featname}) + z(模型打分)，'
          f'{ITERS} 次场景聚类分层 bootstrap，seed={SEED}\n{"=" * 108}')
    print('  自变量都标准化，故系数可横向比；ΔAUC 是样本内的（加变量只会不降），'
          '所以并排给"打乱打分"的置换零线')
    hdr = (f"  {'路线':<4}{'格':<22}{'β(打分)':>9}{'β 95%CI':>18}"
           f"{'AUC(长度)':>10}{'AUC(长+分)':>11}{'ΔAUC':>8}{'置换零线 p95':>13}{'n':>6}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    out = {}
    for key in order:
        kind, payload = routes[key]
        s = scenario_score(kind, payload, sids)
        ok = ~np.isnan(s)
        yy, ff, ss = y[ok], feat[ok], s[ok]
        X1 = np.column_stack([np.ones(ok.sum()), z(ff)])
        X2 = np.column_stack([X1, z(ss)])
        b2, conv = irls(X2, yy)
        e1 = X1 @ irls(X1, yy)[0]
        a1 = auc_fast(e1[yy == 1], e1[yy == 0])
        a2 = auc_fast((X2 @ b2)[yy == 1], (X2 @ b2)[yy == 0])
        rng = np.random.RandomState(SEED)
        ir, ia = np.where(yy == 1)[0], np.where(yy == 0)[0]
        bs = []
        for _ in range(ITERS):
            idx = np.r_[ir[rng.randint(0, len(ir), len(ir))],
                        ia[rng.randint(0, len(ia), len(ia))]]
            bb, c = irls(X2[idx], yy[idx])
            bs.append(bb[2] if c else np.nan)
        bs = np.array(bs, float)
        lo, hi = np.percentile(bs[~np.isnan(bs)], [2.5, 97.5])
        # Permutation null line: how much ΔAUC still comes for free after shuffling the
        # score column (the reading baseline for the in-sample ΔAUC)
        prng = np.random.RandomState(SEED)
        dperm = []
        for _ in range(PERM):
            Xp = X2.copy()
            Xp[:, 2] = X2[prng.permutation(len(yy)), 2]
            ep = Xp @ irls(Xp, yy)[0]
            dperm.append(auc_fast(ep[yy == 1], ep[yy == 0]) - a1)
        p95 = float(np.percentile(dperm, 95))
        out[key] = (b2[2], lo, hi, a2 - a1, p95)
        print(f"  {key[0]:<4}{key[1]:<22}{b2[2]:>+9.3f}   [{lo:+.3f},{hi:+.3f}]"
              f"{a1:>10.3f}{a2:>11.3f}{a2 - a1:>+8.3f}{p95:>13.3f}{int(ok.sum()):>6}"
              + ('' if conv else '  ⚠未收敛'))
    return out


def main():
    tab = scenario_table()
    routes = load_routes()
    order = ([('L1', f'{m}/{t}') for m in MODELS for t in TEMPLATES]
             + [('L4', k) for k in ('bart-large-mnli', 'distilroberta', 'distilbert')]
             + [('L5', m) for m in MODELS])
    missing = [k for k in order if k not in routes]
    if missing:
        raise SystemExit(f'缺读数：{missing}')

    print('# 长度分层重分析（纯 CPU 重分析，无新生成）')
    print(f'# 预注册见本脚本 docstring。bootstrap {ITERS} 次，seed={SEED}，'
          f'报箱条件 REQ/ALT 各 ≥{MIN_PER_CLASS} 场景')
    print(f'# 场景表：{len(tab)} 个场景；lv 分布 {tab.lv.value_counts().to_dict()}；'
          f'REQ {int(tab.req.sum())} / ALT {int((~tab.req).sum())}')
    print('# V1 通过：孪生共享 γ1/γ3/agreement_lv/金标类别（分层变量是场景级属性的前提）')

    sub = {'全量 872': tab.index.to_numpy(),
           'lv=5 391（主读数集）': tab.index[tab.lv == 5].to_numpy()}
    for tag, sids in sub.items():
        print(f'#   {tag}: n={len(sids)}  REQ={int(tab.loc[sids, "req"].sum())} '
              f'ALT={int((~tab.loc[sids, "req"]).sum())}')

    # ------------------------------------------------------------------ positive control
    print(f'\n{"=" * 108}\n【0 正对照】分层前先复现三条路线的既有头条数——'
          f'用**既有脚本自己的函数**算，复现不出就停\n{"=" * 108}')
    rel = [json.loads(l) for l in open(os.path.join(D_REL, 'time_t1_Qwen_Qwen2.5_7B_Instruct.jsonl'))]
    got = {'L1 relation@7B (全量 1744 行合并)':
           auc_ref([r['relation_delta_mean'] for r in rel if r['ground_truth'] == 'c'],
                   [r['relation_delta_mean'] for r in rel if r['ground_truth'] != 'c'])}
    nli = [json.loads(l) for l in open(os.path.join(D_NLI, 'time_t1_facebook_bart_large_mnli.jsonl'))]
    nli5 = [r for r in nli if r['agreement_lv'] == 5]
    got['L4 bart-large-mnli (lv=5)'] = auc_ref([r['delta'] for r in nli5 if r['ground_truth'] == 'c'],
                                               [r['delta'] for r in nli5 if r['ground_truth'] != 'c'])
    ic = {o: [json.loads(l) for l in
              open(os.path.join(D_IC, f'time_t1_Qwen_Qwen2.5_7B_Instruct_{o}.jsonl'))]
          for o in ('req_first', 'alt_first')}
    got['L5 7B 宽口径 (lv=5, 两排法均值)'] = float(np.mean(
        [balacc_ref([r for r in v if r['agreement_lv'] == 5])[0] for v in ic.values()]))
    ok_all = True
    for k, anchor in ANCHORS.items():
        d = abs(got[k] - anchor)
        ok_all &= d <= 0.002
        print(f'  {k:<40} 登记 {anchor:.3f}   复算 {got[k]:.3f}   Δ={d:.4f}   '
              + ('✅ 通过' if d <= 0.002 else '❌ **不通过**'))
    if not ok_all:
        raise SystemExit('正对照不通过 → 报"仪器坏了"，不继续分层。')

    # My vectorized implementation must agree bit-for-bit with the existing functions;
    # otherwise the stratified numbers below are a different quantity
    sids5 = tab.index[tab.lv == 5].to_numpy()
    req5 = tab.loc[sids5, 'req'].to_numpy()
    for key in (('L1', '7B/relation'), ('L4', 'bart-large-mnli')):
        kind, payload = routes[key]
        mat = to_matrix(kind, payload, sids5)
        mine = metric(kind, mat, np.where(req5)[0], np.where(~req5)[0])
        ref = auc_ref(list(mat[np.where(req5)[0]].ravel()), list(mat[np.where(~req5)[0]].ravel()))
        if abs(mine - ref) > 1e-9:
            raise SystemExit(f'V2 失败：{key} 我的 AUC {mine} != 既有实现 {ref}')
    kind, payload = routes[('L5', '7B')]
    mine = metric(kind, to_matrix(kind, payload, sids5), np.where(req5)[0], np.where(~req5)[0])
    if abs(mine - got['L5 7B 宽口径 (lv=5, 两排法均值)']) > 1e-9:
        raise SystemExit(f'V2 失败：L5 我的平衡准确率 {mine} != 既有实现')
    print('  V2 通过：本脚本的向量化实现与既有脚本的函数逐位相同（差 <1e-9），'
          '故下面分层的是同一个量')

    # sklearn cross-check: the hand-written IRLS must agree with the penalty=None implementation
    from sklearn.linear_model import LogisticRegression
    rs = np.random.RandomState(0)
    Xc = np.column_stack([np.ones(300), rs.randn(300), rs.randn(300)])
    yc = (rs.rand(300) < 1 / (1 + np.exp(-(0.3 + Xc[:, 1] - 0.5 * Xc[:, 2])))).astype(float)
    lr = LogisticRegression(penalty=None, max_iter=2000).fit(Xc[:, 1:], yc)
    bmine, _ = irls(Xc, yc)
    dmax = max(abs(bmine[0] - lr.intercept_[0]), *(abs(bmine[1:] - lr.coef_[0])))
    if dmax > 1e-4:
        raise SystemExit(f'V3 失败：自写 IRLS 与 sklearn 差 {dmax}')
    print(f'  V3 通过：自写 IRLS 与 sklearn(penalty=None) 系数最大差 {dmax:.2e}')

    # Strength of the stratification variable itself (recompute STATUS §6.10 to confirm
    # I'm reading the same column)
    print('\n  分层变量自身对金标的 AUC（复算 §6.10，确认字段读对了）：')
    for tag, sids in sub.items():
        r = tab.loc[sids, 'req'].to_numpy()
        for col, nm in (('g3_len', 'γ3 字符数'), ('d_len', 'γ3−γ1 字符差'),
                        ('g1_len', 'γ1 字符数（空对照）')):
            v = tab.loc[sids, col].to_numpy(float)
            print(f'    {tag:<22} {nm:<20} AUC {auc_fast(v[r], v[~r]):.3f}')

    # Extra diagnostic (not a readout, not adjudicated): correlation of the score itself
    # with the stratification variable. Without this column, "L1 generally rises after
    # stratification" would look like a bug; with it we know the score is **inversely**
    # correlated with length, i.e. the confound had been **suppressing** L1, not inflating it.
    print('\n  附加诊断：Spearman corr(场景级打分, γ3−γ1 字符差)；'
          '负值 = 打分偏好更短的 γ3，与金标方向相反')
    for tag, sids in sub.items():
        rk = rankdata(tab.loc[sids, 'd_len'].to_numpy(float))
        cells = []
        for key in order:
            kind, payload = routes[key]
            s = scenario_score(kind, payload, sids)
            ok = ~np.isnan(s)
            c = np.corrcoef(rankdata(s[ok]), rankdata(tab.loc[sids, 'd_len'].to_numpy(float)[ok]))[0, 1]
            cells.append(f'{key[0]}:{key[1]}={c:+.2f}')
        print(f'    [{tag}] ' + '  '.join(cells[:9]))
        print(f'    {" " * (len(tag) + 3)}' + '  '.join(cells[9:18]))
        print(f'    {" " * (len(tag) + 3)}' + '  '.join(cells[18:]))
        del rk

    # ------------------------------------------------------------------ main / null control
    keep_res = {}
    for feat_col, feat_tag, label in (('d_len', 'γ3−γ1 字符差', '主读数'),
                                      ('g1_len', 'γ1 字符数', '空对照')):
        for tag, sids in sub.items():
            feat = tab.loc[sids, feat_col].to_numpy(float)
            r = stratified_block(f'【{label}】按 {feat_tag} 四分位分层 ｜ 子集 {tag}',
                                 routes, tab, sids, feat, order)
            keep_res[(label, tag)] = r

    # The secondary readout is adjudicated only on the primary readout set, but both
    # subsets are printed (the scope requires "report both")
    for tag, sids in sub.items():
        print(f'\n[子集 {tag}]')
        covariate_block(routes, tab, sids, tab.loc[sids, 'd_len'].to_numpy(float),
                        order, 'γ3−γ1 字符差')

    # ------------------------------------------------------------------ adjudication
    print(f'\n{"=" * 108}\n【判定】按冻结判据：以 lv=5 上、分层后各路线的最好成绩为准\n'
          f'  ≤0.55 原信号可由长度解释 ｜ ≥0.60 且三条路线方向一致 → 有残留信号 ｜'
          f' 其余如实报为不确定\n{"=" * 108}')
    main_res = keep_res[('主读数', 'lv=5 391（主读数集）')]
    null_res = keep_res[('空对照', 'lv=5 391（主读数集）')]
    print(f"  {'线':<4}{'分层前最好':>12}{'分层后最好(格)':>28}{'方向严读':>12}{'空对照分层后最好':>18}")
    best = {}
    for line in ('L1', 'L4', 'L5'):
        cells = {k: v for k, v in main_res.items() if k[0] == line}
        bk = max(cells, key=lambda k: cells[k][1])
        pre_best = max(v[0] for v in cells.values())
        same = len({v[1] > 0.5 for v in cells.values()}) == 1
        nb = max(v[1] for k, v in null_res.items() if k[0] == line)
        best[line] = (cells[bk][1], same)
        print(f'  {line:<4}{pre_best:>12.3f}{f"{cells[bk][1]:.3f} ({bk[1]})":>28}'
              f'{("一致" if same else "**不一致**"):>12}{nb:>18.3f}')
    b = max(v[0] for v in best.values())
    strict = len({v[0] > 0.5 for v in best.values()}) == 1 and all(v[1] for v in best.values())
    lenient = len({v[0] > 0.5 for v in best.values()}) == 1
    print(f'\n  三条路线分层后的最好成绩 = {b:.3f}')
    print(f'  方向一致（严读：每条线内所有格同侧且三线同侧）= {strict}')
    print(f'  方向一致（宽读：只看每条线最好的那格）        = {lenient}')
    if b <= 0.55:
        v = '**原信号可由长度解释**——控住 γ3 长度后，三条路线全部落到 ≤0.55'
    elif b >= 0.60 and strict:
        v = '**存在长度之外的残留信号**——分层后仍 ≥0.60 且三条路线方向一致'
    elif b >= 0.60:
        v = '**不确定**：分层后有 ≥0.60 的格，但方向不一致 → 按冻结判据不许判为残留信号'
    else:
        v = '**不确定**：落在 0.55–0.60 的判据未覆盖区，不硬套档位'
    print(f'  → {v}')


if __name__ == '__main__':
    main()
