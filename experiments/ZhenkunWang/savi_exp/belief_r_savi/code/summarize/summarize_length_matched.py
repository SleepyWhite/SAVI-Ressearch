#!/usr/bin/env python
"""B1: rescore the already-saved L1 / L4 / L5 outputs on length-matched subsets. Pure CPU, no new generation.

Criteria, subset definitions, gates, and known limitations are frozen in
`PREREG_length_matched.md` (2026-08-03, written before looking at any model readouts).
This script only prints against that set of criteria; no thresholds are improvised here.

Usage: python scripts/summarize_length_matched.py > outputs/ci/length_matched.txt
"""
import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_oracle_state import parse_cond          # noqa: E402  single source, don't write another parser

ROOT = os.path.join(HERE, '..')
# Belief-R dataset CSV. The data is not distributed with this repo (see the README's data section);
# point env var BELIEF_R_CSV at a local copy; falls back to <repo>/data/queries_time_t1.csv if unset.
QUERIES = os.environ.get('BELIEF_R_CSV') or os.path.join(
    ROOT, '..', 'data', 'queries_time_t1.csv')
BOOT, SEED = 5000, 20260803

# Frozen subset thresholds (PREREG §2). First-round V2 failed → demoted to "range
# truncation" controls, not adjudicated.
THRESHOLDS = (15, 25)
# Main subset SM: bin width for coarsened exact matching (PREREG §5b revision 1)
CEM_BIN = 15

# Registered anchors (PREREG §5 V1). Failure to reproduce means the rescoring pipeline is
# wrong, not a new finding.
ANCHORS = {'L1_3B_relation_lv5': 0.611, 'L4_bart_lv5': 0.560, 'L5_range_lv5': (0.507, 0.545)}

NAME = {'Qwen_Qwen2.5_0.5B_Instruct': '0.5B', 'Qwen_Qwen2.5_1.5B_Instruct': '1.5B',
        'Qwen_Qwen2.5_3B_Instruct': '3B', 'Qwen_Qwen2.5_7B_Instruct': '7B',
        'Qwen_Qwen3_4B': 'Qwen3-4B', 'meta_llama_Llama_3.1_8B_Instruct': 'Llama-8B'}
ORDER = ('0.5B', '1.5B', '3B', '7B', 'Qwen3-4B', 'Llama-8B')


# ---------------------------------------------------------------- subsets
def build_subsets():
    """Scenario-level subset table. Δ = len(r) − len(p), both are **antecedents**, parsed from γ1/γ3."""
    d = pd.read_csv(QUERIES)
    d = d[d.agreement_lv <= 5]                      # the 4 lv=6 rows are a data anomaly, consistently excluded
    rows, bad = [], 0
    for _, x in d.iterrows():
        lines = x.questions.split('\n')
        c1, c3 = parse_cond(lines[0]), parse_cond(lines[2])
        if not c1 or not c3:
            bad += 1
            continue
        rows.append(dict(dataset_id=x.dataset_id, lv=x.agreement_lv,
                         intent=x.dataset_id.split('-')[1],
                         gold=x.ground_truth, d=len(c3[0]) - len(c1[0])))
    f = pd.DataFrame(rows).drop_duplicates('dataset_id').reset_index(drop=True)
    print(f'[V0] 解析失败 {bad} 行；场景 {len(f)}（预期 870）')
    assert bad == 0 and len(f) == 870, 'V0 失败：解析或场景数与登记不符'
    f['REQ'] = (f['gold'] == 'c').astype(int)
    return f


def cem_match(f):
    """Coarsened exact matching: bin Δ, and within each bin take min(n_REQ, n_ALT) of each class.

    Thus the Δ distribution is identical bin by bin between the two classes → the length
    rule is ≈0.5 by construction on this subset (V2).
    Selection is sorted by dataset_id — deterministic, reproducible, no random numbers.
    The first round's |Δ|≤T only truncated the value range without matching the
    distribution, which is why V2 failed (AUC 0.687).
    """
    lv5 = f[f.lv == 5].copy()
    lv5['bin'] = (lv5.d // CEM_BIN).astype(int)
    keep = []
    for _, g in lv5.groupby('bin'):
        req = g[g.REQ == 1].sort_values('dataset_id')
        alt = g[g.REQ == 0].sort_values('dataset_id')
        k = min(len(req), len(alt))
        if k:
            keep += list(req.dataset_id[:k]) + list(alt.dataset_id[:k])
    return set(keep)


def auc(score, label):
    s = np.asarray(score, float)
    y = np.asarray(label)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_auc(score, label, sid, rng):
    """Scenario-clustered resampling. Rows with the same sid enter/leave together."""
    by = collections.defaultdict(list)
    for i, k in enumerate(sid):
        by[k].append(i)
    keys = list(by)
    s, y = np.asarray(score, float), np.asarray(label)
    out = []
    for _ in range(BOOT):
        idx = [i for k in rng.choice(len(keys), len(keys), replace=True) for i in by[keys[k]]]
        v = auc(s[idx], y[idx])
        if not np.isnan(v):
            out.append(v)
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)


def paired_delta(score, label, sid, member, rng):
    """Δ = AUC(subset) − AUC(full set), paired resampling over the same batch of scenarios."""
    by = collections.defaultdict(list)
    for i, k in enumerate(sid):
        by[k].append(i)
    keys = list(by)
    s, y, m = np.asarray(score, float), np.asarray(label), np.asarray(member)
    out = []
    for _ in range(BOOT):
        idx = [i for k in rng.choice(len(keys), len(keys), replace=True) for i in by[keys[k]]]
        idx = np.asarray(idx)
        sub = idx[m[idx]]
        a_all, a_sub = auc(s[idx], y[idx]), auc(s[sub], y[sub])
        if not (np.isnan(a_all) or np.isnan(a_sub)):
            out.append(a_sub - a_all)
    return np.mean(out), np.percentile(out, 2.5), np.percentile(out, 97.5)


# ---------------------------------------------------------------- ceiling
def ceiling(sub):
    """Balanced accuracy with the design intent (strong/weak) as a perfect scorer."""
    pred = (sub.intent == 'strong').astype(int).values
    y = sub.REQ.values
    return ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()) / 2


def ceiling_permutation(f, thr, rng):
    """V5: randomly redraw subsets of the same size and see which quantile of the null distribution the observed ceiling falls in.

    Null hypothesis = "any lv=5 subset of the same size has a ceiling no higher than the full set".
    """
    lv5 = f[f.lv == 5]
    obs = ceiling(lv5[lv5.d.abs() <= thr])
    n = int((lv5.d.abs() <= thr).sum())
    null = []
    for _ in range(BOOT):
        s = lv5.iloc[rng.choice(len(lv5), n, replace=False)]
        if s.REQ.nunique() < 2:
            continue
        null.append(ceiling(s))
    null = np.asarray(null)
    return obs, float((null >= obs).mean()), null.mean()


# ---------------------------------------------------------------- readouts of the three routes
def load_L1():
    """Long table of (sid, wording, model, delta_mean, REQ).

    Scope = `delta_**mean**`, row-level — the default of `summarize_probe_relation.py
    --metric`, and the source of the registered 3B/relation 0.611. In the first round I
    used delta_sum, which is why V1 failed.
    """
    out = []
    for fp in sorted(glob.glob(os.path.join(ROOT, 'outputs', 'probe_relation', '*.jsonl'))):
        for line in open(fp):
            r = json.loads(line)
            key = os.path.basename(fp)[len('time_t1_'):-len('.jsonl')]
            for w in ('sufficiency', 'relation', 'certainty'):
                out.append(dict(sid=r['dataset_id'], model=NAME.get(key, key), wording=w,
                                score=r[f'{w}_delta_mean'], REQ=int(r['ground_truth'] == 'c'),
                                lv=r['agreement_lv']))
    return pd.DataFrame(out)


def load_L4():
    out = []
    for fp in sorted(glob.glob(os.path.join(ROOT, 'outputs', 'probe_nli', '*.jsonl'))):
        for line in open(fp):
            r = json.loads(line)
            out.append(dict(sid=r['dataset_id'], model=r['model'], score=r['delta'],
                            REQ=int(r['ground_truth'] == 'c'), lv=r['agreement_lv']))
    return pd.DataFrame(out)


def load_L5():
    """picked_req was already parsed at the probe stage. strict scope: unhonored rows count as wrong."""
    out = []
    for fp in sorted(glob.glob(os.path.join(ROOT, 'outputs', 'incontext_choice', '*.jsonl'))):
        for line in open(fp):
            r = json.loads(line)
            key = [k for k in NAME if k in os.path.basename(fp)]
            out.append(dict(sid=r['dataset_id'], model=NAME[key[0]] if key else '?',
                            order=r['order'], picked=r['picked_req'],
                            REQ=int(r['gold_is_req']), lv=r['agreement_lv']))
    return pd.DataFrame(out)


def bal_acc(g, strict=False):
    """Balanced accuracy. Lenient scope = only rows where RELATION parsing succeeded
    (**this is what the registered headline uses**); strict = unhonored rows count as
    wrong. In the first round I treated strict as the main cell, which is why V1 failed."""
    pool = g if strict else g[g.picked.notna()]
    req, alt = pool[pool.REQ == 1], pool[pool.REQ == 0]
    if len(req) == 0 or len(alt) == 0:
        return float('nan')
    return ((req.picked == True).mean() + (alt.picked == False).mean()) / 2   # noqa: E712


def main():
    rng = np.random.default_rng(SEED)
    f = build_subsets()
    SM = cem_match(f)
    memb = {'SM': SM}
    memb.update({t: set(f[(f.lv == 5) & (f.d.abs() <= t)].dataset_id) for t in THRESHOLDS})
    lv5 = set(f[f.lv == 5].dataset_id)
    COLS = ['SM', 15, 25]

    print('\n' + '=' * 78)
    print('子集与仪器效度检查（PREREG §5，含 §5b 修订）')
    print('=' * 78)
    sub_all = f[f.lv == 5]
    print(f'  参照全集 F : n={len(sub_all)}  REQ占比={sub_all.REQ.mean():.3f}  '
          f'天花板={ceiling(sub_all):.3f}')
    print(f'    [V2 正对照] 长度规则在全集上 AUC = '
          f'{auc(sub_all.d.values, sub_all.REQ.values):.3f}  （登记 0.800/0.802）')
    for k in COLS:
        sub = f[f.dataset_id.isin(memb[k])]
        tag = ('SM 粗化精确匹配【主子集】' if k == 'SM' else f'S{k} 范围截断（对照，不参与判定）')
        print(f'\n  {tag}: n={len(sub)}  REQ占比={sub.REQ.mean():.3f}  '
              f'strong占比={(sub.intent == "strong").mean():.3f}')
        a = auc(sub.d.values, sub.REQ.values)
        lo, hi = boot_auc(sub.d.values, sub.REQ.values, sub.dataset_id.values, rng)
        ok = abs(a - 0.5) < 0.10
        print(f'    [V2] 长度规则 Δ→REQ 的 AUC = {a:.3f} [{lo:.3f},{hi:.3f}]  '
              f'{"✅ 配平成立" if ok else "❌ 配平未做到 → 该子集不能读作长度干净"}')
        print(f'    [V3] 天花板（设计意图为完美打分器） = {ceiling(sub):.3f}')

    print('\n  [V5] 天花板"升了"的置换检验（随机重划同样大小的 lv=5 子集）')
    for k in COLS:
        r2 = np.random.default_rng(SEED + (0 if k == 'SM' else k))
        sub = f[f.dataset_id.isin(memb[k])]
        obs, n = ceiling(sub), len(sub)
        null = []
        for _ in range(BOOT):
            s = sub_all.iloc[r2.choice(len(sub_all), n, replace=False)]
            if s.REQ.nunique() > 1:
                null.append(ceiling(s))
        null = np.asarray(null)
        print(f'    {str(k):<3}: 观测 {obs:.3f}，零分布均值 {null.mean():.3f}，'
              f'p(零分布 ≥ 观测) = {(null >= obs).mean():.4f}')

    # ---------------------------------------------------------------- L1
    print('\n' + '=' * 78)
    print('L1 枚举+似然打分（主格 = relation 措辞；另两套措辞不参与判定）')
    print('=' * 78)
    L1 = load_L1()
    L1 = L1[L1.sid.isin(lv5)]
    for w in ('relation', 'sufficiency', 'certainty'):
        tag = '【主格】' if w == 'relation' else '（不参与判定）'
        print(f'\n  -- {w} {tag}')
        print(f'    {"模型":<10}{"全集lv5":>10}{"SM":>10}{"SM 95%CI":>18}'
              f'{"S15":>8}{"S25":>8}{"Δ(SM−全集)":>22}')
        for mdl in ORDER:
            g = L1[(L1.model == mdl) & (L1.wording == w)]
            if g.empty:
                continue
            a_f = auc(g.score.values, g.REQ.values)
            gm = g[g.sid.isin(SM)]
            am = auc(gm.score.values, gm.REQ.values)
            lo, hi = boot_auc(gm.score.values, gm.REQ.values, gm.sid.values, rng)
            a15 = auc(*[g[g.sid.isin(memb[15])][c].values for c in ('score', 'REQ')])
            a25 = auc(*[g[g.sid.isin(memb[25])][c].values for c in ('score', 'REQ')])
            d, dlo, dhi = paired_delta(g.score.values, g.REQ.values, g.sid.values,
                                       g.sid.isin(SM).values, rng)
            print(f'    {mdl:<10}{a_f:>10.3f}{am:>10.3f}{f"[{lo:.3f},{hi:.3f}]":>18}'
                  f'{a15:>8.3f}{a25:>8.3f}{f"{d:+.3f} [{dlo:+.3f},{dhi:+.3f}]":>22}')
    # V1 anchor
    g = L1[(L1.model == '3B') & (L1.wording == 'relation')]
    a = auc(g.score.values, g.REQ.values)
    print(f'\n  [V1] 锚点 3B/relation 全集lv5 = {a:.3f}（登记 {ANCHORS["L1_3B_relation_lv5"]}）'
          f'  {"✅" if abs(a - ANCHORS["L1_3B_relation_lv5"]) < 0.005 else "❌ 管道口径不符"}')
    # G-dir
    print('\n  [G-dir] 三套措辞方向一致性（SM 主子集）')
    ok = 0
    for mdl in ORDER:
        signs = []
        for w in ('sufficiency', 'relation', 'certainty'):
            g = L1[(L1.model == mdl) & (L1.wording == w) & (L1.sid.isin(SM))]
            if not g.empty:
                signs.append(int(np.sign(auc(g.score.values, g.REQ.values) - 0.5)))
        same = len(set(signs)) == 1
        ok += same
        print(f'    {mdl:<10} 符号 {signs}  {"一致" if same else "不一致"}')
    print(f'    → {ok}/6 个模型三套措辞同号（判据要 ≥4/6）')

    # ---------------------------------------------------------------- L4
    print('\n' + '=' * 78)
    print('L4 外部 NLI 证据（主格 = bart-large-mnli，预注册的那个）')
    print('=' * 78)
    L4 = load_L4()
    L4 = L4[L4.sid.isin(lv5)]
    print(f'    {"模型":<42}{"全集lv5":>10}{"SM":>10}{"SM 95%CI":>18}{"S15":>8}{"S25":>8}')
    for mdl, g in L4.groupby('model'):
        a_f = auc(g.score.values, g.REQ.values)
        gm = g[g.sid.isin(SM)]
        am = auc(gm.score.values, gm.REQ.values)
        lo, hi = boot_auc(gm.score.values, gm.REQ.values, gm.sid.values, rng)
        a15 = auc(*[g[g.sid.isin(memb[15])][c].values for c in ('score', 'REQ')])
        a25 = auc(*[g[g.sid.isin(memb[25])][c].values for c in ('score', 'REQ')])
        tag = ' 【主格】' if 'bart' in mdl else ''
        print(f'    {mdl[:40]:<42}{a_f:>10.3f}{am:>10.3f}'
              f'{f"[{lo:.3f},{hi:.3f}]":>18}{a15:>8.3f}{a25:>8.3f}{tag}')
        if 'bart' in mdl:
            print(f'      [V1] 锚点 bart 全集lv5 = {a_f:.3f}（登记 {ANCHORS["L4_bart_lv5"]}）'
                  f'  {"✅" if abs(a_f - ANCHORS["L4_bart_lv5"]) < 0.005 else "❌"}')

    # ---------------------------------------------------------------- L5
    print('\n' + '=' * 78)
    print('L5 上下文内显式选择（主格 = strict 口径，两排法平均）')
    print('=' * 78)
    L5 = load_L5()
    L5 = L5[L5.sid.isin(lv5)]
    print(f'    {"模型":<10}{"全集lv5":>10}{"SM":>10}{"S15":>8}{"S25":>8}'
          f'{"|全集strict":>12}{"SM strict":>11}   （宽口径为主，两排法平均）')
    anchors_ok = []
    for mdl in ORDER:
        g = L5[L5.model == mdl]
        if g.empty:
            continue
        wide, strict = [], []
        for sel in (lv5, SM, memb[15], memb[25]):
            wide.append(np.nanmean([bal_acc(g[(g.order == o) & (g.sid.isin(sel))])
                                    for o in sorted(g.order.unique())]))
        for sel in (lv5, SM):
            strict.append(np.nanmean([bal_acc(g[(g.order == o) & (g.sid.isin(sel))], True)
                                      for o in sorted(g.order.unique())]))
        anchors_ok.append(wide[0])
        print(f'    {mdl:<10}{wide[0]:>10.3f}{wide[1]:>10.3f}{wide[2]:>8.3f}{wide[3]:>8.3f}'
              f'{strict[0]:>12.3f}{strict[1]:>11.3f}')
    lo_a, hi_a = ANCHORS['L5_range_lv5']
    ok = all(lo_a - 0.005 <= v <= hi_a + 0.005 for v in anchors_ok)
    print(f'    [V1] 锚点 六模型全集lv5 宽口径落在 [{min(anchors_ok):.3f},{max(anchors_ok):.3f}]'
          f'（登记 {lo_a}–{hi_a}）  {"✅" if ok else "❌ 管道口径不符"}')

    print('\n' + '=' * 78)
    print('判据（PREREG §4）：S15 主格 ≥0.70 且 G-dir 过 → 门开；≤0.60 → 结案；'
          '中间或闸门不过 → 不确定')
    print('=' * 78)


if __name__ == '__main__':
    main()
