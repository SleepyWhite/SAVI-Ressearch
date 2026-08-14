#!/usr/bin/env python
"""Summarize the NLI entrenchment-contrast probe. Criteria are frozen in the docstring of `probe_nli_entrenchment.py`.

After running, copy the criteria verbatim — do not invent thresholds here. This file does
only three things:
  1. compute the P-1 / P-2 / D-1 readouts + the V3 / V4 controls
  2. check the consistency gate (P-1 and P-2 agree in direction; multiple models agree in
     direction)
  3. print the conclusion per the frozen four-tier criteria

Usage: python scripts/summarize_probe_nli.py
"""
import collections
import glob
import json
import math
import os

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'probe_nli')
BOOT = 2000
SEED = 20260801


def auc(pos, neg):
    """AUC = P(a random pos value > a random neg value), ties count 0.5. Rank method, handles ties."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # ties get the average rank
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def auc_ci(pos, neg, rng):
    """Bootstrap 95% CI resampling by scenario (this probe's unit of analysis is the scenario)."""
    vals = []
    for _ in range(BOOT):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        vals.append(auc(p, n))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def sign_test(k, n):
    """Exact p-value of the two-sided sign test (H0: p=0.5). n is small, enumerate directly."""
    if n == 0:
        return float('nan')
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def verdict(a, a_jac, gate_ok):
    if not gate_ok:
        return '判负（一致性闸门未过：方向不一致，不许挑一个来报）'
    if a < 0.45:
        return '**显著反向 → 话题相似度签名**。报负结果，不许翻符号'
    if a < 0.55:
        return '无信号，结案'
    if a < 0.65:
        return '有信号但不可用（BU 基线 0.015，这个强度驱动不了解码）。记录，不建设'
    if a - a_jac < 0.05:
        return '不可行动：未优于 Jaccard 对照 ≥0.05，NLI 没提供超出表面重叠的东西'
    return '**可行动**：外部证据源成立，下一步才谈接进 SAVI'


def main():
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(D, '*.jsonl')))
    if not files:
        raise SystemExit(f'没有找到读数：{D}')

    per_model = {}
    for f in files:
        rs = [json.loads(l) for l in open(f)]
        name = rs[0]['model']
        per_model[name] = rs

        print('=' * 76)
        print(f'=== {name}   n={len(rs)} 场景 ===')
        print('=' * 76)

        for tag, sub in (('全量（次要）', rs),
                         ('agreement_lv=5（**主读数集**）', [r for r in rs if r['agreement_lv'] == 5])):
            bu = [r['delta'] for r in sub if r['ground_truth'] == 'c']
            bm = [r['delta'] for r in sub if r['ground_truth'] != 'c']
            if not bu or not bm:
                continue
            a = auc(bu, bm)
            lo, hi = auc_ci(bu, bm, rng)
            aj = auc([r['delta_jac'] for r in sub if r['ground_truth'] == 'c'],
                     [r['delta_jac'] for r in sub if r['ground_truth'] != 'c'])
            print(f'\n  [P-1] {tag}  n(BU)={len(bu)} n(BM)={len(bm)}')
            print(f'        AUC(Δ)      = {a:.3f}  [{lo:.3f}, {hi:.3f}]')
            print(f'        AUC(Δ_jac)  = {aj:.3f}   ← V3 词汇重叠对照')
            print(f'        Δ 均值: BU {np.mean(bu):+.4f} / BM {np.mean(bm):+.4f}')

        # D-1: against design intent (strong/weak) rather than gold labels
        st = [r['delta'] for r in rs if r['intent'] == 'strong']
        wk = [r['delta'] for r in rs if r['intent'] == 'weak']
        print(f'\n  [D-1] AUC(Δ; intent strong vs weak) = {auc(st, wk):.3f}'
              f'   n={len(st)}/{len(wk)}')
        print('        若明显高于 P-1 → "信号在、标签脏"，与"没信号"是两回事')

        # P-2: minimal-pair paired sign test (p and q equal within a pair, so the p term cancels)
        by_base = collections.defaultdict(dict)
        for r in rs:
            by_base[r['base']][r['intent']] = r
        pairs = [(v['strong'], v['weak']) for v in by_base.values()
                 if 'strong' in v and 'weak' in v
                 and v['strong']['p'] == v['weak']['p'] and v['strong']['q'] == v['weak']['q']]
        split = [(s, w) for s, w in pairs if (s['ground_truth'] == 'c') != (w['ground_truth'] == 'c')]
        lv5 = [(s, w) for s, w in split if s['agreement_lv'] == 5 and w['agreement_lv'] == 5]
        print(f'\n  [P-2] 内容配对 {len(pairs)} / 金标分开 {len(split)} / 双侧 lv=5 {len(lv5)}')
        for tag, ps in (('金标分开', split), ('双侧 lv=5', lv5)):
            if not ps:
                continue
            # the REQ side (gold c) should have the larger Δ
            k = sum((s['delta'] > w['delta']) if s['ground_truth'] == 'c'
                    else (w['delta'] > s['delta']) for s, w in ps)
            print(f'        {tag}: {k}/{len(ps)} 方向正确  p={sign_test(k, len(ps)):.4f}'
                  f'  {"(功效有限，确认项)" if len(ps) < 40 else ""}')

        # V4 length control
        d = np.array([r['delta'] for r in rs])
        dl = np.array([r['len_r'] - r['len_p'] for r in rs], float)
        dj = np.array([r['delta_jac'] for r in rs])
        print(f'\n  [V4] corr(Δ, len_r−len_p) = {np.corrcoef(d, dl)[0, 1]:+.3f}'
              f'   ← 排除"NLI 偏好长前提"')
        print(f'  [V3] corr(Δ, Δ_jac)       = {np.corrcoef(d, dj)[0, 1]:+.3f}'
              f'   ← 越接近 1 越像只是词汇重叠')

    # ---------------- consistency gate ----------------
    print('\n' + '=' * 76)
    print('一致性闸门（L1 的直接教训：单一仪器会产出模型/措辞伪影）')
    print('=' * 76)
    dirs = {}
    for name, rs in per_model.items():
        sub = [r for r in rs if r['agreement_lv'] == 5] or rs
        a = auc([r['delta'] for r in sub if r['ground_truth'] == 'c'],
                [r['delta'] for r in sub if r['ground_truth'] != 'c'])
        dirs[name] = a
        print(f'  {name:<40} lv5 AUC = {a:.3f}  方向 {"+" if a > 0.5 else "−"}')
    if len(dirs) < 2:
        print('  ⚠ 只跑了 1 个 NLI 模型 —— **闸门未满足**，按预注册须 ≥2 个模型方向一致')
        gate = False
    else:
        gate = len({a > 0.5 for a in dirs.values()}) == 1
        print(f'  多模型方向一致: {"是" if gate else "**否 → 判负**"}')

    print('\n' + '=' * 76)
    print('结论（按冻结判据）')
    print('=' * 76)
    for name, rs in per_model.items():
        sub = [r for r in rs if r['agreement_lv'] == 5] or rs
        a = auc([r['delta'] for r in sub if r['ground_truth'] == 'c'],
                [r['delta'] for r in sub if r['ground_truth'] != 'c'])
        aj = auc([r['delta_jac'] for r in sub if r['ground_truth'] == 'c'],
                 [r['delta_jac'] for r in sub if r['ground_truth'] != 'c'])
        print(f'  {name}: AUC={a:.3f} (jac {aj:.3f}) → {verdict(a, aj, gate)}')


if __name__ == '__main__':
    main()
