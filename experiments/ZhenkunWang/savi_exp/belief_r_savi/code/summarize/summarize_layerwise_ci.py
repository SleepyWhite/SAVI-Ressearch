#!/usr/bin/env python
"""Layer-wise probe curve with CIs + significance test for the "last-layer drop". CPU only, reads only on-disk features.

Criteria frozen in `PREREG_layerwise_ci.md` (evening of 2026-08-10). This script only
mechanically matches results against that criteria set; it writes no interpretation and
settles nothing.

Mechanism source (PREREG §2 "reuse, don't reimplement"): fold structure / inner C selection /
outer out-of-fold prediction / cluster bootstrap are all imported from
`summarize_hidden_probe.py` (L7's single source). That script's `layer_curve()` returns only
(layer, balanced accuracy) and keeps no per-layer out-of-fold predictions, while the paired
difference needs per-scenario hits, so `layer_curve_preds()` here replays it with **the same
task order, the same `_final_task`, the same per-fold-per-layer C** and keeps the
predictions; it then does a **bitwise-consistency assertion** against the original
`layer_curve()` return value (`--no-mirror-check` disables it; on by default).

Anchors (PREREG §2, stop on failure): the per-layer point estimates and the main readout
must reproduce the values registered on 2026-08-03, copied verbatim from
`outputs/ci/hidden_probe.txt` and `outputs/ci/hidden_probe_llama.txt`.

Usage:
  python scripts/summarize_layerwise_ci.py                       # full run, three models
  python scripts/summarize_layerwise_ci.py --models Qwen/Qwen3-4B \
      --txt /tmp/smoke.txt --json /tmp/smoke.json                # smoke: single-model anchor reproduction
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_hidden_probe import (  # noqa: E402  single source of the layer-wise mechanism, no reimplementation
    CGRID, N_OUTER, N_INNER, BOOT, SEED,
    _final_task, bal_acc, boot_ci, layer_curve, load_reps, nested_cv)

OUT = os.path.join(HERE, '..', 'outputs')
MODELS = ('Qwen/Qwen3-4B', 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct')
TOL = 0.002  # PREREG §2: same code path should be bitwise identical; outside ±0.002 → stop, no CI

# ---- anchors: values registered 2026-08-03 (copied verbatim, see file header) ----
ANCHOR_MAIN = {'Qwen/Qwen3-4B': 0.692,
               'Qwen/Qwen2.5-7B-Instruct': 0.716,
               'meta-llama/Llama-3.1-8B-Instruct': 0.734}
ANCHOR_CHOSEN = {  # (layer, C) chosen by each outer fold; diagnostic, not a stop condition
    'Qwen/Qwen3-4B': [(21, 0.01), (9, 10.0), (13, 0.01), (19, 1.0), (20, 0.01)],
    'Qwen/Qwen2.5-7B-Instruct': [(19, 10.0), (10, 0.01), (10, 10.0), (19, 0.01), (10, 0.01)],
    'meta-llama/Llama-3.1-8B-Instruct': [(10, 0.1), (12, 0.01), (10, 0.1), (8, 0.01), (10, 0.1)],
}
ANCHOR_CURVE = {
    'Qwen/Qwen3-4B': [
        0.500, 0.590, 0.612, 0.677, 0.678, 0.682, 0.665, 0.684, 0.718, 0.697,
        0.696, 0.695, 0.708, 0.710, 0.708, 0.727, 0.706, 0.706, 0.729, 0.735,
        0.725, 0.701, 0.702, 0.690, 0.688, 0.658, 0.679, 0.688, 0.669, 0.660,
        0.679, 0.675, 0.677, 0.688, 0.698, 0.696, 0.688],
    'Qwen/Qwen2.5-7B-Instruct': [
        0.500, 0.564, 0.580, 0.598, 0.621, 0.653, 0.654, 0.682, 0.701, 0.707,
        0.729, 0.699, 0.726, 0.726, 0.728, 0.705, 0.694, 0.724, 0.712, 0.726,
        0.722, 0.687, 0.690, 0.682, 0.666, 0.654, 0.671, 0.673, 0.652],
    'meta-llama/Llama-3.1-8B-Instruct': [
        0.500, 0.601, 0.623, 0.666, 0.701, 0.680, 0.718, 0.713, 0.722, 0.739,
        0.739, 0.709, 0.718, 0.714, 0.685, 0.714, 0.712, 0.729, 0.722, 0.682,
        0.676, 0.716, 0.678, 0.678, 0.659, 0.680, 0.678, 0.659, 0.660, 0.675,
        0.662, 0.664, 0.669],
}
# The entries PREREG §2 calls out by name (peak layer / last layer / embedding), printed separately
ANCHOR_NAMED = {'Qwen/Qwen3-4B': [(19, 0.735), (36, 0.688), (0, 0.500)],
                'Qwen/Qwen2.5-7B-Instruct': [(10, 0.729), (28, 0.652), (0, 0.500)],
                'meta-llama/Llama-3.1-8B-Instruct': [(9, 0.739), (32, 0.669), (0, 0.500)]}


class Tee:
    """Write complete, ordered output to disk and screen simultaneously (no shell redirection, so stderr can't interleave and scramble the order)."""

    def __init__(self, path):
        self.f = open(path, 'w')

    def write(self, s):
        self.f.write(s)
        sys.__stdout__.write(s)

    def flush(self):
        self.f.flush()
        sys.__stdout__.flush()


def layer_curve_preds(X, y, groups, cv, layers, cgrid, n_jobs):
    """Same-mechanism replay of summarize_hidden_probe.layer_curve, except it keeps the per-layer out-of-fold predictions.

    Task order (fold outer, layer inner), `_final_task`, and the per-fold-per-layer C are
    verbatim identical to the original function; correctness is backed by the bitwise
    assertion against the original function's return value in run_model."""
    tasks = []
    for k, (tr, te) in enumerate(cv['folds']):
        for li, L in enumerate(layers):
            tasks.append((tr, te, L, cgrid[int(cv['inner_mats'][k][li].argmax())]))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_final_task)(X, y, tr, te, L, C) for tr, te, L, C in tasks)
    it = iter(res)
    preds_by_layer = {L: np.full(len(y), -1) for L in layers}
    for k, (tr, te) in enumerate(cv['folds']):
        for L in layers:
            p, _, _, _ = next(it)
            preds_by_layer[L][te] = p
    return preds_by_layer


def paired_delta(y, groups, pred_a, pred_b):
    """Paired difference A−B: main scale = difference in balanced accuracy (same scale as the anchors), side registration = difference in per-scenario hits.

    Both arms use the same bootstrap resampling (the rng inside boot_ci is re-created from
    the same seed → identical sampling sequence), which is what guarantees the pairing.
    Returns a dict."""
    y = np.asarray(y)
    ba = lambda i: bal_acc(y[i], pred_a[i]) - bal_acc(y[i], pred_b[i])  # noqa: E731
    lo, hi = boot_ci(ba, groups, np.random.default_rng(SEED), BOOT)
    d = (np.asarray(pred_a) == y).astype(float) - (np.asarray(pred_b) == y).astype(float)
    alo, ahi = boot_ci(lambda i: d[i].mean(), groups, np.random.default_rng(SEED), BOOT)
    return dict(bal=dict(point=float(ba(np.arange(len(y)))), ci=[float(lo), float(hi)]),
                acc=dict(point=float(d.mean()), ci=[float(alo), float(ahi)]))


def tally(lo, hi):
    """PREREG §1 mechanical matching. Match only, no interpretation."""
    if lo > 0:
        return '末层回落显著（CI 下界 > 0）'
    if lo <= 0 <= hi:
        return ('回落在本样本量下不可分辨（CI 含 0）→ 引用逐层曲线一律写'
                '"数值回落但未达显著"，报告/幻灯措辞同步降格')
    return '判据未覆盖：CI 整体 < 0（主读数低于对照层）→ 如实报告为不确定，交人裁决'


def run_model(model_name, n_jobs, mirror_check):
    last, _g3, meta = load_reps(model_name, smoke=False)
    taps = last.shape[1]
    layers = list(range(taps))
    print(f'\n{"=" * 78}\n{model_name}  (N={len(meta)}, taps={taps}, hidden={last.shape[2]})\n{"=" * 78}')

    # main set: lv=5 ponens (same construction and assertions as summarize_hidden_probe.run_model)
    main = meta[(meta.agreement_lv == 5) & (meta.modus == 'ponens')]
    idx = main.index.values
    assert len(idx) == 391, f'主集行数 {len(idx)} != 391'
    Xm, Gm = last[idx], main['atomic_idx'].values
    ym = (main['gold'] == 'c').astype(int).values
    assert pd.Series(Gm).nunique() == 152

    cv = nested_cv(Xm, ym, Gm, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    b_main = bal_acc(ym, cv['oof_pred'])
    lo_m, hi_m = boot_ci(lambda i: bal_acc(ym[i], cv['oof_pred'][i]), Gm,
                         np.random.default_rng(SEED), BOOT)

    preds = layer_curve_preds(Xm, ym, Gm, cv, layers, CGRID, n_jobs)
    curve = [(int(L), bal_acc(ym, preds[L])) for L in layers]
    if mirror_check:
        ref = layer_curve(Xm, ym, Gm, cv, layers, CGRID, n_jobs)
        mism = [(L, b, rb) for (L, b), (rL, rb) in zip(curve, ref) if L != rL or b != rb]
        assert not mism, f'重放与冻结 layer_curve 不逐位一致：{mism[:5]}'
        print(f'[镜像检查] layer_curve_preds 的逐层平衡准确率与冻结 layer_curve() '
              f'逐位一致（{len(curve)} 层全过）')

    # ---- anchors (PREREG §2, stop on failure) ----
    print(f'\n[锚点] 主读数 {b_main:.3f}（登记 {ANCHOR_MAIN[model_name]:.3f}，'
          f'差 {b_main - ANCHOR_MAIN[model_name]:+.4f}）')
    chosen = [(L, C) for L, C, _ in cv['chosen']]
    print(f'[锚点] 各折选中 (层,C) = {chosen}')
    print(f'         登记         = {ANCHOR_CHOSEN[model_name]}  '
          f'{"一致" if chosen == ANCHOR_CHOSEN[model_name] else "不一致（诊断，非停机条件）"}')
    ac = ANCHOR_CURVE[model_name]
    assert len(ac) == len(curve), f'登记曲线层数 {len(ac)} != 本次 {len(curve)}'
    diffs = [abs(b - a) for (_, b), a in zip(curve, ac)]
    exact3 = sum(round(b, 3) == a for (_, b), a in zip(curve, ac))
    for L, a in ANCHOR_NAMED[model_name]:
        print(f'[锚点] L{L}: 复现 {curve[L][1]:.3f} vs 登记 {a:.3f}  差 {curve[L][1] - a:+.4f}')
    print(f'[锚点] 全层曲线：3 位小数完全相同 {exact3}/{len(curve)} 层；'
          f'最大绝对差 {max(diffs):.4f}（阈 {TOL}）')
    bad = [(L, b, a) for (L, b), a, d in zip(curve, ac, diffs) if d > TOL]
    assert abs(b_main - ANCHOR_MAIN[model_name]) <= TOL, \
        f'锚点不过：主读数 {b_main:.4f} vs 登记 {ANCHOR_MAIN[model_name]}，停，不出 CI'
    assert not bad, f'锚点不过：逐层点估计偏离 >{TOL}：{bad[:8]}，停，不出 CI'
    print('[锚点] 全部通过 → 出 CI')

    # ---- main criterion Δ_cons / registered readout Δ_peak ----
    L_last = layers[-1]
    L_peak, v_peak = max(curve, key=lambda t: t[1])
    d_cons = paired_delta(ym, Gm, cv['oof_pred'], preds[L_last])
    d_peak = paired_delta(ym, Gm, preds[L_peak], preds[L_last])
    t_cons = tally(*d_cons['bal']['ci'])
    t_peak = tally(*d_peak['bal']['ci'])

    print(f'\n[主读数] 嵌套选层 out-of-fold 平衡准确率 = {b_main:.3f} [{lo_m:.3f},{hi_m:.3f}]')
    print(f'[末层]   固定层 L{L_last} out-of-fold 平衡准确率 = {curve[L_last][1]:.3f}')
    print(f'[峰层]   固定层 L{L_peak} out-of-fold 平衡准确率 = {v_peak:.3f}'
          f'  —— 事后选层，偏乐观，不承重')
    print(f'\n[Δ_cons 主判据] 主读数 − 末层 L{L_last}（配对，atomic 簇 bootstrap '
          f'{BOOT} 次 seed {SEED}）')
    print(f'  平衡准确率之差 = {d_cons["bal"]["point"]:+.4f} '
          f'[{d_cons["bal"]["ci"][0]:+.4f},{d_cons["bal"]["ci"][1]:+.4f}]')
    print(f'  （附登记：逐场景命中之差 = {d_cons["acc"]["point"]:+.4f} '
          f'[{d_cons["acc"]["ci"][0]:+.4f},{d_cons["acc"]["ci"][1]:+.4f}]，'
          f'非平衡尺度，见分歧登记）')
    print(f'  → 对号：{t_cons}')
    print(f'\n[Δ_peak 登记读数] 峰层 L{L_peak} − 末层 L{L_last}'
          f'  **事后选层，偏乐观，不承重**')
    print(f'  平衡准确率之差 = {d_peak["bal"]["point"]:+.4f} '
          f'[{d_peak["bal"]["ci"][0]:+.4f},{d_peak["bal"]["ci"][1]:+.4f}]')
    print(f'  （附登记：逐场景命中之差 = {d_peak["acc"]["point"]:+.4f} '
          f'[{d_peak["acc"]["ci"][0]:+.4f},{d_peak["acc"]["ci"][1]:+.4f}]）')
    print(f'  → 对号：{t_peak}')

    # ---- full-layer CI curve (pure description, for plotting) ----
    rows = []
    for L, b in curve:
        lo, hi = boot_ci(lambda i, L=L: bal_acc(ym[i], preds[L][i]), Gm,
                         np.random.default_rng(SEED), BOOT)
        rows.append(dict(layer=int(L), bal_acc=float(b), ci=[float(lo), float(hi)]))
    print(f'\n[描述] 全层固定层 out-of-fold 平衡准确率 + 95% CI（簇 bootstrap，纯描述）：')
    for r in rows:
        mark = ''
        if r['layer'] == L_peak:
            mark = '  ← 峰层'
        if r['layer'] == L_last:
            mark += '  ← 末层'
        print(f"  L{r['layer']:>2}: {r['bal_acc']:.3f} "
              f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]{mark}")

    return dict(model=model_name, n=int(len(ym)), n_groups=int(pd.Series(Gm).nunique()),
                taps=int(taps), pos=int(ym.sum()), neg=int((1 - ym).sum()),
                main=dict(bal_acc=float(b_main), ci=[float(lo_m), float(hi_m)],
                          anchor=ANCHOR_MAIN[model_name],
                          chosen=[[int(L), float(C)] for L, C in chosen]),
                last_layer=int(L_last), peak_layer=int(L_peak),
                delta_cons=dict(**d_cons, tally=t_cons),
                delta_peak=dict(**d_peak, tally=t_peak,
                                note='事后选层，偏乐观，不承重'),
                layers=rows,
                anchor_curve_max_abs_diff=float(max(diffs)),
                anchor_curve_exact3=int(exact3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--n_jobs', type=int, default=16)
    ap.add_argument('--txt', default=os.path.join(OUT, 'ci', 'layerwise_ci.txt'))
    ap.add_argument('--json', default=os.path.join(OUT, 'layerwise_ci.json'))
    ap.add_argument('--no-mirror-check', dest='mirror', action='store_false',
                    help='跳过与冻结 layer_curve() 的逐位一致断言（默认执行）')
    args = ap.parse_args()

    sys.stdout = Tee(args.txt)
    print('# 逐层探针曲线补 CI（PREREG_layerwise_ci.md，2026-08-10 冻结）')
    print('# 机制 import 自 summarize_hidden_probe.py（L7 单一来源）：折结构/内层选 C/'
          '折外预测/聚类 bootstrap')
    print(f'# 主集 lv=5 ponens；外层 GroupKFold {N_OUTER}（组=atomic_idx）；'
          f'内层 {N_INNER} 折联合选 层×C∈{CGRID}')
    print(f'# CI：atomic_idx 聚类 bootstrap {BOOT} 次，seed {SEED}；三模型分别报，不聚合')
    print('# 主判据 Δ_cons = 主读数 − 末层固定层（配对）；Δ_peak = 峰层 − 末层（事后选层，'
          '偏乐观，只登记）')

    try:
        res = [run_model(m, args.n_jobs, args.mirror) for m in args.models.split(',')]
    except AssertionError as e:      # anchor/assertion failure: write the stop reason into the output, then re-raise
        print(f'\n[停机] 断言失败 → {e}')
        print('[停机] 按 PREREG §2「锚点不过即停」：不出 CI，上报。')
        sys.stdout.flush()
        raise

    print(f'\n{"=" * 78}\n机械对号汇总（PREREG §1；只对号，不解读）\n{"=" * 78}')
    for r in res:
        c, p = r['delta_cons']['bal'], r['delta_peak']['bal']
        print(f"\n{r['model']}")
        print(f"  Δ_cons = {c['point']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] "
              f"→ {r['delta_cons']['tally']}")
        print(f"  Δ_peak = {p['point']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] "
              f"→ {r['delta_peak']['tally']}  【事后选层，偏乐观，不承重】")
    print('\n本脚本只对号，不写解读、不定案；最终措辞与落档留给人。')

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, 'w') as f:
        json.dump(dict(prereg='PREREG_layerwise_ci.md', boot=BOOT, seed=SEED,
                       cgrid=list(CGRID), n_outer=N_OUTER, n_inner=N_INNER,
                       models={r['model']: r for r in res}), f, indent=1,
                  ensure_ascii=False)
    print(f'\n产物：{args.txt}\n      {args.json}')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
