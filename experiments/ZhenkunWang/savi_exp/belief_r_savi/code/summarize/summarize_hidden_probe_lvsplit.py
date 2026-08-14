#!/usr/bin/env python
"""Out-of-protocol exploration: does the hidden-layer probe signal hold only on lv=5 (unanimous annotators)? Pure CPU.

**Not governed by the PREREG_hidden_probe.md §6 criteria; does not change the primary readout.**
Motivation = the user asked on 2026-08-03 whether "the good result holds only on lv5, where
expert opinion is unanimous".
The primary readout and the six gates are as recorded in outputs/ci/hidden_probe.txt; this
script only adds an lv-dimension description, no adjudication. All pipeline parts are
imported from the frozen deliverables summarize_hidden_probe.py and
summarize_length_matched.py; this file implements no new statistical logic.

Four arms (two models × main position last_tok):
  L5-anchor  lv=5 ponens (391): rerun the main pipeline, assert agreement with the
           registered readout (4B 0.692 / 7B 0.716, ±0.002) — non-reproduction = this
           script's pipeline is wrong, raise, output no readouts;
  A        lv=4 ponens (479) standalone nested CV;
  B        lv∈{4,5} ponens pooled (870) nested CV, out-of-fold reported stratified by lv
           (the same atomic seed can spawn both lv4 and lv5 scenarios — 118/204 groups
           overlap — pooling lets grouped CV handle that overlap naturally);
  C        lv4 gates: full-permutation label shuffle (must return to 0.5) + TF-IDF surface
           baseline (same folds, same protocol).

Reference lines (recomputed and asserted in-script, ±0.005): lv4 ceiling 0.728 / length-rule
AUC 0.712; lv5 counterparts 0.843 / 0.802. Each arm also prints the ratio
(readout−0.5)/(ceiling−0.5) as description.
CV parameters identical to the main script (5 outer × 4 inner, groups=atomic_idx, C grid,
seed 20260803, bootstrap 5000); the V3 cross-fold assertion stays pinned inside the
imported nested_cv.

Usage:
  python scripts/summarize_hidden_probe_lvsplit.py --smoke   # verify the path with _smoke artifacts
  python scripts/summarize_hidden_probe_lvsplit.py > outputs/ci/hidden_probe_lvsplit.txt
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_hidden_probe import (BOOT, CGRID, MODELS, N_INNER, N_OUTER,  # noqa: E402
                                    SEED, bal_acc, boot_ci, load_reps, nested_cv,
                                    shuffle_labels, tfidf_cv)
from summarize_length_matched import build_subsets, auc  # noqa: E402  single source

# Registered values (all computed elsewhere before this script was written; failure to
# reproduce = this script's pipeline is wrong, not a new finding)
ANCHOR_MAIN = {'Qwen/Qwen3-4B': 0.692, 'Qwen/Qwen2.5-7B-Instruct': 0.716}
TOL_MAIN = 0.002
REF = {4: dict(ceiling=0.728, len_auc=0.712), 5: dict(ceiling=0.843, len_auc=0.802)}
TOL_REF = 0.005
# Registered anchors for the --n_perm permutation null distribution (the observed arm and
# the single shuffle are both deterministic reruns, must reproduce ±0.002):
# lv4 observed/single shuffle = first-round full hidden_probe_lvsplit.txt; lv5 = main table hidden_probe.txt G-2
PERM_REG = {
    'Qwen/Qwen3-4B':            {4: dict(obs=0.604, shuf=0.511), 5: dict(obs=0.692, shuf=0.502)},
    'Qwen/Qwen2.5-7B-Instruct': {4: dict(obs=0.564, shuf=0.444), 5: dict(obs=0.716, shuf=0.518)},
}
N_PERM_LV5 = 10   # permutations for the lv5 symmetric reference (main G-2 already passed; only gives the null center a reference point)


def ratio(b, ceiling):
    """Descriptive quantity: (readout−0.5)/(ceiling−0.5)."""
    return (b - 0.5) / (ceiling - 0.5)


def arm(X, y, groups, layers, n_jobs, tag, ceiling):
    """One arm = main-pipeline nested CV + clustered bootstrap CI + ceiling ratio. Returns (bal, (lo,hi), cv)."""
    cv = nested_cv(X, y, groups, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    b = bal_acc(y, cv['oof_pred'])
    lo, hi = boot_ci(lambda i: bal_acc(np.asarray(y)[i], cv['oof_pred'][i]), groups,
                     np.random.default_rng(SEED), BOOT)
    print(f'  {tag}: 平衡准确率 = {b:.3f} [{lo:.3f},{hi:.3f}]  '
          f'(读数−0.5)/(天花板−0.5) = {ratio(b, ceiling):.3f}')
    print(f'    各外层折选中 (层, C, 内层bal_acc): {cv["chosen"]}')
    return b, (lo, hi), cv


def run_model(model_name, n_jobs, smoke):
    last, g3, meta = load_reps(model_name, smoke)
    taps = last.shape[1]
    layers = list(range(taps))
    print(f'\n{"=" * 78}\n{model_name}  (N={len(meta)}, taps={taps}, hidden={last.shape[2]})'
          f'\n{"=" * 78}')

    if smoke:
        # Path verification, same mechanism as the main script's smoke branch:
        # synthetic groups (2 rows each, one per class) + shrunken grid.
        n = len(meta)
        assert n >= 8 and n % 2 == 0, f'smoke 需要 ≥8 的偶数行，得到 {n}'
        gsyn = np.arange(n) % (n // 2)
        y = (np.arange(n) >= n // 2).astype(int)
        lay = [0, taps // 2, taps - 1]
        cg = (0.1, 1.0)
        cv = nested_cv(last, y, gsyn, lay, cg, 2, 2, n_jobs)
        assert (cv['oof_pred'] != -1).all()
        # shuffle_labels itself must be exercised (assert it is a permutation); but at n=8
        # a true full permutation can create single-class folds (an artifact of smoke scale,
        # not a pipeline property), so the CV path is fed a permutation that keeps one row
        # of each class per group.
        y_sh = shuffle_labels(y, SEED)
        assert sorted(y_sh) == sorted(y), 'shuffle_labels 不是排列'
        cv2 = nested_cv(last, np.roll(y, 1), gsyn, lay, cg, 2, 2, n_jobs)
        assert (cv2['oof_pred'] != -1).all()
        tp, _ = tfidf_cv(meta['questions'].values, y, gsyn, cg, 2, 2, n_jobs)
        assert (tp != -1).all()
        boot_ci(lambda i: bal_acc(y[i], cv['oof_pred'][i]), gsyn,
                np.random.default_rng(SEED), 50)
        # Stratified reporting mechanism: synthesize two strata, compute bal_acc + CI per stratum
        stratum = np.arange(n) % 2
        for s in (0, 1):
            m = stratum == s
            bal_acc(y[m], cv['oof_pred'][m])
        print('[smoke] 嵌套CV/打乱标签/TF-IDF/聚类bootstrap/分层报告通路全部跑通。'
              '样本太小，无读数；锚点与参照线断言按预案跳过。SMOKE PASS')
        return

    pon = meta[meta.modus == 'ponens']
    lv4 = pon[pon.agreement_lv == 4]
    lv5 = pon[pon.agreement_lv == 5]
    lv45 = pon[pon.agreement_lv <= 5]
    assert len(lv4) == 479, f'lv4 ponens 行数 {len(lv4)} != 479'
    assert (meta.agreement_lv == 6).sum() == 4, 'lv=6 行数 != 4（应恰排除 4 行）'
    assert len(lv45) == 870, f'lv∈{{4,5}} ponens 行数 {len(lv45)} != 870'

    def XyG(sub):
        return (last[sub.index.values], (sub['gold'] == 'c').astype(int).values,
                sub['atomic_idx'].values)

    # ---- Arm L5-anchor ----
    print('\n[臂 L5-锚点] lv=5 ponens 重跑主管线（断言 = 与 hidden_probe.txt 登记一致）')
    X5, y5, G5 = XyG(lv5)
    b5, ci5, _ = arm(X5, y5, G5, layers, n_jobs, 'lv=5 (n=391)', REF[5]['ceiling'])
    reg = ANCHOR_MAIN[model_name]
    assert abs(b5 - reg) <= TOL_MAIN, \
        f'L5 锚点失败：{b5:.4f} vs 登记 {reg}（±{TOL_MAIN}）——本脚本管线错，读数无效'
    print(f'  [锚点] {b5:.3f} == 登记 {reg}（±{TOL_MAIN}）  ✅')

    # ---- Arm A ----
    print('\n[臂 A] lv=4 ponens 单独嵌套 CV')
    X4, y4, G4 = XyG(lv4)
    b4, ci4, _ = arm(X4, y4, G4, layers, n_jobs, 'lv=4 (n=479)', REF[4]['ceiling'])

    # ---- Arm B ----
    print('\n[臂 B] lv∈{4,5} ponens 合池嵌套 CV，out-of-fold 按 lv 分层')
    Xb, yb, Gb = XyG(lv45)
    lv_of = lv45['agreement_lv'].values
    cvb = nested_cv(Xb, yb, Gb, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    bb = bal_acc(yb, cvb['oof_pred'])
    lob, hib = boot_ci(lambda i: bal_acc(yb[i], cvb['oof_pred'][i]), Gb,
                       np.random.default_rng(SEED), BOOT)
    print(f'  合池全体 (n=870): 平衡准确率 = {bb:.3f} [{lob:.3f},{hib:.3f}]')
    print(f'    各外层折选中 (层, C, 内层bal_acc): {cvb["chosen"]}')
    strata = {}
    for lv in (4, 5):
        m = lv_of == lv
        bs = bal_acc(yb[m], cvb['oof_pred'][m])
        los, his = boot_ci(lambda i: bal_acc(yb[m][i], cvb['oof_pred'][m][i]), Gb[m],
                           np.random.default_rng(SEED), BOOT)
        strata[lv] = bs
        print(f'  合池内 lv={lv} 层 (n={int(m.sum())}): 平衡准确率 = {bs:.3f} '
              f'[{los:.3f},{his:.3f}]  比值 = {ratio(bs, REF[lv]["ceiling"]):.3f}')

    # ---- Arm C (lv4 gates) ----
    print('\n[臂 C] lv4 闸门（协议同主脚本 G-2/G-3）')
    y4_sh = shuffle_labels(y4, SEED)
    cvs = nested_cv(X4, y4_sh, G4, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    bsh = bal_acc(y4_sh, cvs['oof_pred'])
    losh, hish = boot_ci(lambda i: bal_acc(y4_sh[i], cvs['oof_pred'][i]), G4,
                         np.random.default_rng(SEED), BOOT)
    ok_sh = losh <= 0.5 <= hish
    print(f'  全排列打乱标签: 平衡准确率 = {bsh:.3f} [{losh:.3f},{hish:.3f}]  '
          f'{"✅ CI 含 0.5" if ok_sh else "❌ CI 不含 0.5 → lv4 臂管线泄漏，读数不可信"}')
    tf_pred, tf_C = tfidf_cv(lv4['questions'].values, y4, G4, CGRID, N_OUTER, N_INNER, n_jobs)
    btf = bal_acc(y4, tf_pred)
    print(f'  TF-IDF(1-2gram) 词面基线: 平衡准确率 = {btf:.3f}（各折 C={tf_C}）  '
          f'lv4 探针−基线 = {b4 - btf:+.3f}')

    # ---- lv4 length-rule record ----
    print(f'  长度规则参照：lv4 AUC 0.712 / lv5 AUC 0.802（脚本头部已断言复算一致）')

    return dict(b5=b5, b4=b4, bb=bb, strata=strata, ok_sh=ok_sh, btf=btf)


def run_perm(model_name, n_jobs, n_perm):
    """--n_perm path: permutation null distributions for the standalone lv4 (N=n_perm) / lv5 (N=10 symmetric reference) arms.

    Motivation (added 2026-08-03): 7B lv4 single shuffle 0.444 with CI below 0.5 — a leakage
    signature should sit **above** 0.5; below looks more like CV's pessimistic bias under
    the null plus single-permutation sampling noise (both models get the same shuffle vector
    from the same SEED, 4B 0.511 / 7B 0.444 — same targets, different features). One
    permutation cannot settle it, so a null distribution is added here: seeds =
    SEED+1…SEED+N, each a full-permutation shuffle + complete nested CV.
    The observed arm and the seed-baseline single shuffle are deterministic reruns; assert
    agreement with the registered values first, then draw the null distribution.
    """
    last, _, meta = load_reps(model_name, False)
    layers = list(range(last.shape[1]))
    print(f'\n{"=" * 78}\n{model_name}  (taps={len(layers)})\n{"=" * 78}')
    pon = meta[meta.modus == 'ponens']
    for lv, n_null in ((4, n_perm), (5, N_PERM_LV5)):
        sub = pon[pon.agreement_lv == lv]
        X = last[sub.index.values]
        y = (sub['gold'] == 'c').astype(int).values
        G = sub['atomic_idx'].values
        reg = PERM_REG[model_name][lv]

        cv = nested_cv(X, y, G, layers, CGRID, N_OUTER, N_INNER, n_jobs)
        b_obs = bal_acc(y, cv['oof_pred'])
        assert abs(b_obs - reg['obs']) <= TOL_MAIN, \
            f'锚点失败：lv{lv} 观测 {b_obs:.4f} vs 登记 {reg["obs"]}——管线漂移，零分布无效'
        y_s = shuffle_labels(y, SEED)
        cvs = nested_cv(X, y_s, G, layers, CGRID, N_OUTER, N_INNER, n_jobs)
        b_shuf = bal_acc(y_s, cvs['oof_pred'])
        assert abs(b_shuf - reg['shuf']) <= TOL_MAIN, \
            f'锚点失败：lv{lv} 单次打乱 {b_shuf:.4f} vs 登记 {reg["shuf"]}——管线漂移'

        null = []
        for i in range(1, n_null + 1):
            y_p = shuffle_labels(y, SEED + i)
            cvp = nested_cv(X, y_p, G, layers, CGRID, N_OUTER, N_INNER, n_jobs)
            null.append(bal_acc(y_p, cvp['oof_pred']))
        null = np.asarray(null)
        k_shuf = int((null <= b_shuf).sum())
        k_obs = int((null >= b_obs).sum())
        print(f'\n[lv={lv}] n={len(y)}  观测 = {b_obs:.3f}（登记 {reg["obs"]} ✅）  '
              f'seed 基准单次打乱 = {b_shuf:.3f}（登记 {reg["shuf"]} ✅）')
        print(f'  置换零分布 N={n_null}（seeds {SEED}+1…+{n_null}，每次完整嵌套 CV）:')
        print(f'    均值 = {null.mean():.3f}  SD = {null.std(ddof=1):.3f}  '
              f'[2.5%,97.5%] 分位 = [{np.percentile(null, 2.5):.3f},'
              f'{np.percentile(null, 97.5):.3f}]  min/max = [{null.min():.3f},{null.max():.3f}]')
        print(f'    单次打乱 {b_shuf:.3f} 的零分布位置：#null ≤ 它 = {k_shuf}/{n_null}')
        p_txt = (f'p < 1/{n_null}（加一校正 p ≈ {1 / (n_null + 1):.3f}）' if k_obs == 0 else
                 f'p = {k_obs}/{n_null} = {k_obs / n_null:.3f}'
                 f'（加一校正 {(k_obs + 1) / (n_null + 1):.3f}）')
        print(f'    观测经验 p：#null ≥ {b_obs:.3f} = {k_obs}/{n_null} → {p_txt}；'
              f'{n_null} 次分辨率有限，p 只报到该粒度')
        print(f'    零分布逐值(升序): {np.array2string(np.sort(null), precision=3)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--n_jobs', type=int, default=16)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n_perm', type=int, default=0,
                    help='>0：只跑置换零分布路径（lv4 N=n_perm，lv5 N=10 参照）；'
                         '0=现行为不变。仅全量路径生效，smoke 不受影响。')
    args = ap.parse_args()

    if args.n_perm > 0 and not args.smoke:
        print('# 协议外探索·追加：lv4/lv5 打乱标签臂的置换零分布（不下判定，判定归主会话）')
        print(f'# 嵌套 CV 参数与主脚本一致；置换 seeds = {SEED}+1…+{args.n_perm}（lv4）'
              f'/ +{N_PERM_LV5}（lv5 对称参照）')
        for m in args.models.split(','):
            run_perm(m, args.n_jobs, args.n_perm)
        return

    print('# 协议外探索：探针信号的 agreement_lv 分解（不受 PREREG §6 管辖，不改主读数）')
    print(f'# CV 参数与主脚本一致：外层 {N_OUTER} × 内层 {N_INNER}，组=atomic_idx，'
          f'C∈{CGRID}，seed {SEED}，bootstrap {BOOT}')

    if not args.smoke:
        # Reference-line recomputation assertions (±0.005): ceiling (intent as predictor)
        # and length-rule AUC, stratified by lv
        f = build_subsets()
        for lv in (4, 5):
            s = f[f.lv == lv]
            pred = (s.intent == 'strong').astype(int).values
            yy = s.REQ.values
            ceil = ((pred[yy == 1] == 1).mean() + (pred[yy == 0] == 0).mean()) / 2
            la = auc(s.d.values, s.REQ.values)
            print(f'[参照] lv={lv}: n={len(s)}  天花板 = {ceil:.3f}（登记 {REF[lv]["ceiling"]}）'
                  f'  长度规则 AUC = {la:.3f}（登记 {REF[lv]["len_auc"]}）')
            assert abs(ceil - REF[lv]['ceiling']) <= TOL_REF, f'参照线失败：lv{lv} 天花板不复现'
            assert abs(la - REF[lv]['len_auc']) <= TOL_REF, f'参照线失败：lv{lv} 长度规则不复现'
        print('[参照] 全部复现，断言通过')

    for m in args.models.split(','):
        run_model(m, args.n_jobs, args.smoke)

    if args.smoke:
        print('\n[smoke] 全部模型通路检查完成。无读数，无判定。')
        return

    print(f'\n{"=" * 78}\n解读对照表（描述用；不下判定，判定归主会话）\n{"=" * 78}')
    print('  lv4 比值 ≈ lv5 比值（按比例缩）   → 指向：真信号 + 标签噪声稀释')
    print('  lv4 读数 ≈ 0.5（塌回随机）        → 指向：信号为 lv5（全票一致）特有')
    print('  lv4 绝对读数 ≈ lv5（不缩反平）    → 指向：偏向记忆/构造痕迹解释')


if __name__ == '__main__':
    main()
