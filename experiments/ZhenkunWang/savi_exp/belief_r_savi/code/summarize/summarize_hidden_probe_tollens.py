#!/usr/bin/env python
"""Off-protocol exploration: full probe test on tollens rows (upgraded from the "agreement-rate diagnostic"). Pure CPU.

**Not governed by the PREREG_hidden_probe.md §6 criteria; does not change the main readout.**
Motivation = on 2026-08-03 the user pointed out that the main pipeline tested only ponens
and used tollens merely as an agreement-rate diagnostic, without sufficient justification.
The main readout and the six gates remain those of outputs/ci/hidden_probe.txt; this script
only upgrades the tollens rows to a full test and makes no verdict. All pipeline parts are
imported, reusing the frozen deliverables; zero changes to existing scripts.

Labels: for tollens rows, y = (gold=='c') holds just the same — in REQ scenarios both
twins' gold is c; in ALT scenarios the tollens gold is b (still != 'c'). In-script
assertions: lv=5 tollens has 391 rows; y agrees scenario-by-scenario with the same
scenario's ponens row; atomic_idx agrees scenario-by-scenario.

Four arms (two models × main position last_tok, lv=5):
  T1  tollens standalone nested CV (protocol identical to the main pipeline: 5×4,
      groups=atomic_idx, same seed/grid);
  T2  ponens→tollens transfer: re-run the ponens main pipeline (first assert the anchors
      0.692/0.716 reproduce), use each outer fold's model to predict the tollens rows of
      that fold's test scenarios (groups never seen), accuracy vs y + clustered CI;
      recompute the same-scenario agreement rate and assert == the main-table registered
      values (0.711/0.611);
  T3  reverse transfer (train on tollens → predict ponens), symmetric protocol, as reference;
  T4  gates: tollens-arm TF-IDF lexical baseline (same folds, same protocol) + shuffled-label
      permutation null distribution N=10 (last round's lesson: a single permutation is no
      gate; give the null distribution and the empirical p of the observation directly).

Reference lines: the ceiling 0.843 and the length-rule AUC 0.802 are **scenario-level
quantities**, identical for ponens/tollens; not recomputed, only cited.

Usage:
  python scripts/summarize_hidden_probe_tollens.py --smoke
  python scripts/summarize_hidden_probe_tollens.py > outputs/ci/hidden_probe_tollens.txt
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
from summarize_hidden_probe_lvsplit import ANCHOR_MAIN, TOL_MAIN  # noqa: E402  single source of the anchors

CEILING_LV5, LEN_AUC_LV5 = 0.843, 0.802   # scenario-level quantities, same for ponens/tollens (cited, not recomputed)
# Agreement-rate diagnostic registered in the main table hidden_probe.txt (T2's recomputation
# must reproduce it digit-for-digit, anchoring the transfer pipeline as isomorphic to the main table)
AGREE_REG = {'Qwen/Qwen3-4B': 0.711, 'Qwen/Qwen2.5-7B-Instruct': 0.611}
N_PERM = 10


def ratio(b):
    """Descriptive quantity: (readout−0.5)/(ceiling−0.5); ceiling = scenario-level 0.843."""
    return (b - 0.5) / (CEILING_LV5 - 0.5)


def transfer_predict(cv, sid_src, rows_by_sid, last):
    """Use each outer fold's model from the nested CV to predict the twin rows (other modus) of that fold's test scenarios.

    The group (atomic_idx) never appeared in training — outer folds split by group, and twin
    rows share group and scenario. Returns {sid: pred}; each scenario is predicted exactly
    once (asserted).
    """
    out = {}
    for k, (tr, te) in enumerate(cv['folds']):
        L, C, sc, clf = cv['models'][k]
        sids = sid_src[te]
        rows = [rows_by_sid[s] for s in sids]
        p = clf.predict(sc.transform(last[rows][:, L].astype(np.float32)))
        for s, pi in zip(sids, p):
            assert s not in out, f'场景 {s} 被预测两次'
            out[s] = int(pi)
    return out


def perm_null(X, y, G, layers, n_jobs, n_perm):
    """Shuffled-label permutation null distribution: seeds SEED+1…SEED+n_perm, full nested CV each time."""
    null = []
    for i in range(1, n_perm + 1):
        y_p = shuffle_labels(y, SEED + i)
        cvp = nested_cv(X, y_p, G, layers, CGRID, N_OUTER, N_INNER, n_jobs)
        null.append(bal_acc(y_p, cvp['oof_pred']))
    return np.asarray(null)


def run_model(model_name, n_jobs, smoke):
    last, _, meta = load_reps(model_name, smoke)
    taps = last.shape[1]
    layers = list(range(taps))
    print(f'\n{"=" * 78}\n{model_name}  (N={len(meta)}, taps={taps}, hidden={last.shape[2]})'
          f'\n{"=" * 78}')

    if smoke:
        # Path check (same mechanism as the lvsplit smoke): synthetic groups/labels + self-transfer to verify the transfer mechanism.
        n = len(meta)
        assert n >= 8 and n % 2 == 0, f'smoke 需要 ≥8 的偶数行，得到 {n}'
        gsyn = np.arange(n) % (n // 2)
        y = (np.arange(n) >= n // 2).astype(int)
        lay = [0, taps // 2, taps - 1]
        cg = (0.1, 1.0)
        cv = nested_cv(last, y, gsyn, lay, cg, 2, 2, n_jobs)
        assert (cv['oof_pred'] != -1).all()
        # transfer mechanism: synthetic sid=row number, self-transfer (twin row = itself), verify each is covered exactly once
        sids = np.arange(n)
        preds = transfer_predict(cv, sids, {i: i for i in range(n)}, last)
        assert len(preds) == n
        y_sh = shuffle_labels(y, SEED)
        assert sorted(y_sh) == sorted(y), 'shuffle_labels 不是排列'
        cv2 = nested_cv(last, np.roll(y, 1), gsyn, lay, cg, 2, 2, n_jobs)
        assert (cv2['oof_pred'] != -1).all()
        tp, _ = tfidf_cv(meta['questions'].values, y, gsyn, cg, 2, 2, n_jobs)
        assert (tp != -1).all()
        boot_ci(lambda i: bal_acc(y[i], cv['oof_pred'][i]), gsyn,
                np.random.default_rng(SEED), 50)
        print('[smoke] 嵌套CV/自迁移/打乱标签/TF-IDF/聚类bootstrap 通路全部跑通。'
              '样本太小，无读数；标签与锚点断言按预案跳过。SMOKE PASS')
        return

    pon = meta[(meta.agreement_lv == 5) & (meta.modus == 'ponens')]
    tol = meta[(meta.agreement_lv == 5) & (meta.modus == 'tollens')]
    assert len(tol) == 391, f'lv=5 tollens 行数 {len(tol)} != 391'
    assert set(pon.dataset_id) == set(tol.dataset_id), '孪生场景集合不一致'
    tol_row = dict(zip(tol['dataset_id'], tol.index))
    pon_row = dict(zip(pon['dataset_id'], pon.index))
    tol_by_sid = tol.set_index('dataset_id')
    for _, r in pon.iterrows():
        tw = tol_by_sid.loc[r['dataset_id']]
        assert (r['gold'] == 'c') == (tw['gold'] == 'c'), \
            f"场景 {r['dataset_id']} 孪生 y 不一致"
        assert r['atomic_idx'] == tw['atomic_idx'], \
            f"场景 {r['dataset_id']} 孪生 atomic_idx 不一致"
    print('[标签] lv=5 tollens 391 行；y=(gold==c) 与同场景 ponens 逐场景一致；'
          'atomic_idx 逐场景一致  —— 断言通过')

    def XyG(sub):
        return (last[sub.index.values], (sub['gold'] == 'c').astype(int).values,
                sub['atomic_idx'].values)

    Xt, yt, Gt = XyG(tol)
    Xp, yp, Gp = XyG(pon)
    rng = np.random.default_rng

    # ---- T1 tollens standalone nested CV ----
    print('\n[T1] tollens lv=5 单独嵌套 CV（协议与主管线完全一致）')
    cv_t = nested_cv(Xt, yt, Gt, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    b_t1 = bal_acc(yt, cv_t['oof_pred'])
    lo, hi = boot_ci(lambda i: bal_acc(yt[i], cv_t['oof_pred'][i]), Gt, rng(SEED), BOOT)
    print(f'  平衡准确率 = {b_t1:.3f} [{lo:.3f},{hi:.3f}]  '
          f'(读数−0.5)/(天花板−0.5) = {ratio(b_t1):.3f}')
    print(f'  各外层折选中 (层, C, 内层bal_acc): {cv_t["chosen"]}')

    # ---- T2 ponens→tollens transfer ----
    print('\n[T2] ponens→tollens 迁移（各外层折模型预测该折测试场景的 tollens 行）')
    cv_p = nested_cv(Xp, yp, Gp, layers, CGRID, N_OUTER, N_INNER, n_jobs)
    b_p = bal_acc(yp, cv_p['oof_pred'])
    reg = ANCHOR_MAIN[model_name]
    assert abs(b_p - reg) <= TOL_MAIN, \
        f'锚点失败：ponens 主管线 {b_p:.4f} vs 登记 {reg}——管线漂移，迁移读数无效'
    print(f'  [锚点] ponens 主管线 = {b_p:.3f} == 登记 {reg}（±{TOL_MAIN}）  ✅')
    preds2 = transfer_predict(cv_p, pon['dataset_id'].values, tol_row, last)
    pred_t = np.array([preds2[s] for s in tol['dataset_id']])
    b_t2 = bal_acc(yt, pred_t)
    lo2, hi2 = boot_ci(lambda i: bal_acc(yt[i], pred_t[i]), Gt, rng(SEED), BOOT)
    # Agreement-rate recomputation = the diagnostic registered in the main table (isomorphism anchor)
    pon_pos = {s: j for j, s in enumerate(pon['dataset_id'])}
    agree = np.mean([preds2[s] == cv_p['oof_pred'][pon_pos[s]] for s in pon['dataset_id']])
    assert abs(agree - AGREE_REG[model_name]) <= TOL_MAIN, \
        f'锚点失败：一致率复算 {agree:.4f} vs 主表登记 {AGREE_REG[model_name]}'
    print(f'  对金标平衡准确率 = {b_t2:.3f} [{lo2:.3f},{hi2:.3f}]  比值 = {ratio(b_t2):.3f}')
    print(f'  [锚点] 同场景一致率复算 = {agree:.3f} == 主表登记 '
          f'{AGREE_REG[model_name]}（±{TOL_MAIN}）  ✅')

    # ---- T3 reverse transfer ----
    print('\n[T3] tollens→ponens 反向迁移（协议对称，作参照）')
    preds3 = transfer_predict(cv_t, tol['dataset_id'].values, pon_row, last)
    pred_p = np.array([preds3[s] for s in pon['dataset_id']])
    b_t3 = bal_acc(yp, pred_p)
    lo3, hi3 = boot_ci(lambda i: bal_acc(yp[i], pred_p[i]), Gp, rng(SEED), BOOT)
    print(f'  对金标平衡准确率 = {b_t3:.3f} [{lo3:.3f},{hi3:.3f}]  比值 = {ratio(b_t3):.3f}')

    # ---- T4 gates ----
    print('\n[T4] tollens 臂闸门')
    tf_pred, tf_C = tfidf_cv(tol['questions'].values, yt, Gt, CGRID, N_OUTER, N_INNER, n_jobs)
    btf = bal_acc(yt, tf_pred)
    print(f'  TF-IDF(1-2gram) 词面基线: 平衡准确率 = {btf:.3f}（各折 C={tf_C}）  '
          f'T1 探针−基线 = {b_t1 - btf:+.3f}')
    null = perm_null(Xt, yt, Gt, layers, n_jobs, N_PERM)
    k_obs = int((null >= b_t1).sum())
    p_txt = (f'p < 1/{N_PERM}（加一校正 p ≈ {1 / (N_PERM + 1):.3f}）' if k_obs == 0 else
             f'p = {k_obs}/{N_PERM} = {k_obs / N_PERM:.3f}'
             f'（加一校正 {(k_obs + 1) / (N_PERM + 1):.3f}）')
    print(f'  打乱标签置换零分布 N={N_PERM}（seeds {SEED}+1…+{N_PERM}，每次完整嵌套 CV）:')
    print(f'    均值 = {null.mean():.3f}  SD = {null.std(ddof=1):.3f}  '
          f'[2.5%,97.5%] 分位 = [{np.percentile(null, 2.5):.3f},'
          f'{np.percentile(null, 97.5):.3f}]  min/max = [{null.min():.3f},{null.max():.3f}]')
    print(f'    T1 观测经验 p：#null ≥ {b_t1:.3f} = {k_obs}/{N_PERM} → {p_txt}；'
          f'{N_PERM} 次分辨率有限，p 只报到该粒度')
    print(f'    零分布逐值(升序): {np.array2string(np.sort(null), precision=3)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--n_jobs', type=int, default=16)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    print('# 协议外探索：tollens 行完整探针测试（不受 PREREG §6 管辖，不改主读数）')
    print(f'# CV 参数与主脚本一致：外层 {N_OUTER} × 内层 {N_INNER}，组=atomic_idx，'
          f'C∈{CGRID}，seed {SEED}，bootstrap {BOOT}')
    print(f'# 参照线（场景级量，ponens/tollens 相同，引用不重算）：'
          f'天花板 {CEILING_LV5} / 长度规则 AUC {LEN_AUC_LV5}')

    for m in args.models.split(','):
        run_model(m, args.n_jobs, args.smoke)

    if args.smoke:
        print('\n[smoke] 全部模型通路检查完成。无读数，无判定。')
        return
    print(f'\n{"=" * 78}\n（描述用输出；不下判定，判定归主会话）\n{"=" * 78}')


if __name__ == '__main__':
    main()
