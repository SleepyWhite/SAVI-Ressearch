#!/usr/bin/env python
"""B3 summary: relation bal-acc main readout + paired comparisons + end-to-end BREU secondary readout + gates. Pure CPU.

Criteria are frozen in `PREREG_b3.md` (2026-08-04; revision 1 only corrected
wording). This script only prints "falls in tier X" per the criteria; it
**does not make the final call**.

============================================================================
Protocol
============================================================================
- The statistics for the **main readout** (relation bal-acc) are newly
  implemented in this script (the existing single source only computes BREU):
  scenario as the unit, bootstrap stratified by REQ/ALT, B=10000 seed=20260803
  (frozen in PREREG §4). Parse failures (pred_state=None) count as wrong.
  **Anchors (identity cells, V-S1/V-S2)**: the same functions first recompute
  (a) lv5-ponens bal-acc of the probe OOF predictions in the arm A artifact
      == 0.6917 (V-A1 registered value);
  (b) arm C step1 self-judgement lv=5 == 0.519 (§6.19 registered value,
      tolerance ±0.005).
  If not reproduced, raise; no B3 readout is produced.
- **Secondary readout** (BREU) with SM stratification and paired tests: reuses
  `summarize_probe_decode`'s `try_strata/restrict/line/delta` verbatim
  (underneath is the summarize_generative_ci single source, 5000 iterations
  seed=0 — same protocol as the registered numbers of arms A/B/C; not mixed
  with the main readout's B=10000).
- **G-B3-1 fold discipline**: each row's fold field must == the recomputed
  `fold_of_atomic` value (OOF assertion).

Usage
  python scripts/summarize_b3.py > outputs/ci/b3.txt
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from summarize_probe_decode import (GEN, PD, delta, line, recs_of,  # noqa: E402
                                    restrict, try_strata)
from summarize_length_matched import build_subsets, cem_match  # noqa: E402
from run_probe_decode import main_folds, fold_of_atomic  # noqa: E402

B_ITERS, B_SEED = 10000, 20260803
PROBE_ANCHOR, PROBE_TOL = 0.6917, 5e-4     # V-A1 registered value (§6.19)
C_ANCHOR, C_TOL = 0.519, 5e-3              # §6.19 registered value (0.519, registered to 3dp only)
REF = dict(probe=0.692, c_self=0.519, a_breu=0.656, c_breu=0.487, greedy=0.492)
STEP1_SHA_REG = '51e915d1f862944b3e2dc14b9d8720c11f312a3b5d4c358bb8949f7fd6b85e5c'
TEMPLATE_SHA_REG = '5a104f449cb08aba418c5d9d69fd974d27491f846baa5f3282497025366a9376'


# ------------------------------------------------------------------ bal-acc (scenario-clustered)
def gold_of(r):
    """Gold relation. The arm A artifact has no gold_state field, so derive from ground_truth
    (same rule as arm C); when both exist, assert they agree — one more cross-check of the
    rule, not redundancy."""
    d = 'REQ' if r['ground_truth'] == 'c' else 'ALT'
    if 'gold_state' in r:
        assert r['gold_state'] == d, f'idx {r.get("idx")} gold_state 与 ground_truth 推导不符'
    return d


def to_groups(rows):
    """rows (dicts with dataset_id/ground_truth/pred_state) → {sid: (cls, [correct,...])}.
    Both rows of a scenario (ponens/tollens) share the class, pinned by assertion — class is
    the stratification axis; a scenario spanning classes would make stratification meaningless."""
    g = {}
    for r in rows:
        cls = gold_of(r)
        ok = r['pred_state'] == cls
        sid = r['dataset_id']
        if sid in g:
            assert g[sid][0] == cls, f'场景 {sid} 两行 gold_state 不一致'
            g[sid][1].append(ok)
        else:
            g[sid] = (cls, [ok])
    return g


def balacc_point(groups):
    acc = {}
    for cls in ('REQ', 'ALT'):
        flat = [ok for c, oks in groups.values() if c == cls for ok in oks]
        acc[cls] = float(np.mean(flat)) if flat else float('nan')
    return (acc['REQ'] + acc['ALT']) / 2, acc


def balacc_ci(groups, iters=B_ITERS, seed=B_SEED):
    """Bootstrap CI stratified by class, with scenario as the unit."""
    rng = np.random.default_rng(seed)
    per = {cls: [oks for c, oks in groups.values() if c == cls] for cls in ('REQ', 'ALT')}
    vals = []
    for _ in range(iters):
        s = 0.0
        for cls in ('REQ', 'ALT'):
            arr = per[cls]
            pick = rng.integers(0, len(arr), len(arr))
            flat = [ok for j in pick for ok in arr[j]]
            s += float(np.mean(flat))
        vals.append(s / 2)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired_balacc(groups_x, groups_y, label):
    """Δ = y − x; paired by scenario, resampled stratified by class. x/y must cover the same scenarios."""
    assert set(groups_x) == set(groups_y), f'{label}: 两臂场景集不同,配对失义'
    sids = {cls: [s for s in groups_x if groups_x[s][0] == cls] for cls in ('REQ', 'ALT')}
    rng = np.random.default_rng(B_SEED)
    dx, _ = balacc_point(groups_x)
    dy, _ = balacc_point(groups_y)
    vals = []
    for _ in range(B_ITERS):
        sx = sy = 0.0
        for cls in ('REQ', 'ALT'):
            arr = sids[cls]
            pick = rng.integers(0, len(arr), len(arr))
            fx = [ok for j in pick for ok in groups_x[arr[j]][1]]
            fy = [ok for j in pick for ok in groups_y[arr[j]][1]]
            sx += float(np.mean(fx)); sy += float(np.mean(fy))
        vals.append((sy - sx) / 2)
    lo, hi = float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
    verdict = '不可区分' if lo <= 0 <= hi else ('高于' if lo > 0 else '低于')
    print(f'  {label:<40} Δbal-acc={dy - dx:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}]  → {verdict}')
    return dy - dx, lo, hi


def pick(recs, lv=None, modus=None):
    return [r for r in recs if (lv is None or r['agreement_lv'] == lv)
            and (modus is None or r['modus'] == modus)]


def show_balacc(name, rows, ci=True):
    g = to_groups(rows)
    pt, acc = balacc_point(g)
    tail = ''
    if ci:
        lo, hi = balacc_ci(g)
        tail = f' [{lo:.4f},{hi:.4f}]'
    print(f'  {name:<40} n行={len(rows):>4} n场景={len(g):>4}  '
          f'bal-acc={pt:.4f}{tail}  (REQ {acc["REQ"]:.4f} / ALT {acc["ALT"]:.4f})')
    return pt, g


# ------------------------------------------------------------------ main flow
def main():
    print('# B3 汇总(PREREG_b3.md,2026-08-04 冻结;修订 1 只订正措辞)')
    print(f'# 主读数统计:场景分层 bootstrap B={B_ITERS} seed={B_SEED};'
          f'BREU 沿用 summarize_generative_ci 单一来源')
    print(f'# 参照线:探针 {REF["probe"]} / C 自判 {REF["c_self"]} / 随机 0.500;'
          f'BREU 对表 A {REF["a_breu"]} / C {REF["c_breu"]} / greedy {REF["greedy"]}')

    # ---- load + gates ----
    parts = sorted(p for p in glob.glob(os.path.join(PD, 'B3_fold*.jsonl'))
                   if not p.endswith('_smoke.jsonl'))
    rs = [r for p in parts for r in recs_of(p)]
    a_recs = recs_of(os.path.join(PD, 'A_Qwen_Qwen3_4B.jsonl'))
    c_recs = recs_of(os.path.join(PD, 'C_twostep.jsonl'))
    assert a_recs and c_recs, '缺 A/C 臂产物,锚点与配对都无从谈起'

    # ---- anchors (identity cells): recompute the A/C registered values with the same bal-acc functions ----
    print(f'\n{"=" * 78}\n锚点(V-S1/V-S2):本脚本的 bal-acc 机器复算已登记值\n{"=" * 78}')
    a5p = pick(a_recs, lv=5, modus='ponens')
    pa, _ = show_balacc('V-S1 探针 OOF(A 臂 pred_state,lv5-pon)', a5p, ci=False)
    assert abs(pa - PROBE_ANCHOR) <= PROBE_TOL, \
        f'V-S1 失败:{pa:.4f} != {PROBE_ANCHOR} —— bal-acc 机器或 A 产物有问题,不读 B3'
    c5 = pick(c_recs, lv=5)
    pc, _ = show_balacc('V-S2 臂 C 自判(lv=5 全行)', c5, ci=False)
    assert abs(pc - C_ANCHOR) <= C_TOL, \
        f'V-S2 失败:{pc:.4f} != {C_ANCHOR} —— 与 §6.19 登记值不符,不读 B3'
    print('  两锚点复现 ✅')

    if not rs:
        print('\n[无产物] B3_fold*.jsonl 不存在——只跑了锚点自检。')
        return
    print(f'\n{"=" * 78}\n臂 B3(折文件 {len(parts)} 个,n行={len(rs)})\n{"=" * 78}')

    # G-B3-0/1 + completeness
    assert all(r['step1_sha'] == STEP1_SHA_REG and r['template_sha'] == TEMPLATE_SHA_REG
               for r in rs), 'G-B3-0 失败:sha 字段与登记值不符'
    assert not any(r['smoke'] for r in rs), '混入 smoke 记录'
    assert len(rs) == 782 and len({r['idx'] for r in rs}) == 782, \
        f'行数 {len(rs)} != 782 或 idx 重复(5 折未跑齐/跑串)'
    folds, groups, _ = main_folds()
    f_of = fold_of_atomic(folds, groups)
    assert all(r['fold'] == f_of[r['atomic_idx']] for r in rs), \
        'G-B3-1 失败:某行的 fold 字段 != 折结构重算值 —— OOF 纪律破了'
    print('[G-B3-0] sha 一致 ✅   [G-B3-1] 782 行 OOF 纪律逐行断言 ✅')

    # per-fold dev curve registration
    print('  各折 dev bal-acc(best_epoch): ' + ', '.join(
        f'fold{f}={d:.4f}(ep{e})' for f, d, e in sorted(
            {(r['fold'], r['dev_balacc'], r['best_epoch']) for r in rs})))

    # G-B3-2 / G-B3-3
    n = len(rs)
    n_fb = sum(r['fallback'] for r in rs)
    parse_ok = (n - n_fb) / n
    trc1 = np.mean([r['step1_truncated'] for r in rs])
    print(f'[G-B3-2] step1 解析成功率={parse_ok:.4f} {"✅" if parse_ok >= 0.8 else "❌<0.8 仪器坏,主读数无效"}  '
          f'step1 截断率={trc1:.4f} {"✅" if trc1 <= 0.05 else "❌>0.05 仪器坏"}')
    n_req = sum(1 for r in rs if r['pred_state'] == 'REQ')
    n_alt = sum(1 for r in rs if r['pred_state'] == 'ALT')
    top = max(n_req, n_alt) / max(n_req + n_alt, 1)
    print(f'[G-B3-3] 预测边缘分布 REQ {n_req} / ALT {n_alt} / 失败 {n_fb}'
          f'{"  ⚠️ 塌缩形态(某类 ≥0.9),解读须写明" if top >= 0.9 else "  ✅ 未塌缩"}')

    # ---- main readout: relation bal-acc ----
    print(f'\n  ---- 主读数:step1 关系判断折外 bal-acc ----')
    b3_5p = pick(rs, lv=5, modus='ponens')
    pt_main, g_main = show_balacc('B3 lv5-ponens(主口径,=探针口径)', b3_5p)
    show_balacc('B3 lv=5 全 782 行(附报)', pick(rs, lv=5))
    show_balacc('B3 lv5-tollens(附报,题面 γ2 不同)', pick(rs, lv=5, modus='tollens'))

    # ---- paired comparisons (same 391 rows) ----
    print(f'\n  ---- 配对比较(lv5-ponens 同行,场景分层 bootstrap)----')
    idx_c = {r['idx']: r for r in c_recs}
    idx_a = {r['idx']: r for r in a_recs}
    c_rows = [dict(dataset_id=r['dataset_id'], ground_truth=r['ground_truth'],
                   pred_state=idx_c[r['idx']]['pred_state']) for r in b3_5p]
    a_rows = [dict(dataset_id=r['dataset_id'], ground_truth=r['ground_truth'],
                   pred_state=idx_a[r['idx']]['pred_state']) for r in b3_5p]
    assert all(idx_c[r['idx']]['gold_state'] == r['gold_state'] for r in b3_5p)
    paired_balacc(to_groups(c_rows), g_main, 'B3 − C 自判(提示 vs 梯度,同结构先验)')
    paired_balacc(to_groups(a_rows), g_main, 'B3 − 探针(梯度 vs 表征读出)')

    # ---- secondary readout: end-to-end BREU (reuses the existing single source) ----
    merged = os.path.join(PD, 'B3_merged.jsonl')
    with open(merged, 'w') as f:
        for r in sorted(rs, key=lambda x: x['idx']):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    sm = set(cem_match(build_subsets()))
    fol = [r for r in rs if not r['fallback']]
    print(f'\n  ---- 次读数:端到端 BREU(lv=5 折外)----')
    print(f'  执行健康:跟随率(非回退)={np.mean([r["follows"] for r in fol]):.4f}  '
          f'格式兑现率={np.mean([r["has_final_answer"] for r in rs]):.4f}  '
          f'step2 截断率={np.mean([r["truncated"] for r in rs]):.4f}')
    st5 = try_strata(merged, 5, 'B3/lv5')
    b5 = line('B3', st5)
    st5_sm = restrict(st5, sm)
    if len(st5_sm[0]) and len(st5_sm[1]):
        print('  ---- lv=5 × SM 长度配平分层 ----')
        line('B3 / SM', st5_sm)
    gd5 = try_strata(os.path.join(GEN, 'time_t1_Qwen_Qwen3_4B_dp.jsonl'), 5, 'dp/lv5')
    if gd5 is not None:
        print('  ---- 配对检验(BREU)----')
        delta(gd5, st5, 'B3 − greedy(DP), lv=5')
        if len(st5_sm[0]) and len(st5_sm[1]):
            delta(restrict(gd5, sm), st5_sm, 'B3 − greedy(DP), lv=5×SM')

    # ---- tier assignment (PREREG_b3 §1, no final call) ----
    print(f'\n{"=" * 78}\n按 PREREG_b3 §1 落档(本脚本只对表,不定案)\n{"=" * 78}')
    if pt_main < 0.55:
        t = ('机制结论干净成立档(<0.55):该配置下梯度提取不出表征里线性可读的区分;'
             'B 臂失败不是监督目标不公平的伪影')
    elif pt_main >= 0.65:
        t = ('头条改写档(≥0.65):"端到端微调失败,结构化微调可以";'
             '探针独特性降级为免训练/可解释/样本效率')
    else:
        t = '中间带(0.55–0.65):部分提取,如实报,交人裁;看与 C 的配对差值'
    print(f'  B3 主读数(lv5-ponens bal-acc)={pt_main:.4f} → {t}')
    print(f'  次读数 BREU={b5:.4f}(对表 A {REF["a_breu"]} / C {REF["c_breu"]});'
          f'一致性检查用,不改档位')
    print('  最终判定留给人。')


if __name__ == '__main__':
    main()
