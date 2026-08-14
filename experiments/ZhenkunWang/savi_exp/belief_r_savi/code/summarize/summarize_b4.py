#!/usr/bin/env python
"""B4 (lm_head-only LoRA) summary. Pure CPU. Criteria frozen in `PREREG_b4.md`; checks against them only, no final call.

All statistical pieces are imported from `summarize_b3` (bal-acc
scenario-stratified bootstrap B=10000 seed=20260803, incl. the V-S1/V-S2
anchor discipline) and `summarize_probe_decode` (BREU single source).
The two probe reference lines are different things and must not be mixed
(reminder from the coding subagent):
  - 0.692 = arm A OOF probe (nested CV layer selection) — used for the paired
    comparison (has row-level predictions);
  - 0.688 = last-layer L36 fixed-layer probe of the L7 layerwise curve — the
    anchor for the PREREG_b4 §1 point prediction, checked against only.

Usage
  python scripts/summarize_b4.py > outputs/ci/b4.txt
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from summarize_b3 import (STEP1_SHA_REG, TEMPLATE_SHA_REG,  # noqa: E402
                          balacc_point, paired_balacc, pick, show_balacc, to_groups)
from summarize_probe_decode import (GEN, PD, delta, line, recs_of,  # noqa: E402
                                    restrict, try_strata)
from summarize_length_matched import build_subsets, cem_match  # noqa: E402
from run_probe_decode import main_folds, fold_of_atomic  # noqa: E402

REF = dict(probe_oof=0.692, probe_l36=0.688, b3=0.6407, c_self=0.519,
           b3_range=0.2778, greedy=0.492)


def load_arm(prefix):
    parts = sorted(p for p in glob.glob(os.path.join(PD, f'{prefix}_fold*.jsonl'))
                   if not p.endswith('_smoke.jsonl'))
    return parts, [r for p in parts for r in recs_of(p)]


def fold_ponens_balacc(rs):
    """Per-fold lv5-ponens row-level bal-acc (same estimator as the main readout, restricted to the fold)."""
    out = {}
    for f in sorted({r['fold'] for r in rs}):
        sub = [r for r in pick(rs, lv=5, modus='ponens') if r['fold'] == f]
        out[f] = balacc_point(to_groups(sub))[0]
    return out


def main():
    print('# B4 汇总(PREREG_b4.md,2026-08-04 冻结;修订 1 = 四条实现层登记)')
    print(f'# 参照线:A 臂 OOF 探针 {REF["probe_oof"]}(配对用)/ 末层 L36 固定层探针 '
          f'{REF["probe_l36"]}(点预测锚,只对表)/ B3 {REF["b3"]} / C 自判 {REF["c_self"]}')

    parts, rs = load_arm('B4')
    b3_parts, b3 = load_arm('B3')
    a_recs = recs_of(os.path.join(PD, 'A_Qwen_Qwen3_4B.jsonl'))
    c_recs = recs_of(os.path.join(PD, 'C_twostep.jsonl'))
    assert rs and b3 and a_recs and c_recs

    # ---- gates ----
    print(f'\n{"=" * 78}\n臂 B4(折文件 {len(parts)} 个,n行={len(rs)})\n{"=" * 78}')
    assert all(r['step1_sha'] == STEP1_SHA_REG and r['template_sha'] == TEMPLATE_SHA_REG
               for r in rs), 'G-B3-0 失败'
    assert not any(r['smoke'] for r in rs) and len(rs) == 782 \
        and len({r['idx'] for r in rs}) == 782
    folds, groups, _ = main_folds()
    f_of = fold_of_atomic(folds, groups)
    assert all(r['fold'] == f_of[r['atomic_idx']] for r in rs), 'OOF 纪律破了'
    print('[G-B3-0] sha 一致 ✅   [G-B3-1] 782 行 OOF 纪律逐行断言 ✅')
    print('  各折 dev bal-acc(best_epoch): ' + ', '.join(
        f'fold{f}={d:.4f}(ep{e})' for f, d, e in sorted(
            {(r['fold'], r['dev_balacc'], r['best_epoch']) for r in rs})))
    n = len(rs)
    n_fb = sum(r['fallback'] for r in rs)
    trc1 = np.mean([r['step1_truncated'] for r in rs])
    print(f'[G-B3-2] step1 解析成功率={(n - n_fb) / n:.4f} ✅  截断率={trc1:.4f} ✅')
    n_req = sum(1 for r in rs if r['pred_state'] == 'REQ')
    n_alt = sum(1 for r in rs if r['pred_state'] == 'ALT')
    top = max(n_req, n_alt) / max(n_req + n_alt, 1)
    print(f'[G-B3-3] 预测边缘分布 REQ {n_req} / ALT {n_alt} / 失败 {n_fb}'
          f'{"  ⚠️ 塌缩" if top >= 0.9 else "  ✅ 未塌缩"}')

    # ---- main readout ----
    print(f'\n  ---- 主读数:step1 关系判断折外 bal-acc ----')
    b4_5p = pick(rs, lv=5, modus='ponens')
    pt_main, g_main = show_balacc('B4 lv5-ponens(主口径)', b4_5p)
    show_balacc('B4 lv=5 全 782 行(附报)', pick(rs, lv=5))
    show_balacc('B4 lv5-tollens(附报)', pick(rs, lv=5, modus='tollens'))

    # ---- per-fold stability (half of the criterion; B3 recomputed with the same estimator — don't quote the row-level numbers of §6.20) ----
    fp4, fp3 = fold_ponens_balacc(rs), fold_ponens_balacc(b3)
    r4 = max(fp4.values()) - min(fp4.values())
    r3 = max(fp3.values()) - min(fp3.values())
    print(f'\n  ---- 逐折稳定性(lv5-ponens,同一估计量)----')
    print(f'  B4 各折: ' + '  '.join(f'f{k}={v:.4f}' for k, v in fp4.items())
          + f'   最差 {min(fp4.values()):.4f}  极差 {r4:.4f}')
    print(f'  B3 各折: ' + '  '.join(f'f{k}={v:.4f}' for k, v in fp3.items())
          + f'   最差 {min(fp3.values()):.4f}  极差 {r3:.4f}')

    # ---- paired comparisons (same 391 lv5-ponens rows) ----
    print(f'\n  ---- 配对比较(场景分层 bootstrap)----')
    idx_b3 = {r['idx']: r for r in b3}
    idx_a = {r['idx']: r for r in a_recs}
    idx_c = {r['idx']: r for r in c_recs}
    mk = lambda src: to_groups([dict(dataset_id=r['dataset_id'],
                                     ground_truth=r['ground_truth'],
                                     pred_state=src[r['idx']]['pred_state'])
                                for r in b4_5p])
    paired_balacc(mk(idx_b3), g_main, 'B4 − B3(接口 vs 深层,同监督同折)')
    paired_balacc(mk(idx_a), g_main, 'B4 − 探针(A 臂 OOF)')
    paired_balacc(mk(idx_c), g_main, 'B4 − C 自判')

    # ---- secondary readout: end-to-end BREU ----
    merged = os.path.join(PD, 'B4_merged.jsonl')
    with open(merged, 'w') as f:
        for r in sorted(rs, key=lambda x: x['idx']):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    sm = set(cem_match(build_subsets()))
    fol = [r for r in rs if not r['fallback']]
    print(f'\n  ---- 次读数:端到端 BREU(lv=5 折外)----')
    print(f'  执行健康:跟随率(非回退)={np.mean([r["follows"] for r in fol]):.4f}  '
          f'格式兑现率={np.mean([r["has_final_answer"] for r in rs]):.4f}  '
          f'step2 截断率={np.mean([r["truncated"] for r in rs]):.4f}')
    st5 = try_strata(merged, 5, 'B4/lv5')
    b5 = line('B4', st5)
    st5_sm = restrict(st5, sm)
    if len(st5_sm[0]) and len(st5_sm[1]):
        print('  ---- lv=5 × SM 长度配平分层 ----')
        line('B4 / SM', st5_sm)
    gd5 = try_strata(os.path.join(GEN, 'time_t1_Qwen_Qwen3_4B_dp.jsonl'), 5, 'dp/lv5')
    if gd5 is not None:
        print('  ---- 配对检验(BREU)----')
        delta(gd5, st5, 'B4 − greedy(DP), lv=5')
        if len(st5_sm[0]) and len(st5_sm[1]):
            delta(restrict(gd5, sm), st5_sm, 'B4 − greedy(DP), lv=5×SM')

    # ---- tier assignment (PREREG_b4 §2, no final call) ----
    print(f'\n{"=" * 78}\n按 PREREG_b4 §2 落档(本脚本只对表,不定案)\n{"=" * 78}')
    worst = min(fp4.values())
    if pt_main >= 0.65 and worst >= 0.60:
        t = '接口假说成立档(pooled ≥0.65 且最差折 ≥0.60)'
    elif pt_main < 0.55:
        t = '先查仪器档(<0.55)'
    else:
        t = f'中间带(pooled {"≥0.65 但最差折 <0.60" if pt_main >= 0.65 else "落 0.55–0.65"}),如实报交人裁'
    print(f'  B4 主读数={pt_main:.4f}  最差折={worst:.4f}  → {t}')
    print(f'  点预测对表:末层 L36 固定层探针 {REF["probe_l36"]}(差 {pt_main - REF["probe_l36"]:+.4f})')
    print(f'  稳定性对表:B4 极差 {r4:.4f} vs B3 极差 {r3:.4f}(同一估计量)')
    print('  提示:五折 best_epoch 是否全在最后一轮、dev 是否仍在上升(欠训练迹象)见上方登记。')
    print('  最终判定留给人。')


if __name__ == '__main__':
    main()
