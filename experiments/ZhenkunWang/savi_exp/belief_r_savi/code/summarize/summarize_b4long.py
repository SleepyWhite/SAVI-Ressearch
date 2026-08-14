#!/usr/bin/env python
"""B4L (relaxed-budget lm_head LoRA) summary. Pure CPU. Criteria frozen in `PREREG_b4long.md` §3; compares against the table only, makes no final call.

Statistical parts imported from `summarize_b3`/`summarize_b4` (bal-acc scenario-stratified
bootstrap B=10000 seed=20260803; BREU via the summarize_probe_decode single source).
Convergence check per the §2 frozen definition: early stop triggered, or d[-1]−d[-4] < 0.005;
if ≥2 folds ran the full 20 epochs without converging → record only, assign no tier.

Usage
  python scripts/summarize_b4long.py > outputs/ci/b4long.txt
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from summarize_b3 import (STEP1_SHA_REG, TEMPLATE_SHA_REG,  # noqa: E402
                          balacc_point, paired_balacc, pick, show_balacc, to_groups)
from summarize_b4 import fold_ponens_balacc, load_arm  # noqa: E402
from summarize_probe_decode import (GEN, PD, delta, line, recs_of,  # noqa: E402
                                    restrict, try_strata)
from summarize_length_matched import build_subsets, cem_match  # noqa: E402
from run_probe_decode import main_folds, fold_of_atomic  # noqa: E402

REF = dict(b4=0.6306, b3=0.6407, probe_oof=0.692, probe_l36=0.688, c_self=0.519)
CONV_TAIL_EPS, MAX_EP = 0.005, 20
ANCHOR_FLOOR = REF['b4'] - 0.05   # V-B4L: more budget must not be worse


def main():
    print('# B4L 汇总(PREREG_b4long.md §3,2026-08-05 冻结)')
    print(f'# 参照线:B4 {REF["b4"]} / B3 {REF["b3"]} / A 臂 OOF 探针 {REF["probe_oof"]} / '
          f'末层 L36 探针 {REF["probe_l36"]}(点预测锚) / C 自判 {REF["c_self"]}')

    parts, rs = load_arm('B4L')
    parts = [p for p in parts if 'lv4' not in os.path.basename(p)]
    rs = [r for r in rs if r['arm'] == 'B4L']
    b4_parts, b4 = load_arm('B4')
    b4 = [r for r in b4 if r['arm'] == 'B4']
    b3_parts, b3 = load_arm('B3')
    a_recs = recs_of(os.path.join(PD, 'A_Qwen_Qwen3_4B.jsonl'))
    c_recs = recs_of(os.path.join(PD, 'C_twostep.jsonl'))
    assert rs and b4 and b3 and a_recs and c_recs

    print(f'\n{"=" * 78}\n臂 B4L(折文件 {len(parts)} 个,n行={len(rs)})\n{"=" * 78}')
    assert all(r['step1_sha'] == STEP1_SHA_REG and r['template_sha'] == TEMPLATE_SHA_REG
               for r in rs), 'G-B3-0 失败'
    assert not any(r['smoke'] for r in rs) and len(rs) == 782 \
        and len({r['idx'] for r in rs}) == 782
    folds, groups, _ = main_folds()
    f_of = fold_of_atomic(folds, groups)
    assert all(r['fold'] == f_of[r['atomic_idx']] for r in rs), 'OOF 纪律破了'
    print('[G-B3-0] sha 一致 ✅   [G-B3-1] 782 行 OOF 纪律逐行断言 ✅')
    n = len(rs)
    n_fb = sum(r['fallback'] for r in rs)
    print(f'[G-B3-2] step1 解析成功率={(n - n_fb) / n:.4f} ✅  '
          f'截断率={np.mean([r["step1_truncated"] for r in rs]):.4f} ✅')

    # ---- Convergence check (§2 frozen) ----
    print(f'\n  ---- 收敛判定(早停触发 或 d[-1]−d[-4] < {CONV_TAIL_EPS}) ----')
    meta = {r['fold']: r for r in rs}
    n_unconv = 0
    for f in sorted(meta):
        m = meta[f]
        dc = m['dev_curve']
        tail = dc[-1] - dc[-4] if len(dc) >= 4 else float('nan')
        conv = m['early_stopped'] or (tail < CONV_TAIL_EPS)
        n_unconv += (not conv)
        print(f'  fold{f}: epochs={m["n_epochs_run"]}/{MAX_EP}  early_stop={m["early_stopped"]}'
              f'  best_ep={m["best_epoch"]}  best_dev={m["dev_balacc"]:.4f}'
              f'  tail={tail:+.4f}  → {"已收敛" if conv else "未收敛"}')
    if n_unconv >= 2:
        print(f'  ⚠️ {n_unconv} 折未收敛 ≥2 —— 按 §2 只登记不落档')
    else:
        print(f'  未收敛折数 = {n_unconv} < 2  → 可落档 ✅')

    # ---- Main readout ----
    print(f'\n  ---- 主读数:step1 关系判断折外 bal-acc ----')
    b4l_5p = pick(rs, lv=5, modus='ponens')
    pt_main, g_main = show_balacc('B4L lv5-ponens(主口径)', b4l_5p)
    show_balacc('B4L lv=5 全 782 行(附报)', pick(rs, lv=5))
    show_balacc('B4L lv5-tollens(附报)', pick(rs, lv=5, modus='tollens'))
    anchor_ok = pt_main >= ANCHOR_FLOOR
    print(f'  [V-B4L] pooled {pt_main:.4f} ≥ B4 {REF["b4"]} − 0.05 = {ANCHOR_FLOOR:.4f}'
          f'  {"✅" if anchor_ok else "❌ 更多预算反而更差 —— 查仪器"}')

    # ---- Per-fold stability ----
    fp = fold_ponens_balacc(rs)
    fp4 = fold_ponens_balacc(b4)
    print(f'\n  ---- 逐折(lv5-ponens,同一估计量) ----')
    print('  B4L 各折: ' + '  '.join(f'f{k}={v:.4f}' for k, v in fp.items())
          + f'   最差 {min(fp.values()):.4f}  极差 {max(fp.values()) - min(fp.values()):.4f}')
    print('  B4  各折: ' + '  '.join(f'f{k}={v:.4f}' for k, v in fp4.items())
          + f'   最差 {min(fp4.values()):.4f}  极差 {max(fp4.values()) - min(fp4.values()):.4f}')

    # ---- Paired comparisons ----
    print(f'\n  ---- 配对比较(lv5-ponens 同 391 行,场景分层 bootstrap) ----')
    mk = lambda recs_list: to_groups([dict(dataset_id=r['dataset_id'],
                                           ground_truth=r['ground_truth'],
                                           pred_state={x['idx']: x for x in recs_list}
                                           [r['idx']]['pred_state']) for r in b4l_5p])
    paired_balacc(mk(b4), g_main, 'B4L − B4(预算的净效应)')
    paired_balacc(mk(b3), g_main, 'B4L − B3')
    paired_balacc(mk(a_recs), g_main, 'B4L − 探针(A 臂 OOF)')
    paired_balacc(mk(c_recs), g_main, 'B4L − C 自判')

    # ---- Secondary readout: BREU ----
    merged = os.path.join(PD, 'B4L_merged.jsonl')
    with open(merged, 'w') as f:
        for r in sorted(rs, key=lambda x: x['idx']):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    sm = set(cem_match(build_subsets()))
    print(f'\n  ---- 次读数:端到端 BREU(lv=5 折外) ----')
    st5 = try_strata(merged, 5, 'B4L/lv5')
    b5 = line('B4L', st5)
    st5_sm = restrict(st5, sm)
    if len(st5_sm[0]) and len(st5_sm[1]):
        line('B4L / SM', st5_sm)
    gd5 = try_strata(os.path.join(GEN, 'time_t1_Qwen_Qwen3_4B_dp.jsonl'), 5, 'dp/lv5')
    if gd5 is not None:
        delta(gd5, st5, 'B4L − greedy(DP), lv=5')
        if len(st5_sm[0]) and len(st5_sm[1]):
            delta(restrict(gd5, sm), st5_sm, 'B4L − greedy(DP), lv=5×SM')

    # ---- Tier assignment (§3) ----
    print(f'\n{"=" * 78}\n按 PREREG_b4long §3 落档(只对表,不定案)\n{"=" * 78}')
    worst = min(fp.values())
    if n_unconv >= 2:
        t = '未收敛 ≥2 折 → 只登记不落档'
    elif not anchor_ok:
        t = 'V-B4L 锚点不过 → 查仪器'
    elif pt_main >= 0.65 and worst >= 0.60:
        t = ('接口假说完整成立档(≥0.65 且最差折 ≥0.60)'
             + (',满额恢复(0.688±0.02)' if abs(pt_main - REF['probe_l36']) <= 0.02 else ''))
    elif pt_main >= 0.55:
        t = ('残差为真档(0.55–0.65 且已收敛):低秩 r16 / token-CE≠逻辑回归 / '
             'modus 不变性——探针保有水平优势(不只稳定优势)')
    else:
        t = '先查仪器档(<0.55)'
    print(f'  B4L 主读数={pt_main:.4f}  最差折={worst:.4f}  未收敛折={n_unconv}  → {t}')
    print(f'  点预测对表:末层 L36 探针 {REF["probe_l36"]}(差 {pt_main - REF["probe_l36"]:+.4f})')
    print('  最终判定留给人。')


if __name__ == '__main__':
    main()
