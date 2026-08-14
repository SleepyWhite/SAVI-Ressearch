#!/usr/bin/env python
"""lv=4 dose-response summary (PREREG_lv4_dose.md, incl. revision 1: main combination rule = OOF assignment). Pure CPU.

Dose = representation signal (probe bal-acc, lv5 0.692 → lv4 0.604, registered
in L7 §6.16); response = each arm's Δ(arm − greedy) BREU, paired within the
lv=5 and lv=4 strata (scenario-clustered bootstrap, existing single source,
5000 iterations seed=0). Three arms: A probe+inject / C prompt self-judgement
/ B4L last-layer head. lv=4 ceiling = intent scorer computed on lv=4-only
(same method as compute_ceiling).

Usage
  python scripts/summarize_lv4_dose.py > outputs/ci/lv4_dose.txt
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from summarize_b3 import balacc_point, pick, show_balacc, to_groups  # noqa: E402
from summarize_probe_decode import (GEN, PD, delta, line, recs_of,  # noqa: E402
                                    restrict, try_strata)
from run_b4long_lora import load_lv4_step1, majority_vote  # noqa: E402
from run_probe_decode import CSV  # noqa: E402
from compute_ceiling import balanced_acc, boot_ci  # noqa: E402

SIG = dict(lv5=0.692, lv4=0.604)   # dose side: probe representation signal (L7 §6.15/§6.16 registered values)
DP = os.path.join(GEN, 'time_t1_Qwen_Qwen3_4B_dp.jsonl')


def ceiling_lv4():
    """Intent-scorer bal-acc on lv=4-only scenarios (same method as compute_ceiling, only the filter changes)."""
    df = pd.read_csv(CSV)
    sub = df[df.agreement_lv == 4].copy()
    sub['is_req'] = sub['ground_truth'] == 'c'
    sub['intent'] = sub['dataset_id'].str.split('-').str[-1]
    chk = sub.groupby('dataset_id').agg(ni=('intent', 'nunique'), ng=('is_req', 'nunique'))
    assert not len(chk[(chk.ni > 1) | (chk.ng > 1)]), 'lv=4 孪生 intent/关系不一致'
    s = sub.drop_duplicates('dataset_id')
    pred = (s['intent'] == 'strong').to_numpy()
    gold = s['is_req'].to_numpy()
    ba = balanced_acc(pred, gold)
    lo, hi = boot_ci(pred, gold)
    return ba, lo, hi, len(s), int(gold.sum())


def arm_stratum(path, lv, tag, gd, sm_keep=None):
    """One arm, one stratum: BREU line + paired Δ vs greedy. Returns (breu, (d,lo,hi,verdict))."""
    st = try_strata(path, lv, tag)
    if st is None:
        print(f'  [{tag}] 分层不全,跳过')
        return None, None
    if sm_keep is not None:
        st = restrict(st, sm_keep)
        gd = restrict(gd, sm_keep)
    b = line(tag, st)
    d = delta(gd, st, f'{tag} − greedy')
    return b, d


def rel_balacc_of(recs, lv):
    """Relation bal-acc for one arm and stratum; missing/None pred_state (fallback rows) counts as wrong."""
    rows = [dict(dataset_id=r['dataset_id'], ground_truth=r['ground_truth'],
                 pred_state=r.get('pred_state'))
            for r in recs if r['agreement_lv'] == lv]
    return balacc_point(to_groups(rows))[0], len(rows)


def main():
    print('# lv=4 剂量-响应汇总(PREREG_lv4_dose.md,修订 1=OOF 指派主规则)')
    print(f'# 剂量端(登记值):探针表征信号 lv5 {SIG["lv5"]} → lv4 {SIG["lv4"]}')

    ba, lo, hi, n_sc, n_req = ceiling_lv4()
    print(f'# lv=4-only 意图天花板(实算)= {ba:.4f} [{lo:.4f},{hi:.4f}]'
          f'(场景 {n_sc},REQ {n_req};对表 lv5 0.843 / 全量混合 0.777)')

    a_path = os.path.join(PD, 'A_Qwen_Qwen3_4B.jsonl')
    c_path = os.path.join(PD, 'C_twostep.jsonl')
    b4l_path = os.path.join(PD, 'B4L_lv4_exec.jsonl')
    b4l5_path = os.path.join(PD, 'B4L_merged.jsonl')
    exec_recs = recs_of(b4l_path)
    assert len(exec_recs) == 958 and not any(r['smoke'] for r in exec_recs)
    n_fb = sum(r['fallback'] for r in exec_recs)
    print(f'# B4L lv4 主规则:assigned {sum(1 for r in exec_recs if r["rule"] == "assigned")}'
          f' / majority {sum(1 for r in exec_recs if r["rule"] == "majority")};'
          f'回退率 {n_fb / 958:.4f}(闸门 ≤0.2 {"✅" if n_fb / 958 <= 0.2 else "❌"})')

    # ---- relation bal-acc strata (dose-side reference) ----
    print(f'\n{"=" * 78}\n关系判断 bal-acc 分层(次读数)\n{"=" * 78}')
    for name, recs, lv in (('A 探针 pred_state lv5', recs_of(a_path), 5),
                           ('A 探针 pred_state lv4', recs_of(a_path), 4),
                           ('C 自判 lv5', recs_of(c_path), 5),
                           ('C 自判 lv4', recs_of(c_path), 4)):
        v, n = rel_balacc_of(recs, lv)
        print(f'  {name:<28} bal-acc = {v:.4f}  (n行={n})')
    v, n = rel_balacc_of(exec_recs, 4)
    print(f'  {"B4L 主规则(指派) lv4":<28} bal-acc = {v:.4f}  (n行={n};'
          f'lv5 对表=B4L 汇总 0.63 档)')
    clean = [r for r in exec_recs if r['rule'] == 'majority']
    vc = balacc_point(to_groups([dict(dataset_id=r['dataset_id'],
                                      ground_truth=r['ground_truth'],
                                      pred_state=r['pred_state']) for r in clean]))[0]
    print(f'  {"B4L lv4 × atomic 干净 212 行":<28} bal-acc = {vc:.4f}')

    # additionally reported: full 958-row five-fold majority-vote version (pure CPU, recomputed from step1 artifacts; no execution run)
    idxs, per = load_lv4_step1(PD, '')
    mv_rows, n_diff = [], 0
    ex_by = {r['idx']: r for r in exec_recs}
    for i in idxs:
        rel, vs, n_ab, n1, n2 = majority_vote([per[k][i]['relation'] for k in range(5)])
        ps = None if rel is None else ('REQ' if rel == '1' else 'ALT')
        r0 = per[0][i]
        mv_rows.append(dict(dataset_id=r0['dataset_id'], ground_truth=r0['ground_truth'],
                            pred_state=ps))
        n_diff += (ps != ex_by[i]['pred_state'])
    vm = balacc_point(to_groups(mv_rows))[0]
    print(f'  {"附报:全 958 行多数票版":<28} bal-acc = {vm:.4f}'
          f'(与主规则分岔 {n_diff} 行——家族暴露模型混入投票的影响量)')

    # ---- response side: Δ(arm − greedy) BREU, two strata ----
    print(f'\n{"=" * 78}\n响应端:端到端 BREU 与 Δ(臂 − greedy),分层配对\n{"=" * 78}')
    res = {}
    for lv in (5, 4):
        gd = try_strata(DP, lv, f'greedy/lv{lv}')
        print(f'\n  ---- lv={lv}(greedy 参照)----')
        line(f'greedy lv{lv}', gd)
        for name, path in (('A 探针+注入', a_path), ('C 提示自判', c_path),
                           ('B4L 末层头', b4l5_path if lv == 5 else b4l_path)):
            b, d = arm_stratum(path, lv, f'{name}/lv{lv}', gd)
            res[(name, lv)] = (b, d)
        if lv == 4:
            keep = {r['dataset_id'] for r in clean}
            print('  ---- lv=4 × atomic 干净 212 行分层 ----')
            for name, path in (('A/干净212', a_path), ('B4L/干净212', b4l_path)):
                arm_stratum(path, 4, name, try_strata(DP, 4, 'gd'), sm_keep=keep)

    # ---- dose-response table and reading (PREREG §4, no final call) ----
    print(f'\n{"=" * 78}\n剂量-响应表(按 PREREG_lv4_dose §4 判读,不定案)\n{"=" * 78}')
    print(f'  剂量:表征信号 0.692 → 0.604(缩 {SIG["lv4"] - SIG["lv5"]:+.3f});'
          f'天花板 0.843 → {ba:.3f}')
    for name in ('A 探针+注入', 'C 提示自判', 'B4L 末层头'):
        d5, d4 = res[(name, 5)][1], res[(name, 4)][1]
        if d5 is None or d4 is None:
            continue
        shrink = d4[0] - d5[0]
        print(f'  {name:<14} Δlv5={d5[0]:+.4f}[{d5[1]:+.3f},{d5[2]:+.3f}]  '
              f'Δlv4={d4[0]:+.4f}[{d4[1]:+.3f},{d4[2]:+.3f}]  增益变化={shrink:+.4f}'
              f'  → {"缩小/归零方向" if shrink < 0 else "未缩小"}')
    print('  判读表:各臂 lv4 增益 < lv5 且方向仍正(或归零)→ H-dose 成立;'
          '某臂不缩且 CI 可区分 → 对该臂证伪;全在噪声里 → 不确定。')
    print('  C 臂为阴性对照:两层增益都应 ≈0,否则先查仪器。最终判定留给人。')


if __name__ == '__main__':
    main()
