#!/usr/bin/env python
"""B3L (all-linear LoRA with relaxed budget) summary. Pure CPU. Criteria frozen in
`PREREG_b3long.md`; this only checks against the table, it does not decide.

Statistical parts are imported from `summarize_b3`/`summarize_b4` (bal-acc scenario-stratified
bootstrap B=10000 seed=20260803; BREU goes through the summarize_probe_decode single source),
so the scope matches the registered B3/B4/B4L numbers. Three things specific to B3L:

1. **Convergence criterion** (frozen in §1, verbatim from B4L): early stop triggered, or
   d[-1]−d[-4] < 0.005; if ≥2 folds run the full 20 epochs without converging → record only,
   no tier assignment.
2. **Collapse-shape triple** (§1; B4L doesn't have this): (i) whether the start is an all-REQ
   collapse (REQ prediction share at the 1st dev point ≥0.95); (ii) whether it climbs out
   (final bal-acc ≥0.55); (iii) per-fold range (>0.15 counts as "instability still present"
   in the §2 table). The B3 side is recomputed with the same scope — **(ii)(iii) come from
   B3's existing jsonl; (i) can only come from the `[epoch 0]` line of `logs/B3_fold{k}.log`**
   (B3's jsonl has no dev-curve field; the "recompute from existing jsonl" in PREREG §4 is
   infeasible for (i) — recorded as-is, not silently reconciled).
3. **Resampling unit of the paired difference = atomic cluster** (frozen in §2: atomic-cluster
   bootstrap 5000, seed 20260808). This is **not the same estimator** as the one behind the
   registered B3/B4/B4L paired differences (scenario-stratified bootstrap B=10000
   seed=20260803); both are printed — the one specified in §2 is the main table, the other is
   a comparability cross-check. Recorded as-is, not silently reconciled.

Usage
  python scripts/summarize_b3long.py > outputs/ci/b3long.txt
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from summarize_b3 import (C_ANCHOR, C_TOL, PROBE_ANCHOR, PROBE_TOL,  # noqa: E402
                          STEP1_SHA_REG, TEMPLATE_SHA_REG, balacc_point,
                          paired_balacc, pick, show_balacc, to_groups)
from summarize_b4 import fold_ponens_balacc, load_arm  # noqa: E402
from summarize_probe_decode import (GEN, PD, delta, line, recs_of,  # noqa: E402
                                    restrict, try_strata)
from summarize_length_matched import build_subsets, cem_match  # noqa: E402
from run_probe_decode import main_folds, fold_of_atomic  # noqa: E402

LOGS = os.path.join(HERE, '..', 'logs')
REF = dict(b3=0.6407, b4l=0.6231, b4=0.6306, probe_oof=0.692, c_self=0.519,
           a_breu=0.656, c_breu=0.487, greedy=0.492)
CONV_TAIL_EPS, MAX_EP = 0.005, 20                 # §1 convergence criterion
COLLAPSE_REQ, CLIMB_OUT, RANGE_CAP = 0.95, 0.55, 0.15   # §1 collapse-shape triple thresholds
ANCHOR_FLOOR = REF['b3'] - 0.05                   # V-B3L: more budget should not be worse
CL_ITERS, CL_SEED = 5000, 20260808                # §2 frozen paired-difference resampling


# ------------------------------------------------------------------ §2 paired difference: atomic-cluster bootstrap
def _balacc(rows, key):
    req = [r[key] for r in rows if r['gold'] == 'REQ']
    alt = [r[key] for r in rows if r['gold'] == 'ALT']
    if not req or not alt:
        return None
    return (float(np.mean(req)) + float(np.mean(alt))) / 2


def paired_balacc_atomic(rows, label, iters=CL_ITERS, seed=CL_SEED):
    """Δ = y − x, paired per scenario, **resampling unit = atomic cluster** (frozen in PREREG_b3long §2).

    rows: [{atomic, gold, ok_x, ok_y}], each row carries both arms' correctness on this
    scenario — the pairing lives here; the bootstrap only supplies the sampling
    distribution of Δ.
    """
    by_a = {}
    for r in rows:
        by_a.setdefault(r['atomic'], []).append(r)
    keys = list(by_a)
    dx, dy = _balacc(rows, 'ok_x'), _balacc(rows, 'ok_y')
    rng = np.random.default_rng(seed)
    vals, skipped = [], 0
    for _ in range(iters):
        pickk = rng.integers(0, len(keys), len(keys))
        flat = [r for j in pickk for r in by_a[keys[j]]]
        bx, by = _balacc(flat, 'ok_x'), _balacc(flat, 'ok_y')
        if bx is None or by is None:
            skipped += 1
            continue
        vals.append(by - bx)
    lo, hi = float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
    verdict = '不可区分' if lo <= 0 <= hi else ('高于' if lo > 0 else '低于')
    tail = f'  (簇 {len(keys)}，弃 {skipped} 次重采样)' if skipped else f'  (簇 {len(keys)})'
    print(f'  {label:<40} Δbal-acc={dy - dx:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}]'
          f'  → {verdict}{tail}')
    return dy - dx, lo, hi


def paired_rows(main_recs, other_by_idx):
    """Join the main arm's lv5-ponens rows with the control arm's same-idx predictions into paired rows."""
    out = []
    for r in main_recs:
        o = other_by_idx[r['idx']]
        g = 'REQ' if r['ground_truth'] == 'c' else 'ALT'
        out.append(dict(atomic=int(r['atomic_idx']), gold=g,
                        ok_x=(o['pred_state'] == g), ok_y=(r['pred_state'] == g)))
    return out


# ------------------------------------------------------------------ collapse shape
EP0_RE = re.compile(r'\[epoch 0\][^\n]*?dev n=(\d+)[^\n]*?预测 REQ=(\d+)/ALT=(\d+)')


def b3_ep0_req_frac(fold):
    """B3's REQ prediction share at the 1st dev point — only obtainable from B3's run logs (see module docstring)."""
    p = os.path.join(LOGS, f'B3_fold{fold}.log')
    if not os.path.exists(p):
        return None
    m = EP0_RE.search(open(p, errors='replace').read())
    if not m:
        return None
    n, n_req = int(m.group(1)), int(m.group(2))
    return n_req / n


def collapse_row(tag, fold, ep0_frac, final_balacc):
    st = ('起步全 REQ 塌缩' if ep0_frac is not None and ep0_frac >= COLLAPSE_REQ
          else ('未塌缩起步' if ep0_frac is not None else '起步占比不可得'))
    climb = final_balacc >= CLIMB_OUT
    print(f'  {tag} fold{fold}: 第1个dev点 REQ 占比='
          f'{"n/a" if ep0_frac is None else f"{ep0_frac:.4f}"} → {st}；'
          f'该折主读数={final_balacc:.4f} → {"爬出" if climb else "没爬出"}'
          f'(阈 ≥{CLIMB_OUT})')
    return (ep0_frac is not None and ep0_frac >= COLLAPSE_REQ), climb


# ------------------------------------------------------------------ main flow
def main():
    print('# B3L 汇总(PREREG_b3long.md,2026-08-10 冻结;只对表,不定案,不写解读)')
    print(f'# 参照线:B3 {REF["b3"]} / B4L {REF["b4l"]} / B4 {REF["b4"]} / '
          f'A 臂 OOF 探针 {REF["probe_oof"]} / C 自判 {REF["c_self"]} / 随机 0.500')
    print(f'# 配对差(§2 冻结)= 逐场景配对 + atomic 簇 bootstrap {CL_ITERS} 次 '
          f'seed {CL_SEED};另附 B3/B4L 已登记数用的场景分层 bootstrap 作可比性核对')

    parts, rs = load_arm('B3L')
    rs = [r for r in rs if r['arm'] == 'B3L']
    b3_parts, b3 = load_arm('B3')
    b4l_parts, b4l = load_arm('B4L')
    b4l = [r for r in b4l if r['arm'] == 'B4L']
    a_recs = recs_of(os.path.join(PD, 'A_Qwen_Qwen3_4B.jsonl'))
    c_recs = recs_of(os.path.join(PD, 'C_twostep.jsonl'))
    assert b3 and a_recs and c_recs, '缺 B3/A/C 产物,锚点与配对无从谈起'

    # ---- Anchors (identity cells V-S1/V-S2, B3's existing assertions copied verbatim) ----
    print(f'\n{"=" * 78}\n锚点(V-S1/V-S2):本脚本的 bal-acc 机器复算已登记值\n{"=" * 78}')
    pa, _ = show_balacc('V-S1 探针 OOF(A 臂 pred_state,lv5-pon)',
                        pick(a_recs, lv=5, modus='ponens'), ci=False)
    assert abs(pa - PROBE_ANCHOR) <= PROBE_TOL, \
        f'V-S1 失败:{pa:.4f} != {PROBE_ANCHOR} —— bal-acc 机器或 A 产物有问题,不读 B3L'
    pc, _ = show_balacc('V-S2 臂 C 自判(lv=5 全行)', pick(c_recs, lv=5), ci=False)
    assert abs(pc - C_ANCHOR) <= C_TOL, f'V-S2 失败:{pc:.4f} != {C_ANCHOR},不读 B3L'
    pb3, _ = show_balacc('V-S3 B3 已登记主读数(lv5-pon,应 = 0.6407)',
                         pick(b3, lv=5, modus='ponens'), ci=False)
    assert abs(pb3 - REF['b3']) <= 5e-4, f'V-S3 失败:B3 复算 {pb3:.4f} != {REF["b3"]}'
    print('  三锚点复现 ✅')

    if not rs:
        print('\n[无产物] B3L_fold*.jsonl 不存在——只跑了锚点自检。')
        return

    # ---- Gates (B3's existing assertions copied verbatim) ----
    print(f'\n{"=" * 78}\n臂 B3L(折文件 {len(parts)} 个,n行={len(rs)})\n{"=" * 78}')
    assert all(r['step1_sha'] == STEP1_SHA_REG and r['template_sha'] == TEMPLATE_SHA_REG
               for r in rs), 'G-B3-0 失败:sha 字段与登记值不符'
    assert not any(r['smoke'] for r in rs), '混入 smoke 记录'
    assert len(rs) == 782 and len({r['idx'] for r in rs}) == 782, \
        f'行数 {len(rs)} != 782 或 idx 重复(5 折未跑齐/跑串)'
    folds, groups, _ = main_folds()
    f_of = fold_of_atomic(folds, groups)
    assert all(r['fold'] == f_of[r['atomic_idx']] for r in rs), \
        'G-B3-1 失败:某行 fold != 折结构重算值 —— OOF 纪律破了'
    print('[G-B3-0] sha 一致 ✅   [G-B3-1] 782 行 OOF 纪律逐行断言 ✅')
    n = len(rs)
    n_fb = sum(r['fallback'] for r in rs)
    parse_ok = (n - n_fb) / n
    trc1 = float(np.mean([r['step1_truncated'] for r in rs]))
    print(f'[G-B3-2] step1 解析成功率={parse_ok:.4f} '
          f'{"✅" if parse_ok >= 0.8 else "❌<0.8 仪器坏,主读数无效"}  '
          f'截断率={trc1:.4f} {"✅" if trc1 <= 0.05 else "❌>0.05 仪器坏"}')
    n_req = sum(1 for r in rs if r['pred_state'] == 'REQ')
    n_alt = sum(1 for r in rs if r['pred_state'] == 'ALT')
    top = max(n_req, n_alt) / max(n_req + n_alt, 1)
    print(f'[G-B3-3] 预测边缘分布 REQ {n_req} / ALT {n_alt} / 失败 {n_fb}'
          f'{"  ⚠️ 塌缩形态(某类 ≥0.9)" if top >= 0.9 else "  ✅ 未塌缩"}')

    # ---- Convergence criterion (frozen in §1) ----
    print(f'\n  ---- 收敛判定(早停触发 或 d[-1]−d[-4] < {CONV_TAIL_EPS};'
          f'≥2 折未收敛 → 只登记不落档) ----')
    meta = {r['fold']: r for r in rs}
    n_unconv = 0
    for f in sorted(meta):
        m = meta[f]
        dc = m['dev_curve']
        tail = dc[-1] - dc[-4] if len(dc) >= 4 else float('nan')
        conv = m['early_stopped'] or (tail < CONV_TAIL_EPS)
        n_unconv += (not conv)
        print(f'  fold{f}: epochs={m["n_epochs_run"]}/{m["max_epochs"]}  '
              f'early_stop={m["early_stopped"]}  best_ep={m["best_epoch"]}  '
              f'best_dev={m["dev_balacc"]:.4f}  tail={tail:+.4f}  '
              f'→ {"已收敛" if conv else "未收敛"}')
        print(f'         dev 曲线 = {[round(x, 4) for x in dc]}')
        print(f'         dev REQ 占比 = {[round(x, 4) for x in m["dev_req_frac_curve"]]}')
    print(f'  未收敛折数 = {n_unconv}' + (' ≥2 → 按 §1 只登记不落档 ⚠️'
                                          if n_unconv >= 2 else ' < 2 → 可落档 ✅'))

    # ---- Primary readout ----
    print(f'\n  ---- 主读数:step1 关系判断折外 bal-acc(lv5-ponens,同 B3 一把尺) ----')
    b3l_5p = pick(rs, lv=5, modus='ponens')
    pt_main, g_main = show_balacc('B3L lv5-ponens(主口径)', b3l_5p)
    show_balacc('B3L lv=5 全 782 行(附报)', pick(rs, lv=5))
    show_balacc('B3L lv5-tollens(附报,题面 γ2 不同)', pick(rs, lv=5, modus='tollens'))
    anchor_ok = pt_main >= ANCHOR_FLOOR
    print(f'  [V-B3L] pooled {pt_main:.4f} ≥ B3 {REF["b3"]} − 0.05 = {ANCHOR_FLOOR:.4f}'
          f'  {"✅" if anchor_ok else "❌ 更多预算反而更差 —— 查仪器"}')

    # ---- Per-fold + collapse-shape triple (§1) ----
    fp, fp3 = fold_ponens_balacc(rs), fold_ponens_balacc(b3)
    rg, rg3 = (max(fp.values()) - min(fp.values())), (max(fp3.values()) - min(fp3.values()))
    print(f'\n  ---- 逐折(lv5-ponens,同一估计量) ----')
    print('  B3L 各折: ' + '  '.join(f'f{k}={v:.4f}' for k, v in fp.items())
          + f'   最差 {min(fp.values()):.4f}  极差 {rg:.4f}')
    print('  B3  各折: ' + '  '.join(f'f{k}={v:.4f}' for k, v in fp3.items())
          + f'   最差 {min(fp3.values()):.4f}  极差 {rg3:.4f}')
    if b4l:
        fp4 = fold_ponens_balacc(b4l)
        print('  B4L 各折: ' + '  '.join(f'f{k}={v:.4f}' for k, v in fp4.items())
              + f'   最差 {min(fp4.values()):.4f}  '
                f'极差 {max(fp4.values()) - min(fp4.values()):.4f}')

    print(f'\n  ---- 塌缩形态三元组(§1;(i) 阈 ≥{COLLAPSE_REQ} / (ii) 阈 ≥{CLIMB_OUT} '
          f'/ (iii) 阈 >{RANGE_CAP}) ----')
    print('  B3L(起步占比取自落盘的 dev_req_frac_curve[0]):')
    n_collapse = n_climb = 0
    for f in sorted(fp):
        c, cl = collapse_row('B3L', f, meta[f]['dev_req_frac_curve'][0], fp[f])
        n_collapse += c
        n_climb += cl
    print('  B3(同口径复算;起步占比取自 logs/B3_fold{k}.log 的 [epoch 0] 行 —— '
          'B3 的 jsonl 无此字段,见模块 docstring 的登记):')
    n_collapse3 = n_climb3 = 0
    for f in sorted(fp3):
        c, cl = collapse_row('B3 ', f, b3_ep0_req_frac(f), fp3[f])
        n_collapse3 += c
        n_climb3 += cl
    print(f'  三元组小结  B3L:(i) 起步塌缩 {n_collapse}/5 折  '
          f'(ii) 爬出 {n_climb}/5 折  (iii) 极差 {rg:.4f} '
          f'{">" if rg > RANGE_CAP else "≤"} {RANGE_CAP}')
    print(f'              B3 :(i) 起步塌缩 {n_collapse3}/5 折  '
          f'(ii) 爬出 {n_climb3}/5 折  (iii) 极差 {rg3:.4f} '
          f'{">" if rg3 > RANGE_CAP else "≤"} {RANGE_CAP}')

    # ---- Paired comparisons ----
    idx_b3 = {r['idx']: r for r in b3}
    idx_a = {r['idx']: r for r in a_recs}
    idx_c = {r['idx']: r for r in c_recs}
    print(f'\n  ---- 配对比较(lv5-ponens 同 {len(b3l_5p)} 行)｜主表 = §2 冻结的'
          f' atomic 簇 bootstrap {CL_ITERS} 次 seed {CL_SEED} ----')
    paired_balacc_atomic(paired_rows(b3l_5p, idx_b3), 'B3L − B3(预算的净效应)')
    if b4l:
        paired_balacc_atomic(paired_rows(b3l_5p, {r['idx']: r for r in b4l}),
                             'B3L − B4L(深层 vs 末层,同预算)')
    paired_balacc_atomic(paired_rows(b3l_5p, idx_a), 'B3L − 探针(A 臂 OOF)')
    paired_balacc_atomic(paired_rows(b3l_5p, idx_c), 'B3L − C 自判(附报)')

    print(f'\n  ---- 同三对的可比性核对:B3/B4/B4L 已登记数用的估计量'
          f'(场景分层 bootstrap B=10000 seed=20260803) ----')
    mk = lambda src: to_groups([dict(dataset_id=r['dataset_id'],
                                     ground_truth=r['ground_truth'],
                                     pred_state=src[r['idx']]['pred_state'])
                                for r in b3l_5p])
    paired_balacc(mk(idx_b3), g_main, 'B3L − B3(预算的净效应)')
    if b4l:
        paired_balacc(mk({r['idx']: r for r in b4l}), g_main, 'B3L − B4L')
    paired_balacc(mk(idx_a), g_main, 'B3L − 探针(A 臂 OOF)')
    paired_balacc(mk(idx_c), g_main, 'B3L − C 自判(附报)')

    # ---- Secondary readout: end-to-end BREU (B3 scope, reusing the existing single source) ----
    merged = os.path.join(PD, 'B3L_merged.jsonl')
    with open(merged, 'w') as f:
        for r in sorted(rs, key=lambda x: x['idx']):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    sm = set(cem_match(build_subsets()))
    fol = [r for r in rs if not r['fallback']]
    print(f'\n  ---- 次读数:端到端 BREU(lv=5 折外,照 B3 口径;附报不判定) ----')
    print(f'  执行健康:跟随率(非回退)={np.mean([r["follows"] for r in fol]):.4f}  '
          f'格式兑现率={np.mean([r["has_final_answer"] for r in rs]):.4f}  '
          f'step2 截断率={np.mean([r["truncated"] for r in rs]):.4f}')
    st5 = try_strata(merged, 5, 'B3L/lv5')
    b5 = line('B3L', st5)
    st5_sm = restrict(st5, sm)
    if len(st5_sm[0]) and len(st5_sm[1]):
        print('  ---- lv=5 × SM 长度配平分层 ----')
        line('B3L / SM', st5_sm)
    gd5 = try_strata(os.path.join(GEN, 'time_t1_Qwen_Qwen3_4B_dp.jsonl'), 5, 'dp/lv5')
    if gd5 is not None:
        print('  ---- 配对检验(BREU)----')
        delta(gd5, st5, 'B3L − greedy(DP), lv=5')
        if len(st5_sm[0]) and len(st5_sm[1]):
            delta(restrict(gd5, sm), st5_sm, 'B3L − greedy(DP), lv=5×SM')

    # ---- Mechanical table match (PREREG_b3long §2 table) ----
    print(f'\n{"=" * 78}\n按 PREREG_b3long §2 机械对号(只对号,不解读,不定案)\n{"=" * 78}')
    worst = min(fp.values())
    shape_alive = (n_climb < 5) or (rg > RANGE_CAP)
    if n_unconv >= 2:
        t = '未收敛(≥2 折)→ 只登记"预算仍不足",不落档'
    elif not anchor_ok:
        t = f'V-B3L 锚点不过(pooled < {ANCHOR_FLOOR:.4f})→ 先查仪器'
    elif pt_main < 0.55:
        t = 'pooled <0.55 → 先查仪器'
    elif pt_main >= 0.70 and worst >= 0.65:
        t = ('"深层预算不足"成立档(pooled ≥0.70 且最差折 ≥0.65):B3 已登记档必须改写;'
             '与探针 0.692 的配对差另报,决定"微调追平探针"是否成立')
    elif pt_main < 0.70 and shape_alive:
        t = ('"优化困难为真"的直接证据档(pooled 0.55–0.70,已收敛,塌缩形态仍在):'
             '预算解决不了深层不稳定;B3 档从间接证据升级')
    elif pt_main < 0.70:
        t = ('拆开写档(pooled 0.55–0.70,已收敛,五折全爬出且极差 ≤0.15):'
             '不稳定被预算解决(这半句改写 B3 档),水平残差为真(与探针配对差报告)')
    else:
        t = (f'pooled {pt_main:.4f} ≥0.70 但最差折 {worst:.4f} <0.65 —— '
             f'§2 表没有这一格,不对号,交人裁')
    print(f'  pooled={pt_main:.4f}  最差折={worst:.4f}  极差={rg:.4f}  '
          f'未收敛折={n_unconv}  爬出={n_climb}/5  起步塌缩={n_collapse}/5')
    print(f'  → {t}')
    print(f'  V-B3L 锚点:{pt_main:.4f} vs 下限 {ANCHOR_FLOOR:.4f} '
          f'→ {"过" if anchor_ok else "不过"}')
    print(f'  次读数 BREU={b5:.4f}(对表 A {REF["a_breu"]} / C {REF["c_breu"]} / '
          f'greedy {REF["greedy"]});附报不判定,不改档位')
    print('  最终判定留给人。')


if __name__ == '__main__':
    main()
