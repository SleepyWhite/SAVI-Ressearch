#!/usr/bin/env python
"""Three-arm probe-decode summary: BREU / strata / paired tests / gates / §6 tier printing. Pure CPU.

Criteria and gates are frozen in `PREREG_probe_decode.md` (2026-08-03). This script only
prints "falls into tier X by those criteria" — it does **not** make the final call.

============================================================================
Conventions (all reuse existing single sources; this script invents no new statistics)
============================================================================
- Scenario-clustered bootstrap CI and paired test = `summarize_generative_ci`'s
  `load_strata` / `stats` / `paired` (statistical unit = dataset_id, resampling within
  BU/BM strata, 5000 iterations seed=0). It carries its own V0–V3 self-checks (gold
  reconciliation / two rows per scenario / twins never straddle BU-BM / twins share lv);
  any arm's jsonl failing these raises immediately.
- SM (length-matched) subset = `summarize_length_matched.build_subsets/cem_match`,
  the same scenario table as L7 gate G-5.
- Probe sample-efficiency curve = `summarize_hidden_probe`'s `nested_cv/bal_acc/load_reps`;
  folds are the same set as `run_probe_decode.main_folds`.

Main convention = lv=5; SM is a stratum; lv=4 is flagged **exploratory** (the probe was
never trained on lv=4, and lv=4's gold agreement is low by dataset design). lv=6 has only
2 scenarios; recorded, not reported.

Reference lines (PREREG §5): greedy 0.497 / oracle injection 0.994 (4B BU-pon) /
intent ceiling 0.843 (SM 0.764) / probe out-of-fold bal-acc 0.692 (4B), 0.716 (7B).

Usage
  python scripts/summarize_probe_decode.py > outputs/ci/probe_decode.txt
  python scripts/summarize_probe_decode.py --sample_curve
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_generative_ci import ITERS, SEED, load_strata, paired, stats  # noqa: E402
from summarize_length_matched import build_subsets, cem_match  # noqa: E402
from summarize_hidden_probe import (CGRID, N_INNER, N_OUTER, SEED as PSEED,  # noqa: E402
                                    bal_acc, load_reps, nested_cv, _fit)
from summarize_hidden_probe_lvsplit import ANCHOR_MAIN, TOL_MAIN  # noqa: E402
from run_probe_decode import main_folds  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

GEN = os.path.join(HERE, '..', 'outputs', 'generative')
PD = os.path.join(HERE, '..', 'outputs', 'probe_decode')
GREEDY_REF, ORACLE_REF, CEIL_LV5, CEIL_SM = 0.497, 0.994, 0.843, 0.764
CURVE_LEVELS = (391, 200, 100, 50)
CURVE_REPS = 5


# ------------------------------------------------------------------ small helpers
def restrict(strata, keep):
    """Restrict load_strata's product to the dataset_ids in keep."""
    bu, bm, bi, mi = strata
    kb = [j for j, s in enumerate(bi) if s in keep]
    km = [j for j, s in enumerate(mi) if s in keep]
    return (bu[kb], bm[km], [bi[j] for j in kb], [mi[j] for j in km])


def try_strata(path, agreement, tag):
    if not os.path.exists(path):
        return None
    st = load_strata(path, agreement, tag)
    if not len(st[0]) or not len(st[1]):
        return None
    return st


def line(name, st):
    pt, ci = stats(st[0], st[1], ITERS, SEED)
    print(f"  {name:<34} n场景={len(st[0]) + len(st[1]):>4}  "
          f"BU={pt['BU']:.4f}[{ci['BU'][0]:.4f},{ci['BU'][1]:.4f}]  "
          f"BM={pt['BM']:.4f}[{ci['BM'][0]:.4f},{ci['BM'][1]:.4f}]  "
          f"BREU={pt['BREU']:.4f}[{ci['BREU'][0]:.4f},{ci['BREU'][1]:.4f}]")
    print(f"  {'':<34} 四格 " + '  '.join(
        f"{k}={pt[k]:.3f}[{ci[k][0]:.3f},{ci[k][1]:.3f}]"
        for k in ('BU-pon', 'BU-tol', 'BM-pon', 'BM-tol')))
    return pt['BREU']


def delta(a, b, label):
    """Paired ΔBREU of b − a; returns (d, lo, hi, verdict string)."""
    d, lo, hi, n = paired(a, b, ITERS, SEED)
    verdict = '不可区分' if lo <= 0 <= hi else ('高于' if lo > 0 else '低于')
    print(f'  {label:<34} ΔBREU={d:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}]  '
          f'n场景={n:>4}  → {verdict}')
    return d, lo, hi, verdict


def recs_of(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


# ------------------------------------------------------------------ arm A
def arm_a(sm, models):
    out = {}
    for path in sorted(glob.glob(os.path.join(PD, 'A_*.jsonl'))):
        safe = os.path.basename(path)[2:-len('.jsonl')]
        rs = recs_of(path)
        print(f'\n{"=" * 78}\n臂 A  {safe}  (n行={len(rs)})\n{"=" * 78}')
        print(f'  template_sha = {rs[0]["template_sha"] if rs else "?"}')
        greedy = os.path.join(GEN, f'time_t1_{safe}_dp.jsonl')

        # --- Gate (ii): condition-following rate / format-compliance rate / truncation rate (check budget before accuracy) ---
        for tag, sub in (('全 1,744 行', rs),
                         ('lv=5', [r for r in rs if r['agreement_lv'] == 5]),
                         ('lv=4', [r for r in rs if r['agreement_lv'] == 4])):
            if not sub:
                continue
            fol = np.mean([r['follows'] for r in sub])
            fmt = np.mean([r['has_final_answer'] for r in sub])
            trc = np.mean([r['truncated'] for r in sub])
            req = [r for r in sub if r['pred_state'] == 'REQ']
            alt = [r for r in sub if r['pred_state'] == 'ALT']
            print(f'  [{tag:<12}] 跟随率={fol:.4f} {"✅" if fol >= 0.8 else "❌<0.8"}  '
                  f'格式兑现率={fmt:.4f} {"✅" if fmt >= 0.8 else "❌<0.8"}  '
                  f'截断率={trc:.4f}  '
                  f'预测 REQ {len(req)}（跟随 '
                  f'{np.mean([r["follows"] for r in req]) if req else float("nan"):.3f}）/ '
                  f'ALT {len(alt)}（跟随 '
                  f'{np.mean([r["follows"] for r in alt]) if alt else float("nan"):.3f}）')
            g = [r for r in sub if r['ground_truth'] == 'c']
            h = [r for r in sub if r['ground_truth'] != 'c']
            if g and h:
                sb = (np.mean([r['pred_state'] == 'REQ' for r in g])
                      + np.mean([r['pred_state'] == 'ALT' for r in h])) / 2
                print(f'  {"":<14} 注入态对金标的平衡准确率 = {sb:.4f}'
                      f'（对表：探针折外登记值）')

        # --- Main convention lv=5 + SM stratum + lv=4 exploratory ---
        st5 = try_strata(path, 5, f'A/{safe}/lv5')
        if st5 is None:
            print('  [跳过] lv=5 分层不全')
            continue
        print('\n  ---- lv=5 主口径 ----')
        b5 = line('臂 A', st5)
        st5_sm = restrict(st5, sm)
        if len(st5_sm[0]) and len(st5_sm[1]):
            print('  ---- lv=5 × SM 长度配平分层 ----')
            line('臂 A / SM', st5_sm)
        st4 = try_strata(path, 4, f'A/{safe}/lv4')
        if st4 is not None:
            print('  ---- lv=4（探索性，探针未在此训练；不进主判据）----')
            line('臂 A / lv4', st4)

        gd5 = try_strata(greedy, 5, f'dp/{safe}/lv5')
        res = dict(breu=b5, st5=st5, st5_sm=st5_sm)
        if gd5 is not None:
            print('  ---- 与 greedy(DP) 的配对检验 ----')
            line('greedy DP', gd5)
            res['vs_greedy'] = delta(gd5, st5, 'A − greedy(DP), lv=5')
            gsm = restrict(gd5, sm)
            if len(gsm[0]) and len(gsm[1]):
                res['vs_greedy_sm'] = delta(gsm, st5_sm, 'A − greedy(DP), lv=5×SM')
        out[safe] = res
    return out


# ------------------------------------------------------------------ arm B
def arm_b(sm, a_res):
    out = {}
    for variant in ('B1', 'B2'):
        parts = sorted(glob.glob(os.path.join(PD, f'B_{variant}_fold*.jsonl')))
        parts = [p for p in parts if not p.endswith('_smoke.jsonl')]
        if not parts:
            continue
        rs = [r for p in parts for r in recs_of(p)]
        merged = os.path.join(PD, f'B_{variant}_merged.jsonl')
        with open(merged, 'w') as f:
            for r in sorted(rs, key=lambda x: x['idx']):
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'\n{"=" * 78}\n臂 B  {variant}  (折文件 {len(parts)} 个，n行={len(rs)})\n{"=" * 78}')
        print('  各折 dev BREU: ' + ', '.join(
            f"fold{p[0]}={p[1]:.4f}" for p in sorted(
                {(r['fold'], r['dev_breu']) for r in rs})))
        print(f'  格式兑现率={np.mean([r["has_final_answer"] for r in rs]):.4f}  '
              f'截断率={np.mean([r["truncated"] for r in rs]):.4f}')
        if len(rs) != 782:
            print(f'  ⚠️ 行数 {len(rs)} != 782（5 折未跑齐），以下读数不完整')

        st5 = try_strata(merged, 5, f'B/{variant}/lv5')
        if st5 is None:
            continue
        print('  ---- lv=5 主口径 ----')
        b5 = line(f'臂 B {variant}', st5)
        st5_sm = restrict(st5, sm)
        if len(st5_sm[0]) and len(st5_sm[1]):
            print('  ---- lv=5 × SM 分层（PREREG §3 闸门：全集涨而 SM 不涨 = 学到构造痕迹）----')
            line(f'臂 B {variant} / SM', st5_sm)
        safe = 'Qwen_Qwen3_4B'
        gd5 = try_strata(os.path.join(GEN, f'time_t1_{safe}_dp.jsonl'), 5, 'dp/lv5')
        res = dict(breu=b5, st5=st5, st5_sm=st5_sm)
        if gd5 is not None:
            print('  ---- 配对检验 ----')
            res['vs_greedy'] = delta(gd5, st5, f'{variant} − greedy(DP), lv=5')
            res['vs_greedy_sm'] = delta(restrict(gd5, sm), st5_sm,
                                        f'{variant} − greedy(DP), lv=5×SM')
        if safe in a_res:
            res['vs_a'] = delta(a_res[safe]['st5'], st5, f'{variant} − A, lv=5')
        out[variant] = res
    return out


# ------------------------------------------------------------------ arm C
def arm_c(sm, a_res):
    path = os.path.join(PD, 'C_twostep.jsonl')
    rs = recs_of(path)
    if not rs:
        return None
    print(f'\n{"=" * 78}\n臂 C  两步分类提示  (n行={len(rs)})\n'
          f'{"=" * 78}')
    print(f'  step1_sha = {rs[0]["step1_sha"]}   template_sha = {rs[0]["template_sha"]}')
    for tag, sub in (('全 1,744 行', rs), ('lv=5', [r for r in rs if r['agreement_lv'] == 5])):
        if not sub:
            continue
        fb = np.mean([r['fallback'] for r in sub])
        ok = [r for r in sub if not r['fallback']]
        fol = np.mean([r['follows'] for r in ok]) if ok else float('nan')
        g = [r for r in ok if r['ground_truth'] == 'c']
        h = [r for r in ok if r['ground_truth'] != 'c']
        sb = ((np.mean([r['pred_state'] == 'REQ' for r in g])
               + np.mean([r['pred_state'] == 'ALT' for r in h])) / 2) if g and h else float('nan')
        print(f'  [{tag:<12}] 回退率={fb:.4f}  step1截断率='
              f'{np.mean([r["step1_truncated"] for r in sub]):.4f}  '
              f'跟随率(非回退)={fol:.4f}  '
              f'格式兑现率={np.mean([r["has_final_answer"] for r in sub]):.4f}  '
              f'截断率={np.mean([r["truncated"] for r in sub]):.4f}')
        print(f'  {"":<14} step1 关系判断对金标的平衡准确率 = {sb:.4f}'
              f'（对表：L5 上下文内选择 0.545 / 探针 0.692 / 天花板 0.843）')
        print(f'  {"":<14} 预测 REQ {sum(1 for r in ok if r["pred_state"] == "REQ")} / '
              f'ALT {sum(1 for r in ok if r["pred_state"] == "ALT")}')

    st5 = try_strata(path, 5, 'C/lv5')
    if st5 is None:
        return None
    print('  ---- lv=5 主口径 ----')
    b5 = line('臂 C', st5)
    st5_sm = restrict(st5, sm)
    if len(st5_sm[0]) and len(st5_sm[1]):
        print('  ---- lv=5 × SM 分层 ----')
        line('臂 C / SM', st5_sm)
    st4 = try_strata(path, 4, 'C/lv4')
    if st4 is not None:
        print('  ---- lv=4（探索性）----')
        line('臂 C / lv4', st4)
    res = dict(breu=b5, st5=st5)
    gd5 = try_strata(os.path.join(GEN, 'time_t1_Qwen_Qwen3_4B_dp.jsonl'), 5, 'dp/lv5')
    if gd5 is not None:
        print('  ---- 配对检验 ----')
        res['vs_greedy'] = delta(gd5, st5, 'C − greedy(DP), lv=5')
    if 'Qwen_Qwen3_4B' in a_res:
        res['vs_a'] = delta(a_res['Qwen_Qwen3_4B']['st5'], st5, 'C − A, lv=5')
    return res


# ------------------------------------------------------------------ sample-efficiency curve
def sample_curve(models, n_jobs):
    """Out-of-fold bal-acc at training-scenario counts 391→200→100→50. Layer/C frozen to the
    full pipeline's per-fold selection — the curve is meant to isolate **sample size**, not
    selection noise; the N=391 tier must reproduce the anchor (identity cell)."""
    print(f'\n{"=" * 78}\n附送（纯 CPU）：探针样本效率曲线\n{"=" * 78}')
    print(f'# 每折训练集按 atomic_idx 组子抽样到目标场景数（组整体进出），'
          f'层/C 固定为全量管线该折的选择；{CURVE_REPS} 次重抽样报 均值[min,max]')
    for model_name in models:
        last, _, meta = load_reps(model_name, smoke=False)
        layers = list(range(last.shape[1]))
        folds, groups, positions = main_folds()
        main = meta.loc[positions]
        X = last[positions]
        y = (main['gold'] == 'c').astype(int).values
        G = main['atomic_idx'].values
        cv = nested_cv(X, y, G, layers, CGRID, N_OUTER, N_INNER, n_jobs)
        b = bal_acc(y, cv['oof_pred'])
        assert abs(b - ANCHOR_MAIN[model_name]) <= TOL_MAIN, \
            f'锚点失败：{b:.4f} vs {ANCHOR_MAIN[model_name]} —— 曲线无效'
        print(f'\n  {model_name}: 锚点 {b:.4f} ✅  '
              f'各折训练行数 {[len(tr) for tr, _ in cv["folds"]]}')
        for N in CURVE_LEVELS:
            vals, sizes = [], []
            reps = 1 if N >= len(y) else CURVE_REPS
            for rep in range(reps):
                rng = np.random.default_rng(PSEED + rep)
                oof = np.full(len(y), -1)
                tot = 0
                for k, (tr, te) in enumerate(cv['folds']):
                    L, C, _, _ = cv['models'][k]
                    gs = list(dict.fromkeys(G[tr].tolist()))
                    rng.shuffle(gs)
                    pick, cnt = set(), 0
                    for gg in gs:
                        m = int((G[tr] == gg).sum())
                        if cnt + m > N and pick:
                            continue
                        pick.add(gg); cnt += m
                        if cnt >= N:
                            break
                    sub = tr[np.isin(G[tr], list(pick))]
                    tot += len(sub)
                    Xtr = X[sub, L].astype(np.float32)
                    sc = StandardScaler().fit(Xtr)
                    clf = _fit(sc.transform(Xtr), y[sub], C)
                    oof[te] = clf.predict(sc.transform(X[te, L].astype(np.float32)))
                vals.append(bal_acc(y, oof))
                sizes.append(tot / len(cv['folds']))
            v = np.array(vals)
            note = '  ← 全量档，应复现锚点' if N >= len(y) else ''
            print(f'    训练场景目标 {N:>3}（实得均 {np.mean(sizes):.0f}/折）: '
                  f'折外 bal-acc = {v.mean():.4f} [min {v.min():.4f}, max {v.max():.4f}]'
                  f'（{reps} 次重抽样）{note}')
            if N >= len(y):
                assert abs(v[0] - b) <= 1e-12, '全量档未复现锚点 —— 子抽样逻辑写错'


# ------------------------------------------------------------------ tier assignment
def verdicts(a, b, c):
    print(f'\n{"=" * 78}\n按 PREREG §6 判据落档（本脚本只对表，不定案）\n{"=" * 78}')
    for safe, r in a.items():
        v = r.get('vs_greedy')
        sig = v is not None and v[1] > 0
        if r['breu'] >= 0.65:
            t = ('修复演示成立档（≥0.65 且配对显著高于 greedy）' if sig else
                 '判据未覆盖：≥0.65 但与 greedy 的配对不显著 → 交人裁决')
        elif r['breu'] >= 0.55:
            t = '部分成立档（0.55–0.65；按跟随率定位损耗在执行还是预测）'
        else:
            t = '组合失败档（<0.55；查注入干扰，负结果照写）'
        print(f'  臂 A / {safe}: BREU={r["breu"]:.4f} → {t}')
    for variant, r in b.items():
        d = r.get('vs_a')
        if d is None:
            t = '缺臂 A 对照，无法判 B vs A'
        elif abs(d[0]) >= 0.05 and not (d[1] <= 0 <= d[2]):
            t = f'B {"优于" if d[0] > 0 else "劣于"} A（|Δ|≥0.05 且 CI 不含 0）'
        else:
            t = 'B 与 A 不可区分（|Δ|<0.05 或 CI 含 0）'
        print(f'  臂 B / {variant}: BREU={r["breu"]:.4f} → {t}')
        print(f'    SM 闸门：全集与 SM 的对 greedy 增量 = '
              f'{r.get("vs_greedy", (float("nan"),))[0]:+.4f} / '
              f'{r.get("vs_greedy_sm", (float("nan"),))[0]:+.4f}'
              f'  —— 只在全集涨、SM 不涨 = 学到构造痕迹，判"未修复"')
    if c is not None:
        t = ('符合先验档（0.45–0.55）' if 0.45 <= c['breu'] <= 0.55 else
             '意外正结果，单独报（≥0.60，且它是免训练的）' if c['breu'] >= 0.60 else
             '判据未覆盖：落 0.55–0.60 或 <0.45 → 交人裁决')
        print(f'  臂 C: BREU={c["breu"]:.4f} → {t}')
    print('  最终判定留给人。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_curve', action='store_true')
    ap.add_argument('--models', default='Qwen/Qwen3-4B,Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--n_jobs', type=int, default=16)
    args = ap.parse_args()

    print('# probe-decode 三臂汇总（PREREG_probe_decode.md，2026-08-03 冻结）')
    print(f'# CI = 场景聚类分层 bootstrap {ITERS} 次 seed={SEED}（复用 '
          f'summarize_generative_ci 的 load_strata/stats/paired，含其 V0–V3 自查）')
    print(f'# 参照线：greedy {GREEDY_REF} / oracle 注入 {ORACLE_REF}（4B BU-pon）/ '
          f'天花板 {CEIL_LV5}（SM {CEIL_SM}）/ 探针折外 {ANCHOR_MAIN}')

    sm = set(cem_match(build_subsets()))
    print(f'# SM 长度配平场景数 = {len(sm)}（build_subsets/cem_match 单一来源）')

    if args.sample_curve:
        sample_curve(args.models.split(','), args.n_jobs)
        return

    a = arm_a(sm, args.models.split(','))
    b = arm_b(sm, a)
    c = arm_c(sm, a)
    if not a and not b and c is None:
        print('\n[无产物] outputs/probe_decode/ 下没有任何一臂的 jsonl。')
        return
    verdicts(a, b, c)


if __name__ == '__main__':
    main()
