#!/usr/bin/env python
"""R1/R2 hint-ladder summary. Adjudication logic frozen in the docstring of `run_hint_ladder.py`.

Scope fully reuses `summarize_generative_ci.py` (`load_strata` with the V0–V3 self-checks,
`stats`, `paired`) — no second implementation: statistical unit = scenario, resampling
within BU/BM strata, paired bootstrap 5000 draws seed=0.

The report order is deliberate:
  0 degenerate solutions and format (**look at this first**) → 1 BREU and the four cells →
  2 paired ΔBREU vs R0 → 3 gates

**Why BU-Acc cannot be the primary readout**: always answering c gets BU=1.000 /
BM=0.000 / BREU=0.500. In this project Qwen2.5-0.5B measures exactly this shape
(BU 0.9953 / BM 0.0015 / BREU 0.498, nearly identical to 7B's 0.497). Any hint in the
"overturn" direction pushes BU up — **reading BU will inevitably score it a big success**.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_generative_ci import load_strata, stats, paired, ITERS, SEED, LV  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_DIR = os.path.join(HERE, '..', 'outputs', 'hint_ladder')
R0_DIR = os.path.join(HERE, '..', 'outputs', 'generative')
ARMS = ('license', 'strict', 'pragmatic', 'placebo', 'loose')


def degenerate(path, agreement):
    """Degenerate-solution and format diagnostics: answer distribution + rows without `final answer`.

    BU alone is deceptive, so this section comes first: if an arm's share of c exceeds
    0.9, its BU gain is a degenerate solution, not capability.
    """
    cnt, nofa, n = collections.Counter(), 0, 0
    for line in open(path):
        r = json.loads(line)
        if agreement and LV[int(r['idx'])] != agreement:
            continue
        n += 1
        cnt[r['extracted'] or '∅'] += 1
        # R0 files only have format_ok, and R7 records that field as a false positive
        # (upstream get_final_answer: with no `final answer`, rfind returns −1 and it
        # degenerates to grabbing the first letter, emitting illegal letters like m/n/e).
        # So always recompute from raw_output, same method as summarize_generative.py.
        nofa += 'final answer' not in r['raw_output'].lower()
    return {k: v / n for k, v in cnt.items()}, nofa, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agreement', type=int, default=0)
    ap.add_argument('--iters', type=int, default=ITERS)
    args = ap.parse_args()
    tag = f'agreement_lv={args.agreement}' if args.agreement else '全集'
    print(f'# 提示阶梯，场景级聚类配对 bootstrap（{tag}，{args.iters} 次，seed={SEED}）')
    print('# R0 = outputs/generative/*_dp.jsonl（同一 harness，提示逐字节相同，已过同一性检查）\n')

    models = sorted({os.path.basename(f)[len('time_t1_'):-len('.jsonl')].rsplit('_', 1)[0]
                     for f in glob.glob(os.path.join(ARM_DIR, 'time_t1_*.jsonl'))
                     if os.path.basename(f)[:-len('.jsonl')].rsplit('_', 1)[1] in ARMS})

    for model in models:
        r0_path = os.path.join(R0_DIR, f'time_t1_{model}_dp.jsonl')
        if not os.path.exists(r0_path):
            print(f'[skip] {model}: 没有 R0 基线')
            continue
        print(f'\n{"=" * 104}\n{model}\n{"=" * 104}')

        rows, store = [], {}
        for arm in ('R0',) + ARMS:
            p = r0_path if arm == 'R0' else os.path.join(ARM_DIR, f'time_t1_{model}_{arm}.jsonl')
            if not os.path.exists(p):
                continue
            dist, nofa, n = degenerate(p, args.agreement)
            s = load_strata(p, args.agreement, f'{model}/{arm}')
            pt, ci = stats(s[0], s[1], args.iters, SEED)
            store[arm] = (s, pt, ci)
            rows.append((arm, pt, ci, dist, nofa, n))

        print(f'\n【0 退化解与格式】答案分布（a/b/c/其他），c 占比 >0.90 = 退化解嫌疑')
        print(f'  {"臂":<12}{"a":>7}{"b":>7}{"c":>7}{"其他":>7}{"无FA行":>8}   n')
        for arm, _, _, dist, nofa, n in rows:
            other = 1 - sum(dist.get(k, 0) for k in 'abc')
            flag = '  ⚠️ 退化解嫌疑' if dist.get('c', 0) > 0.90 else ''
            print(f'  {arm:<12}{dist.get("a",0):>7.3f}{dist.get("b",0):>7.3f}'
                  f'{dist.get("c",0):>7.3f}{other:>7.3f}{nofa:>8}   {n}{flag}')

        print(f'\n【1 主读数】BREU = (BU+BM)/2，**不是 BU**')
        hdr = (f'  {"臂":<12}{"BU":>7}{"BM":>7}{"BREU":>8}{"BREU 95%CI":>19}'
               f'{"BU-pon":>8}{"BU-tol":>8}{"BM-pon":>8}{"BM-tol":>8}')
        print(hdr); print('  ' + '-' * (len(hdr) - 2))
        for arm, pt, ci, _, _, _ in rows:
            lo, hi = ci['BREU']
            print(f'  {arm:<12}{pt["BU"]:>7.3f}{pt["BM"]:>7.3f}{pt["BREU"]:>8.4f}'
                  f'   [{lo:.4f},{hi:.4f}]{pt["BU-pon"]:>8.3f}{pt["BU-tol"]:>8.3f}'
                  f'{pt["BM-pon"]:>8.3f}{pt["BM-tol"]:>8.3f}')

        print(f'\n【2 配对 ΔBREU vs R0】同一批场景同时套两臂；**区间重叠 ≠ 差值含 0**')
        print(f'  {"臂":<12}{"ΔBREU":>9}{"95%CI":>21}{"ΔBU":>9}{"ΔBM":>9}   判读')
        deltas = {}
        for arm in ARMS:
            if arm not in store:
                continue
            pt, lo, hi, _ = paired(store['R0'][0], store[arm][0], args.iters, SEED)
            dbu = store[arm][1]['BU'] - store['R0'][1]['BU']
            dbm = store[arm][1]['BM'] - store['R0'][1]['BM']
            deltas[arm] = (pt, lo, hi, dbu, dbm)
            sig = '含 0（与零不可区分）' if lo <= 0 <= hi else '**CI 不含 0**'
            print(f'  {arm:<12}{pt:>+9.4f}   [{lo:>+.4f},{hi:>+.4f}]{dbu:>+9.3f}{dbm:>+9.3f}   {sig}')

        # Once the placebo has an effect of its own, "vs R0" is no longer the right
        # control — it conflates "a sentence was added" with "what that sentence said".
        # The real question: **what remains after subtracting the placebo**.
        if 'placebo' in store:
            print(f'\n【2b 配对 ΔBREU vs placebo】← 扣掉"加了句话"本身的效应')
            print(f'  {"臂":<12}{"ΔBREU":>9}{"95%CI":>21}{"ΔBU":>9}{"ΔBM":>9}   判读')
            for arm in ARMS:
                if arm == 'placebo' or arm not in store:
                    continue
                pt, lo, hi, _ = paired(store['placebo'][0], store[arm][0], args.iters, SEED)
                dbu = store[arm][1]['BU'] - store['placebo'][1]['BU']
                dbm = store[arm][1]['BM'] - store['placebo'][1]['BM']
                sig = '含 0（与安慰剂不可区分）' if lo <= 0 <= hi else '**CI 不含 0**'
                print(f'  {arm:<12}{pt:>+9.4f}   [{lo:>+.4f},{hi:>+.4f}]'
                      f'{dbu:>+9.3f}{dbm:>+9.3f}   {sig}')

        print(f'\n【3 闸门】')
        # G-rev: license and strict are a pair of oppositely-directed hints. If their
        # ΔBREU have the same sign and magnitude, the change is unrelated to hint content —
        # just "a sentence was added" — judged a placebo effect.
        if 'license' in deltas and 'strict' in deltas:
            dl, ds = deltas['license'][0], deltas['strict'][0]
            same = (dl * ds > 0) and abs(abs(dl) - abs(ds)) < 0.5 * max(abs(dl), abs(ds), 1e-9)
            print(f'  G-rev  license ΔBREU {dl:+.4f} vs strict ΔBREU {ds:+.4f} → '
                  + ('❌ 同号同量级，判为安慰剂效应' if same else '✅ 方向可分辨'))
            # Core pre-registered expectation: license should give BU↑ BM↓, strict the reverse, with BREU unmoved
            bl, bs = deltas['license'], deltas['strict']
            print(f'         ΔBU  license {bl[3]:+.3f} / strict {bs[3]:+.3f}   '
                  f'ΔBM  license {bl[4]:+.3f} / strict {bs[4]:+.3f}   '
                  + ('← 对称互换，符合"只是移动阈值"的预注册预期'
                     if bl[3] * bs[3] < 0 and bl[4] * bs[4] < 0 else ''))
        if 'placebo' in deltas:
            dp = deltas['placebo'][0]
            print(f'  G-pla  placebo ΔBREU {dp:+.4f} → 任何真实效应必须**超过**这一条')
        for arm, (_, _, _, dist, nofa, n) in ((r[0], r) for r in rows):
            if dist.get('c', 0) > 0.90:
                print(f'  G-deg  ⚠️ {arm} 的 c 占比 {dist["c"]:.3f} > 0.90 → 该臂的 BU 提升是退化解')


if __name__ == '__main__':
    main()
