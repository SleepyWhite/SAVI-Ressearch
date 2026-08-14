#!/usr/bin/env python
"""PREREG_bon32.md §2 gates + primary readout + mechanical table match + appendix. Match only, no interpretation.

Single source for scoring: only calls readouts()/agg()/load_greedy() of summarize_bon_v2.py;
no second scoring implementation. εmiss(cell) = 1 - agg(..., 'any').

Output (stdout; outer tee into outputs/ci/bon32.txt):
  G-1 anchor bit-for-bit reproduction / G-2 new-pool standalone εmiss comparison / G-3
  (subprocess call to check_bon32_g3.py) / G-4 BM positive control; six-cell εmiss@32
  primary readout + mechanical match against the two lines;
  appendix = any@N curves (N=1..32, nested) / hit rate of previously missed items in the
  new pool / vote-BoN@32 (§6.2b scope).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize_bon_v2 as S  # noqa: E402  single source for scoring

BR = os.path.join(HERE, '..')
ORIG_DIR = os.path.join(BR, 'outputs', 'bon_v2')
INC_DIR = os.path.join(BR, 'outputs', 'bon_v2_inc')
MERGED_DIR = os.path.join(BR, 'outputs', 'bon_v2_merged32')

METHODS = ('dp', 'cot', 'ps')
MODI = ('ponens', 'tollens')

# STATUS §6.2 registered anchors (e16, 3 decimal places) — frozen in PREREG §2
ANCHOR_BU = {('dp', 'ponens'): 0.957, ('cot', 'ponens'): 0.957, ('ps', 'ponens'): 0.940,
             ('dp', 'tollens'): 0.931, ('cot', 'tollens'): 0.929, ('ps', 'tollens'): 0.913}
ANCHOR_BM = {('dp', 'ponens'): 0.003, ('cot', 'ponens'): 0.009, ('ps', 'ponens'): 0.003,
             ('dp', 'tollens'): 0.009, ('cot', 'tollens'): 0.003, ('ps', 'tollens'): 0.003}


def load(path):
    return [json.loads(l) for l in open(path)]


def emiss(recs, n, modus, bu, last16=False):
    """εmiss = 1 - mean(any@n), via summarize_bon_v2's readouts/agg."""
    if last16:
        recs = [{**r, 'chains': r['chains'][16:]} for r in recs]
    pairs = [(r, S.readouts(r, n)) for r in recs]
    sel = lambda r: (r['modus'] == modus) and ((r['ground_truth'] == 'c') == bu)
    mean, cnt, _ = S.agg(pairs, sel, 'any')
    return 1.0 - mean, cnt


def main():
    merged = {m: load(os.path.join(MERGED_DIR, f'time_t1_Qwen_Qwen3_4B_{m}_N32.jsonl')) for m in METHODS}
    orig = {m: load(os.path.join(ORIG_DIR, f'time_t1_Qwen_Qwen3_4B_{m}_N16.jsonl')) for m in METHODS}

    gates = {}

    # ---------------- G-1 anchor: recompute first 16 chains of merged files; six-cell εmiss must reproduce §6.2 bit-for-bit ----------------
    # Three mechanical readings printed side by side, not silently reconciled (register the
    # discrepancy, leave the ruling to a human):
    #  (a) literal: |recomputed − 3dp registered| <= 1e-4 — the books carry only 3 decimals,
    #      so no cell can satisfy this (e.g. dp-pon exactly reproduces 0.957169 vs 0.957,
    #      Δ=1.69e-4); the criterion defect is reported as-is;
    #  (b) bookkeeping-precision digitwise: round(recomputed, 3) == registered value;
    #  (c) instrument identity: recompute of merged first 16 ≡ recompute of original file
    #      (<=1e-4) — the real "instrument broke" detector.
    print('================ G-1 锚点复现（合并文件前 16 条，BU 六格；三读法并列） ================')
    print(f'{"cell":<14} {"e16(merged前16)":>16} {"e16(原文件重算)":>16} {"|Δ|c":>10} '
          f'{"round3":>8} {"锚点":>7} {"(a)1e-4":>8} {"(b)3dp":>7} {"(c)同一":>7}')
    ok_a = ok_b = ok_c = True
    for m in METHODS:
        for mo in MODI:
            e_merged, cnt = emiss(merged[m], 16, mo, True)
            e_orig, cnt2 = emiss(orig[m], 16, mo, True)
            d = abs(e_merged - e_orig)
            anchor = ANCHOR_BU[(m, mo)]
            ca = abs(e_merged - anchor) <= 1e-4
            cb = round(e_merged, 3) == anchor
            cc = d <= 1e-4
            ok_a &= ca
            ok_b &= cb
            ok_c &= cc
            print(f'BU-{mo[:3]:<3} {m:<6} {e_merged:>16.6f} {e_orig:>16.6f} {d:>10.2e} '
                  f'{round(e_merged, 3):>8.3f} {anchor:>7.3f} '
                  f'{"PASS" if ca else "FAIL":>8} {"PASS" if cb else "FAIL":>7} '
                  f'{"PASS" if cc else "FAIL":>7}  (n={cnt})')
    gates['G-1'] = ok_b and ok_c
    print(f'[G-1] (a)字面1e-4: {"PASS" if ok_a else "FAIL"}   (b)账面3dp逐位: {"PASS" if ok_b else "FAIL"}   '
          f'(c)仪器同一: {"PASS" if ok_c else "FAIL"}')
    print('[G-1] 汇总行取 (b) AND (c)。已知登记事项：ps BU-tol 原文件重算 = 0.912477'
          '（summarize CLI 4dp 打印 0.9125），账面 0.913 为 0.9125 的半进 3dp 转写；'
          '读法 (a) 对 3dp 账面在所有格上按构造不可满足。两点均为账面/判据字面问题，'
          '登记不调和，最终裁决留人。')

    # ---------------- G-2 new-pool comparison: new 16 chains only, εmiss@16_new, per cell |e16_new - e16| <= 0.03 ----------------
    print('\n================ G-2 新池单独 εmiss@16_new（BU 六格，|Δ| <= 0.03） ================')
    print(f'{"cell":<14} {"e16_new":>10} {"e16锚点":>8} {"|Δ|":>8} 判')
    ok = True
    for m in METHODS:
        for mo in MODI:
            e_new, cnt = emiss(merged[m], 16, mo, True, last16=True)
            anchor = ANCHOR_BU[(m, mo)]
            d = abs(e_new - anchor)
            cell_ok = d <= 0.03
            ok &= cell_ok
            print(f'BU-{mo[:3]:<3} {m:<6} {e_new:>10.4f} {anchor:>8.3f} {d:>8.4f} '
                  f'{"PASS" if cell_ok else "FAIL"}  (n={cnt})')
    gates['G-2'] = ok
    print(f'[G-2] {"PASS" if ok else "FAIL"}')

    # ---------------- G-3 parameter and prompt identity (subprocess, full row-by-row) ----------------
    print('\n================ G-3 参数与提示同一（check_bon32_g3.py，全量） ================')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'check_bon32_g3.py')],
                       capture_output=True, text=True)
    print(p.stdout, end='')
    if p.stderr:
        print(p.stderr, end='')
    gates['G-3'] = (p.returncode == 0)
    print(f'[G-3] {"PASS" if gates["G-3"] else "FAIL"}')

    # ---------------- G-4 BM positive control: each BM cell εmiss@32 <= 0.02 ----------------
    print('\n================ G-4 BM 正对照（εmiss@32 <= 0.02） ================')
    print(f'{"cell":<14} {"e32(BM)":>10} {"e16登记":>8} 判')
    ok = True
    for m in METHODS:
        for mo in MODI:
            e32, cnt = emiss(merged[m], 32, mo, False)
            cell_ok = e32 <= 0.02
            ok &= cell_ok
            print(f'BM-{mo[:3]:<3} {m:<6} {e32:>10.4f} {ANCHOR_BM[(m, mo)]:>8.3f} '
                  f'{"PASS" if cell_ok else "FAIL"}  (n={cnt})')
    gates['G-4'] = ok
    print(f'[G-4] {"PASS" if ok else "FAIL"}')

    print('\n================ 闸门汇总 ================')
    for g in ('G-1', 'G-2', 'G-3', 'G-4'):
        print(f'  {g}: {"PASS" if gates[g] else "FAIL"}')
    if not all(gates.values()):
        print('  仪器闸门未全过 —— 按 PREREG §2/§3 不读主数，停下上报')
        # The primary readout is still printed below for troubleshooting, but the match conclusion is void

    # ---------------- Primary readout: six-cell εmiss@32 + mechanical table match ----------------
    print('\n================ 主读数：六格 εmiss(BU)@32 + 机械对号（PREREG §2） ================')
    print(f'{"cell":<14} {"e16锚点":>8} {"e32":>10} {"平线 e16-0.02":>13} {"几何线 e16^2+0.02":>17} 档')
    bins = []
    for m in METHODS:
        for mo in MODI:
            e32, cnt = emiss(merged[m], 32, mo, True)
            a = ANCHOR_BU[(m, mo)]
            flat_line, geo_line = a - 0.02, a * a + 0.02
            if e32 >= flat_line:
                b = '平的（池里没有）'
            elif e32 <= geo_line:
                b = '几何衰减（池子太小）'
            else:
                b = '中间带（不确定）'
            bins.append(b)
            print(f'BU-{mo[:3]:<3} {m:<6} {a:>8.3f} {e32:>10.4f} {flat_line:>13.4f} '
                  f'{geo_line:>17.4f} {b}  (n={cnt})')
    if len(set(bins)) == 1:
        print(f'总判（机械）：六格全部同档 -> 落「{bins[0]}」档')
    else:
        from collections import Counter
        print(f'总判（机械）：六格不同档 -> 不确定档，逐格如实报（分布 {dict(Counter(bins))}）')
    print('措辞附带（PREREG §2 强制）：εmiss 只能对每链命中率 q 的量级设界，不能证明 q=0；'
          'N=32 下 0 命中仅给出每题 q < ~0.09 的 95% 上界。最终判定留人。')

    # ---------------- Appendix (no adjudication) ----------------
    print('\n================ 附报 1：any@N 曲线（N=1..32，嵌套子采样，BU 六格） ================')
    for m in METHODS:
        for mo in MODI:
            recs = [r for r in merged[m] if r['modus'] == mo and r['ground_truth'] == 'c']
            curve = []
            for n in range(1, 33):
                pairs = [(r, S.readouts(r, n)) for r in recs]
                mean, _, _ = S.agg(pairs, lambda r: True, 'any')
                curve.append(mean)
            print(f'BU-{mo[:3]} {m:<4} any@N: ' + ' '.join(f'{v:.4f}' for v in curve))

    print('\n================ 附报 2：前 16 条未命中题在新 16 条中的命中率（逐格） ================')
    print(f'{"cell":<14} {"未命中数":>8} {"新池命中":>8} {"命中率":>8}')
    for bu in (True, False):
        for m in METHODS:
            for mo in MODI:
                recs = [r for r in merged[m]
                        if r['modus'] == mo and (r['ground_truth'] == 'c') == bu]
                miss = [r for r in recs if not S.readouts(r, 16)['any']]
                hit_new = [r for r in miss
                           if S.readouts({**r, 'chains': r['chains'][16:]}, 16)['any']]
                rate = len(hit_new) / len(miss) if miss else float('nan')
                print(f'{"BU" if bu else "BM"}-{mo[:3]:<3} {m:<6} {len(miss):>8} '
                      f'{len(hit_new):>8} {rate:>8.4f}')

    print('\n================ 附报 3：vote/BoN@32（§6.2b 口径，BU 行级；greedy=v1 单遍） ================')
    print(f'{"method":<7} {"greedy":>8} {"vote_all":>9} {"vote_valid":>10} {"bon_sum":>8} {"bon_mean":>9} {"any@32":>8}')
    for m in METHODS:
        recs = [r for r in merged[m] if r['ground_truth'] == 'c']
        pairs = [(r, S.readouts(r, 32)) for r in recs]
        greedy = S.load_greedy('Qwen_Qwen3_4B', m)
        vals = {}
        for k in ('greedy', 'vote_all', 'vote_valid', 'bon_sum', 'bon_mean', 'any'):
            vals[k], _, _ = S.agg(pairs, lambda r: True, k, greedy)
        print(f'{m:<7} {vals["greedy"]:>8.4f} {vals["vote_all"]:>9.4f} {vals["vote_valid"]:>10.4f} '
              f'{vals["bon_sum"]:>8.4f} {vals["bon_mean"]:>9.4f} {vals["any"]:>8.4f}')

    sys.exit(0 if all(gates.values()) else 1)


if __name__ == '__main__':
    main()
