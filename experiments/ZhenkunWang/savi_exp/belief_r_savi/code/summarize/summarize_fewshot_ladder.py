#!/usr/bin/env python
"""R3 few-shot ladder summary. Criteria frozen in the internal experiment log §2b.7 and in run_fewshot_ladder.py's docstring.

The protocol reuses existing pieces throughout, nothing reimplemented:
  statistics/CI/pairing   ← summarize_generative_ci.load_strata / stats / paired (incl. V0–V3 self-checks)
  degenerate solutions & format ← diag() in this file (restricted to the scenario set, see its docstring)
  length-stratification variable ← summarize_length_stratified.scenario_table (γ3−γ1 character count, scenario level, incl. V1)
  eval set & example pool  ← run_fewshot_ladder.select_pool (incl. the frozen assertions of §2b.3)

**R0 must be recomputed on the same 852 scenarios** (§2b.10). The 0.4970 of
§6.1 is a 872-scenario number and cannot be quoted directly. This script
restricts every arm and R0 to those 852 scenarios before comparing; if they
cannot be cut to match, it raises.

The two families are reported separately:
  dp family  R0 anchor = outputs/generative/*_dp.jsonl   (no trigger sentence on the tested question)
  cot family R0 anchor = outputs/generative/*_cot.jsonl  (tested question gets R0's CoT trigger sentence)
The two families differ in readout depth (examples give only the answer → at
large k there are zero reasoning tokens; dp-family k16 median output is 17
chars); **a conclusion counts only if both families agree**; anything holding
in one family only gets no readout, per L6's G3 discipline.

Usage
  python scripts/summarize_fewshot_ladder.py                 # full set 852
  python scripts/summarize_fewshot_ladder.py --agreement 5   # the lv=5 main criterion lives here
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_fewshot_ladder import CSV, select_pool  # noqa: E402
from summarize_generative_ci import ITERS, LV, SEED, load_strata, paired, stats  # noqa: E402
from summarize_length_stratified import scenario_table  # noqa: E402

import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_DIR = os.path.join(HERE, '..', 'outputs', 'fewshot_ladder')
R0_DIR = os.path.join(HERE, '..', 'outputs', 'generative')
MODELS = ('Qwen_Qwen2.5_7B_Instruct', 'meta_llama_Llama_3.1_8B_Instruct')
# (family, R0 file suffix, arms of that family). Arm names match drive_fewshot_ladder.sh.
FAMILIES = (('dp', 'dp', ('k2_chat', 'k2_shuf_chat', 'k4_chat', 'k4_shuf_chat',
                          'k8_chat', 'k8_shuf_chat',
                          'k16_chat', 'k16_shuf_chat')),
            ('cot', 'cot', ('k16_chat_cot', 'k16_shuf_chat_cot')))
PAIRS = {'k16_chat': 'k16_shuf_chat', 'k8_chat': 'k8_shuf_chat',
         'k4_chat': 'k4_shuf_chat', 'k2_chat': 'k2_shuf_chat',
         'k16_chat_cot': 'k16_shuf_chat_cot'}


def eval_scenes():
    """The 852 eval scenarios (= all scenarios minus the example seeds); also re-asserts G-leak."""
    df = pd.read_csv(CSV)
    _, _, seeds = select_pool(df)
    pon = df[df.modus == 'ponens']
    keep = set(pon[~pon.atomic_idx.isin(seeds)].dataset_id)
    assert not (set(pon[pon.atomic_idx.isin(seeds)].dataset_id) & keep) and len(keep) == 852
    return keep, seeds


def restrict(s, keep):
    """Restrict load_strata's return to the scenarios in keep."""
    bu, bm, bi, mi = s
    kb = [i for i, d in enumerate(bi) if d in keep]
    km = [i for i, d in enumerate(mi) if d in keep]
    return bu[kb], bm[km], [bi[i] for i in kb], [mi[i] for i in km]


def complete(path):
    """Whether the arm finished. A half-written file would be caught by load_strata's V1
    (a scenario with only one twin), but that is a crash, not a skip — summarizing while a
    batch is still running is routine, so block it here first."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for _ in f) == 852 * 2


def diag(path, agreement, keep):
    """Output length / example copying / no-FA / answer distribution, **restricted to the same scenarios as the statistics**.

    summarize_hint_ladder.degenerate is not used directly because it does not restrict to a
    scenario set; R0 is 872 scenarios, the arms are 852 — side by side that would be two
    denominators. Length and "example copying" are the two failure shapes found in the
    2026-08-03 smoke (zero reasoning tokens at large k; inline Llama continuing the example
    block 32/32), so they are computed here together.
    """
    L, copy, nofa, cnt, n = [], 0, 0, {}, 0
    for line in open(path):
        r = json.loads(line)
        if (agreement and LV[int(r['idx'])] != agreement) or r['dataset_id'] not in keep:
            continue
        n += 1
        L.append(len(r['raw_output']))
        copy += 'Example' in r['raw_output']
        nofa += 'final answer' not in r['raw_output'].lower()   # R7: don't trust format_ok
        cnt[r['extracted'] or '∅'] = cnt.get(r['extracted'] or '∅', 0) + 1
    L.sort()
    return L[len(L) // 2], copy, nofa, {k: v / n for k, v in cnt.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agreement', type=int, default=0)
    ap.add_argument('--iters', type=int, default=ITERS)
    args = ap.parse_args()
    keep, _ = eval_scenes()
    tab = scenario_table()
    tag = f'agreement_lv={args.agreement}' if args.agreement else '全集'
    print(f'# R3 少样本阶梯（{tag}；评测集 = 852 场景，R0 已裁到同一批；'
          f'{args.iters} 次配对 bootstrap，seed={SEED}）')
    print('# 主判据 = lv=5 上 k16 − k16_shuf 的配对 ΔBREU：≥+0.05 可学 / <+0.02 无效 / '
          '中间带或闸门不过 = 不确定\n')

    for fam, r0_suffix, arms in FAMILIES:
        print(f'\n{"#" * 100}\n# {fam} 族（R0 锚 = *_{r0_suffix}.jsonl）\n{"#" * 100}')
        for model in MODELS:
            r0p = os.path.join(R0_DIR, f'time_t1_{model}_{r0_suffix}.jsonl')
            have = [a for a in arms
                    if complete(os.path.join(ARM_DIR, f'time_t1_{model}_{a}.jsonl'))]
            if not have or not os.path.exists(r0p):
                print(f'\n[skip] {model}/{fam}: 臂 {len(have)}/{len(arms)}，R0 '
                      f'{"有" if os.path.exists(r0p) else "无"}')
                continue
            print(f'\n{"=" * 100}\n{model}  [{fam}]  已有臂 {len(have)}/{len(arms)}\n{"=" * 100}')

            store, rows = {}, []
            for arm in ['R0'] + have:
                p = r0p if arm == 'R0' else os.path.join(ARM_DIR, f'time_t1_{model}_{arm}.jsonl')
                s = restrict(load_strata(p, args.agreement, f'{model}/{arm}'), keep)
                if len(s[0]) + len(s[1]) != (391 - 16 if args.agreement == 5 else 852):
                    raise AssertionError(f'{arm} 裁完场景数不对：{len(s[0]) + len(s[1])}')
                pt, ci = stats(s[0], s[1], args.iters, SEED)
                med, copy, nofa, dist, n = diag(p, args.agreement, keep)
                store[arm] = (s, pt, ci)
                rows.append((arm, pt, ci, dist, nofa, med, copy, n))

            print('\n【0 仪器】输出中位长度 / 抄示例 / 无FA / 答案分布  ← 先看这三个数')
            print(f'  {"臂":<20}{"中位长":>7}{"抄示例":>8}{"无FA":>7}'
                  f'{"a":>7}{"b":>7}{"c":>7}{"其他":>7}')
            for arm, _, _, dist, nofa, med, copy, n in rows:
                other = 1 - sum(dist.get(k, 0) for k in 'abc')
                flag = ''
                if copy: flag += f'  ⚠️ {copy} 行抄了示例（读数不可信）'
                if dist.get('c', 0) > 0.90: flag += '  ⚠️ 退化解嫌疑'
                print(f'  {arm:<20}{med:>7}{copy:>8}{nofa:>7}{dist.get("a", 0):>7.3f}'
                      f'{dist.get("b", 0):>7.3f}{dist.get("c", 0):>7.3f}{other:>7.3f}{flag}')

            print('\n【1 主读数】BREU = (BU+BM)/2，**不是 BU**')
            print(f'  {"臂":<20}{"BU":>7}{"BM":>7}{"BREU":>8}{"BREU 95%CI":>20}'
                  f'{"BU-pon":>8}{"BU-tol":>8}{"BM-pon":>8}{"BM-tol":>8}')
            for arm, pt, ci, *_ in rows:
                lo, hi = ci['BREU']
                print(f'  {arm:<20}{pt["BU"]:>7.3f}{pt["BM"]:>7.3f}{pt["BREU"]:>8.4f}'
                      f'   [{lo:.4f},{hi:.4f}]{pt["BU-pon"]:>8.3f}{pt["BU-tol"]:>8.3f}'
                      f'{pt["BM-pon"]:>8.3f}{pt["BM-tol"]:>8.3f}')

            print('\n【2 配对 ΔBREU vs R0（同一 852 场景）】区间重叠 ≠ 差值含 0')
            for arm in have:
                pt, lo, hi, _ = paired(store['R0'][0], store[arm][0], args.iters, SEED)
                print(f'  {arm:<20}{pt:>+9.4f}   [{lo:>+.4f},{hi:>+.4f}]   '
                      + ('含 0' if lo <= 0 <= hi else '**CI 不含 0**'))

            print('\n【2b 主判据：配对 ΔBREU vs 同 k 的打乱标签臂】← 扣掉"给了 k 个例子"本身')
            main_delta = {}
            for a, b in PAIRS.items():
                if a in store and b in store:
                    pt, lo, hi, _ = paired(store[b][0], store[a][0], args.iters, SEED)
                    main_delta[a] = (pt, lo, hi)
                    verdict = ('约定可学' if pt >= 0.05 else '无效' if pt < 0.02 else '中间带')
                    print(f'  {a} − {b}: {pt:>+8.4f}  [{lo:>+.4f},{hi:>+.4f}]   '
                          f'{"含 0" if lo <= 0 <= hi else "**CI 不含 0**"}   → 落档：{verdict}'
                          + ('（须再过四道闸门）' if pt >= 0.02 else ''))

            print('\n【3 闸门】')
            # G-k dose response: real learning should be monotone in k; format effects saturate at k=2
            ks = [(int(a[1:].split('_')[0]), store[a][1]['BREU'])
                  for a in have if '_shuf' not in a]
            ks.sort()
            if len(ks) >= 3:
                mono = all(b > a for (_, a), (_, b) in zip(ks, ks[1:]))
                print(f'  G-k    BREU vs k: ' + '  '.join(f'k{k}={v:.4f}' for k, v in ks)
                      + f' → {"✅ 单调上升" if mono else "❌ 不单调"}')
            # G-shuf: how much the shuffled-label arm itself gains over R0 (that gain is format/priming, not the convention)
            for b in set(PAIRS.values()) & set(store):
                pt, lo, hi, _ = paired(store['R0'][0], store[b][0], args.iters, SEED)
                print(f'  G-shuf {b} 自己 vs R0: {pt:+.4f} [{lo:+.4f},{hi:+.4f}]'
                      f'   ← 这部分是"给了例子"的效应，不算约定')
            # G-len: whether gains show up only in the extreme bins of γ3−γ1 length
            for a, b in PAIRS.items():
                if a not in store or b not in store:
                    continue
                sc = sorted(keep & set(tab.index))
                if args.agreement:
                    sc = [s for s in sc if tab.loc[s, 'lv'] == args.agreement]
                v = tab.loc[sc, 'd_len'].to_numpy()
                cuts = np.quantile(v, [.25, .5, .75])
                cells = []
                for q in range(4):
                    ids = {s for s, x in zip(sc, np.digitize(v, cuts, right=False)) if x == q}
                    sa, sb = restrict(store[a][0], ids), restrict(store[b][0], ids)
                    if min(len(sa[0]), len(sa[1])) < 10:
                        cells.append(f'Q{q + 1}=样本不足'); continue
                    d, lo, hi, _ = paired(sb, sa, args.iters, SEED)
                    cells.append(f'Q{q + 1}={d:+.3f}')
                print(f'  G-len  {a}−{b} 按 γ3−γ1 四分位分箱: ' + '  '.join(cells)
                      + '   ← 只在极端箱有增益 = 学到的是长度')
            print('  G-leak ✅ 评测集与示例的 atomic_idx 交集为空（eval_scenes 已断言）')


if __name__ == '__main__':
    main()
