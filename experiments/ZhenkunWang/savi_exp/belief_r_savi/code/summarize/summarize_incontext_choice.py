#!/usr/bin/env python
"""Summary for candidate (ii) in-context two-way choice. Criteria frozen in the docstring of `probe_incontext_choice.py`.

Prints per that set of criteria only; no thresholds are invented here.

Usage: python scripts/summarize_incontext_choice.py
"""
import collections
import glob
import json
import os

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'incontext_choice')
BOOT, SEED = 2000, 20260802

# reference lines (all computed within this project, see STATUS)
L1_BEST = 0.588      # AUC ceiling of the L1 scoring axis over 6 models (Qwen2.5-7B / relation)
# Balanced-accuracy ceiling on lv=5 with the "design intent" as a perfect scorer.
# Produced by `scripts/compute_ceiling.py` (added 2026-08-02; previously a hardcoded
# constant with no producing code):
#   lv=5 scenarios 391, REQ recall 0.9844 / ALT recall 0.7015 → 0.8430, 95%CI [0.8019, 0.8822]
# Note this is the ceiling of the "recover the design intent" family of strategies;
# **the absolute ceiling is 1.000** (lv=5 gold labels are unanimous).
CEILING = 0.8430
ORDER = ('0.5B', '1.5B', '3B', '7B', 'Qwen3-4B', 'Llama-8B')
NAME = {'Qwen_Qwen2.5_0.5B_Instruct': '0.5B', 'Qwen_Qwen2.5_1.5B_Instruct': '1.5B',
        'Qwen_Qwen2.5_3B_Instruct': '3B', 'Qwen_Qwen2.5_7B_Instruct': '7B',
        'Qwen_Qwen3_4B': 'Qwen3-4B', 'meta_llama_Llama_3.1_8B_Instruct': 'Llama-8B'}


def bal_acc(rs, strict=False):
    """Balanced accuracy = (REQ recall + ALT recall)/2.

    Balanced rather than plain accuracy: BU is 65.7% of lv=5 scenarios, so
    "always pick REQ" scores 0.657 with zero information. A binary scorer's AUC
    equals its balanced accuracy, so this number compares directly to L1's AUC.

    Two conventions:
      strict=False  denominator includes only rows where RELATION parsed. **On small
                    models this is a self-selected subset** (0.5B's contract
                    fulfilment rate is only ~0.26); it cannot be read as ability.
      strict=True   rows where the contract was not fulfilled count as wrong. Not a
                    newly added criterion — it is the natural extension to the
                    relation line of the "format errors count as wrong" rule already
                    locked in STATUS §3.
    """
    pool = rs if strict else [r for r in rs if r['picked_req'] is not None]
    req = [r for r in pool if r['gold_is_req']]
    alt = [r for r in pool if not r['gold_is_req']]
    if not req or not alt:
        return float('nan'), 0
    return (np.mean([r['picked_req'] is True for r in req])
            + np.mean([r['picked_req'] is False for r in alt])) / 2, len(pool)


def boot_ci(rs, rng):
    """Scenario-clustered bootstrap (twins of the same dataset_id are resampled together)."""
    by = collections.defaultdict(list)
    for r in rs:
        by[r['dataset_id']].append(r)
    keys = list(by)
    out = []
    for _ in range(BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        s = [x for i in pick for x in by[keys[i]]]
        b, _n = bal_acc(s)
        if not np.isnan(b):
            out.append(b)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (
        float('nan'), float('nan'))


def main():
    rng = np.random.default_rng(SEED)
    per = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(D, '*.jsonl'))):
        rs = [json.loads(l) for l in open(f)]
        if not rs:
            continue
        tag = NAME.get(os.path.basename(f)[len('time_t1_'):-len('.jsonl')].rsplit('_', 2)[0],
                       os.path.basename(f))
        per[tag][rs[0]['order']] = rs

    print('=' * 88)
    print('仪器有效性（先看这个：契约没兑现的话下面的数都不用看）')
    print('=' * 88)
    print(f"  {'模型':<11}{'排法':<11}{'n':>6}{'RELATION解析':>13}{'答案槽合法':>11}{'有FinalAns':>11}")
    for tag in ORDER:
        for od, rs in sorted(per.get(tag, {}).items()):
            n = len(rs)
            print(f"  {tag:<11}{od:<11}{n:>6}"
                  f"{sum(r['relation_slot'] is not None for r in rs) / n:>13.3f}"
                  f"{sum(r['extracted_valid'] for r in rs) / n:>11.3f}"
                  f"{sum(r['has_final_answer'] for r in rs) / n:>11.3f}")

    print('\n' + '=' * 88)
    print('P-1 主检验：RELATION 选择的平衡准确率（agreement_lv=5 为主读数集）')
    print(f'   参照：随机 0.500 ｜ L1 打分轴上限 {L1_BEST:.3f} ｜ 天花板 {CEILING:.3f}')
    print('=' * 88)
    print(f"  {'模型':<11}{'req_first':>20}{'alt_first':>20}{'均值(宽)':>10}"
          f"{'均值(严)':>10}{'选REQ率':>9}")
    print('   宽 = 只算契约兑现的行（小模型上是自选子集）｜ 严 = 契约未兑现计错')
    summary = {}
    for tag in ORDER:
        d = per.get(tag)
        if not d or len(d) < 2:
            continue
        cells, picks, strict = {}, {}, {}
        for od in ('req_first', 'alt_first'):
            sub = [r for r in d[od] if r['agreement_lv'] == 5]
            b, n = bal_acc(sub)
            lo, hi = boot_ci(sub, rng)
            cells[od] = (b, lo, hi)
            strict[od] = bal_acc(sub, strict=True)[0]
            v = [r for r in sub if r['picked_req'] is not None]
            picks[od] = np.mean([r['picked_req'] for r in v]) if v else float('nan')
        m = np.mean([cells['req_first'][0], cells['alt_first'][0]])
        ms = np.mean(list(strict.values()))
        summary[tag] = (m, cells, picks, ms)
        print(f"  {tag:<11}"
              + ''.join(f"{cells[o][0]:>9.3f} [{cells[o][1]:.2f},{cells[o][2]:.2f}]"
                        for o in ('req_first', 'alt_first'))
              + f"{m:>10.3f}{ms:>10.3f}{np.mean(list(picks.values())):>9.3f}")

    print('\n' + '=' * 88)
    print('P-2 位置对照闸门：|Δ平衡准确率| > 0.10，或两种排法下"选第1项"都 >0.70 → 读数作废')
    print('=' * 88)
    gates = {}
    for tag, (m, cells, picks, _ms) in summary.items():
        gap = abs(cells['req_first'][0] - cells['alt_first'][0])
        # "picked item 1" rate: under req_first = REQ-pick rate; under alt_first = 1 − REQ-pick rate
        first_rate = (picks['req_first'], 1 - picks['alt_first'])
        pos_dom = all(x > 0.70 for x in first_rate)
        ok = gap <= 0.10 and not pos_dom
        gates[tag] = ok
        print(f"  {tag:<11} |Δ|={gap:.3f}  选第1项={first_rate[0]:.2f}/{first_rate[1]:.2f}"
              f"  → {'通过' if ok else '**作废**'}")

    print('\n' + '=' * 88)
    print('C-1 自洽率（答案 == 模型自己选的读法所蕴含的答案）｜ C-2 孪生一致 ｜ 端到端')
    print('   C-1 与 C-2 都不需要金标，故不受 47.4% 标签噪声影响')
    print('=' * 88)
    print(f"  {'模型':<11}{'C-1自洽':>10}{'C-2孪生一致':>13}{'BU':>8}{'BM':>8}{'BREU':>8}")
    for tag in ORDER:
        d = per.get(tag)
        if not d or len(d) < 2:
            continue
        rs = d['req_first'] + d['alt_first']
        coh = [r['coherent'] for r in rs if r['coherent'] is not None]
        tw = collections.defaultdict(dict)
        for r in rs:
            if r['picked_req'] is not None:
                tw[(r['dataset_id'], r['order'])][r['modus']] = r['picked_req']
        pairs = [v for v in tw.values() if len(v) == 2]
        agree = np.mean([v['ponens'] == v['tollens'] for v in pairs]) if pairs else float('nan')
        bu = np.mean([r['correct'] for r in rs if r['ground_truth'] == 'c'])
        bm = np.mean([r['correct'] for r in rs if r['ground_truth'] != 'c'])
        print(f"  {tag:<11}{np.mean(coh):>10.3f}{agree:>13.3f}{bu:>8.3f}{bm:>8.3f}"
              f"{(bu + bm) / 2:>8.3f}")

    print('\n' + '=' * 88)
    print('结论（按冻结判据，以 lv=5 两排法均值为准）')
    print('=' * 88)
    for tag in ORDER:
        if tag not in summary:
            continue
        m, _c, _p, ms = summary[tag]
        if not gates[tag]:
            v = '读数作废（P-2 位置闸门未过）'
        elif m <= 0.55:
            v = '**无信号** —— 这条路在 Belief-R 上关闭'
        elif m < 0.65:
            v = f'与 L1 同带（{L1_BEST:.3f}），干预形态没带来东西。记录，不建设'
        else:
            v = f'**真信号** —— 超出 L1 打分轴上限 {L1_BEST:.3f}，归因于干预形态'
        print(f'  {tag:<11} 宽={m:.3f} 严={ms:.3f}  → {v}')


if __name__ == '__main__':
    main()
