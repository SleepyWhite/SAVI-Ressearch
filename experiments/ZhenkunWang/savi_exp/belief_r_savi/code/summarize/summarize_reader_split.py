#!/usr/bin/env python
"""Reader vs question-writer split readouts (2026-08-05, computed directly per user
instruction; the reading draft predates this computation, in
report_2026-08-05/REPORT.md §5 and the STATUS changelog entry 2026-08-05 e, but was not
formally frozen by a human — results are registered as "exploratory, read against the draft").

Grouping (scenario level): intent = dataset_id suffix (strong→REQ / weak→ALT,
same protocol as compute_ceiling.py); gold = derived back from ground_truth
(REQ↔c / ALT↔a|b). intent==gold → agree group; intent!=gold → disagree group.
lv=agreement_lv (6 = data anomaly, excluded).

Readouts: arm C (two-step self-judgement) and arm A (probe pred_state, 4B main
/ 7B secondary) on each (lv, group): accuracy against gold (= fraction picking
the majority-vote side), per-gold-class recall, bal-acc;
CI = bootstrap clustered by dataset_id, 5000 iterations, seed 20260805.

Anchor self-check (no main numbers if it fails): A-4B lv5 0.6835 / lv4 0.6039;
C lv5 0.5189 / lv4 0.5605 (registered in the relation-judgement strata table of
outputs/ci/lv4_dose.txt).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PD = os.path.join(BASE, 'outputs', 'probe_decode')
B = 5000
SEED = 20260805


def load(path):
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows:
        lv = r['agreement_lv']
        gold = 'REQ' if r['ground_truth'] == 'c' else 'ALT'
        if 'gold_state' in r:
            assert r['gold_state'] == gold, (r['dataset_id'], r['modus'])
        intent = 'REQ' if r['dataset_id'].rsplit('-', 1)[1] == 'strong' else 'ALT'
        out.append(dict(sid=r['dataset_id'], lv=lv, gold=gold, intent=intent,
                        pred=r['pred_state'], modus=r['modus']))
    return out


def balacc(rows):
    rec = {}
    for c in ('REQ', 'ALT'):
        sub = [r for r in rows if r['gold'] == c]
        rec[c] = (sum(r['pred'] == c for r in sub) / len(sub)) if sub else None
    both = [v for v in rec.values() if v is not None]
    return sum(both) / len(both), rec


def acc_ci(rows):
    """Accuracy against gold + dataset_id-clustered bootstrap CI."""
    by = defaultdict(list)
    for r in rows:
        by[r['sid']].append(r['pred'] == r['gold'])
    sids = sorted(by)
    acc = np.mean([x for s in sids for x in by[s]])
    rng = np.random.default_rng(SEED)
    stats = []
    arr = [np.array(by[s], dtype=float) for s in sids]
    for _ in range(B):
        idx = rng.integers(0, len(arr), len(arr))
        stats.append(np.concatenate([arr[i] for i in idx]).mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return acc, lo, hi


def cell(rows, label):
    sids = {r['sid'] for r in rows}
    acc, lo, hi = acc_ci(rows)
    ba, rec = balacc(rows)
    rr = ' '.join(f'{c}召回={rec[c]:.3f}' if rec[c] is not None else f'{c}召回=—'
                  for c in ('REQ', 'ALT'))
    print(f'  {label:<26} 场景={len(sids):>3} 行={len(rows):>4} '
          f'对金标acc={acc:.4f} [{lo:.3f},{hi:.3f}]  bal-acc={ba:.4f}  {rr}')
    return acc, lo, hi


def main():
    arms = {
        'C 两步自判': load(os.path.join(PD, 'C_twostep.jsonl')),
        'A 探针(4B)': load(os.path.join(PD, 'A_Qwen_Qwen3_4B.jsonl')),
        'A 探针(7B)': load(os.path.join(PD, 'A_Qwen_Qwen2.5_7B_Instruct.jsonl')),
    }
    # V1 anchors (checked against lv4_dose.txt registered values; bal-acc protocol, all rows)
    anchors = {('A 探针(4B)', 5): 0.6835, ('A 探针(4B)', 4): 0.6039,
               ('C 两步自判', 5): 0.5189, ('C 两步自判', 4): 0.5605}
    print('# V1 锚点复算(bal-acc,全行)')
    for (arm, lv), want in anchors.items():
        got, _ = balacc([r for r in arms[arm] if r['lv'] == lv])
        ok = abs(got - want) < 5e-4
        print(f'  {arm} lv={lv}: {got:.4f} (登记 {want}) {"✅" if ok else "❌"}')
        if not ok:
            sys.exit('锚点不复现,停止')

    # V2 scenario-level consistency + lv6 exclusion
    ref = arms['C 两步自判']
    per_sid = defaultdict(set)
    for r in ref:
        per_sid[r['sid']].add((r['lv'], r['gold'], r['intent']))
    assert all(len(v) == 1 for v in per_sid.values()), '孪生行 lv/gold/intent 不一致'
    n6 = sum(r['lv'] == 6 for r in ref)
    print(f'# V2 孪生一致 ✅;lv=6 排除 {n6} 行(预期 4)\n')

    # group counts (scenario level; arm C is the full set, 1,744 rows)
    print('# 分组数量(意图=strong→REQ/weak→ALT;金标=多数票反推)')
    for lv in (5, 4):
        sc = {r['sid']: r for r in ref if r['lv'] == lv}.values()
        agree = [s for s in sc if s['intent'] == s['gold']]
        dis = [s for s in sc if s['intent'] != s['gold']]
        d_sa = sum(s['intent'] == 'REQ' for s in dis)   # strong→gold ALT
        d_wr = len(dis) - d_sa                           # weak→gold REQ
        print(f'  lv={lv}: 一致 {len(agree)} 场景 / 分歧 {len(dis)} 场景'
              f'(strong→ALT 翻转 {d_sa},weak→REQ 翻转 {d_wr})')
    print()

    for arm, rows in arms.items():
        rows = [r for r in rows if r['lv'] in (4, 5)]
        print(f'== {arm} ==')
        for lv in (5, 4):
            for g, name in (('agree', '一致组'), ('dis', '分歧组')):
                sub = [r for r in rows if r['lv'] == lv and
                       ((r['intent'] == r['gold']) == (g == 'agree'))]
                cell(sub, f'lv={lv} {name}')
        for g, name in (('agree', '一致组'), ('dis', '分歧组')):
            sub = [r for r in rows if (r['intent'] == r['gold']) == (g == 'agree')]
            cell(sub, f'lv4+lv5 合计 {name}')
        print()

    print('# 判读对照(草案见 REPORT §5;正式裁决留给人):')
    print('#  1 跟读者成立 = C 分歧组显著>0.5 且 lv4 一致组与 lv5 的 0.519 差别不大')
    print('#  2 更懂意图   = C 分歧组显著<0.5')
    print('#  3 不确定     = C 分歧组 CI 含 0.5 或组间无差')


if __name__ == '__main__':
    main()
