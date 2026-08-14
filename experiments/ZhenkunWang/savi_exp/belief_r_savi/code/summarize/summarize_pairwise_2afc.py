#!/usr/bin/env python
"""L6 summary: main readout of the pairwise comparison (2AFC) + four gates. All decision logic frozen in PLAN §3 "L6".

The reporting order is **deliberate** (L5's lesson: check truncation rate before accuracy):
  0 instrument health (truncation rate / format-compliance rate) → 1 main readout →
  2 four gates → 3 secondary readouts → 4 verdict

[Main readout] pair accuracy = fraction pointing to the strong side, **design-intent labels
    × 63 pairs × mean over both orders**, ponens rendering. Chance 0.500. Both denominators
    are reported:
      wide   = only rows that complied with the format (self-selects a subset, so it must
               be read together with the compliance rate)
      strict = non-compliance counted as wrong (consistent with §3 "format errors count as wrong")
[Four gates]
  G1 order swap: accuracy difference between the two orders > 0.10, or "picks slot 1" > 0.70
     under both orders → that model is voided
  G2 lexical control: build one rule scorer from γ3 length and one from Jaccard with q, and
     take **the better of the two directions** (i.e. the ceiling of this family of lexical
     rules, the strictest form); the model must exceed both by ≥0.05
  G3 cross-model direction agreement: if the 6 models disagree in direction the verdict is
     negative — **no picking one model to report**
  G4 models with format-compliance rate < 0.8: their "wide" reading stays out of the main table
[Verdict] ≥0.70 new finding | 0.65–0.70 not covered by the criteria, report honestly as
    uncertain | 0.55–0.65 same band as L1/L4/L5 | ≤0.55 no signal, L series closed

Two CIs are reported; their difference is itself a readout:
  - clustered bootstrap by **pair** (n=63) — conventional
  - clustered bootstrap by **atomic_idx** (24 seeds) — the independence discount required
    by PLAN §3 L6 known limitation 4
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_pairwise_2afc import load_pairs  # noqa: E402
from run_generative import BELIEF_REV  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'pairwise_2afc')
ORDERS = ('strong_first', 'weak_first')
ITERS, SEED = 5000, 0
CEILING, L1, L5 = 0.843, 0.588, 0.545


def load(path):
    return [json.loads(line) for line in open(path)]


def acc_by_order(rows, strict):
    """Return {order: accuracy}. With strict=True, format non-compliance counts as wrong."""
    out = {}
    for o in ORDERS:
        sub = [r for r in rows if r['order'] == o]
        if strict:
            out[o] = float(np.mean([bool(r['picked_strong']) for r in sub])) if sub else float('nan')
        else:
            ok = [r for r in sub if r['format_ok']]
            out[o] = float(np.mean([r['picked_strong'] for r in ok])) if ok else float('nan')
    return out


def boot_ci(per_unit, iters=ITERS, seed=SEED):
    """per_unit: {cluster key: [0/1 for each row in that cluster]}. Resample whole clusters, then average over all rows."""
    rng = np.random.RandomState(seed)
    keys = list(per_unit)
    vals = []
    for _ in range(iters):
        pick = [per_unit[keys[i]] for i in rng.randint(0, len(keys), len(keys))]
        flat = [v for grp in pick for v in grp]
        vals.append(np.mean(flat))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def lexical_arms(pairs):
    """Two lexical rule scorers. Returns {name: (naive-direction accuracy, better-of-both-directions ceiling)}.

    The naive directions are hard-coded here — no picking after the run:
      length  — predicts the **longer** γ3 is strong ("the appended condition is written
                more concretely and at greater length")
      jaccard — predicts the γ3 with **higher** lexical overlap with the conclusion q is strong
    But the gate uses max(acc, 1-acc): a rule with the wrong direction is equivalent to its
    reversed rule, so to win, the model must beat this rule family's ceiling.
    """
    def jac(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

    out = {}
    for name, score in (('length', lambda p, s: len(s)),
                        ('jaccard_q', lambda p, s: jac(s, p['q']))):
        hits = []
        for p in pairs:
            ss, sw = score(p, p['g3_strong']), score(p, p['g3_weak'])
            if ss == sw:
                hits.append(0.5)          # tie: score as a coin flip, no favor to the rule
            else:
                hits.append(float(ss > sw))
        a = float(np.mean(hits))
        out[name] = (a, max(a, 1 - a))
    return out


def report(modus, pairs_all, files, gold_only=False, tag=''):
    pairs = [p for p in pairs_all if p['modus'] == modus]
    if gold_only:
        pairs = [p for p in pairs if p['gold_separates']]
    keep = {p['pair_id'] for p in pairs}
    n = len(pairs)
    lex = lexical_arms(pairs)
    lex_best = max(v[1] for v in lex.values())

    print(f'\n{"=" * 92}\n{tag}  modus={modus}  n={n} 组 × 2 顺序 = {n * 2} 次生成')
    print(f'{"=" * 92}')
    print(f'词面对照臂（G2 的参照）：'
          + '  '.join(f'{k} 朴素 {v[0]:.3f} / 取好 {v[1]:.3f}' for k, v in lex.items())
          + f'   → 闸门线 = {lex_best:.3f} + 0.05 = {lex_best + 0.05:.3f}')

    hdr = (f'\n{"模型":<26} {"截断":>6} {"兑现":>6} | {"宽":>6} {"严":>6} '
           f'{"95%CI(配对)":>17} {"95%CI(种子)":>17} | {"S1":>6} {"S2":>6} {"Δ顺序":>7} {"选槽1":>7}')
    print(hdr)
    print('-' * len(hdr))

    summary = []
    for path in files:
        rows = [r for r in load(path) if r['pair_id'] in keep]
        if not rows:
            continue
        model = rows[0]['model'].split('/')[-1]
        trunc = float(np.mean([r['truncated'] for r in rows]))
        fmt = float(np.mean([r['format_ok'] for r in rows]))
        wide, strict = acc_by_order(rows, False), acc_by_order(rows, True)
        w = float(np.mean(list(wide.values())))
        s = float(np.mean(list(strict.values())))
        # Main readout uses "strict" (format errors count as wrong), matching the decision locked in §3; the wide reading is reported alongside
        per_pair, per_seed = {}, {}
        for r in rows:
            per_pair.setdefault(r['pair_id'], []).append(float(bool(r['picked_strong'])))
            per_seed.setdefault(r['atomic_idx'], []).append(float(bool(r['picked_strong'])))
        lo1, hi1 = boot_ci(per_pair)
        lo2, hi2 = boot_ci(per_seed)
        d_order = abs(strict[ORDERS[0]] - strict[ORDERS[1]])
        # Fraction "picks slot 1", computed once per order (G1's second condition)
        slot1 = {o: float(np.mean([r['picked_slot'] == '1' for r in rows
                                   if r['order'] == o and r['format_ok']] or [np.nan]))
                 for o in ORDERS}
        print(f'{model:<26} {trunc:>6.3f} {fmt:>6.3f} | {w:>6.3f} {s:>6.3f} '
              f'  [{lo1:.3f},{hi1:.3f}]   [{lo2:.3f},{hi2:.3f}] | '
              f'{strict[ORDERS[0]]:>6.3f} {strict[ORDERS[1]]:>6.3f} {d_order:>7.3f} '
              f'{slot1[ORDERS[0]]:>3.2f}/{slot1[ORDERS[1]]:<3.2f}')
        summary.append({'model': model, 'wide': w, 'strict': s, 'fmt': fmt, 'trunc': trunc,
                        'd_order': d_order, 'slot1': slot1, 'ci_pair': (lo1, hi1),
                        'ci_seed': (lo2, hi2)})

    if not summary or gold_only:
        return summary

    # ------------------------------------------------ position-debiasing diagnostic (additive; does not change the main readout)
    # The standard 2AFC reading: under the two orders, does the model point to the **same content**?
    # A position-constant model drops to 0 on this quantity (it always picks the same slot while the content swaps).
    print('\n位置去偏诊断（附加，不参与判定）：内容一致率 = 两种顺序下指向同一条 γ3 的比例；'
          '一致子集准确率 = 在这些组里指对 strong 的比例')
    for path in files:
        rows = [r for r in load(path) if r['pair_id'] in keep]
        if not rows:
            continue
        model = rows[0]['model'].split('/')[-1]
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r['pair_id'], {})[r['order']] = r
        cons, hit = [], []
        for pid, d in by_pair.items():
            if len(d) < 2 or not all(x['format_ok'] for x in d.values()):
                continue
            same = d[ORDERS[0]]['picked_strong'] == d[ORDERS[1]]['picked_strong']
            cons.append(float(same))
            if same:
                hit.append(float(d[ORDERS[0]]['picked_strong']))
        c = float(np.mean(cons)) if cons else float('nan')
        h = float(np.mean(hit)) if hit else float('nan')
        print(f'  {model:<26} 内容一致率 {c:.3f} (n={len(cons)})   '
              f'一致子集准确率 {h:.3f} (n={len(hit)})')

    # ---------------------------------------------------------------- four gates
    print('\n四道闸门：')
    for m in summary:
        bad = m['gate_fail'] = []
        if m['d_order'] > 0.10:
            bad.append(f"G1 顺序差 {m['d_order']:.3f} > 0.10")
        if all(v > 0.70 for v in m['slot1'].values()):
            bad.append(f"G1 两顺序下选槽1 均 > 0.70 ({m['slot1'][ORDERS[0]]:.2f}/"
                       f"{m['slot1'][ORDERS[1]]:.2f})")
        if m['strict'] < lex_best + 0.05:
            bad.append(f"G2 未超词面上限+0.05（{m['strict']:.3f} < {lex_best + 0.05:.3f}）"
                       + ('【注：词面上限已达 1.000，闸门线 >1，**任何**模型都不可能通过'
                          '——这是探针被混淆作废，不是该模型的过失】'
                          if lex_best >= 1.0 else ''))
        if m['fmt'] < 0.8:
            bad.append(f"G4 兑现率 {m['fmt']:.3f} < 0.8，宽口径不进主表")
        print(f"  {m['model']:<26} " + ('✅ 全过' if not bad else '❌ ' + '；'.join(bad)))

    above = sum(m['strict'] > 0.5 for m in summary)
    g3_ok = above in (0, len(summary))
    print(f"\n  G3 跨模型方向一致：{above}/{len(summary)} 个模型 > 0.500 → "
          + ('✅ 一致' if g3_ok else '❌ 不一致（按判据判负，不许挑一个来报）'))

    # ---------------------------------------------------------------- verdict
    # ⚠️ The verdict is made only over models that **passed the gates**. Gate-voided models
    # must not be slotted into tiers — that is exactly the "pick one after the fact to
    # report" pattern, and it was the core reason L1 was judged negative.
    valid = [m for m in summary if not m['gate_fail']] if g3_ok else []
    mean = float(np.mean([m['strict'] for m in summary]))
    print(f'\n（全部模型的严口径均值 = {mean:.3f}，最高 '
          f'{max(summary, key=lambda m: m["strict"])["model"]} '
          f'{max(m["strict"] for m in summary):.3f}——**以下判定只看过闸的模型**）')
    print(f'参照线：随机 0.500 ｜ L5 单题 {L5} ｜ L1 打分轴 {L1} ｜ 天花板 {CEILING} ｜ 上限 1.000')
    if not valid:
        print(f'\n判定：**无有效读数**——{len(summary)}/{len(summary)} 个模型被闸门作废。'
              f'\n      这是"仪器坏了"，不是"真阴性"（CLAUDE.md 要求把两者分开）。'
              f'\n      不得由此宣称 H-L6 成立或不成立；本轮不产出主读数。')
        return summary
    best = max(valid, key=lambda m: m['strict'])
    print(f'过闸模型 {len(valid)}/{len(summary)}：最好 {best["model"]} = {best["strict"]:.3f}')
    v = best['strict']
    if v >= 0.70:
        verdict = '≥0.70 → **成对形态可提取，H-L6 成立，是新发现**'
    elif v > 0.65:
        verdict = '0.65–0.70 → **判据未覆盖区，如实报为不确定，交人裁决**（不硬套档位）'
    elif v > 0.55:
        verdict = '0.55–0.65 → 与 L1/L4/L5 同带，**形态没带来东西**。记录，不建设'
    else:
        verdict = '≤0.55 → **无信号**。"关闭"推到输入形态这一层，L 系列结案'
    print(f'判定（以最好的模型为准，且须过闸门）：{verdict}')
    return summary


def length_confound():
    """The G2 lexical arm gives 1.000 on the 63 pairs; this section re-checks it on the **entire dataset**.

    Reading: γ1 is the scenario-shared, unmanipulated sentence, so it must be a null control
    (AUC ≈ 0.5); if γ1 also carries signal, what is measured is a generic length artifact,
    not a construction trace on γ3.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv'))
    L = df['questions'].str.split('\n')
    df['g1'], df['g3'] = L.str[0], L.str[2]
    df['intent'] = df['dataset_id'].str.split('-').str[-1]
    pon = df[df.modus == 'ponens'].drop_duplicates('dataset_id')

    def boot(y, s, iters=ITERS, seed=SEED):
        y, s = np.asarray(y), np.asarray(s)
        rng = np.random.RandomState(seed)
        gi, ai = np.where(y)[0], np.where(~y)[0]
        out = []
        for _ in range(iters):
            g, a = gi[rng.randint(0, len(gi), len(gi))], ai[rng.randint(0, len(ai), len(ai))]
            out.append(roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(a))], np.r_[s[g], s[a]]))
        return np.percentile(out, [2.5, 97.5])

    print(f'\n{"=" * 92}\n【G2 的词面臂推到全数据集：γ3 长度与金标的关系】\n{"=" * 92}')
    for tag, sub in (('全量 872', pon), ('**lv=5 391（主读数集）**', pon[pon.agreement_lv == 5])):
        y = (sub.ground_truth == 'c').to_numpy()
        for feat, name in ((sub.g3.str.len(), 'γ3 字符数'),
                           (sub.g1.str.len(), 'γ1 字符数（空对照）'),
                           (sub.g3.str.len() - sub.g1.str.len(), 'γ3−γ1 字符差')):
            a = roc_auc_score(y, feat.to_numpy())
            lo, hi = boot(y, feat.to_numpy())
            print(f'  {tag:<24} 金标 REQ ~ {name:<18} AUC {a:.3f}  95%CI [{lo:.3f},{hi:.3f}]')
        yi = (sub.intent == 'strong').to_numpy()
        print(f'  {tag:<24} 设计意图 ~ γ3 字符数      AUC '
              f'{roc_auc_score(yi, sub.g3.str.len().to_numpy()):.3f}')
    print(f'  对照：L1 {L1} / L4 0.562 / L5 {L5} / 天花板 {CEILING}')


def main():
    pairs = load_pairs()
    length_confound()
    files = sorted(glob.glob(os.path.join(OUT, 'pairwise_*.jsonl')))
    files = [f for f in files if os.path.getsize(f) > 0]
    print(f'读入 {len(files)} 个模型的输出：')
    for f in files:
        print('  ', os.path.basename(f))

    report('ponens', pairs, files, tag='【主读数：设计意图标签 × 63 组 × ponens 渲染】')
    report('ponens', pairs, files, gold_only=True,
           tag='【次要读数：金标标签 × 39 组 × ponens 渲染】')
    report('tollens', pairs, files, tag='【稳健性臂（不参与判定）：tollens 渲染】')

    tol_clean = [p for p in pairs if p['modus'] == 'tollens' and not p['g2_identical']]
    print(f'\n注：tollens 臂里有 {len(tol_clean)} 组的 γ2 两侧写法不同'
          f'（{[p["base"] for p in tol_clean]}），那几组在 tollens 上不是真最小对；'
          f'主读数（ponens）不含此问题。')
    n_lv5 = sum(p['lv5_both'] and p['modus'] == 'ponens' for p in pairs)
    print(f'注：双侧 lv=5 只有 {n_lv5} 组、两者兼备 6 组，n 太小，'
          f'按预注册**只作定性参考，不参与判定**，故不单独出表。')


if __name__ == '__main__':
    main()
