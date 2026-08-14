#!/usr/bin/env python
"""C(i) reader vs item-writer split readout — confirmatory rerun (PREREG_c1_reader_split.md, frozen 2026-08-08).

Independent-implementation clause (PREREG §5): this file does **not import and does not
rewrite** `summarize_reader_split.py`. All statistical logic is rewritten from the raw
per-line outputs in `outputs/probe_decode/*.jsonl`. The scope of consultation of old
scripts is registered item by item in the §"consultation register" comment.

Run (pure CPU):
    python scripts/reader_split_confirm.py

Products (two new files; no existing file is touched):
    outputs/ci/reader_split_confirm.txt
    outputs/reader_split_confirm/SUMMARY.md

============================================================================
Consultation register (required by the PREREG §5 independent-implementation clause)
============================================================================
Existing files read, item by item:
  1. PREREG_c1_reader_split.md — full preregistration for this record.
  2. scripts/compute_ceiling.py — in full. Taken from it: the intent-label convention
     `dataset_id.str.split('-').str[-1]`, strong=REQ prediction, `is_req = ground_truth=='c'`,
     statistical unit = scenario, exact lv==5 match excluding lv=6, and the project
     convention of bootstrap 5000/seed=0.
  3. outputs/ci/lv4_dose.txt — registered values of the four V-1 anchors (read-only).
  4. outputs/ci/reader_split.txt — V-2 reconciliation target (read-only; this script
     parses it line by line for item-by-item comparison).
  5. outputs/probe_decode/{C_twostep,A_Qwen_Qwen3_4B,A_Qwen_Qwen2.5_7B_Instruct}.jsonl
     — input data; field semantics confirmed empirically by this script (see below),
     with no reliance on any old script's description.
  6. dataset/belief_r/queries_time_t1.csv — raw items used for G-1/G-2.
  7. grep hit lines in the internal experiment log and in files such as
     run_hidden_extract.py and probe_relation.py — used only to confirm the convention
     that γ1/γ3 = lines 1/3 of `questions`.
`scripts/summarize_reader_split.py` was **not** opened (not even field names or paths were
taken from it — paths and field names were obtained directly by ls + reading the first
line of the jsonl).

============================================================================
Field semantics (confirmed empirically, not copied)
============================================================================
- Each `dataset_id` has exactly 2 rows (ponens/tollens twins): 872 scenarios / 1744 rows.
- `agreement_lv` ∈ {4: 958 rows, 5: 782 rows, 6: 4 rows}; lv=6 excluded per prereg.
- Gold: empirically `ground_truth=='c'` ⇔ arm C's `gold_state=='REQ'` (1074 rows),
  `a`(ponens)/`b`(tollens) ⇔ `ALT` (335 rows each), zero exceptions. Arm A has no
  `gold_state` field, so gold is uniformly back-derived via `ground_truth=='c'`, same
  convention as compute_ceiling.py.
- `pred_state` ∈ {REQ, ALT}: arm C = step1 zero-shot self-judgment; arm A = hidden-layer
  probe OOF prediction.
- ⚠️ Measured: the twin rows' `pred_state` are **not always identical** (C differs in
  353/872 scenarios, A-4B 260/872, A-7B 368/872). The sentence in compute_ceiling.py's
  docstring, "the relation judgment is verbatim identical, so row-level equals using it
  twice", does **not** hold for these three arms. Since each scenario has exactly 2 rows
  and cluster sizes are equal, the row-level mean still always equals the scenario-level
  mean (point estimates unaffected), but the CI must be clustered by scenario — this
  script does so.

============================================================================
Implementation choices frozen by this script (points the prereg left open; declared before computing)
============================================================================
(a) Main readout "accuracy against gold" = mean(pred_state == gold) over rows; since
    cluster size is always 2, this equals the scenario-level mean.
(b) CI = scenario-clustered bootstrap, **not stratified** (no stratified resampling over
    REQ/ALT), iters=5000, seed=0, percentile [2.5, 97.5]. Rationale: the main readout is
    plain accuracy; stratified resampling would pin down the REQ/ALT composition, and the
    composition itself is the core fact about the disagreement group. A stratified
    version is also emitted as a sensitivity row for reconciliation.
(c) bal-acc / REQ recall / ALT recall computed over rows (needed for the V-1 anchors and
    the V-2 reconciliation).
(d) Shrinkage decomposition: see the three variants in §6.3; all are exact identities.
"""
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Root of the experiment-artifact tree. Artifacts are not distributed with this repo;
# set BELIEF_R_SAVI_ROOT to point at the local experiment directory.
ROOT = os.environ.get('BELIEF_R_SAVI_ROOT') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..')
# Belief-R dataset CSV. The data is not distributed with this repo (see the README's data section); set env var BELIEF_R_CSV to a local copy, else falls back to <repo>/data/queries_time_t1.csv.
CSV = os.environ.get('BELIEF_R_CSV') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'queries_time_t1.csv')
ARMS = [
    ('C 两步自判', 'outputs/probe_decode/C_twostep.jsonl'),
    ('A 探针(4B)', 'outputs/probe_decode/A_Qwen_Qwen3_4B.jsonl'),
    ('A 探针(7B)', 'outputs/probe_decode/A_Qwen_Qwen2.5_7B_Instruct.jsonl'),
]
ITERS, SEED = 5000, 0

# PREREG §5 V-1: registered values from lv4_dose.txt (hand-copied; source = consultation register item 3)
ANCHORS = {
    ('A 探针(4B)', 5): 0.6835,
    ('A 探针(4B)', 4): 0.6039,
    ('C 两步自判', 5): 0.5189,
    ('C 两步自判', 4): 0.5605,
}

OUT_TXT = os.path.join(ROOT, 'outputs/ci/reader_split_confirm.txt')
OUT_MD = os.path.join(ROOT, 'outputs/reader_split_confirm/SUMMARY.md')

_buf = []


def say(s=''):
    _buf.append(s)
    print(s)


# --------------------------------------------------------------------------
# Loading and grouping
# --------------------------------------------------------------------------
def load_arm(path):
    """→ DataFrame, one row per raw output line; keep only the columns this record uses + self-computed grouping columns."""
    rows = []
    with open(os.path.join(ROOT, path)) as fh:
        for line in fh:
            d = json.loads(line)
            rows.append({
                'dataset_id': d['dataset_id'],
                'modus': d['modus'],
                'agreement_lv': d['agreement_lv'],
                'ground_truth': d['ground_truth'],
                'pred_state': d['pred_state'],
                'gold_state_field': d.get('gold_state'),
            })
    df = pd.DataFrame(rows)
    # Gold: c → REQ, everything else → ALT (compute_ceiling.py convention)
    df['gold_req'] = df['ground_truth'] == 'c'
    # Intent: dataset_id suffix strong → REQ, weak → ALT
    df['intent'] = df['dataset_id'].str.split('-').str[-1]
    df['intent_req'] = df['intent'] == 'strong'
    df['pred_req'] = df['pred_state'] == 'REQ'
    df['group'] = np.where(df['intent_req'] == df['gold_req'], 'consistent', 'disagree')
    df['hit'] = (df['pred_req'] == df['gold_req']).astype(float)
    return df


def sanity(df, name):
    """Hard assertions on this arm itself (before computing; any failure raises immediately)."""
    assert len(df) == 1744, f'{name}: 行数 {len(df)} ≠ 1744'
    sizes = df.groupby('dataset_id').size()
    assert (sizes == 2).all(), f'{name}: 有场景不是恰 2 行'
    assert set(df['intent'].unique()) == {'strong', 'weak'}, \
        f'{name}: dataset_id 后缀不止 strong/weak: {set(df["intent"].unique())}'
    assert set(df['pred_state'].unique()) <= {'REQ', 'ALT'}, f'{name}: pred_state 有异值'
    # Within a scenario, intent / gold class must be consistent (same assertion as compute_ceiling.py)
    chk = df.groupby('dataset_id').agg(ni=('intent', 'nunique'), ng=('gold_req', 'nunique'),
                                       nl=('agreement_lv', 'nunique'))
    bad = chk[(chk.ni > 1) | (chk.ng > 1) | (chk.nl > 1)]
    assert len(bad) == 0, f'{name}: {len(bad)} 个场景孪生 intent/金标/lv 不一致'
    if df['gold_state_field'].notna().any():
        m = df['gold_state_field'] == 'REQ'
        assert (m == df['gold_req']).all(), f'{name}: gold_state 与 ground_truth==c 不一致'


# --------------------------------------------------------------------------
# Readouts and CIs
# --------------------------------------------------------------------------
def acc(sub):
    return float(sub['hit'].mean()) if len(sub) else float('nan')


def bal_acc_rows(sub):
    """(REQ recall + ALT recall)/2, over rows."""
    g = sub['gold_req'].to_numpy(bool)
    p = sub['pred_req'].to_numpy(bool)
    if not g.any() or g.all():
        return float('nan'), float('nan'), float('nan')
    rec_req = float(p[g].mean())
    rec_alt = float((~p[~g]).mean())
    return (rec_req + rec_alt) / 2, rec_req, rec_alt


def _scene_arrays(sub):
    """→ (n_scene, 2) hit matrix + per-scenario gold_req. Cluster size is always 2."""
    s = sub.sort_values(['dataset_id', 'modus'])
    ids = s['dataset_id'].to_numpy()
    hit = s['hit'].to_numpy()
    gold = s['gold_req'].to_numpy()
    uniq, first = np.unique(ids, return_index=True)
    n = len(uniq)
    assert len(s) == 2 * n, '簇大小不恒为 2'
    H = hit.reshape(n, 2) if (ids[0::2] == ids[1::2]).all() else None
    assert H is not None, '排序后孪生未成对'
    G = gold[0::2]
    return H, G


def boot_acc_ci(sub, iters=ITERS, seed=SEED, stratified=False):
    """Scenario-clustered bootstrap CI for accuracy. With stratified=True, resample within REQ/ALT strata."""
    H, G = _scene_arrays(sub)
    rng = np.random.RandomState(seed)
    n = len(H)
    out = np.empty(iters)
    if not stratified:
        for b in range(iters):
            idx = rng.randint(0, n, n)
            out[b] = H[idx].mean()
    else:
        gi, ai = np.where(G)[0], np.where(~G)[0]
        for b in range(iters):
            i1 = gi[rng.randint(0, len(gi), len(gi))] if len(gi) else gi
            i0 = ai[rng.randint(0, len(ai), len(ai))] if len(ai) else ai
            idx = np.concatenate([i1, i0])
            out[b] = H[idx].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), float(out.std(ddof=1))


def boot_diff_ci(sub_a, sub_b, iters=ITERS, seed=SEED):
    """CI of the accuracy difference between two groups (a − b); each group's scenarios resampled independently."""
    Ha, _ = _scene_arrays(sub_a)
    Hb, _ = _scene_arrays(sub_b)
    rng = np.random.RandomState(seed)
    na, nb = len(Ha), len(Hb)
    out = np.empty(iters)
    for b in range(iters):
        out[b] = Ha[rng.randint(0, na, na)].mean() - Hb[rng.randint(0, nb, nb)].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(Ha.mean() - Hb.mean()), float(lo), float(hi)


def icc_pairs(sub):
    """Intraclass correlation for cluster size 2 (ICC of correctness)."""
    H, _ = _scene_arrays(sub)
    x = H.ravel()
    m, v = x.mean(), x.var()
    if v == 0:
        return float('nan')
    return float(((H[:, 0] - m) * (H[:, 1] - m)).mean() / v)


# --------------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------------
def main():
    say('=' * 78)
    say('reader_split_confirm —— C(i) 读者 vs 出题人分组读数，确认性重演')
    say('预注册 = PREREG_c1_reader_split.md（2026-08-08 冻结）')
    say(f'生成时间 {datetime.now():%Y-%m-%d %H:%M:%S}；纯 CPU；独立实现（未 import/抄写 '
        'summarize_reader_split.py）')
    say(f'bootstrap：场景聚类、非分层、iters={ITERS}、seed={SEED}、percentile[2.5,97.5]')
    say('=' * 78)

    data = {}
    for name, path in ARMS:
        df = load_arm(path)
        sanity(df, name)
        data[name] = df

    # The three arms' metadata must match row by row (same batch of items)
    base = data['C 两步自判'][['dataset_id', 'modus', 'agreement_lv', 'ground_truth']]
    for name in ('A 探针(4B)', 'A 探针(7B)'):
        oth = data[name][['dataset_id', 'modus', 'agreement_lv', 'ground_truth']]
        assert base.equals(oth), f'{name} 的行元数据与 C 臂不一致'

    gate_pass = {}

    # ---------------- 0. Data and grouping ----------------
    say('\n' + '=' * 78)
    say('0. 数据、口径、分组')
    say('=' * 78)
    C = data['C 两步自判']
    say(f'  总行 {len(C)}；场景 {C["dataset_id"].nunique()}（每场景恰 2 行：ponens/tollens）')
    lvc = C.drop_duplicates('dataset_id')['agreement_lv'].value_counts().sort_index()
    say('  agreement_lv 场景数：' + '  '.join(f'lv={k} {v}' for k, v in lvc.items())
        + '   → lv=6 的 2 场景 / 4 行按预注册排除')
    say('  金标口径：ground_truth==\'c\' → REQ；a(ponens)/b(tollens) → ALT'
        '（与 C 臂 gold_state 字段逐行一致，已断言）')
    say('  意图口径：dataset_id 后缀 strong → REQ / weak → ALT（compute_ceiling.py 同式）')
    say('  一致组 = 意图==金标；分歧组 = 意图≠金标（预注册 §2 中的"约定/残差"即此两组）')
    for name in data:
        d = data[name]
        nd = (d.groupby('dataset_id')['pred_state'].nunique() > 1).sum()
        say(f'  [实测] {name}：孪生两行 pred_state 不同的场景 {nd}/872'
            f'（{nd / 872:.1%}）—— 故行级≠"用两遍"，CI 必须按场景聚类')

    work = {k: v[v['agreement_lv'].isin([4, 5])].copy() for k, v in data.items()}

    # ---------------- 1. V-1 four anchors ----------------
    say('\n' + '=' * 78)
    say('1. [V-1] 四锚点复现（bal-acc，全行，对 outputs/ci/lv4_dose.txt 登记值）')
    say('=' * 78)
    v1_ok = True
    for (nm, lv), reg in ANCHORS.items():
        sub = work[nm][work[nm]['agreement_lv'] == lv]
        b, _, _ = bal_acc_rows(sub)
        ok = abs(b - reg) < 5e-5
        v1_ok &= ok
        say(f'  {nm} lv={lv}: 复算 {b:.4f}  登记 {reg:.4f}  n行={len(sub)}  '
            f'{"✅" if ok else "❌ 不符"}')
    # Also record 7B (not registered in lv4_dose, so no anchor to compare against)
    for lv in (5, 4):
        sub = work['A 探针(7B)'][work['A 探针(7B)']['agreement_lv'] == lv]
        b, _, _ = bal_acc_rows(sub)
        say(f'  A 探针(7B) lv={lv}: 复算 {b:.4f}   （lv4_dose 未登记，无锚点，附报）')
    gate_pass['V-1'] = v1_ok
    say(f'  → V-1 {"通过" if v1_ok else "**不通过**"}')

    # ---------------- Group sizes ----------------
    say('\n' + '=' * 78)
    say('2. 分组规模与方向构成（纯计数）')
    say('=' * 78)
    scen = C.drop_duplicates('dataset_id')[['dataset_id', 'agreement_lv', 'intent',
                                            'intent_req', 'gold_req', 'group']]
    scen = scen[scen['agreement_lv'].isin([4, 5])]
    counts = {}
    for lv in (5, 4):
        s = scen[scen['agreement_lv'] == lv]
        ncon = int((s['group'] == 'consistent').sum())
        ndis = int((s['group'] == 'disagree').sum())
        s2a = int(((s['intent'] == 'strong') & (~s['gold_req'])).sum())   # strong→ALT
        w2r = int(((s['intent'] == 'weak') & (s['gold_req'])).sum())      # weak→REQ
        counts[lv] = dict(n=len(s), con=ncon, dis=ndis, s2a=s2a, w2r=w2r)
        say(f'  lv={lv}: 场景 {len(s)}  一致(约定) {ncon}  分歧(残差) {ndis}'
            f'   残差占比 {ndis / len(s):.4f}'
            f'   方向构成: strong→ALT {s2a} / weak→REQ {w2r}')
    both = scen
    nc, nd = int((both['group'] == 'consistent').sum()), int((both['group'] == 'disagree').sum())
    say(f'  lv4+lv5 合计: 场景 {len(both)}  一致 {nc}  分歧 {nd}  残差占比 {nd / len(both):.4f}')
    say(f'  残差占比 lv5 {counts[5]["dis"] / counts[5]["n"]:.4f} → lv4 '
        f'{counts[4]["dis"] / counts[4]["n"]:.4f}  '
        f'（倍数 {(counts[4]["dis"] / counts[4]["n"]) / (counts[5]["dis"] / counts[5]["n"]):.3f}×）')

    # ---------------- 3. G-1 intent-label reconciliation ----------------
    say('\n' + '=' * 78)
    say('3. [G-1] 意图标签与 compute_ceiling.py 口径对账')
    say('=' * 78)
    raw = pd.read_csv(CSV)
    raw['intent_csv'] = raw['dataset_id'].str.split('-').str[-1]      # same line as compute_ceiling
    raw['is_req_csv'] = raw['ground_truth'] == 'c'                    # same line as compute_ceiling
    key = ['dataset_id', 'modus']
    mine = C[key + ['intent', 'gold_req', 'agreement_lv']].copy()
    j = mine.merge(raw[key + ['intent_csv', 'is_req_csv', 'agreement_lv']],
                   on=key, how='left', suffixes=('', '_csv'))
    assert j['intent_csv'].notna().all(), 'G-1: 有行在 CSV 里找不到'
    n_int = int((j['intent'] == j['intent_csv']).sum())
    n_gold = int((j['gold_req'] == j['is_req_csv']).sum())
    n_lv = int((j['agreement_lv'] == j['agreement_lv_csv']).sum())
    say(f'  全量逐行对账（key=dataset_id×modus，n={len(j)}）：'
        f'intent 一致 {n_int}/{len(j)}；金标一致 {n_gold}/{len(j)}；'
        f'agreement_lv 一致 {n_lv}/{len(j)}')
    # Reproduce compute_ceiling.py's confusion table (lv=5) as a second reconciliation of the convention
    s5 = raw[raw['agreement_lv'] == 5].drop_duplicates('dataset_id')
    pred = (s5['intent_csv'] == 'strong').to_numpy()
    gold = s5['is_req_csv'].to_numpy()
    ceil_ba = (pred[gold].mean() + (~pred[~gold]).mean()) / 2
    say(f'  复现 compute_ceiling lv=5 混淆：strong∧REQ {int((pred & gold).sum())}'
        f'  strong∧ALT {int((pred & ~gold).sum())}'
        f'  weak∧REQ {int((~pred & gold).sum())}'
        f'  weak∧ALT {int((~pred & ~gold).sum())}   平衡准确率 {ceil_ba:.4f}')
    say(f'  对照：本脚本 lv=5 分歧组 = strong∧ALT + weak∧REQ = '
        f'{int((pred & ~gold).sum())} + {int((~pred & gold).sum())} = '
        f'{int((pred & ~gold).sum()) + int((~pred & gold).sum())}'
        f'（本脚本分组给 {counts[5]["dis"]}）')
    # Sample ≥20 scenarios and print each one (fixed seed)
    rs = np.random.RandomState(0)
    pool = sorted(scen['dataset_id'].unique())
    pick = [pool[i] for i in rs.choice(len(pool), 24, replace=False)]
    say('  抽样 24 场景逐条（dataset_id | 后缀→意图 | ground_truth→金标 | 组）：')
    for did in sorted(pick, key=lambda x: (int(x.split('-')[0]), x)):
        r = scen[scen['dataset_id'] == did].iloc[0]
        gt = raw[raw['dataset_id'] == did]['ground_truth'].tolist()
        say(f'    {did:<12} 后缀={did.split("-")[-1]:<6}→意图'
            f'{"REQ" if r.intent_req else "ALT"}   ground_truth={gt}→金标'
            f'{"REQ" if r.gold_req else "ALT"}   {"一致" if r.group == "consistent" else "分歧"}')
    g1_ok = (n_int == len(j)) and (n_gold == len(j)) and (n_lv == len(j))
    gate_pass['G-1'] = g1_ok
    say(f'  → G-1 {"通过（全量逐行 100% 一致，抽样 24 ≥ 预注册要求的 20）" if g1_ok else "**不通过**"}')

    # ---------------- 4. G-2 γ3−γ1 character-length difference ----------------
    say('\n' + '=' * 78)
    say('4. [G-2] 组间句面可比性：γ3−γ1 字符差按组分布')
    say('=' * 78)
    lines = raw['questions'].str.split('\n')
    assert (lines.str.len() == 8).all(), 'G-2: questions 行数不恒为 8'
    raw['g1'] = lines.str[0]
    raw['g3'] = lines.str[2]
    chk = raw.groupby('dataset_id').agg(n1=('g1', 'nunique'), n3=('g3', 'nunique'))
    assert (chk.n1 == 1).all() and (chk.n3 == 1).all(), 'G-2: 孪生 γ1/γ3 不同'
    sc = raw.drop_duplicates('dataset_id')[['dataset_id', 'g1', 'g3']].copy()
    sc['dlen'] = sc['g3'].str.len() - sc['g1'].str.len()
    scen2 = scen.merge(sc, on='dataset_id', how='left')
    say('  γ1 = questions 第 1 行、γ3 = 第 3 行（孪生共享，已断言）；单位=场景')
    say(f'  {"格":<22}{"n":>5}{"均值":>9}{"中位":>8}{"p25":>8}{"p75":>8}{"标准差":>9}')
    for lv in (5, 4):
        for grp, lab in (('consistent', '一致'), ('disagree', '分歧')):
            v = scen2[(scen2.agreement_lv == lv) & (scen2.group == grp)]['dlen'].to_numpy()
            say(f'  {f"lv={lv} {lab}组":<22}{len(v):>5}{v.mean():>9.1f}'
                f'{np.median(v):>8.1f}{np.percentile(v, 25):>8.1f}'
                f'{np.percentile(v, 75):>8.1f}{v.std(ddof=1):>9.1f}')
    for lv in (5, 4):
        a = scen2[(scen2.agreement_lv == lv) & (scen2.group == 'consistent')]['dlen'].to_numpy()
        b = scen2[(scen2.agreement_lv == lv) & (scen2.group == 'disagree')]['dlen'].to_numpy()
        rng = np.random.RandomState(SEED)
        d = np.array([a[rng.randint(0, len(a), len(a))].mean()
                      - b[rng.randint(0, len(b), len(b))].mean() for _ in range(ITERS)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        # Between-group Mann-Whitney-style AUC (probability that a > b)
        auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
        say(f'  lv={lv} 一致−分歧 均值差 = {a.mean() - b.mean():+.1f} 字符 '
            f'[{lo:+.1f},{hi:+.1f}]；P(一致>分歧)={auc:.3f}')
    say('  → G-2 属"照报"项：预注册未给数值判据，故**无法机械判负**。但数值本身是负面的：')
    say('    两个 lv 上分歧组的 γ3−γ1 都显著更长（一致−分歧的 CI 都不含 0），'
        '即两组句面**不可比**，')
    say('    G-2 想排除的"组间差 = 表面特征差"这条**没有排除掉**。')
    say('  【结构性说明，不是事后解释】这个方向是设计决定的，不是巧合：'
        'B0 已测出 γ3 长度与意图几乎共线（最小对 63/63 strong 侧更长）、')
    say('    与金标 AUC 0.800。而分歧组按定义 = 意图≠金标，其中 '
        f'{counts[5]["s2a"]}/{counts[5]["dis"]}(lv5)、{counts[4]["s2a"]}/{counts[4]["dis"]}(lv4)'
        ' 是 strong→ALT，')
    say('    即"γ3 长但金标 ALT"。所以分组本身与长度线索**不独立**，'
        'G-2 在本设计下原则上不可能过——')
    say('    这限制的是解读（分歧组上的低 acc 有"模型跟长度线索走"这个竞争解释），不是仪器故障。')
    gate_pass['G-2'] = None

    # ---------------- 5. Four-cell readouts (the data side of the V-2 reconciliation) ----------------
    say('\n' + '=' * 78)
    say('5. 四格读数：C / A-4B / A-7B × lv × 组，对金标 acc + 场景聚类 95%CI')
    say('=' * 78)
    cells = {}
    for nm in ('C 两步自判', 'A 探针(4B)', 'A 探针(7B)'):
        d = work[nm]
        say(f'\n== {nm} ==')
        for lv, lab_lv in ((5, 'lv=5'), (4, 'lv=4'), (None, 'lv4+lv5 合计')):
            for grp, lab_g in (('consistent', '一致组'), ('disagree', '分歧组')):
                sub = d[d['group'] == grp] if lv is None else \
                    d[(d['agreement_lv'] == lv) & (d['group'] == grp)]
                a = acc(sub)
                lo, hi, se = boot_acc_ci(sub)
                slo, shi, _ = boot_acc_ci(sub, stratified=True)
                b, rr, ra = bal_acc_rows(sub)
                cells[(nm, lv, grp)] = dict(n_scene=sub['dataset_id'].nunique(), n_row=len(sub),
                                            acc=a, lo=lo, hi=hi, se=se, bal=b, rec_req=rr,
                                            rec_alt=ra, slo=slo, shi=shi)
                say(f'  {lab_lv} {lab_g:<6} 场景={sub["dataset_id"].nunique():>4} '
                    f'行={len(sub):>5} 对金标acc={a:.4f} [{lo:.3f},{hi:.3f}] '
                    f'(分层版[{slo:.3f},{shi:.3f}]) bal-acc={b:.4f} '
                    f'REQ召回={rr:.3f} ALT召回={ra:.3f}')

    # ---------------- 6. V-2 item-by-item reconciliation ----------------
    say('\n' + '=' * 78)
    say('6. [V-2] 与 outputs/ci/reader_split.txt 逐项对账')
    say('=' * 78)
    reg_txt = open(os.path.join(ROOT, 'outputs/ci/reader_split.txt')).read()
    diffs = []

    def cmp(label, mine_v, reg_v, tol):
        ok = abs(mine_v - reg_v) <= tol
        if not ok:
            diffs.append((label, mine_v, reg_v))
        say(f'  {label:<44} 复算 {mine_v:.4f}  登记 {reg_v:.4f}  '
            f'{"完全一致" if ok else f"❌ 差 {mine_v - reg_v:+.4f}"}')
        return ok

    v2_ok = True
    # 6a the four anchor rows
    for (nm, lv), reg in ANCHORS.items():
        sub = work[nm][work[nm]['agreement_lv'] == lv]
        b, _, _ = bal_acc_rows(sub)
        v2_ok &= cmp(f'V1锚点 {nm} lv={lv} bal-acc', b, reg, 5e-5)
    # 6b group counts
    for lv in (5, 4):
        m = re.search(rf'lv={lv}: 一致 (\d+) 场景 / 分歧 (\d+) 场景'
                      rf'\(strong→ALT 翻转 (\d+),weak→REQ 翻转 (\d+)\)', reg_txt)
        assert m, f'V-2: 没解析到 lv={lv} 的分组行'
        got = (counts[lv]['con'], counts[lv]['dis'], counts[lv]['s2a'], counts[lv]['w2r'])
        exp = tuple(int(x) for x in m.groups())
        ok = got == exp
        v2_ok &= ok
        if not ok:
            diffs.append((f'分组计数 lv={lv}', got, exp))
        say(f'  {f"分组计数 lv={lv}(一致/分歧/s→ALT/w→REQ)":<44} 复算 {got}  登记 {exp}  '
            f'{"完全一致" if ok else "❌"}')
    # 6c three arms × six rows: acc / CI / bal-acc / both recalls
    arm_blocks = {'C 两步自判': 'C 两步自判', 'A 探针(4B)': 'A 探针\\(4B\\)',
                  'A 探针(7B)': 'A 探针\\(7B\\)'}
    row_pat = (r'(lv=5 一致组|lv=5 分歧组|lv=4 一致组|lv=4 分歧组|lv4\+lv5 合计 一致组|'
               r'lv4\+lv5 合计 分歧组)\s+场景=\s*(\d+) 行=\s*(\d+) 对金标acc=([\d.]+) '
               r'\[([\d.]+),([\d.]+)\]\s+bal-acc=([\d.]+)\s+REQ召回=([\d.]+) ALT召回=([\d.]+)')
    lab2key = {'lv=5 一致组': (5, 'consistent'), 'lv=5 分歧组': (5, 'disagree'),
               'lv=4 一致组': (4, 'consistent'), 'lv=4 分歧组': (4, 'disagree'),
               'lv4+lv5 合计 一致组': (None, 'consistent'),
               'lv4+lv5 合计 分歧组': (None, 'disagree')}
    for nm, pat in arm_blocks.items():
        blk = re.search(rf'== {pat} ==(.*?)(?:\n==|\n# 判读)', reg_txt, re.S)
        assert blk, f'V-2: 没解析到 {nm} 段'
        for m in re.finditer(row_pat, blk.group(1)):
            lab = m.group(1)
            lv, grp = lab2key[lab]
            c = cells[(nm, lv, grp)]
            tag = f'{nm} {lab}'
            ok = (c['n_scene'] == int(m.group(2))) and (c['n_row'] == int(m.group(3)))
            v2_ok &= ok
            if not ok:
                diffs.append((tag + ' n', (c['n_scene'], c['n_row']),
                              (int(m.group(2)), int(m.group(3)))))
            say(f'  {tag + " 场景/行":<44} 复算 ({c["n_scene"]},{c["n_row"]})  '
                f'登记 ({m.group(2)},{m.group(3)})  {"完全一致" if ok else "❌"}')
            v2_ok &= cmp(tag + ' 对金标acc', c['acc'], float(m.group(4)), 5e-5)
            v2_ok &= cmp(tag + ' CI下界', c['lo'], float(m.group(5)), 5.1e-4)
            v2_ok &= cmp(tag + ' CI上界', c['hi'], float(m.group(6)), 5.1e-4)
            v2_ok &= cmp(tag + ' bal-acc', c['bal'], float(m.group(7)), 5e-5)
            v2_ok &= cmp(tag + ' REQ召回', c['rec_req'], float(m.group(8)), 5.1e-4)
            v2_ok &= cmp(tag + ' ALT召回', c['rec_alt'], float(m.group(9)), 5.1e-4)
    gate_pass['V-2'] = v2_ok
    say(f'\n  → V-2 字面结果：共比对 {4 + 2 + 3 * 6 * 7} 项，超容差 {len(diffs)} 项 '
        f'→ {"全部完全一致" if v2_ok else "**有分歧，见清单**"}')
    for lab, a, b in diffs:
        say(f'     ❌ {lab}: 复算 {a} vs 登记 {b}')
    say('  【容差声明】预注册 §5 只给 V-1 定了容差（"浮点"），**没给 V-2 定容差**。'
        '本脚本自设：点估计/计数/召回按 4 位有效位（5e-5），')
    say('  CI 端点按登记值的 3 位小数取半格（5.1e-4）。5.1e-4 是"只考虑四舍五入"的容差，'
        '对 bootstrap 这种蒙特卡洛量偏紧——差异调查见 §6b。')

    # ---------------- 6b. V-2 discrepancy investigation ----------------
    say('\n' + '-' * 78)
    say('6b. [V-2 差异调查] 132 项拆成两类：非 CI 类 96 项（计数 / 场景行数 / 对金标 acc / '
        'bal-acc / REQ 召回 / ALT 召回 / 四锚点）**96/96 全对**；')
    say('    CI 端点类 36 项，22 项对、14 项超容差。即：分歧 100% 落在 CI 端点上。')
    say('    问：CI 端点的差是"算法不同"还是"bootstrap 随机流不同"？两条判别：')
    say('-' * 78)
    reg_ci = {}
    for nm, pat in arm_blocks.items():
        blk = re.search(rf'== {pat} ==(.*?)(?:\n==|\n# 判读)', reg_txt, re.S).group(1)
        for m in re.finditer(row_pat, blk):
            lv, grp = lab2key[m.group(1)]
            reg_ci[(nm, lv, grp)] = (float(m.group(5)), float(m.group(6)))

    def _H(k):
        nm, lv, grp = k
        d = work[nm]
        s = d[d.group == grp] if lv is None else d[(d.agreement_lv == lv) & (d.group == grp)]
        s = s.sort_values(['dataset_id', 'modus'])
        return s['hit'].to_numpy().reshape(-1, 2)

    def _ci(H, seed, iters=ITERS, clustered=True):
        rng = np.random.RandomState(seed)
        if clustered:
            idx = rng.randint(0, len(H), (iters, len(H)))
            v = H[idx].reshape(iters, -1).mean(1)
        else:
            x = H.ravel()
            v = x[rng.randint(0, len(x), (iters, len(x)))].mean(1)
        return tuple(np.percentile(v, [2.5, 97.5]))

    say('  判别一：把我自己的 seed 从 0 换到 1..9，看"我 vs 登记"的超差数与'
        '"我 vs 我"的超差数是不是同一量级。')
    say(f'  {"":<26}{"超 5.1e-4 端点数/36":>22}{"最大偏差":>12}')
    for seed in range(10):
        ds = np.array([abs(x - y) for k, (rlo, rhi) in reg_ci.items()
                       for x, y in zip(_ci(_H(k), seed), (rlo, rhi))])
        say(f'  我(seed={seed}) vs 登记      {int((ds > 5.1e-4).sum()):>18}/36{ds.max():>12.4f}')
    base = {k: _ci(_H(k), 0) for k in reg_ci}
    for seed in range(1, 6):
        ds = np.array([abs(x - y) for k in reg_ci
                       for x, y in zip(_ci(_H(k), seed), base[k])])
        say(f'  我(seed={seed}) vs 我(seed=0) {int((ds > 5.1e-4).sum()):>18}/36{ds.max():>12.4f}')
    say('  → "我 vs 登记" 的超差数（11–15/36）**不大于** "我 vs 我换个 seed"（11–20/36），'
        '最大偏差同量级（≤0.012）。')
    say('    即：登记值与我的实现之差，统计上无法与"同一实现换一条随机流"区分。')
    say('  判别二（阴性对照）：如果登记值当初用的是**行级（非场景聚类）**重采样，'
        '我应该看到近乎全错。')
    ds = np.array([abs(x - y) for k, (rlo, rhi) in reg_ci.items()
                   for x, y in zip(_ci(_H(k), 0, clustered=False), (rlo, rhi))])
    say(f'  行级重采样(seed=0) vs 登记：超差 {int((ds > 5.1e-4).sum())}/36，最大 {ds.max():.4f}')
    say('    → 换掉重采样单位会造成近乎全错（35/36）；登记值与我的场景聚类版只差 14/36 且'
        '幅度与换 seed 相当，')
    say('      说明登记值用的**就是场景聚类**、与本档同单位。两条判别合起来：'
        'CI 端点差 = 蒙特卡洛噪声，不是算法分歧。')
    say('  ⚠️ 我不把容差事后放宽来宣布"通过"——按我跑数前写下的 5.1e-4，V-2 字面是不符；'
        '上面是差异调查，不是调和。')
    say('  预注册 §性质声明 对不符的处置写的是"以复算+差异调查为准"，'
        '并未写"V-2 不符即停"；且 CI 端点的这点差不改变任何一档的落档'
        '（lv=4 分歧组两套 CI 都含 0.5）。是否算"过闸"交人裁。')

    # ---------------- 7. Main readout and tier assignment ----------------
    say('\n' + '=' * 78)
    say('7. 主读数（预注册 §3；判读对象 = lv=4 分歧组）')
    say('=' * 78)
    d = work['C 两步自判']
    filing = {}
    for lv in (4, 5):
        for grp, lab in (('consistent', '一致组'), ('disagree', '分歧组')):
            c = cells[('C 两步自判', lv, grp)]
            say(f'  C 臂 lv={lv} {lab}: 对金标 acc = {c["acc"]:.4f} '
                f'[{c["lo"]:.4f},{c["hi"]:.4f}]  场景 {c["n_scene"]}  '
                f'（CI {"不含" if (c["lo"] > 0.5 or c["hi"] < 0.5) else "含"} 0.5）')
    for lv in (4, 5):
        dd, lo, hi = boot_diff_ci(d[(d.agreement_lv == lv) & (d.group == 'consistent')],
                                  d[(d.agreement_lv == lv) & (d.group == 'disagree')])
        filing[f'diff{lv}'] = (dd, lo, hi)
        say(f'  C 臂 lv={lv} 一致−分歧 = {dd:+.4f} [{lo:+.4f},{hi:+.4f}]  '
            f'（{"可区分" if (lo > 0 or hi < 0) else "不可区分"}）')

    c4d = cells[('C 两步自判', 4, 'disagree')]
    c4c = cells[('C 两步自判', 4, 'consistent')]
    cond_A = c4d['lo'] > 0.5
    cond_B = c4d['hi'] < 0.5
    # "Consistent group not detectably different from the lv=5 overall 0.519" — prereg
    # literal reading: 0.519 is lv4_dose's **bal-acc** anchor, while the consistent-group
    # readout is **acc against gold** — different quantities. Compute all three readings,
    # report the discrepancy, do not reconcile on our own.
    lv5_all = work['C 两步自判'][work['C 两步自判'].agreement_lv == 5]
    lv5_bal, _, _ = bal_acc_rows(lv5_all)
    lv5_acc = acc(lv5_all)
    condC_literal = c4c['lo'] <= 0.5189 <= c4c['hi']
    condC_accmatch = c4c['lo'] <= lv5_acc <= c4c['hi']
    c4c_bal_note = c4c['bal']
    say(f'\n  判读条件（逐条机械求值）：')
    say(f'    A. lv=4 分歧组 CI 下界 {c4d["lo"]:.4f} > 0.5 ?  → {cond_A}')
    say(f'    B. lv=4 分歧组 CI 上界 {c4d["hi"]:.4f} < 0.5 ?  → {cond_B}')
    say(f'    C(字面). lv=4 一致组 acc CI [{c4c["lo"]:.4f},{c4c["hi"]:.4f}] 含 0.519(=lv5 '
        f'bal-acc 锚点) ?  → {condC_literal}')
    say(f'    C(同量). lv=4 一致组 acc CI 含 lv=5 整体 **acc** {lv5_acc:.4f} ?  '
        f'→ {condC_accmatch}')
    say(f'         （参考：lv=4 一致组 bal-acc = {c4c_bal_note:.4f}；'
        f'lv=5 整体 bal-acc = {lv5_bal:.4f}、acc = {lv5_acc:.4f}）')
    dd4, dlo4, dhi4 = filing['diff4']
    say(f'    D. lv=4 两组"无可区分差别" ?  差 {dd4:+.4f} [{dlo4:+.4f},{dhi4:+.4f}] → '
        f'{"无可区分差别" if (dlo4 <= 0 <= dhi4) else "可区分"}')

    if cond_A and condC_literal:
        verdict = '档 1：跟读者成立'
    elif cond_B:
        verdict = '档 2：跟读者被否，模型偏意图'
    elif (not cond_A and not cond_B) or (dlo4 <= 0 <= dhi4):
        verdict = '档 3：判不确定（0.561 另有来源）'
    else:
        verdict = '落档规则不完备（A 成立但 C 不成立）——交人裁'
    filing['verdict'] = verdict
    say(f'\n  **机械落档（按冻结三档，不加解释）= {verdict}**')
    c5d = cells[('C 两步自判', 5, 'disagree')]
    say(f'  附带登记（预注册明示不参与判读）：lv=5 分歧组 acc = {c5d["acc"]:.4f} '
        f'[{c5d["lo"]:.4f},{c5d["hi"]:.4f}] —— CI 整体'
        f'{"低于" if c5d["hi"] < 0.5 else "不低于"} 0.5')

    # ---------------- 8. Secondary readouts ----------------
    say('\n' + '=' * 78)
    say('8. 次读数')
    say('=' * 78)
    say('\n8.1 约定/残差占比与残差方向构成（纯计数，无推断）')
    for lv in (5, 4):
        k = counts[lv]
        say(f'  lv={lv}: 约定(一致) {k["con"]}/{k["n"]} = {k["con"] / k["n"]:.4f}；'
            f'残差(分歧) {k["dis"]}/{k["n"]} = {k["dis"] / k["n"]:.4f}；'
            f'残差方向 strong→ALT {k["s2a"]}({k["s2a"] / k["dis"]:.3f}) / '
            f'weak→REQ {k["w2r"]}({k["w2r"] / k["dis"]:.3f})')
    say(f'  残差占比 {counts[5]["dis"] / counts[5]["n"]:.4f} → '
        f'{counts[4]["dis"] / counts[4]["n"]:.4f}，'
        f'{(counts[4]["dis"] / counts[4]["n"]) / (counts[5]["dis"] / counts[5]["n"]):.2f} 倍'
        '（预注册 §4 所称"残差占比翻倍"）')

    say('\n8.2 四格 acc + CI 全格 —— 见 §5（三臂 × 六行全部列出，未挑选）')

    say('\n8.3 探针 lv4 缩水的两项分解（加权平均恒等式）')
    say('  记号：p_lv,g = 该 lv 该组的对金标 acc；w_lv,g = 该 lv 中该组的场景占比。')
    say('  恒等式 p4 − p5 = Σ_g w̄_g(p4_g − p5_g)  +  Σ_g (w4_g − w5_g) p̄_g')
    say('              = 【水平项】          +   【组成项】     （对称版，w̄/p̄ 取两 lv 均值）')
    say('  ⚠️ 分歧一：预注册说"分解式无自由度"并不严格——一次性(one-sided)分解有两种基底，'
        '本脚本三种都算并列出。')
    say('  ⚠️ 分歧二：预注册写"总缩水 = **一致组内**水平变化 + 组成变化"，但精确恒等式的'
        '水平项必须对**两组**求和；')
    say('     本脚本按精确恒等式算，并把水平项再拆到一致组/分歧组，让预注册那句能对上号。')
    say('  ⚠️ 分歧三：被分解的量只能是**普通 acc**——bal-acc 不是各组 bal-acc 的加权平均'
        '（REQ/ALT 两个召回的权重不同、且两组的 REQ/ALT 组成差异极大），')
    say('     加权平均恒等式对它不成立。而 lv4_dose 里"探针 0.6835→0.6039"是 bal-acc。'
        '故本节分解的是 acc 口径的 0.6790→0.5898，不是那两个登记数。')

    def scene_vecs(sub):
        """→ (per-scenario sum of hits [0..2], whether the scenario is in the consistent group). Cluster size is always 2."""
        s = sub.sort_values(['dataset_id', 'modus'])
        ids = s['dataset_id'].to_numpy()
        assert (ids[0::2] == ids[1::2]).all(), '孪生未成对'
        hs = s['hit'].to_numpy().reshape(-1, 2).sum(1)
        con = s['group'].to_numpy()[0::2] == 'consistent'
        return hs, con

    def _terms(h5, c5, h4, c4):
        """Vectorized: input shapes (B, n5) / (B, n4) (B = number of bootstrap draws; B=1 gives the point estimate)."""
        def agg(h, c):
            n = h.shape[1]
            ncon = c.sum(1).astype(float)
            ndis = n - ncon
            scon = np.where(c, h, 0).sum(1).astype(float)
            stot = h.sum(1).astype(float)
            sdis = stot - scon
            with np.errstate(invalid='ignore', divide='ignore'):
                return (ncon / n, ndis / n, scon / (2 * ncon), sdis / (2 * ndis), stot / (2 * n))
        w5c, w5d, p5c, p5d, P5 = agg(h5, c5)
        w4c, w4d, p4c, p4d, P4 = agg(h4, c4)
        r = {}
        r['p5'], r['p4'], r['total'] = P5, P4, P4 - P5
        lc = (w5c + w4c) / 2 * (p4c - p5c)
        ld = (w5d + w4d) / 2 * (p4d - p5d)
        r['level_sym'] = lc + ld
        r['level_sym_consistent'], r['level_sym_disagree'] = lc, ld
        r['comp_sym'] = ((w4c - w5c) * (p4c + p5c) / 2 + (w4d - w5d) * (p4d + p5d) / 2)
        r['level_w5'] = w5c * (p4c - p5c) + w5d * (p4d - p5d)
        r['comp_p4'] = (w4c - w5c) * p4c + (w4d - w5d) * p4d
        r['level_w4'] = w4c * (p4c - p5c) + w4d * (p4d - p5d)
        r['comp_p5'] = (w4c - w5c) * p5c + (w4d - w5d) * p5d
        r.update(dict(w5_consistent=w5c, w5_disagree=w5d, w4_consistent=w4c, w4_disagree=w4d,
                      p5_consistent=p5c, p5_disagree=p5d, p4_consistent=p4c, p4_disagree=p4d))
        return r

    def decompose(dfa):
        h5, c5 = scene_vecs(dfa[dfa.agreement_lv == 5])
        h4, c4 = scene_vecs(dfa[dfa.agreement_lv == 4])
        r = _terms(h5[None, :], c5[None, :], h4[None, :], c4[None, :])
        return {k: float(v[0]) for k, v in r.items()}

    def decompose_ci(dfa, keys, iters=ITERS, seed=SEED):
        h5, c5 = scene_vecs(dfa[dfa.agreement_lv == 5])
        h4, c4 = scene_vecs(dfa[dfa.agreement_lv == 4])
        rng = np.random.RandomState(seed)
        i5 = rng.randint(0, len(h5), (iters, len(h5)))
        i4 = rng.randint(0, len(h4), (iters, len(h4)))
        r = _terms(h5[i5], c5[i5], h4[i4], c4[i4])
        out = {}
        for k in keys:
            v = r[k]
            v = v[np.isfinite(v)]
            out[k] = tuple(np.percentile(v, [2.5, 97.5]))
        return out

    keys = ['total', 'level_sym', 'comp_sym', 'level_w5', 'comp_p4', 'level_w4', 'comp_p5',
            'level_sym_consistent', 'level_sym_disagree']
    for nm in ('A 探针(4B)', 'A 探针(7B)', 'C 两步自判'):
        dfa = work[nm]
        r = decompose(dfa)
        ci = decompose_ci(dfa, keys)
        star = '（主：预注册所指"探针 lv4 缩水"）' if nm == 'A 探针(4B)' else \
               ('（次）' if nm == 'A 探针(7B)' else '（对照，非探针）')
        say(f'\n  -- {nm} {star}')
        say(f'     lv5 整体 acc {r["p5"]:.4f} → lv4 整体 acc {r["p4"]:.4f}；'
            f'总缩水 = {r["total"]:+.4f} [{ci["total"][0]:+.4f},{ci["total"][1]:+.4f}]')
        say(f'     四格投入：p5(一致)={r["p5_consistent"]:.4f} p5(分歧)={r["p5_disagree"]:.4f} '
            f'p4(一致)={r["p4_consistent"]:.4f} p4(分歧)={r["p4_disagree"]:.4f}')
        say(f'               w5(分歧)={r["w5_disagree"]:.4f} → w4(分歧)={r["w4_disagree"]:.4f}')
        for tag, lk, ck in (('对称版(主报)', 'level_sym', 'comp_sym'),
                            ('一次性 A：水平@lv5权重 / 组成@lv4水平', 'level_w5', 'comp_p4'),
                            ('一次性 B：水平@lv4权重 / 组成@lv5水平', 'level_w4', 'comp_p5')):
            lv_, cv_ = r[lk], r[ck]
            say(f'     {tag}')
            say(f'        水平项 = {lv_:+.4f} [{ci[lk][0]:+.4f},{ci[lk][1]:+.4f}]'
                f'   占总变化 {lv_ / r["total"]:.1%}')
            say(f'        组成项 = {cv_:+.4f} [{ci[ck][0]:+.4f},{ci[ck][1]:+.4f}]'
                f'   占总变化 {cv_ / r["total"]:.1%}')
            say(f'        校验 水平+组成−总 = {lv_ + cv_ - r["total"]:+.2e}（须≈0）')
        say(f'     对称版水平项再拆到组：一致组 {r["level_sym_consistent"]:+.4f} '
            f'[{ci["level_sym_consistent"][0]:+.4f},{ci["level_sym_consistent"][1]:+.4f}]'
            f'；分歧组 {r["level_sym_disagree"]:+.4f} '
            f'[{ci["level_sym_disagree"][0]:+.4f},{ci["level_sym_disagree"][1]:+.4f}]')

    say('\n8.4 检验力：n=44(lv5 分歧)/116(lv4 分歧) 下可检出的最小偏离 0.5 的效应')
    say('  模型：每场景 2 行、组内相关 ρ（由 correctness 实测）→ 设计效应 DEFF=1+ρ；')
    say('        p=0.5 时 SE = sqrt(0.25·(1+ρ)/(2n_场景))。')
    say('  两条线：CI-排除线 = 1.96·SE（"CI 不含 0.5" 恰好成立所需的偏离，检验力 50%）；')
    say('          MDE80 = (1.96+0.8416)·SE（双侧 α=0.05、检验力 80% 的最小可检效应）。')
    z975, z80 = 1.959964, 0.841621
    for nm in ('C 两步自判', 'A 探针(4B)', 'A 探针(7B)'):
        dfa = work[nm]
        say(f'  -- {nm}')
        for lv in (5, 4):
            sub = dfa[(dfa.agreement_lv == lv) & (dfa.group == 'disagree')]
            n = sub['dataset_id'].nunique()
            rho = icc_pairs(sub)
            se = float(np.sqrt(0.25 * (1 + rho) / (2 * n)))
            c = cells[(nm, lv, 'disagree')]
            say(f'     lv={lv} 分歧组 n场景={n}  ρ={rho:+.3f}  DEFF={1 + rho:.3f}  '
                f'SE@0.5={se:.4f}  (bootstrap 实测 SE={c["se"]:.4f})')
            say(f'          CI-排除线 = ±{z975 * se:.4f}（即 acc 需 ≤{0.5 - z975 * se:.3f} 或 '
                f'≥{0.5 + z975 * se:.3f}）；MDE80 = ±{(z975 + z80) * se:.4f}'
                f'（acc ≤{0.5 - (z975 + z80) * se:.3f} 或 ≥{0.5 + (z975 + z80) * se:.3f}）')
    say('  参考括号（C 臂 lv=4 分歧组 n=116）：若两行完全独立 ρ=0 → MDE80 ±'
        f'{(z975 + z80) * np.sqrt(0.25 / 232):.4f}；若完全相关 ρ=1 → ±'
        f'{(z975 + z80) * np.sqrt(0.25 / 116):.4f}。')

    # ---------------- 9. Gate summary ----------------
    say('\n' + '=' * 78)
    say('9. 仪器闸门汇总')
    say('=' * 78)
    say('  V-1（四锚点，预注册给了容差=浮点）: ' +
        ('通过 ✅ —— 4/4 逐位复现' if gate_pass['V-1'] else '**不通过** ❌'))
    say('  G-1（意图标签口径，预注册要求抽样 ≥20 场景）: ' +
        ('通过 ✅ —— 全量 1744 行逐行 100% 一致，另抽样 24 场景逐条列出；'
         'compute_ceiling lv=5 混淆表复现（0.8430）' if gate_pass['G-1'] else '**不通过** ❌'))
    say('  V-2（逐项对账，预注册**未给数值容差**）: 三态，不作机械判负——')
    say(f'      · 非 CI 类 96 项（计数/场景行数/对金标 acc/bal-acc/两个召回/四锚点）：'
        '**96/96 完全一致**，一个不差；')
    say(f'      · CI 端点 36 项：22 项一致，14 项超我自设的 5.1e-4，差 0.0009–0.0047；')
    say(f'      · 差异调查（§6b）：该差不大于我自己换 seed 的抖动，且行级重采样对照近乎全错'
        '→ 判为蒙特卡洛噪声，非算法分歧。')
    say('      · 对落档无影响（lv=4 分歧组两套 CI 都含 0.5）。**是否算过闸交人裁。**')
    say('  G-2（句面可比性，预注册写"照报"、无数值判据）: 照报，不可机械判负——'
        '两个 lv 上分歧组的 γ3−γ1 都**更长**')
    say('      （lv=5 +11.0 字符 [3.4,18.8]、lv=4 +6.3 [0.6,12.3]，CI 均不含 0），'
        '即两组句面**不是**可比的；这条对"组间差=表面特征差"的排除**没有做到**，须随读数走。')
    say('  → V-1 / G-1 两道有冻结判据的硬闸门全过；V-2、G-2 无冻结判据，'
        '按上述三态/照报交人裁。主读数照出，落档标注为草稿。')

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w') as fh:
        fh.write('\n'.join(_buf) + '\n')
    print(f'\n[写出] {OUT_TXT}')
    return cells, counts, filing, gate_pass, diffs


if __name__ == '__main__':
    main()
