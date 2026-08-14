#!/usr/bin/env python
"""Arm A: out-of-fold probe-predicted state → prose/replace injection → free-generation execution.

Spec = `PREREG_probe_decode.md` §2 (frozen 2026-08-03). This script only produces raw JSONL;
adjudication and readouts live in `summarize_probe_decode.py`.

============================================================================
What it does (three stages; the middle one is where all the risk is)
============================================================================
1. **Rebuild the L7 main pipeline**: lv=5 ponens 391 rows, GroupKFold 5 (groups=atomic_idx),
   inner 4 folds jointly selecting layer×C — all parts imported from
   `summarize_hidden_probe`, zero changes, zero copies. **After rebuilding, first assert the
   OOF balanced accuracy reproduces 0.692 / 0.716 (±0.002)**; raise if not: the pipeline has
   drifted and all downstream generation would be wasted compute.
2. **Out-of-fold assignment**: each row is assigned, by its atomic_idx, to the probe of the
   fold in which that atomic sits in the test group. Atomics unique to lv=4 (52 groups /
   212 rows) are in no lv=5 fold — no fold has ever seen them — and are uniformly assigned
   fold 0. Pinned by assertion: **for every row, the training fold of the probe it uses does
   not contain that row's atomic_idx**. tollens / lv=4 rows use **their own representations**
   (reps.npz has all 1,744 rows), not the ponens twin's — so T2's transfer loss
   (0.675/0.644) is not double-counted here.
3. **Injection + free generation**: `condition_text('prose', 'and'|'or', p, r, q)` reuses
   `probe_oracle_state` verbatim; prompt assembly copies the replace mode of
   `probe_oracle_generate` verbatim; T=0 / max_new_tokens=1024 / `get_final_answer`
   extraction / record `has_final_answer` (R7).

============================================================================
Frozen checks (run on every execution; failure raises; never relaxed)
============================================================================
V-A1 anchor: main-pipeline OOF balanced accuracy == ANCHOR_MAIN (±0.002)
V-A2 fold-structure identity: the GroupKFold rebuilt from CSV group labels alone matches
     nested_cv's internal folds bit-for-bit
V-A3 out-of-fold discipline: the training fold of each row's probe does not contain that
     row's atomic_idx (asserted separately for lv=5 and non-lv=5 rows)
V-A4 identity cell: predictions for lv=5 ponens rows under the assignment rule == the main
     pipeline's `oof_pred` bit-for-bit — if the assignment rule is wrong, this cell blows
     up first
V-A5 γ1/γ3 parsing 0 failures (p/r/q extractable from all 1,744 rows)

Usage
  python scripts/run_probe_decode.py --model_name Qwen/Qwen3-4B --selfcheck    # pure CPU
  python scripts/run_probe_decode.py --model_name Qwen/Qwen3-4B --dry_render   # pure CPU
  CUDA_VISIBLE_DEVICES=6 python scripts/run_probe_decode.py \
      --model_name Qwen/Qwen3-4B --limit 8
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from run_generative import BELIEF_REV, FORMATTING  # noqa: E402  single source for DP prompt/FORMATTING
from probe_oracle_state import condition_text, parse_cond  # noqa: E402  single source for the prose template
from summarize_hidden_probe import (CGRID, N_INNER, N_OUTER,  # noqa: E402
                                    bal_acc, load_reps, nested_cv)
from summarize_hidden_probe_lvsplit import ANCHOR_MAIN, TOL_MAIN  # noqa: E402  single source for anchors
from src.prompts.utils import get_final_answer  # noqa: E402

CSV = os.path.join(BELIEF_REV, 'dataset', 'belief_r', 'queries_time_t1.csv')
OUT_DIR = os.path.join(HERE, '..', 'outputs', 'probe_decode')
MAIN_LV, MAIN_MODUS, MAIN_N = 5, 'ponens', 391

# ------------------------------------------------------------------ prompt assembly (frozen verbatim)
# Copied verbatim from the replace branch of probe_oracle_generate.build. Change one
# character and the sha changes, voiding comparability with the L2 execution readout (0.994).
PROMPT_FMT = ('{cond}\n{gamma2}\n\nWhat necessarily had to follow assuming that the above '
              'premises were true?\n(a) {opt_a}\n(b) {opt_b}\n(c) {opt_c}\n\n{formatting}')
_SENTINEL = ('<P>', '<R>', '<Q>')
TEMPLATE_SHA256 = hashlib.sha256(
    ('\x00'.join([condition_text('prose', 'and', *_SENTINEL),
                  condition_text('prose', 'or', *_SENTINEL),
                  PROMPT_FMT, FORMATTING])).encode('utf-8')).hexdigest()


def build_replace_prompt(row, conn):
    """replace mode: the condition statement replaces γ1/γ3, keeps γ2, followed by question + options + FORMATTING.

    conn='and' → REQ necessity framing; conn='or' → ALT sufficiency framing (see probe_oracle_state).
    """
    L = row['questions'].split('\n')
    c1, c3 = parse_cond(L[0]), parse_cond(L[2])
    if not c1 or not c3:
        raise ValueError(f"{row['dataset_id']} 的 γ1/γ3 解析失败，不该发生（V-A5）")
    (p, q), (r, _) = c1, c3
    return PROMPT_FMT.format(cond=condition_text('prose', conn, p, r, q), gamma2=L[1],
                             opt_a=row['a'], opt_b=row['b'], opt_c=row['c'],
                             formatting=FORMATTING)


def implied_answer(conn, modus):
    """The answer **mechanically entailed** by the fed state (ignores gold). The follow rate is computed against this."""
    return 'c' if conn == 'and' else ('a' if modus == 'ponens' else 'b')


# ------------------------------------------------------------------ fold structure (single source shared by arms A/B)
def main_folds(df=None):
    """Main-pipeline fold structure, depending only on the CSV group labels. Returns (folds, groups, main_positions).

    `folds` is the GroupKFold(5) split over the lv=5 ponens 391 rows (indices relative to
    that subset); `main_positions` are those 391 rows' row numbers within the full 1,744.
    Arm B imports this function to get the same folds — "A/B same folds" is half of
    PREREG §5 gate (i).
    """
    if df is None:
        df = pd.read_csv(CSV)
    main = df[(df.agreement_lv == MAIN_LV) & (df.modus == MAIN_MODUS)]
    assert len(main) == MAIN_N, f'主集行数 {len(main)} != {MAIN_N}'
    g = main['atomic_idx'].values
    folds = list(GroupKFold(N_OUTER).split(np.zeros(len(g)), None, g))
    return folds, g, main.index.values


def fold_of_atomic(folds, groups):
    """atomic_idx → the fold in which it sits as a test group."""
    out = {}
    for k, (_, te) in enumerate(folds):
        for a in groups[te]:
            assert out.setdefault(int(a), k) == k, f'atomic {a} 出现在两折的测试集里'
    return out


def assign_folds(atomics, f_of):
    """The PREREG §2 assignment rule: seen atomics → their test fold; unseen → fold 0.

    Fold 0 is safe for "atomics never present in the lv=5 main set": every fold's training
    set is a subset of the main set, which lacks that atomic, so fold 0's probe cannot have
    seen it. This is pinned by the V-A3 assertion, not by argument.
    """
    return np.array([f_of.get(int(a), 0) for a in atomics]), \
        np.array([int(a) in f_of for a in atomics])


# ------------------------------------------------------------------ probe
def build_probe(model_name, n_jobs, verbose=True):
    """Rebuild the main pipeline + the first four of the five frozen checks. Returns a dict."""
    last, _, meta = load_reps(model_name, smoke=False)
    df = pd.read_csv(CSV)
    layers = list(range(last.shape[1]))

    folds_csv, groups, positions = main_folds(df)
    main = meta.loc[positions]
    Xm = last[positions]
    ym = (main['gold'] == 'c').astype(int).values
    Gm = main['atomic_idx'].values
    assert (Gm == groups).all(), 'meta 与 CSV 的 atomic_idx 顺序不一致'

    cv = nested_cv(Xm, ym, Gm, layers, CGRID, N_OUTER, N_INNER, n_jobs)

    # V-A1 anchor
    b = bal_acc(ym, cv['oof_pred'])
    reg = ANCHOR_MAIN[model_name]
    assert abs(b - reg) <= TOL_MAIN, \
        f'V-A1 失败：主管线 OOF 平衡准确率 {b:.4f} vs 登记 {reg}（±{TOL_MAIN}）——管线漂移'

    # V-A2 fold-structure identity
    for k, ((a_tr, a_te), (b_tr, b_te)) in enumerate(zip(folds_csv, cv['folds'])):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te), \
            f'V-A2 失败：第 {k} 折与 nested_cv 内部折不同 —— A/B 同折前提破了'

    f_of = fold_of_atomic(cv['folds'], Gm)
    assert len(f_of) == 152, f'lv=5 atomic 组数 {len(f_of)} != 152'

    # Fold assignment + predictions for all 1,744 rows
    all_atomics = meta['atomic_idx'].values
    kk, seen = assign_folds(all_atomics, f_of)
    pred = np.full(len(meta), -1)
    for k in range(N_OUTER):
        rows = np.where(kk == k)[0]
        if not len(rows):
            continue
        L, C, sc, clf = cv['models'][k]
        pred[rows] = clf.predict(sc.transform(last[rows][:, L].astype(np.float32)))
    assert (pred != -1).all(), '有行没被预测到'

    # V-A3 out-of-fold discipline: check each row's probe training fold row by row
    train_groups = [set(Gm[tr].tolist()) for tr, _ in cv['folds']]
    for i, (a, k) in enumerate(zip(all_atomics, kk)):
        assert int(a) not in train_groups[k], \
            f'V-A3 失败：第 {i} 行 atomic {a} 在折 {k} 的训练组里'

    # V-A4 identity cell: main-set rows' predictions under the assignment rule must equal oof_pred bit-for-bit
    assert np.array_equal(pred[positions], cv['oof_pred']), \
        'V-A4 失败：主集行的指派预测 != 主管线 oof_pred —— 指派规则写错了'

    if verbose:
        print(f'[V-A1] 主管线 OOF 平衡准确率 = {b:.4f} == 登记 {reg}（±{TOL_MAIN}）  ✅')
        print(f'[V-A2] 折结构与 nested_cv 内部逐位相同（各折测试行数 '
              f'{[len(te) for _, te in cv["folds"]]}）  ✅')
        print(f'[V-A3] 全 {len(meta)} 行折外纪律断言通过；'
              f'其中 {int(seen.sum())} 行的 atomic 在 lv=5 折内、'
              f'{int((~seen).sum())} 行（lv=4 独有种子）指派折 0  ✅')
        print(f'[V-A4] 主集 391 行的指派预测 == oof_pred 逐位相同  ✅')
        print(f'  各外层折选中 (层, C, 内层bal_acc): {cv["chosen"]}')
        print(f'  预测状态分布：REQ(and) {int((pred == 1).sum())} / '
              f'ALT(or) {int((pred == 0).sum())}')
    return dict(meta=meta, df=df, pred=pred, fold=kk, seen=seen, cv=cv,
                positions=positions, b_main=b)


# ------------------------------------------------------------------ main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0, help='只对生成段生效；探针始终用全量')
    ap.add_argument('--n_jobs', type=int, default=16)
    ap.add_argument('--selfcheck', action='store_true',
                    help='纯 CPU：折指派断言 + 锚点断言 + 模板 sha，不加载语言模型')
    ap.add_argument('--dry_render', action='store_true',
                    help='纯 CPU：打印 8 行渲染全文（状态取真实折外预测）')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    print(f'# 臂 A probe-decode（PREREG_probe_decode.md §2，2026-08-03 冻结）')
    print(f'# model = {args.model_name}')
    print(f'# TEMPLATE_SHA256 = {TEMPLATE_SHA256}')
    print(f'# 折结构：GroupKFold {N_OUTER}（组=atomic_idx），内层 {N_INNER}，C∈{CGRID}')

    P = build_probe(args.model_name, args.n_jobs)
    meta, df, pred, kk, seen = P['meta'], P['df'], P['pred'], P['fold'], P['seen']

    # V-A5 parsing gate (pre-builds all prompts along the way)
    conns = ['and' if p == 1 else 'or' for p in pred]
    prompts = [build_replace_prompt(df.iloc[i], c) for i, c in enumerate(conns)]
    print(f'[V-A5] γ1/γ3 解析失败 0 行（全 {len(prompts)} 行提示已构建）  ✅')

    if args.dry_render:
        # The 8 rows are not df.head(8): the first 8 rows are all ponens and all the same
        # atomic, covering only two of the four rendering paths (modus × predicted state),
        # and the non-lv=5-fold assignment path would not be visible either.
        # Instead: first row of each design cell + two lv=4 rows + two rows whose atomic is
        # not in any lv=5 fold.
        picks = []
        for m in ('ponens', 'tollens'):
            for s in (1, 0):
                cand = [i for i in range(len(df))
                        if df.iloc[i]['modus'] == m and pred[i] == s]
                if cand:
                    picks.append(cand[0])
        for cond in (lambda i: df.iloc[i]['agreement_lv'] == 4,
                     lambda i: not seen[i]):
            cand = [i for i in range(len(df)) if cond(i) and i not in picks]
            picks += cand[:2]
        picks = picks[:8]
        print(f'\n{"=" * 78}\n渲染样例（8 行，状态 = 真实折外预测；'
              f'覆盖 modus×状态 四格 + lv=4 + 非 lv=5 折指派）\n{"=" * 78}')
        for i in picks:
            r = df.iloc[i]
            print(f'{"#" * 26} idx={i}  dataset_id={r["dataset_id"]}  modus={r["modus"]}  '
                  f'lv={r["agreement_lv"]}  金标={r["ground_truth"]}\n'
                  f'{"#" * 26} atomic={r["atomic_idx"]}  折={kk[i]}  '
                  f'atomic在lv5折内={bool(seen[i])}  '
                  f'预测状态={"REQ" if pred[i] == 1 else "ALT"}(conn={conns[i]})  '
                  f'蕴含答案={implied_answer(conns[i], r["modus"])}')
            print(prompts[i])
            print()

    if args.selfcheck or args.dry_render:
        print('[selfcheck] V-A1…V-A5 全部通过，未加载语言模型，无读数。SELFCHECK PASS')
        return

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda').eval()

    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.model_name.replace('/', '_').replace('-', '_')
    out_path = os.path.join(args.out_dir, f'A_{safe}.jsonl')
    n = args.limit if args.limit else len(df)

    done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['idx'])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [i for i in range(n) if i not in done]

    with open(out_path, 'a' if done else 'w') as fout:
        for st in tqdm(range(0, len(todo), args.batch), desc=f'A {safe}'):
            bidx = todo[st:st + args.batch]
            texts = [tok.apply_chat_template([{'role': 'user', 'content': prompts[i]}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for i in bidx]
            enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
            in_len = enc['input_ids'].shape[1]
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gens = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
            for j, i in enumerate(bidx):
                row = df.iloc[i]
                ids = out[j, in_len:].tolist()
                trunc = (len(ids) >= args.max_new_tokens) and (tok.eos_token_id not in ids)
                n_new = (ids.index(tok.eos_token_id) + 1) if tok.eos_token_id in ids else len(ids)
                gen = gens[j]
                ext = get_final_answer(gen)
                imp = implied_answer(conns[i], row['modus'])
                fout.write(json.dumps({
                    'arm': 'A', 'idx': int(i), 'dataset_id': row['dataset_id'],
                    'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
                    'agreement_lv': int(row['agreement_lv']),
                    'ground_truth': row['ground_truth'],
                    'pred_state': 'REQ' if pred[i] == 1 else 'ALT', 'pred_y': int(pred[i]),
                    'fed_conn': conns[i], 'fold': int(kk[i]),
                    'atomic_in_lv5_folds': bool(seen[i]), 'implied': imp,
                    'model': args.model_name, 'template_sha': TEMPLATE_SHA256,
                    'prompt': prompts[i], 'raw_output': gen, 'extracted': ext,
                    'has_final_answer': 'final answer' in gen.lower(),
                    'n_new_tokens': int(n_new), 'truncated': bool(trunc),
                    'follows': ext == imp, 'correct': ext == row['ground_truth'],
                }, ensure_ascii=False) + '\n')
            fout.flush()
    print(f'[done] {out_path}  n={n}  sha={TEMPLATE_SHA256[:12]}')


if __name__ == '__main__':
    main()
