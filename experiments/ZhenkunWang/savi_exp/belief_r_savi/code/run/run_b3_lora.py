#!/usr/bin/env python
"""Arm B3: structure-matched LoRA (PREREG_b3.md, Qwen3-4B only).

The question it asks: after handing LoRA the **same** two pieces of hand-built prior that
arm A gets for free (decision variable = binary relation, and the mechanical relation→answer
mapping), can gradients acting on the output path extract the linearly readable R in the
representation?
So the difference vs B1 (`run_lora_sft.py`) is **the supervision target only**: B1's
assistant span is the answer letter, B3's assistant span is the relation line. Data, folds,
hyperparameters, early-stop machinery, and assertions are all copied from B1.

============================================================================
Protocol (frozen before running, PREREG_b3.md §2–§3)
============================================================================
- Training sample: user = arm C's step1 prompt **verbatim** (`run_twostep.build_step1`,
  single source, module-level assertion sha == STEP1_SHA256); assistant = `RELATION: 1`
  (gold REQ, i.e. ground_truth == 'c') / `RELATION: 2` (gold ALT). Loss is computed on the
  assistant span only.
- Data and folds: all 782 lv=5 rows (`run_lora_sft.lv5_split`, the same 5 folds as A/B1,
  so the V-B0/V-B1 assertions come for free). Twin rows share the label and are
  **not deduplicated** (fixed in PREREG §2; row count / fold structure bitwise isomorphic
  to B1).
- Hyperparameters copied from B1: all-linear r16 α32 dropout 0.05, lr 1e-4, AdamW constant
  lr, micro 4 × accum 4, epochs ≤5 patience 1, seed 20260803.
- Inference chain (PREREG §3, three steps within each fold): fine-tuned model runs step1
  (greedy 512 tokens, parsed with `prompts/contract.parse_relation`) → free the fine-tuned
  model → **separately load the original model without adapter** to run step2
  (`build_replace_prompt`, greedy 1024 tokens, extracted with `get_final_answer`).
  Parse failure → fall back to the original DP prompt and record `fallback=1` (same
  convention as arm C). Switching the executor to the original model is a design
  requirement, not a shortcut: variable isolation on the classifier.

============================================================================
Implementation-level registration (registered in PREREG §2 + degrees of freedom this script does not pin)
============================================================================
- Dev early-stop metric = **relation bal-acc** on dev rows (mean of per-class accuracy for
  REQ/ALT, parse failure counts as wrong), dev generation budget 32 tokens. The supervision
  target changed so the early-stop metric changes with it, registered in PREREG §2;
  dev only selects the epoch and enters no readout.
- Gradient accumulation, per-epoch shuffling with seed 20260803+epoch: verbatim same as B1.
- Raise when best_state is empty (no epoch produced a comparable dev metric) — B1 would
  silently use the last epoch; on B3 that is "instrument broke but looks like it ran",
  better to blow up.

Usage
  python scripts/run_b3_lora.py --selfcheck                      # CPU only
  CUDA_VISIBLE_DEVICES=6 python scripts/run_b3_lora.py --fold 0 --smoke
  CUDA_VISIBLE_DEVICES=6 python scripts/run_b3_lora.py --fold 0
"""
import argparse
import gc
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
import run_twostep  # noqa: E402  single source of the step1 prompt (build_step1 / STEP1_SHA256)
from run_generative import build_prompt  # noqa: E402  single source of the fallback DP prompt
from run_lora_sft import (ACCUM, LR, MAX_EPOCHS, MICRO_BS, MODEL,  # noqa: E402
                          OUT_DIR, PATIENCE, SEED, TARGETS, breu,
                          check_prompt_identity, collate, encode, lv5_split)
from run_probe_decode import (TEMPLATE_SHA256, build_replace_prompt,  # noqa: E402
                              implied_answer)
from prompts.contract import parse_relation  # noqa: E402  single source of the RELATION regex
from src.prompts.utils import get_final_answer  # noqa: E402

# G-B3-0 input identity (PREREG §5): both template shas are registered here, so the
# assertion is not done by eye. Provenance of registered values: STEP1 = PREREG_b3.md §2;
# TEMPLATE = the `template_sha` written by arms A/C (outputs/probe_decode/C_twostep.jsonl).
# If either template changes by one character, this blows up first.
STEP1_SHA_REG = '51e915d1f862944b3e2dc14b9d8720c11f312a3b5d4c358bb8949f7fd6b85e5c'
TEMPLATE_SHA_REG = '5a104f449cb08aba418c5d9d69fd974d27491f846baa5f3282497025366a9376'
assert run_twostep.STEP1_SHA256 == STEP1_SHA_REG, \
    f'G-B3-0 失败：step1 模板 sha {run_twostep.STEP1_SHA256} != 登记 {STEP1_SHA_REG}'
assert TEMPLATE_SHA256 == TEMPLATE_SHA_REG, \
    f'G-B3-0 失败：注入模板 sha {TEMPLATE_SHA256} != 登记 {TEMPLATE_SHA_REG}'

DEV_MAX_NEW, STEP1_MAX_NEW, STEP2_MAX_NEW = 32, 512, 1024
LORA_LAYERS = None  # all-linear, all layers = B1's variant B1; B3 has only this variant


# ------------------------------------------------------------------ labels (= arm C's gold_state rule)
def gold_state(row):
    """Gold relation. Verbatim the same expression as the `gold_state` written to disk by `run_twostep.main`."""
    return 'REQ' if row['ground_truth'] == 'c' else 'ALT'


def target_line(row):
    """Assistant segment (the only line loss is computed on). REQ→1 / ALT→2, same mechanical mapping as A/C."""
    return 'RELATION: 1' if gold_state(row) == 'REQ' else 'RELATION: 2'


def state_of_relation(rel):
    """Parsed relation → state. Verbatim the same expression as arm C's `pred_state`; None on parse failure."""
    return None if rel is None else ('REQ' if rel == '1' else 'ALT')


def rel_bal_acc(records):
    """Relation bal-acc = (accuracy on REQ rows + accuracy on ALT rows)/2; parse failures count as wrong (pred_state=None)."""
    req = [r['pred_state'] == 'REQ' for r in records if r['gold_state'] == 'REQ']
    alt = [r['pred_state'] == 'ALT' for r in records if r['gold_state'] == 'ALT']
    if not req or not alt:
        return float('nan')
    return (float(np.mean(req)) + float(np.mean(alt))) / 2


# ------------------------------------------------------------------ frozen checks
def check_label_rule(df, idx):
    """V-B3 label round-trip: target line → parse_relation → arm C's state mapping → must return to gold_state.

    Merely asserting "the label rule and the readout rule are inverses" is meaningless
    (an identity); what is asserted here is that **the literal training-target string**,
    passed through the regex and mapping actually used at inference time, comes back to
    the gold label — a swapped 1/2, a target line in a shape beyond what the regex
    tolerates (it does tolerate `RELATION:1`), or a reversed mapping direction all blow
    up in this cell first.
    """
    for i in idx:
        row = df.loc[i]
        t = target_line(row)
        rel = parse_relation(t)
        assert rel is not None, f'idx {i} 的训练目标 {t!r} 过不了 parse_relation'
        assert state_of_relation(rel) == gold_state(row), \
            f'V-B3 失败：idx {i} 目标行 {t!r} 解析回 {state_of_relation(rel)} != 金标 {gold_state(row)}'
    return f'{len(idx)} 行目标行→正则→映射闭环回到金标  ✅'


def check_gold_state_vs_C(df, idx):
    """V-B4 identity cell: this script's gold_state equals arm C's on-disk `gold_state` row by row.

    The two rules are two copies of the same expression; comparing them by eye does not
    count — use C's artifact as the external reference.
    """
    p = os.path.join(OUT_DIR, 'C_twostep.jsonl')
    if not os.path.exists(p):
        return f'跳过（无 {os.path.basename(p)}）'
    want = {int(i) for i in idx}
    n = 0
    for line in open(p):
        r = json.loads(line)
        if r['idx'] not in want:
            continue
        assert r['gold_state'] == gold_state(df.loc[r['idx']]), \
            f'V-B4 失败：idx {r["idx"]} 的 gold_state 与臂 C 不同'
        n += 1
    return f'{n} 行与 C_twostep.jsonl 的 gold_state 逐行相同  ✅'


def twin_step1_report(df, idx):
    """Are the two twin rows' step1 prompts actually identical — count it, don't assume PREREG's wording.

    PREREG §2/§4 say "the twin rows' step1 prompts are verbatim identical". The step1
    problem text takes the first three lines of questions, including γ2, and γ2 is exactly
    where ponens/tollens differ — so that sentence needs to be measured, not quoted.
    This function changes no behavior (PREREG fixes "no dedup", which is followed); it
    only puts the measured numbers on stdout.
    """
    sub = df.loc[list(idx)]
    same = diff = 0
    for _, g in sub.groupby('dataset_id'):
        if len(g) < 2:
            continue
        if len({run_twostep.build_step1(r) for _, r in g.iterrows()}) == 1:
            same += 1
        else:
            diff += 1
    lab = sum(len({gold_state(r) for _, r in g.iterrows()}) > 1
              for _, g in sub.groupby('dataset_id'))
    return f'孪生场景 {same + diff}：step1 提示逐字相同 {same} / 不同 {diff}；标签不一致 {lab}'


# ------------------------------------------------------------------ generation
def gen_batch(model, tok, prompts, max_new, batch, desc):
    """Greedy batched generation → [(text, truncated, n_new_tokens)]. Same protocol as `run_twostep.main.gen`."""
    import torch
    from tqdm import tqdm
    outs = []
    model.eval()
    for st in tqdm(range(0, len(prompts), batch), desc=desc, leave=False):
        texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
                 for p in prompts[st:st + batch]]
        enc = tok(texts, return_tensors='pt', padding=True).to(model.device)
        in_len = enc['input_ids'].shape[1]
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for j, g in enumerate(tok.batch_decode(o[:, in_len:], skip_special_tokens=True)):
            ids = o[j, in_len:].tolist()
            outs.append((g, bool(len(ids) >= max_new and tok.eos_token_id not in ids),
                         (ids.index(tok.eos_token_id) + 1
                          if tok.eos_token_id in ids else len(ids))))
    return outs


def step1_records(model, tok, df, idx, max_new, batch, desc):
    """Run step1 on the idx rows → per-row relation / pred_state / truncation / token count."""
    outs = gen_batch(model, tok, [run_twostep.build_step1(df.loc[i]) for i in idx],
                     max_new, batch, desc)
    recs = []
    for i, (g, trunc, ntok) in zip(idx, outs):
        rel = parse_relation(g)
        recs.append({'idx': int(i), 'relation': rel, 'pred_state': state_of_relation(rel),
                     'gold_state': gold_state(df.loc[i]), 'step1_raw': g,
                     'step1_truncated': trunc, 'step1_n_new_tokens': ntok})
    return recs


# ------------------------------------------------------------------ main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--model_name', default=MODEL)
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--batch', type=int, default=16, help='推理批大小')
    ap.add_argument('--smoke', action='store_true',
                    help='train 64 行 / dev,test 各 32 行 / 1 epoch')
    ap.add_argument('--selfcheck', action='store_true',
                    help='纯 CPU：折与三集不交断言 + sha + 标签闭环 + 样本与推理链渲染')
    args = ap.parse_args()

    df, tr, dv, te, k_dev, A = lv5_split(args.fold)
    print('# 臂 B3 结构匹配 LoRA（PREREG_b3.md §2–§3）')
    print(f'# fold={args.fold}（dev=折{k_dev}）  model={args.model_name}')
    print(f'# targets={TARGETS}  r16 α32 dropout0.05  layers_to_transform={LORA_LAYERS}')
    print(f'# lr={LR} epochs≤{MAX_EPOCHS} patience={PATIENCE} '
          f'micro_bs={MICRO_BS}×accum{ACCUM}  dev_max_new={DEV_MAX_NEW} '
          f'step1_max_new={STEP1_MAX_NEW} step2_max_new={STEP2_MAX_NEW}')
    print(f'# STEP1_SHA256    = {run_twostep.STEP1_SHA256}')
    print(f'# TEMPLATE_SHA256 = {TEMPLATE_SHA256}')
    print('[G-B3-0] step1 模板 sha 与注入模板 sha 均 == 登记值（与臂 C/A 同一串字节）  ✅')
    print('[V-B0] 782 行版折结构 == 391 行版（与 A/B1 同折）  ✅')
    print(f'[V-B1] train/dev/test 行数 = {len(tr)}/{len(dv)}/{len(te)}；'
          f'atomic 组数 = {len(A["train"])}/{len(A["dev"])}/{len(A["test"])}；两两不交  ✅')
    print(f'[V-B2] 回退用 DP 提示识别单元格：{check_prompt_identity(df, args.model_name)}')
    print(f'[V-B3] 标签闭环：{check_label_rule(df, list(tr) + list(dv) + list(te))}')
    print(f'[V-B4] 金标状态：{check_gold_state_vs_C(df, list(tr) + list(dv) + list(te))}')
    n_req = sum(gold_state(df.loc[i]) == 'REQ' for i in tr)
    print(f'[data] 训练集标签分布：REQ {n_req} / ALT {len(tr) - n_req}')
    print(f'[obs] {twin_step1_report(df, list(tr) + list(dv) + list(te))}')

    if args.selfcheck:
        print(f'\n{"=" * 78}\nSFT 样本渲染（2 条；assistant 段即监督目标，'
              f'与 B1 的唯一差别在这一段）\n{"=" * 78}')
        for i in list(tr)[:2]:
            row = df.loc[i]
            print(f'{"#" * 26} idx={i} dataset_id={row["dataset_id"]} '
                  f'modus={row["modus"]} 金标={row["ground_truth"]} '
                  f'金标关系={gold_state(row)}')
            print('--- user（= 臂 C step1 提示逐字）---')
            print(run_twostep.build_step1(row))
            print('--- assistant（只在这一段算 loss）---')
            t = target_line(row)
            assert t in ('RELATION: 1', 'RELATION: 2'), f'监督目标写成了 {t!r}'
            print(t)
            print()

        # Inference chain: 1 ponens + 1 tollens. Render both step2 parse outcomes and the
        # fallback once each — under tollens ALT implies b (not a); if that cell can't be
        # rendered, nothing was checked.
        picks = []
        for m in ('ponens', 'tollens'):
            cand = [i for i in te if df.loc[i, 'modus'] == m]
            if cand:
                picks.append(cand[0])
        print(f'{"=" * 78}\n推理链渲染（1 ponens + 1 tollens，取自本折 test；'
              f'step1 全文 + rel=1/2 两种 step2 + 回退 DP）\n{"=" * 78}')
        for i in picks:
            row = df.loc[i]
            print(f'{"#" * 26} idx={i} dataset_id={row["dataset_id"]} modus={row["modus"]} '
                  f'lv={row["agreement_lv"]} 金标={row["ground_truth"]} '
                  f'金标关系={gold_state(row)}')
            print('----- STEP 1（微调模型跑，贪心 512 token）-----')
            print(run_twostep.build_step1(row))
            for rel in ('1', '2'):
                conn = 'and' if rel == '1' else 'or'
                print(f'----- STEP 2（原模型跑；step1 解析出 RELATION: {rel} → '
                      f'{state_of_relation(rel)} → conn={conn} → '
                      f'蕴含答案 {implied_answer(conn, row["modus"])}）-----')
                print(build_replace_prompt(row, conn))
            print('----- 回退（step1 解析失败 → 原始 DP 提示，fallback=1）-----')
            print(build_prompt(row['questions'], 'dp'))
            print()
        print('[selfcheck] G-B3-0 / V-B0…V-B4 全部通过，未加载语言模型，无读数。')
        print('SELFCHECK PASS')
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16, device_map='cuda')
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=TARGETS, layers_to_transform=LORA_LAYERS))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[LoRA] 模型层数={model.config.num_hidden_layers}  可训练参数={n_tr:,}')

    epochs = 1 if args.smoke else MAX_EPOCHS
    tr_idx, dv_idx, te_idx = list(tr), list(dv), list(te)
    if args.smoke:
        # Test slice stratified by modus, 16 each: the CSV has the ponens block first, so
        # taking the first 32 rows would be all ponens, while step2's implied answer
        # depends on modus (under tollens ALT→b) — that path would get no smoke coverage.
        # Only affects the --smoke branch; the full-run path is untouched.
        tr_idx, dv_idx = tr_idx[:64], dv_idx[:32]
        te_idx = ([i for i in te_idx if df.loc[i, 'modus'] == 'ponens'][:16]
                  + [i for i in te_idx if df.loc[i, 'modus'] == 'tollens'][:16])

    samples = [encode(tok, run_twostep.build_step1(df.loc[i]), target_line(df.loc[i]))
               for i in tr_idx]
    print(f'[data] 训练样本 {len(samples)} 条；assistant 段 token 数 '
          f'{[sum(1 for x in s[1] if x != -100) for s in samples[:3]]}…')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(SEED + ep).permutation(len(samples))
        tot, nb = 0.0, 0
        opt.zero_grad()
        for s in range(0, len(order), MICRO_BS):
            ids, lab, att = collate([samples[j] for j in order[s:s + MICRO_BS]],
                                    tok.pad_token_id, model.device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / ACCUM).backward()
            tot += float(loss); nb += 1
            if (s // MICRO_BS + 1) % ACCUM == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        dev_recs = step1_records(model, tok, df, dv_idx, DEV_MAX_NEW, args.batch,
                                 f'dev ep{ep}')
        d = rel_bal_acc(dev_recs)
        n_fail = sum(1 for r in dev_recs if r['relation'] is None)
        print(f'[epoch {ep}] train loss = {tot / max(nb, 1):.4f}  '
              f'dev 关系 bal-acc = {d:.4f}  (dev n={len(dev_recs)}, '
              f'解析失败={n_fail}, 预测 REQ={sum(1 for r in dev_recs if r["pred_state"] == "REQ")}'
              f'/ALT={sum(1 for r in dev_recs if r["pred_state"] == "ALT")})', flush=True)
        if d > best:
            best, best_ep, bad = d, ep, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items() if 'lora' in k}
        else:
            bad += 1
            if bad > PATIENCE:
                print(f'[early stop] dev 关系 bal-acc 连续 {bad} 轮未提升，停在 epoch {ep}')
                break
    assert best_state is not None, \
        '没有任何一轮拿到可比较的 dev 指标（全 NaN？dev 单类？）——仪器坏了，不出读数'
    model.load_state_dict(best_state, strict=False)
    print(f'[best] dev 关系 bal-acc = {best:.4f}（epoch {best_ep}）')

    # ---- inference chain step 1: fine-tuned model predicts the relation ----
    s1 = step1_records(model, tok, df, te_idx, STEP1_MAX_NEW, args.batch, 'B3 step1')

    # ---- step 2: free the fine-tuned model. step2 must be executed by the
    # **un-fine-tuned original model** (variable isolation on the classifier) ----
    del model, opt, best_state, samples
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, local_files_only=True, dtype=torch.bfloat16,
        device_map='cuda').eval()
    print('[step2] 已释放微调模型，另行加载不带 adapter 的原模型执行注入')

    # ---- step 3: original model executes step2 (parse failure → fall back to DP, fallback=1) ----
    prompts = []
    for r in s1:
        row = df.loc[r['idx']]
        prompts.append(build_prompt(row['questions'], 'dp') if r['relation'] is None
                       else build_replace_prompt(row, 'and' if r['relation'] == '1' else 'or'))
    s2 = gen_batch(base, tok, prompts, STEP2_MAX_NEW, args.batch, 'B3 step2')

    recs = []
    for r, p, (g, trunc, ntok) in zip(s1, prompts, s2):
        row = df.loc[r['idx']]
        rel = r['relation']
        conn = None if rel is None else ('and' if rel == '1' else 'or')
        ext = get_final_answer(g)
        imp = None if conn is None else implied_answer(conn, row['modus'])
        recs.append({
            'arm': 'B3', 'idx': int(r['idx']), 'dataset_id': row['dataset_id'],
            'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
            'agreement_lv': int(row['agreement_lv']), 'ground_truth': row['ground_truth'],
            'model': args.model_name, 'step1_sha': run_twostep.STEP1_SHA256,
            'template_sha': TEMPLATE_SHA256,
            'relation': rel, 'pred_state': r['pred_state'], 'fed_conn': conn,
            'implied': imp, 'fallback': int(rel is None), 'gold_state': r['gold_state'],
            'step1_raw': r['step1_raw'], 'step1_truncated': r['step1_truncated'],
            'step1_n_new_tokens': r['step1_n_new_tokens'],
            'step2_prompt': p, 'raw_output': g, 'extracted': ext,
            'has_final_answer': 'final answer' in g.lower(),
            'n_new_tokens': ntok, 'truncated': trunc,
            'follows': None if imp is None else (ext == imp),
            'correct': ext == row['ground_truth'],
            'fold': args.fold, 'dev_balacc': best, 'best_epoch': best_ep,
            'smoke': bool(args.smoke),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = '_smoke' if args.smoke else ''
    out_path = os.path.join(args.out_dir, f'B3_fold{args.fold}{suffix}.jsonl')
    with open(out_path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n = len(recs)
    n_fb = sum(r['fallback'] for r in recs)
    n_tr1 = sum(1 for r in recs if r['step1_truncated'])
    n_req = sum(1 for r in recs if r['pred_state'] == 'REQ')
    n_alt = sum(1 for r in recs if r['pred_state'] == 'ALT')
    print(f'[done] {out_path}  n={n}')
    print(f'[G-B3-2] step1 解析成功率 = {(n - n_fb) / n:.4f}（闸门 ≥0.8）；'
          f'截断率 = {n_tr1 / n:.4f}（闸门 ≤0.05）')
    print(f'[G-B3-3] step1 预测边缘分布：REQ {n_req} / ALT {n_alt} / 解析失败 {n_fb}'
          f'（某类 ≥0.9 即塌缩形态，解读里须写明）')
    print(f'[read] 折内 test 关系 bal-acc（全 {n} 行口径）= {rel_bal_acc(recs):.4f}；'
          f'ponens 行口径 = '
          f'{rel_bal_acc([r for r in recs if r["modus"] == "ponens"]):.4f}')
    print(f'[read] 折内 test BREU = {breu(recs):.4f}  '
          f'noFA={sum(1 for r in recs if not r["has_final_answer"])}  '
          f'step2 截断={sum(1 for r in recs if r["truncated"])}')
    print('# 主读数与配对比较在汇总脚本里出，本脚本只产原始 JSONL（PREREG §4）')


if __name__ == '__main__':
    main()
