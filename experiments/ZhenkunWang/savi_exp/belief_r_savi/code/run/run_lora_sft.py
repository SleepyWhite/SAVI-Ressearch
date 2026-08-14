#!/usr/bin/env python
"""Arm B: iso-label LoRA end-to-end SFT (control arm, Qwen3-4B only).

Spec = `PREREG_probe_decode.md` §3. The question it answers: given **the gold labels of
the same 391 scenarios**, can gradient fine-tuning reach arm A's
"read representation → inject → execute" pathway?
So the supervision must be iso-label (the answer letter only), not the state-writer format.

============================================================================
Protocol (frozen before running)
============================================================================
- Data: all 782 lv=5 rows (391 scenarios × twin rows). **Twins share a fold**
  (group=atomic_idx guarantees this naturally).
- Folds: the same 5 folds from `run_probe_decode.main_folds()` (A/B sharing folds is
  PREREG §5 gate (i)). Per fold, test=k, dev=(k+1)%5, train=remaining 3 folds. The three
  sets' atomic_idx are pairwise disjoint, pinned by assertion.
- Sample: chat with a single user turn, content = **the DP prompt verbatim identical to
  evaluation** (`run_generative.build_prompt(question, 'dp')`); assistant span =
  `Final Answer [X].`. Loss on the assistant span only (prompt-span label = -100), with an
  assertion that the prompt/full tokenizations share the same prefix.
- Variants: B1 all-linear (q/k/v/o/gate/up/down, all layers) r16 α32 dropout 0.05;
  B2 same targets but `layers_to_transform=range(20,36)` (past the 4B probe's peak
  layer 19).
- Optimization: lr 1e-4, AdamW, constant lr, epochs ≤5, evaluate BREU on dev per epoch,
  patience 1; use the adapter from the epoch with the best dev BREU for test inference.
- Inference: DP prompt, greedy, max_new_tokens=1024, `get_final_answer` extraction, record
  `has_final_answer` and the truncation flag — verbatim same protocol as arm A / the §6.1
  baseline.

============================================================================
Implementation-level registration (does not change the criteria; only degrees of freedom the PREREG left open)
============================================================================
- Dev early stopping uses max_new_tokens=256 (the fine-tuning target is one line
  `Final Answer [X].`, so 256 suffices; dev only selects the epoch and enters no readout).
  **Test inference remains 1024**, same protocol as the baseline.
- Gradient accumulation: micro batch 4 × accum 4 = effective 16. lr/epochs per PREREG.
- Ordering: training samples shuffled each epoch with seed 20260803+epoch.

Usage
  python scripts/run_lora_sft.py --selfcheck                       # CPU only
  CUDA_VISIBLE_DEVICES=6 python scripts/run_lora_sft.py --variant B1 --fold 0 --smoke
  CUDA_VISIBLE_DEVICES=6 python scripts/run_lora_sft.py --variant B1 --fold 0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from run_generative import BELIEF_REV, build_prompt  # noqa: E402  single source of the DP prompt
from run_probe_decode import CSV, MAIN_LV, main_folds, fold_of_atomic  # noqa: E402  single source of the shared folds
from src.prompts.utils import get_final_answer  # noqa: E402

OUT_DIR = os.path.join(HERE, '..', 'outputs', 'probe_decode')
MODEL = 'Qwen/Qwen3-4B'
SEED = 20260803
LR, MAX_EPOCHS, PATIENCE = 1e-4, 5, 1
MICRO_BS, ACCUM = 4, 4
DEV_MAX_NEW, TEST_MAX_NEW = 256, 1024
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
VARIANTS = {
    'B1': dict(layers_to_transform=None),          # all-linear, all layers
    'B2': dict(layers_to_transform=list(range(20, 36))),   # only layers 20–36
}


# ------------------------------------------------------------------ data and folds
def lv5_split(fold):
    """The 782 lv=5 rows + this fold's train/dev/test row indices (relative to the full CSV).

    Folds come from `run_probe_decode.main_folds()` (the same set as arm A). Extra
    assertion here: rerunning GroupKFold on the 782 rows yields the same atomic→fold map
    as the 391-row version — group sizes all double, so GroupKFold's greedy allocation
    stays the same, but that shouldn't rest on argument; assert it.
    """
    df = pd.read_csv(CSV)
    folds, groups, _ = main_folds(df)
    f_of = fold_of_atomic(folds, groups)

    lv5 = df[df.agreement_lv == MAIN_LV]
    assert len(lv5) == 782, f'lv=5 行数 {len(lv5)} != 782'
    from sklearn.model_selection import GroupKFold
    g2 = lv5['atomic_idx'].values
    f2 = list(GroupKFold(len(folds)).split(np.zeros(len(g2)), None, g2))
    f_of2 = fold_of_atomic(f2, g2)
    assert f_of == f_of2, 'V-B0 失败：782 行版折结构与 391 行版不同 —— A/B 同折前提破了'

    k_dev = (fold + 1) % len(folds)
    kk = np.array([f_of[int(a)] for a in g2])
    pos = lv5.index.values
    tr = pos[(kk != fold) & (kk != k_dev)]
    dv = pos[kk == k_dev]
    te = pos[kk == fold]

    A = {n: set(df.loc[ix, 'atomic_idx'].tolist()) for n, ix in
         (('train', tr), ('dev', dv), ('test', te))}
    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        assert not (A[a] & A[b]), f'V-B1 失败：{a}/{b} 的 atomic_idx 相交 {A[a] & A[b]}'
    return df, tr, dv, te, k_dev, A


def rows_to_prompts(df, idx):
    return [build_prompt(df.loc[i, 'questions'], 'dp') for i in idx]


def check_prompt_identity(df, model_name):
    """V-B2 identity cell: this script's DP prompt is verbatim identical to the `prompt` stored in the existing §6.1 greedy outputs.

    Arm B's training input, arm B's evaluation input, and the greedy baseline's input must
    be the same byte string — otherwise neither "iso-supervision" nor "paired comparison
    with greedy" holds.
    """
    safe = model_name.replace('/', '_').replace('-', '_')
    p = os.path.join(HERE, '..', 'outputs', 'generative', f'time_t1_{safe}_dp.jsonl')
    if not os.path.exists(p):
        return f'跳过（无 {os.path.basename(p)}）'
    n = 0
    for l in open(p):
        r = json.loads(l)
        assert r['prompt'] == build_prompt(df.loc[r['idx'], 'questions'], 'dp'), \
            f'V-B2 失败：idx {r["idx"]} 的 DP 提示与 §6.1 产物不逐字相同'
        n += 1
    return f'{n} 行与 §6.1 greedy 产物逐字相同  ✅'


def breu(records):
    """BREU = (BU accuracy + BM accuracy)/2; BU = gold c, BM = gold a/b."""
    bu = [r['correct'] for r in records if r['ground_truth'] == 'c']
    bm = [r['correct'] for r in records if r['ground_truth'] != 'c']
    if not bu or not bm:
        return float('nan')
    return (float(np.mean(bu)) + float(np.mean(bm))) / 2


# ------------------------------------------------------------------ training / inference
def encode(tok, prompt, target):
    """→ (input_ids, labels); labels have values only in the assistant span."""
    pre = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    a = tok(pre, add_special_tokens=False).input_ids
    b = tok(pre + target + tok.eos_token, add_special_tokens=False).input_ids
    assert b[:len(a)] == a, '分词在 prompt/assistant 边界处合并了，label 掩码会错位'
    lab = [-100] * len(a) + b[len(a):]
    return b, lab


def collate(batch, pad_id, device):
    import torch
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for j, (a, b) in enumerate(batch):
        ids[j, :len(a)] = torch.tensor(a)
        lab[j, :len(b)] = torch.tensor(b)
        att[j, :len(a)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def generate_records(model, tok, df, idx, max_new, batch, desc):
    import torch
    from tqdm import tqdm
    prompts = rows_to_prompts(df, idx)
    recs = []
    model.eval()
    for st in tqdm(range(0, len(idx), batch), desc=desc, leave=False):
        sub, ps = idx[st:st + batch], prompts[st:st + batch]
        texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False) for p in ps]
        enc = tok(texts, return_tensors='pt', padding=True).to(model.device)
        in_len = enc['input_ids'].shape[1]
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gens = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
        for j, i in enumerate(sub):
            row, gen = df.loc[i], gens[j]
            ids = out[j, in_len:].tolist()
            ext = get_final_answer(gen)
            recs.append({'idx': int(i), 'dataset_id': row['dataset_id'],
                         'atomic_idx': int(row['atomic_idx']), 'modus': row['modus'],
                         'agreement_lv': int(row['agreement_lv']),
                         'ground_truth': row['ground_truth'],
                         'raw_output': gen, 'extracted': ext,
                         'has_final_answer': 'final answer' in gen.lower(),
                         'n_new_tokens': (ids.index(tok.eos_token_id) + 1
                                          if tok.eos_token_id in ids else len(ids)),
                         'truncated': bool(len(ids) >= max_new
                                           and tok.eos_token_id not in ids),
                         'correct': ext == row['ground_truth']})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='B1', choices=list(VARIANTS))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--model_name', default=MODEL)
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--batch', type=int, default=16, help='推理批大小')
    ap.add_argument('--smoke', action='store_true', help='train 64 行 / dev,test 各 32 行 / 1 epoch')
    ap.add_argument('--selfcheck', action='store_true', help='纯 CPU：折与三集不交断言 + 样本渲染')
    args = ap.parse_args()

    df, tr, dv, te, k_dev, A = lv5_split(args.fold)
    print(f'# 臂 B iso-label LoRA SFT（PREREG_probe_decode.md §3）')
    print(f'# variant={args.variant}  fold={args.fold}（dev=折{k_dev}）  model={args.model_name}')
    print(f'# targets={TARGETS}  r16 α32 dropout0.05  '
          f'layers_to_transform={VARIANTS[args.variant]["layers_to_transform"]}')
    print(f'# lr={LR} epochs≤{MAX_EPOCHS} patience={PATIENCE} '
          f'micro_bs={MICRO_BS}×accum{ACCUM}  dev_max_new={DEV_MAX_NEW} '
          f'test_max_new={TEST_MAX_NEW}')
    print(f'[V-B0] 782 行版折结构 == 391 行版（A/B 同折）  ✅')
    print(f'[V-B1] train/dev/test 行数 = {len(tr)}/{len(dv)}/{len(te)}；'
          f'atomic 组数 = {len(A["train"])}/{len(A["dev"])}/{len(A["test"])}；两两不交  ✅')
    print(f'[V-B2] DP 提示识别单元格：{check_prompt_identity(df, args.model_name)}')

    if args.selfcheck:
        print(f'\n{"=" * 78}\nSFT 样本渲染（2 条；assistant 段即监督目标）\n{"=" * 78}')
        for i in list(tr)[:2]:
            row = df.loc[i]
            print(f'{"#" * 26} idx={i} dataset_id={row["dataset_id"]} '
                  f'modus={row["modus"]} 金标={row["ground_truth"]}')
            print('--- user ---')
            print(build_prompt(row['questions'], 'dp'))
            print('--- assistant（只在这一段算 loss）---')
            print(f'Final Answer [{row["ground_truth"]}].')
            print()
        print('[selfcheck] V-B0/V-B1 通过，未加载语言模型，无读数。SELFCHECK PASS')
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
    n_layer = model.config.num_hidden_layers
    lt = VARIANTS[args.variant]['layers_to_transform']
    if lt is not None:
        assert max(lt) < n_layer, f'layers_to_transform 超出模型层数 {n_layer}'
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=TARGETS, layers_to_transform=lt))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[LoRA] 模型层数={n_layer}  可训练参数={n_tr:,}')

    epochs = 1 if args.smoke else MAX_EPOCHS
    tr_idx, dv_idx, te_idx = list(tr), list(dv), list(te)
    if args.smoke:
        tr_idx, dv_idx, te_idx = tr_idx[:64], dv_idx[:32], te_idx[:32]

    samples = [encode(tok, build_prompt(df.loc[i, 'questions'], 'dp'),
                      f'Final Answer [{df.loc[i, "ground_truth"]}].') for i in tr_idx]
    print(f'[data] 训练样本 {len(samples)} 条；assistant 段 token 数 '
          f'{[sum(1 for x in s[1] if x != -100) for s in samples[:3]]}…')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best, best_state, bad = -1.0, None, 0
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
        dev_recs = generate_records(model, tok, df, dv_idx, DEV_MAX_NEW, args.batch,
                                    f'dev ep{ep}')
        d = breu(dev_recs)
        print(f'[epoch {ep}] train loss = {tot / max(nb, 1):.4f}  dev BREU = {d:.4f}  '
              f'(dev n={len(dev_recs)}, noFA={sum(1 for r in dev_recs if not r["has_final_answer"])})',
              flush=True)
        if d > best:
            best, bad = d, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items() if 'lora' in k}
        else:
            bad += 1
            if bad > PATIENCE:
                print(f'[early stop] dev BREU 连续 {bad} 轮未提升，停在 epoch {ep}')
                break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    print(f'[best] dev BREU = {best:.4f}')

    recs = generate_records(model, tok, df, te_idx, TEST_MAX_NEW, args.batch, 'test')
    for r in recs:
        r.update(arm='B', variant=args.variant, fold=args.fold, model=args.model_name,
                 dev_breu=best, smoke=bool(args.smoke))
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = '_smoke' if args.smoke else ''
    out_path = os.path.join(args.out_dir, f'B_{args.variant}_fold{args.fold}{suffix}.jsonl')
    with open(out_path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[done] {out_path}  n={len(recs)}  test BREU = {breu(recs):.4f}  '
          f'noFA={sum(1 for r in recs if not r["has_final_answer"])}  '
          f'trunc={sum(1 for r in recs if r["truncated"])}')


if __name__ == '__main__':
    main()
