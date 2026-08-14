#!/usr/bin/env python
"""Arm B4: last-layer readout LoRA (PREREG_b4.md, Qwen3-4B only).

The question it asks: if the collapse really happens at the "hidden state →
token readout interface" (last layer linearly readable at 0.688 while behavior
sits at 0.52), then does adding trainable parameters to **only** that interface
— LoRA on `lm_head`, everything else frozen — plus B3's structured supervision,
reliably reach the level the last-layer representation allows?

The **only** difference vs B3 (`run_b3_lora.py`) is the LoRA attachment point:
    B3  target_modules = q/k/v/o/gate/up/down (all-linear, all layers, 33.0M trainable)
    B4  target_modules = ['lm_head']                        (≈2.47M trainable)
Data, 5 folds, supervision target, hyperparameters, early-stopping setup,
inference chain, fallback, and on-disk fields are all imported directly from B3
or copied verbatim. This file modifies no existing file.

============================================================================
Protocol (frozen before running, PREREG_b4.md §3)
============================================================================
- Training samples, label rule, folds, hyperparameters, early stopping,
  inference chain: see the module docstring of `run_b3_lora`. Whatever exists
  as a function in B3 (gold_state / target_line / state_of_relation /
  rel_bal_acc / gen_batch / step1_records / check_*) is imported and reused
  here, not copied. The training-loop body is copied — the two implementations
  differ only by three assertions and one target_modules; building an
  abstraction for that difference is not worth it.
- LoRA: `target_modules=['lm_head']`, r16 α32 dropout 0.05,
  layers_to_transform=None (lm_head belongs to no decoder layer; setting it
  would cause the module to be skipped).
  Expected trainable parameters = 16×(2560+151936) = 2,471,936.

============================================================================
Three B4-specific instrument assertions (PREREG §3, on the GPU path, right after model load)
============================================================================
- [G-B4-1] Every trainable parameter name contains `lm_head`, and the actual
  parameter count is registered. A wrong attachment point blows up here first.
- [G-B4-2] Qwen3-4B is a tie_word_embeddings=True model; lm_head shares one
  weight block with embed_tokens. Assert the output of
  `get_input_embeddings()(ids)` is bitwise identical **before and after**
  attaching LoRA — i.e. the LoRA increment only flows through the lm_head
  forward and does not leak into the embedding via the tied side.
  [G-B4-2b] After training finishes and best_state is loaded, test the same
  thing once more (if it leaks, it leaks during training).
- [G-B4-3] The mean train loss of epoch 0 must be below the loss of that
  epoch's first micro-batch (the lm_head LoRA is actually learning). If not,
  only a ⚠️ instrument warning is printed, no abort — whether to call the
  instrument broken on this basis is decided by a human at summary time.

`--selfcheck` is a pure-CPU path; none of the three above are reachable there
(they need a real model); the end of the selfcheck states this honestly.

Usage
  python scripts/run_b4_lora.py --selfcheck                      # pure CPU
  CUDA_VISIBLE_DEVICES=4 python scripts/run_b4_lora.py --fold 0 --smoke
  CUDA_VISIBLE_DEVICES=4 python scripts/run_b4_lora.py --fold 0
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
# B3 is this arm's skeleton. Importing it also runs its module-level G-B3-0 (both template
# shas == registered values) — those two assertions are the input-identity gate shared by
# A/C/B3/B4, so they are not rewritten here.
from run_b3_lora import (DEV_MAX_NEW, STEP1_MAX_NEW, STEP2_MAX_NEW,  # noqa: E402
                         STEP1_SHA_REG, TEMPLATE_SHA_REG,
                         check_gold_state_vs_C, check_label_rule, gen_batch,
                         gold_state, rel_bal_acc, state_of_relation,
                         step1_records, target_line, twin_step1_report)
from run_probe_decode import (TEMPLATE_SHA256, build_replace_prompt,  # noqa: E402
                              implied_answer)
from src.prompts.utils import get_final_answer  # noqa: E402

# The only experimental design difference vs B3 is the following line.
B4_TARGETS = ['lm_head']
LORA_LAYERS = None
N_TRAINABLE_EXPECT = 2_471_936  # 16×(2560+151936), expected value registered in PREREG §3
EMB_PROBE_TEXT = 'The umbrella is in the car.'  # small fixed text used by [G-B4-2]


# ------------------------------------------------------------------ B4-specific instrument assertions
def assert_trainable_is_lm_head(model):
    """[G-B4-1] every trainable parameter contains lm_head; returns (name list, parameter count)."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names, 'G-B4-1 失败：一个可训练参数都没有 —— LoRA 没挂上'
    bad = [n for n in names if 'lm_head' not in n]
    assert not bad, \
        f'G-B4-1 失败：{len(bad)}/{len(names)} 个可训练参数不在 lm_head 上，例如 {bad[:3]}'
    return names, sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)


def emb_forward(model, ids):
    """Embedding forward (no grad). Taken once before and once after attaching LoRA, compared bitwise."""
    import torch
    with torch.no_grad():
        return model.get_input_embeddings()(ids).detach().clone()


# ------------------------------------------------------------------ main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--model_name', default=MODEL)
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--batch', type=int, default=16, help='推理批大小')
    ap.add_argument('--smoke', action='store_true',
                    help='train 64 行 / dev 32 行 / test 按 modus 分层各 16 行 / 1 epoch')
    ap.add_argument('--selfcheck', action='store_true',
                    help='纯 CPU：折与三集不交断言 + sha + 标签闭环 + 样本与推理链渲染')
    args = ap.parse_args()

    df, tr, dv, te, k_dev, A = lv5_split(args.fold)
    print('# 臂 B4 末层读出 LoRA（PREREG_b4.md §3）')
    print(f'# fold={args.fold}（dev=折{k_dev}）  model={args.model_name}')
    print(f'# targets={B4_TARGETS}  r16 α32 dropout0.05  layers_to_transform={LORA_LAYERS}')
    print(f'# 与 B3 的唯一差别 = target_modules：B3 {TARGETS} → B4 {B4_TARGETS}')
    print(f'# lr={LR} epochs≤{MAX_EPOCHS} patience={PATIENCE} '
          f'micro_bs={MICRO_BS}×accum{ACCUM}  dev_max_new={DEV_MAX_NEW} '
          f'step1_max_new={STEP1_MAX_NEW} step2_max_new={STEP2_MAX_NEW}')
    print(f'# STEP1_SHA256    = {run_twostep.STEP1_SHA256}')
    print(f'# TEMPLATE_SHA256 = {TEMPLATE_SHA256}')
    assert run_twostep.STEP1_SHA256 == STEP1_SHA_REG and TEMPLATE_SHA256 == TEMPLATE_SHA_REG
    print('[G-B3-0] step1 模板 sha 与注入模板 sha 均 == 登记值（与臂 C/A/B3 同一串字节）  ✅')
    print('[V-B0] 782 行版折结构 == 391 行版（与 A/B1/B3 同折）  ✅')
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
              f'与 B3 逐字相同）\n{"=" * 78}')
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
        # fallback — under tollens ALT implies b (not a); if this cell can't be rendered,
        # nothing was checked.
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
        print(f'[B4] 与 B3 的唯一差别 = target_modules：{TARGETS} → {B4_TARGETS}'
              f'（数据/折/监督/超参/早停/推理链/回退/字段全部相同）')
        print(f'[B4] 预期可训练参数 = 16×(2560+151936) = {N_TRAINABLE_EXPECT:,}；'
              f'实际值由 GPU 路径的 [G-B4-1] 登记')
        print('[selfcheck] G-B3-0 / V-B0…V-B4 全部通过，未加载语言模型，无读数。')
        print('[selfcheck] B4 特有的三条断言（G-B4-1 可训练参数名 / G-B4-2 embedding 前向'
              '不变 / G-B4-3 epoch0 loss 下降）需要真模型，纯 CPU 自检里跑不到，'
              '只在 GPU 路径上执行。')
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

    # The premise of [G-B4-2] must itself be measured: do lm_head and embed_tokens really share one weight block.
    tied = (model.get_output_embeddings().weight.data_ptr()
            == model.get_input_embeddings().weight.data_ptr())
    probe_ids = tok(EMB_PROBE_TEXT, return_tensors='pt').input_ids.to(model.device)
    emb_before = emb_forward(model, probe_ids)

    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=B4_TARGETS, layers_to_transform=LORA_LAYERS))

    names, n_tr = assert_trainable_is_lm_head(model)
    print(f'[LoRA] 模型层数={model.config.num_hidden_layers}  可训练参数={n_tr:,}')
    print(f'[G-B4-1] 可训练参数 {len(names)} 个，逐一含 "lm_head"  ✅  {names}')
    print(f'[G-B4-1] 实际可训练参数量 = {n_tr:,}（PREREG 预期 {N_TRAINABLE_EXPECT:,}'
          f'{"，一致" if n_tr == N_TRAINABLE_EXPECT else "，不一致 ⚠️"}）')
    emb_after = emb_forward(model, probe_ids)
    assert torch.equal(emb_before, emb_after), \
        'G-B4-2 失败：挂 LoRA 后 embedding 前向变了 —— 增量漏进了绑定的输入侧'
    print(f'[G-B4-2] 权重绑定 = {tied}；挂 LoRA 后 embedding 前向逐位不变'
          f'（探针 {probe_ids.shape[1]} token，bf16 全等）  ✅')

    epochs = 1 if args.smoke else MAX_EPOCHS
    tr_idx, dv_idx, te_idx = list(tr), list(dv), list(te)
    if args.smoke:
        # Test slice takes 16 per modus stratum: the CSV has the ponens block first, so
        # taking the first 32 rows would be all ponens, while step2's implied answer
        # depends on modus (ALT→b under tollens) — that path would never get smoked.
        # Affects only the --smoke branch; the full path is untouched. Copied from B3.
        tr_idx, dv_idx = tr_idx[:64], dv_idx[:32]
        te_idx = ([i for i in te_idx if df.loc[i, 'modus'] == 'ponens'][:16]
                  + [i for i in te_idx if df.loc[i, 'modus'] == 'tollens'][:16])

    samples = [encode(tok, run_twostep.build_step1(df.loc[i]), target_line(df.loc[i]))
               for i in tr_idx]
    print(f'[data] 训练样本 {len(samples)} 条；assistant 段 token 数 '
          f'{[sum(1 for x in s[1] if x != -100) for s in samples[:3]]}…')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    first_loss = None  # [G-B4-3] first micro-batch loss of epoch 0
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
            if ep == 0 and first_loss is None:
                first_loss = float(loss)
            tot += float(loss); nb += 1
            if (s // MICRO_BS + 1) % ACCUM == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        mean_loss = tot / max(nb, 1)
        if ep == 0:
            if mean_loss < first_loss:
                print(f'[G-B4-3] epoch0 平均 train loss {mean_loss:.4f} < 首个 micro-batch '
                      f'loss {first_loss:.4f}  ✅（lm_head LoRA 在学）')
            else:
                print(f'⚠️ [G-B4-3] 仪器警告：epoch0 平均 train loss {mean_loss:.4f} '
                      f'未低于首个 micro-batch loss {first_loss:.4f} —— '
                      f'lm_head LoRA 可能没在学。不中断，读数按 PREREG §2 先查仪器。')
        dev_recs = step1_records(model, tok, df, dv_idx, DEV_MAX_NEW, args.batch,
                                 f'dev ep{ep}')
        d = rel_bal_acc(dev_recs)
        n_fail = sum(1 for r in dev_recs if r['relation'] is None)
        print(f'[epoch {ep}] train loss = {mean_loss:.4f}  '
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

    # [G-B4-2b] test once more after training + loading best_state: if it leaks into the tied input side, it leaks during training.
    assert torch.equal(emb_before, emb_forward(model, probe_ids)), \
        'G-B4-2b 失败：训练后 embedding 前向变了 —— 梯度漏到了绑定的输入侧'
    print('[G-B4-2b] 训练并载入 best_state 后，embedding 前向仍逐位不变  ✅')

    # ---- inference chain step 1: fine-tuned model judges the relation ----
    s1 = step1_records(model, tok, df, te_idx, STEP1_MAX_NEW, args.batch, 'B4 step1')

    # ---- step 2: release the fine-tuned model. step2 must be executed by the **untuned original model** (variable isolated to the classifier) ----
    del model, opt, best_state, samples, emb_before, emb_after, probe_ids
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
    s2 = gen_batch(base, tok, prompts, STEP2_MAX_NEW, args.batch, 'B4 step2')

    recs = []
    for r, p, (g, trunc, ntok) in zip(s1, prompts, s2):
        row = df.loc[r['idx']]
        rel = r['relation']
        conn = None if rel is None else ('and' if rel == '1' else 'or')
        ext = get_final_answer(g)
        imp = None if conn is None else implied_answer(conn, row['modus'])
        recs.append({
            'arm': 'B4', 'idx': int(r['idx']), 'dataset_id': row['dataset_id'],
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
    out_path = os.path.join(args.out_dir, f'B4_fold{args.fold}{suffix}.jsonl')
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
