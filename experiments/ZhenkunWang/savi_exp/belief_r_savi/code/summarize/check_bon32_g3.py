#!/usr/bin/env python
"""PREREG_bon32.md §2 G-3: parameter and prompt identity check (new pool vs original pool, per idx).

For every row already present in outputs/bon_v2_inc/ (first 8 rows in smoke, 1744 rows in full):
  - record structure: field set isomorphic to the original file; fields within chains isomorphic
  - generation-parameter fields equal field by field (temperature/top_k/top_p/n_samples/model/method),
    except seed (orig=0, new=1; asserted to be exactly these values)
  - metadata (dataset_id/atomic_idx/modus/intent/agreement_lv/ground_truth) equal field by field
  - prompt string verbatim identical to the original file's same-idx row (all rows compared;
    PREREG requires sampling >=20 rows, here the full set is done)
Any assertion failure -> print details and exit nonzero.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BR = os.path.join(HERE, '..')
ORIG_DIR = os.path.join(BR, 'outputs', 'bon_v2')
INC_DIR = os.path.join(BR, 'outputs', 'bon_v2_inc')

PARAM_FIELDS = ('temperature', 'top_k', 'top_p', 'n_samples', 'model', 'method')
META_FIELDS = ('dataset_id', 'atomic_idx', 'modus', 'intent', 'agreement_lv', 'ground_truth')


def main():
    total_checked = 0
    fail = 0
    for m in ('dp', 'cot', 'ps'):
        orig_path = os.path.join(ORIG_DIR, f'time_t1_Qwen_Qwen3_4B_{m}_N16.jsonl')
        inc_path = os.path.join(INC_DIR, f'time_t1_Qwen_Qwen3_4B_{m}_N16_seed1.jsonl')
        if not os.path.exists(inc_path):
            print(f'[G-3] {m}: 新池文件不存在，跳过')
            continue
        orig = {}
        for l in open(orig_path):
            r = json.loads(l)
            orig[r['idx']] = r
        n = 0
        for l in open(inc_path):
            ri = json.loads(l)
            r = orig.get(ri['idx'])
            if r is None:
                print(f'[G-3 FAIL] {m} idx={ri["idx"]}: 原文件无此 idx')
                fail += 1
                continue
            if set(ri.keys()) != set(r.keys()):
                print(f'[G-3 FAIL] {m} idx={ri["idx"]}: 字段集合不同构 '
                      f'only_new={set(ri) - set(r)} only_orig={set(r) - set(ri)}')
                fail += 1
            for c in ri['chains']:
                if set(c.keys()) != set(r['chains'][0].keys()):
                    print(f'[G-3 FAIL] {m} idx={ri["idx"]}: chain 字段不同构 {sorted(c.keys())}')
                    fail += 1
                    break
            for k in PARAM_FIELDS:
                if ri[k] != r[k]:
                    print(f'[G-3 FAIL] {m} idx={ri["idx"]}: 参数 {k} 不等: new={ri[k]!r} orig={r[k]!r}')
                    fail += 1
            if not (r['seed'] == 0 and ri['seed'] == 1):
                print(f'[G-3 FAIL] {m} idx={ri["idx"]}: seed 应为 orig=0/new=1，实际 {r["seed"]}/{ri["seed"]}')
                fail += 1
            for k in META_FIELDS:
                if ri[k] != r[k]:
                    print(f'[G-3 FAIL] {m} idx={ri["idx"]}: 元数据 {k} 不等: new={ri[k]!r} orig={r[k]!r}')
                    fail += 1
            if ri['prompt'] != r['prompt']:
                print(f'[G-3 FAIL] {m} idx={ri["idx"]}: 提示串不同（new len={len(ri["prompt"])} '
                      f'orig len={len(r["prompt"])}）')
                fail += 1
            n += 1
        total_checked += n
        print(f'[G-3] {m}: 比对 {n} 行（参数逐字段 + 提示串逐字 + 结构同构）')
    print(f'[G-3] 共比对 {total_checked} 行，失败断言 {fail} 处 -> '
          + ('PASS' if fail == 0 and total_checked > 0 else 'FAIL'))
    if fail or total_checked == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
