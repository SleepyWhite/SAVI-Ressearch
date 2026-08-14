#!/usr/bin/env python
"""PREREG_bon32.md §1 merge: per idx, concatenate the original 16 chains (original order) + the new 16 chains (original order) into 32-chain rows.

Input:  outputs/bon_v2/time_t1_Qwen_Qwen3_4B_{m}_N16.jsonl (read-only, not one byte touched)
        outputs/bon_v2_inc/time_t1_Qwen_Qwen3_4B_{m}_N16_seed1.jsonl (seed=1 new pool)
Output: outputs/bon_v2_merged32/time_t1_Qwen_Qwen3_4B_{m}_N32.jsonl

Assertions (any failure exits nonzero, nothing written):
  - both sides have exactly 1744 rows and identical idx sets
  - every merged row has 32 chains
  - the merged row's first 16 chains are byte-identical to the original file (serialized comparison)
  - dataset_id / ground_truth / modus / prompt agree across the two sides for the same idx

Row fields: keep all original-row fields; chains replaced by the 32 chains, n_samples set
to 32, seed kept at 0 (the seed of the first 16 chains), plus new fields seed_last16=1 and
merged_from recording provenance.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BR = os.path.join(HERE, '..')
ORIG_DIR = os.path.join(BR, 'outputs', 'bon_v2')
INC_DIR = os.path.join(BR, 'outputs', 'bon_v2_inc')
OUT_DIR = os.path.join(BR, 'outputs', 'bon_v2_merged32')

N_ROWS = 1744


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for m in ('dp', 'cot', 'ps'):
        orig_path = os.path.join(ORIG_DIR, f'time_t1_Qwen_Qwen3_4B_{m}_N16.jsonl')
        inc_path = os.path.join(INC_DIR, f'time_t1_Qwen_Qwen3_4B_{m}_N16_seed1.jsonl')
        out_path = os.path.join(OUT_DIR, f'time_t1_Qwen_Qwen3_4B_{m}_N32.jsonl')

        orig = [json.loads(l) for l in open(orig_path)]
        inc = {}
        for l in open(inc_path):
            r = json.loads(l)
            assert r['idx'] not in inc, f'{m}: inc 文件 idx={r["idx"]} 重复'
            inc[r['idx']] = r

        assert len(orig) == N_ROWS, f'{m}: 原文件行数 {len(orig)} != {N_ROWS}'
        assert len(inc) == N_ROWS, f'{m}: 新池行数 {len(inc)} != {N_ROWS}'
        assert {r['idx'] for r in orig} == set(inc), f'{m}: idx 集合不一致'

        n_out = 0
        with open(out_path, 'w') as fout:
            for r in orig:
                ri = inc[r['idx']]
                for k in ('dataset_id', 'ground_truth', 'modus', 'prompt'):
                    assert r[k] == ri[k], f'{m} idx={r["idx"]}: 字段 {k} 两侧不一致'
                assert len(r['chains']) == 16 and len(ri['chains']) == 16, \
                    f'{m} idx={r["idx"]}: 链数 {len(r["chains"])}/{len(ri["chains"])} != 16'
                merged = dict(r)
                merged['chains'] = r['chains'] + ri['chains']
                merged['n_samples'] = 32
                merged['seed_last16'] = ri['seed']
                merged['merged_from'] = [os.path.basename(orig_path), os.path.basename(inc_path)]
                assert len(merged['chains']) == 32
                assert json.dumps(merged['chains'][:16], ensure_ascii=False) == \
                    json.dumps(r['chains'], ensure_ascii=False), \
                    f'{m} idx={r["idx"]}: 前 16 条与原文件不同'
                fout.write(json.dumps(merged, ensure_ascii=False) + '\n')
                n_out += 1
        assert n_out == N_ROWS
        print(f'[merge] {m}: {n_out} rows x 32 chains -> {out_path}')
    print('[merge] all assertions passed')


if __name__ == '__main__':
    main()
