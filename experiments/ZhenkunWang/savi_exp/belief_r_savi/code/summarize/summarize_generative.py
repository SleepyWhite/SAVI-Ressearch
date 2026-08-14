#!/usr/bin/env python
"""Summarize the P0 generative baseline: BU/BM/BREU (equal weight) + the modus four cells + true format-failure counts.

Format-failure caveat: when 'final answer' is absent from the output, upstream
get_final_answer gets rfind = -1 and then grabs an irrelevant letter from
full_answer[10:], so format_ok can be a false positive. Here the count is
redone as 'does final answer appear in the output', and illegal letters (not
a/b/c) are counted separately. Per the paper protocol format errors always
count as wrong — accuracy is unaffected; only the format diagnostics are.
"""
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'generative')
EXPECTED = 1744


def acc(rows, sel):
    s = [r for r in rows if sel(r)]
    return (sum(r['correct'] for r in s) / len(s), len(s)) if s else (float('nan'), 0)


hdr = (f"{'model':<32} {'m':<4} {'n':>5} {'noFA':>5} {'illeg':>6} {'BU':>7} {'BM':>7} {'BREU':>7} "
       f"{'BU-pon':>7} {'BU-tol':>7} {'BM-pon':>7} {'BM-tol':>7}")
print(hdr)
print('-' * len(hdr))
for f in sorted(glob.glob(os.path.join(OUT, '*.jsonl'))):
    rows = [json.loads(l) for l in open(f)]
    stem = os.path.basename(f)[len('time_t1_'):-len('.jsonl')]
    model, method = stem.rsplit('_', 1)
    nofa = sum('final answer' not in r['raw_output'].lower() for r in rows)
    illegal = sum(r['extracted'] not in ('a', 'b', 'c', '') for r in rows)
    bu, bm = acc(rows, lambda r: r['ground_truth'] == 'c'), acc(rows, lambda r: r['ground_truth'] != 'c')
    cells = [acc(rows, lambda r, g=g, m=m: (r['ground_truth'] == 'c') == g and r['modus'] == m)
             for g in (True, False) for m in ('ponens', 'tollens')]
    flag = '' if len(rows) == EXPECTED else f'  <-- INCOMPLETE {len(rows)}/{EXPECTED}'
    print(f'{model:<32} {method:<4} {len(rows):>5} {nofa:>5} {illegal:>6} '
          f'{bu[0]:>7.4f} {bm[0]:>7.4f} {(bu[0] + bm[0]) / 2:>7.4f} '
          + ' '.join(f'{c[0]:>7.3f}' for c in cells) + flag)
