#!/usr/bin/env python
"""Diff whitelist assertion of PREREG_bon32.md §1: run_bon_v2_inc.py must be a mirror copy of run_bon_v2.py.

Allowed differences (whitelist, asserted line by line):
  (a) default output dir outputs/bon_v2  ->  outputs/bon_v2_inc
  (b) output file name gains a seed suffix (..._N{n}.jsonl -> ..._N{n}_seed{seed}.jsonl)
  (c) comments / docstrings (explanatory text referencing PREREG_bon32)

Any other difference (generation parameters, prompt construction, extraction
logic, teacher-forcing scoring…) = FAIL.

Usage:
  python selfcheck_bon32_mirror.py            # print full diff + line-by-line assertions, PASS/FAIL
  python selfcheck_bon32_mirror.py --negative # negative test: plant one parameter change in a temp copy; the assertion must blow up

Note: the task prompt asked to "write a --selfcheck"; adding that flag to the
mirror copy itself would be a code change outside the whitelist (and the
original script already uses --selfcheck for its scoring-consistency
self-check), hence this standalone script.
"""
import argparse
import ast
import difflib
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, 'run_bon_v2.py')
MIRROR = os.path.join(HERE, 'run_bon_v2_inc.py')

# whitelist (a)/(b): the only two allowed code-line replacements, pinned character for character
EXACT_PAIRS = [
    (
        "                                                     '..', 'outputs', 'bon_v2'))\n",
        "                                                     '..', 'outputs', 'bon_v2_inc'))\n",
    ),
    (
        "        out_path = os.path.join(args.out_dir, f'{args.dataset}_{safe}_{method}_N{args.n_samples}.jsonl')\n",
        "        out_path = os.path.join(args.out_dir, f'{args.dataset}_{safe}_{method}_N{args.n_samples}_seed{args.seed}.jsonl')\n",
    ),
]


def docstring_line_ranges(src_text):
    """Return the (start_lineno, end_lineno) set of all docstrings (module/class/function) in the file, 1-based closed intervals."""
    ranges = []
    tree = ast.parse(src_text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, 'body', [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ranges.append((body[0].lineno, body[0].end_lineno))
    return ranges


def in_ranges(lineno, ranges):
    return any(lo <= lineno <= hi for lo, hi in ranges)


def is_comment(line):
    return line.strip().startswith('#')


def check(orig_path, mirror_path, verbose=True):
    """Return (ok, violations). When verbose, print the full diff and the per-block verdicts."""
    a = open(orig_path).readlines()
    b = open(mirror_path).readlines()
    if verbose:
        print(f'=== full diff: {os.path.basename(orig_path)} -> {os.path.basename(mirror_path)} ===')
        sys.stdout.writelines(difflib.unified_diff(a, b, fromfile=orig_path, tofile=mirror_path, n=2))
        print('=== end diff ===\n')
    doc_a = docstring_line_ranges(''.join(a))
    doc_b = docstring_line_ranges(''.join(b))

    violations = []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        old_block, new_block = a[i1:i2], b[j1:j2]
        # whitelist (a)/(b): the block is exactly one pinned line-replacement pair
        if (len(old_block), len(new_block)) == (1, 1) and (old_block[0], new_block[0]) in EXACT_PAIRS:
            which = 'a' if EXACT_PAIRS.index((old_block[0], new_block[0])) == 0 else 'b'
            if verbose:
                print(f'[whitelist ({which})] line {i1 + 1}: OK  {old_block[0].strip()!r} -> {new_block[0].strip()!r}')
            continue
        # whitelist (c): every line in the block must lie inside a docstring or be a pure comment line
        ok_c = all(in_ranges(i + 1, doc_a) or is_comment(l) for i, l in zip(range(i1, i2), old_block)) \
            and all(in_ranges(j + 1, doc_b) or is_comment(l) for j, l in zip(range(j1, j2), new_block))
        if ok_c:
            if verbose:
                print(f'[whitelist (c)] lines {i1 + 1}-{i2} -> {j1 + 1}-{j2}: OK（docstring/注释）')
            continue
        violations.append((tag, i1 + 1, old_block, new_block))
        if verbose:
            print(f'[VIOLATION] {tag} @ orig line {i1 + 1}:')
            for l in old_block:
                print(f'  - {l.rstrip()}')
            for l in new_block:
                print(f'  + {l.rstrip()}')
    return (not violations), violations


def negative_test():
    """Negative case: plant one parameter change (temperature default 1.0 -> 0.9); check must FAIL."""
    src = open(MIRROR).read()
    needle = "ap.add_argument('--temperature', type=float, default=1.0)"
    assert needle in src, '反例测试前提失败：找不到 temperature 参数行'
    tampered = src.replace(needle, "ap.add_argument('--temperature', type=float, default=0.9)")
    with tempfile.NamedTemporaryFile('w', suffix='_tampered.py', delete=False) as f:
        f.write(tampered)
        tmp = f.name
    try:
        ok, violations = check(ORIG, tmp, verbose=False)
        assert not ok, '反例测试失败：塞了参数改动 check 却 PASS —— 断言器坏了'
        print(f'[negative] 反例测试 PASS：temperature 1.0->0.9 被抓住（{len(violations)} 处违规），断言器有效')
    finally:
        os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--negative', action='store_true')
    args = ap.parse_args()
    if args.negative:
        negative_test()
        return
    ok, violations = check(ORIG, MIRROR, verbose=True)
    print()
    if ok:
        print('[selfcheck] PASS — run_bon_v2_inc.py 与 run_bon_v2.py 的全部差异落在 PREREG_bon32 §1 白名单内')
    else:
        print(f'[selfcheck] FAIL — {len(violations)} 处白名单外差异（见上）')
        sys.exit(1)


if __name__ == '__main__':
    main()
