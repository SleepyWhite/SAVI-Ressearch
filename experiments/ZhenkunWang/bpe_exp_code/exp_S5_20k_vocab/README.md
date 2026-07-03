# exp_S5_20k_vocab — 20k 词表验证实验

对应报告**补充实验 S5**。

## 实验内容

扩大词表至 20,000 次合并（实际 vocab size = 38,001），重复 Token 级检错（Part A/B）和 Word 级检错（D1–D3）实验，验证 10k 词表下的核心结论是否在更大词表下仍然成立。

## 运行方式

```bash
python test_autonomy_dual_domain_v2.py [--n 200] [--L 128] [--device cuda:0] [--smoke]
```

## 依赖数据

所有路径相对于 `data/`：

| 数据 | 路径 |
|---|---|
| Forward 45M 模型 | `model_union_fwd_v1_20000_s512_st64_b32k_n10k/` |
| Backrule 45M 模型 | `model_union_bwd_backrule_v1_20000_s512_st64_b32k_n10k/` |
| Tokflip 45M 模型 | `model_union_bwd_fwdrule_v1_20000_tokflip_s512_st64_b32k_n10k/` |
| Forward 124M 模型 | `model_union_fwd_v1_20000_s512_st64_b16k_n10k_124M/` |
| Backrule 124M 模型 | `model_union_bwd_backrule_v1_20000_s512_st64_b16k_n10k_124M/` |
| 正向 tokenization 验证集 | `tokenized_union_fwd_v1_20000/wiki_val.pt` |
| 逆向 tokenization 验证集 | `tokenized_union_bwd_backrule_v1_20000/wiki_val.pt` |

## 输出

结果保存至 `data/test_autonomy_dual_domain_v2_20k/`：
- `v2_results_L128_n200.json` — Part A/B Token 级指标

> **注**：Word 级和频率梯度的 20k 版本分别由 `exp_word_level`（V3）和 `exp_S3_freq_gradient`（V4.2 `--vocab 20000`）覆盖。
