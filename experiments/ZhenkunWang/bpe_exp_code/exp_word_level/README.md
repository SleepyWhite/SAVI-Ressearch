# exp_word_level — Word 级检错实验

对应报告**第 4 节 核心实验二**。

## 实验内容

将 subword token 聚合成 word（word surprise = max_{token∈word}(surprise)），在统一的 word 粒度下对比 Backrule 和 Tokflip 后向模型的检错能力。

**D1–D3**：分布匹配的错误 token 池下的 Word 级检错。

**D3-CTRL**：给逆向模型注入不匹配分布的错误 token（正向 top-500），验证频率混淆效应。

## 运行方式

```bash
python test_autonomy_dual_domain_v3.py [--n 200] [--L 128] [--device cuda:0] [--smoke]
```

## 依赖数据

所有路径相对于 `data/`：

| 数据 | 路径 |
|---|---|
| Forward 45M 模型 | `model_union_fwd_v1_20000_s512_st64_b32k_n10k/` |
| Backrule 45M 模型 | `model_union_bwd_backrule_v1_20000_s512_st64_b32k_n10k/` |
| Tokflip 45M 模型 | `model_union_bwd_fwdrule_v1_20000_tokflip_s512_st64_b32k_n10k/` |
| 正向 tokenization 验证集 | `tokenized_union_fwd_v1_20000/wiki_val.pt` |
| 逆向 tokenization 验证集 | `tokenized_union_bwd_backrule_v1_20000/wiki_val.pt` |
| 正向 tokenizer | `tokenizer_union_fwd_v1_20000/tokenizer.json` |
| 逆向 tokenizer | `tokenizer_union_bwd_backrule_v1_20000/tokenizer.json` |
| Union 词表 | `vocab_union_v1_20000.json` |

## 输出

结果保存至 `data/test_autonomy_dual_domain_v3_20k/`：
- `v3_results_L128_n200.json` — D1/D2/D3/D3-CTRL 的 Word-AUC、Word-Recall@K

> **注**：当前脚本默认使用 20k 词表（VOCAB_SIZE=38001）。报告第 4 节的 D1–D3 实验使用 10k 词表版本。
