# exp_token_level_S1_S2_S4 — Token 级检错实验

对应报告**第 3 节 核心实验一**，同时覆盖补充实验 **S1**（Part B 生成文本检错）、**S2**（扩展 Recall@K）、**S4**（Surprise-频率关系）。

## 实验内容

在 Wikipedia 真实文本（Part A）和 124M 生成文本（Part B）上注入错误 token，使用 45M 模型检测。评测 Forward（前向 L→R）、Backrule（后向 R→L）、Tokflip（后向 L→R）三条路径的检错能力。

**Part D**：固定错误位置，用不同频率 rank 的 token 替换正确 token，直接测量模型 surprise 与错误 token 频率的关系。

## 运行方式

```bash
python test_autonomy_dual_domain_v5.py [--n 200] [--L 128] [--device cuda:0] [--smoke]
```

- `--n`: 测试序列数（默认 200）
- `--L`: 序列长度（默认 128）
- `--smoke`: 快速冒烟测试（n=2, L=64）

## 依赖数据

所有路径相对于 `data/`：

| 数据 | 路径 |
|---|---|
| Forward 45M 模型 | `model_union_fwd_v1_10000_s512_st64_b65k_n10k/` |
| Backrule 45M 模型 | `model_union_bwd_backrule_v1_10000_s512_st64_b65k_n10k/` |
| Tokflip 45M 模型 | `model_union_bwd_fwdrule_v1_10000_tokflip_s512_st64_b65k_n10k/` |
| Forward 124M 模型 | `model_union_fwd_v1_10000_s512_st64_b32k_n10k_124M/` |
| Backrule 124M 模型 | `model_union_bwd_backrule_v1_10000_s512_st64_b32k_n10k_124M/` |
| 正向 tokenization 验证集 | `tokenized_union_fwd_v1_10000/wiki_val.pt` |
| 逆向 tokenization 验证集 | `tokenized_union_bwd_backrule_v1_10000/wiki_val.pt` |

## 输出

结果保存至 `data/test_autonomy_dual_domain_v5/`：
- `v5_results_L128_n200.json` — Part A/B 指标（AUC, F0.5, R@K, Corr@K）
- `v5_surprise_vs_freq_L128_n200.json` — Part D 频率-surprise 数据
- `v5_surprise_vs_freq_L128_n200.png` — Part D 图表
