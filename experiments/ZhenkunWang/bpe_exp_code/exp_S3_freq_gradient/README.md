# exp_S3_freq_gradient — 频率梯度检错实验

对应报告**补充实验 S3**（频率梯度检错）。

## 实验内容

将 token 按频率排序后划分为 14 个梯度（每 500 rank 一个区间），每个区间内的 token 作为独立的错误注入池，测试 D1（Forward）、D2（Tokflip）、D3（Backrule）在各频率区间的 Word 级检错能力。每个模型使用自己数据上的频率排序。

## 运行方式

```bash
# 10k 词表
python test_autonomy_dual_domain_v4_2.py --vocab 10000 [--n 200] [--L 128] [--device cuda:0] [--smoke]

# 20k 词表
python test_autonomy_dual_domain_v4_2.py --vocab 20000 [--n 200] [--L 128] [--device cuda:0] [--smoke]
```

## 依赖数据

| Vocab | 模型/数据路径模式 |
|---|---|
| 10k | `model_union_fwd_v1_10000_*`, `tokenized_union_fwd_v1_10000`, `tokenized_union_bwd_backrule_v1_10000` |
| 20k | `model_union_fwd_v1_20000_*`, `tokenized_union_fwd_v1_20000`, `tokenized_union_bwd_backrule_v1_20000` |

## 输出

结果保存至 `data/test_autonomy_dual_domain_v4_2/`：
- `v4_2_results_L128_n200.json` — 14 个频率梯度的 W-AUC、W-R@1、W-R@5
- `v4_2_freq_gradient_L128_n200.png` — 频率梯度曲线图
