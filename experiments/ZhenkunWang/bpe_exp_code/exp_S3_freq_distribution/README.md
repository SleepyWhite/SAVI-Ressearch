# exp_S3_freq_distribution — 训练集与测试集词频分布一致性验证

对应报告**补充实验 S3**（频率梯度检错）的推理条件——验证训练集和测试集 token 频率分布一致，确保低频 token 在两者间含义相同。

## 实验内容

验证训练集和测试集（验证集）的 token 频率分布是否一致。计算两者的 Spearman 秩相关、Pearson 相关、top-K 重合度，确认测试集中的低频词是否确实是训练集中的低频词。

## 运行方式

```bash
python test_autonomy_dual_domain_v5_2.py
```

## 依赖数据

| 数据 | 路径 |
|---|---|
| 正向训练集 | `data/tokenized_union_fwd_v1_10000/wiki_train.pt` |
| 正向验证集 | `data/tokenized_union_fwd_v1_10000/wiki_val.pt` |

## 输出

结果保存至 `data/test_autonomy_dual_domain_v5/`：
- `v5_2_train_val_freq.json` — Spearman ρ, Pearson r, top-K 重合度
- `v5_2_train_val_freq_dist.png` — 频率分布对比图
