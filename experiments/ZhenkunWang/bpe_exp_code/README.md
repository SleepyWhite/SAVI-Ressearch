# BPE 实验代码与数据

逆向 BPE 分词 + 后向语言模型检错实验的核心代码、数据与结果。

## 目录结构

```
bpe_exp_code/
├── train_backward_bpe.py              # 训练前向/逆向 BPE 分词器
├── build_union_tokenizers.py          # 构建并集词表分词器
├── train_union_lm.py                  # 训练前向/后向语言模型
├── decode_wiki_to_text.py             # GPT-2 token 解码为原始文本
├── exp_token_level_S1_S2_S4/          # §3 Token级检错 + S1生成文本 + S2扩展Recall + S4 Surprise-频率
├── exp_word_level/                    # §4 Word级检错 (D1-D3)
├── exp_S3_freq_gradient/              # S3 频率梯度检错
├── exp_S3_freq_distribution/          # S3 前提：训练/测试词频分布一致性
├── exp_S5_20k_vocab/                  # S5 20k词表验证
└── data/                              # 训练数据（分词器数据）
```

## 对应报告

详见 issue 。
