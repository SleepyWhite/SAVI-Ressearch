# MuSR 实验代码和报告

## 目录结构

```
musr_savi/
├── report/
│   ├── musr_report.pdf         # 实验报告（5 页，中文；同目录有 .md / .tex 源）
│   ├── musr_report.md
│   └── musr_report.tex
├── code/
│   ├── run_musr_cant.py        # 唯一入口：python code/run_musr_cant.py --stage <阶段>
│   │                           #   阶段依次 audit|a0|a1|a2|a3|b0|b1|b2|g0|g1|g2|g3|g4
│   ├── data_musr.py            # MuSR 数据下载 / 审计（--stage audit 自动拉取）
│   ├── sc_core.py              # 采样协议、多数投票主基线、JSONL 缓存
│   ├── stages.py               # 认证漏斗：F1 预算递增、F2 四门、F3 判定
│   ├── selffacts.py            # F3 自提取控制的两种注入方式
│   ├── stages_b.py             # F4 模板预检；learned-verifier 重排与 gold-facts 掩码读出
│   ├── domain_musr.py          # 跨链归并 trellis、λ 软加权解码 / oracle 掩码 / BoN / SC
│   ├── featurizer.py           # learned-verifier 的隐状态特征
│   ├── verifiers.py            # 步级事实一致性探针与 critic 对照
│   ├── phase_b_split.py        # verifier 训练的故事级防泄漏切分
│   ├── gsd_space.py            # SAVI 支撑轴：逐步枚举可达信念状态
│   ├── gsd_score.py            # 每 (层, 前状态, 候选) 的 TF 似然打分；event 句模板三档
│   ├── gsd_decode.py           # 精确 Viterbi + 终端信念表机械读出
│   ├── gsd_sample.py           # 匹配语境条件采样（g2 主件，g3/g4 复用缓存）
│   ├── gsd_extract.py          # model-written 事件句的提取 / 对齐 / 质量度量
│   ├── stages_g.py             # g0/g1：保真度 ρ、读出消融、知识对照选取、gold arm 主结果
│   ├── stages_g2.py            # 频率打分对照（特异性判别）
│   ├── stages_g3.py            # event 句消融（none / placeholder 两臂）
│   ├── stages_g4.py            # 分叉重放（站 C）+ model-written arm（站 D）
│   ├── belief_schema.py        # BELIEF 行解析与状态规范化
│   ├── facts_oracle.py         # 金标事实、状态一致性、机械读出的金标侧
│   └── prevalence/             # 流行率两脚本（纯 CPU，重放 b1 链缓存）
└── data/
    ├── results_g0.json         # 认证 19 题 + 3 知识对照 + 4 零覆盖题的身份档
    └── PROVENANCE.md           # 身份档来源说明
```

依赖：PyTorch、transformers、numpy、scikit-learn、joblib（可选 matplotlib / datasets /
tqdm）。模型从 Hugging Face 拉取；MuSR 数据由 `--stage audit` 自动获取到 `code/data/`。

## 对应报告

详见 issue。
