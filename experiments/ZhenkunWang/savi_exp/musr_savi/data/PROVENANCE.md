# data/results_g0.json 的来源

`run_musr_cant.py --stage g0` 的原始输出(2026-07-08 运行),从内部实验目录
`musr-cant/outputs/results_g0.json` 逐位拷入(sha256 前 16 位 `0d309668ab0e0a56`)。

随仓收录它只为一件事:**认证集的身份是结果的一部分**。文件里有:

- `scope.ga_patient_ids` — 认证漏斗 F1–F4 之后的 **19 道认证题** id;
- `config.prereg_g.knowledge_control_ids` — **3 道知识对照** id
  (`0020-q1 / 0034-q3 / 0035-q0`,从 F2 第 (ii) 门剔除的 46 道知识失败题按种子抽样,
  抽样代码 `code/stages_g.py:select_knowledge_controls`);
- `config.prereg_g.zero_coverage_ids` — 19 题中的 **4 道零覆盖题** id;
- `scope.ga_tuning_ids` — 40 道调参/流行率切片题 id(流行率的题级分母);
- `ga` — G-A 保真度门的原始读数(median Spearman ρ=0.400);
- `ablation` — 回溯读出消融(机械读出 − 票读出 = +1)的原始读数。

漏斗上游账本(256→107→105→26→21→19 的逐桶 id)在内部的
`results_a1/a2/a3/b1.json` 中,不随仓分发;重跑对应 stage 可再生。
