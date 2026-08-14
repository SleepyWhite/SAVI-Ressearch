# Belief-R 实验代码与报告

## 目录结构

```
belief_r_savi/
├── report/
│   ├── belief_r_report.pdf     # 实验报告（7 页，中文；同目录有 .md / .tex 源）
│   └── figures/                # 报告的 7 张图（PNG）
├── code/
│   ├── run/                    # 实验脚本，需 GPU（19 个）：直接作答基线、隐藏层提取、
│   │                           #   探针写回、LoRA 各臂（B1–B4L/B3L）、采样池 N=16/32、
│   │                           #   提示阶梯 / 少样本、似然打分 / 上下文内选择 / NLI 读法
│   ├── summarize/              # 重算脚本，纯 CPU（34 个）：主读数与 CI、仪器闸门、
│   │                           #   折划分稳健性、长度配平 / 分层、消融与上限
│   ├── figures/                # 作图脚本（2 个，数字硬编码并带出处注释）
│   ├── prompts/                # 冻结的 prompt 契约模块（被 run 脚本 import）
│   └── src/                    # 数据集配套工具（vendored，被 run 脚本 import）
└── data/                       # Belief-R 数据集放置处（不随仓分发，见下）
```

数据：**Belief-R**（Wilie et al., EMNLP 2024）的 `queries_time_t1.csv`，从原作者处获取后
放入 `data/`，或用环境变量 `BELIEF_R_CSV` 指向本地副本。模型（Qwen3-4B /
Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct）从 Hugging Face 拉取。脚本之间是扁平
import，运行前把 `code`、`code/run`、`code/summarize` 加入 `PYTHONPATH`。

## 对应报告

详见 issue；`report/` 下为报告全文与图。
