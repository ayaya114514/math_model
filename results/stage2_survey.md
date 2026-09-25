# 阶段 2：公开数学推理数据集调研（2026-09-25）

方法：只通过 HF datasets-server 抽样读取，没有下载完整数据集。每个数据集按固定 seed 随机取 5 个位置、每处连续 100 行，共 500 行（OpenThoughts3 只统计其中 domain=math 的 300 行）。
长度用学生模型的 Qwen2.5 tokenizer 计算，统计的是**回答部分**（含 `<think>` 推理）的 token 数。
复现：`.venv/bin/python -u src/survey_datasets.py configs/survey_datasets.yaml`，原始统计在 `results/stage2_survey.json`。

## 汇总

| 数据集 | 规模 | 许可证 | 老师模型 | 标准答案字段（能否自动验证） | 回答长度 p50 / p90（token） | ≤1k / ≤2k / ≤4k | 题目难度（抽样来源） | 下载体量 |
|---|---|---|---|---|---|---|---|---|
| **nvidia/OpenMathInstruct-2** | 14M（另有 1M/2M/5M 子集） | CC-BY-4.0 | Llama-3.1-405B-Instruct | ✅ `expected_answer` | 333 / 784 | 99% / 100% / 100% | train_1M：augmented_math 84%，augmented_gsm8k 14%，gsm8k/math 原题 2% | train_1M 0.64 GB |
| **meta-math/MetaMathQA** | 395k | MIT | GPT-3.5-Turbo（据论文，card 未写） | ⚠️ 无单独字段，答案在结尾的 "The answer is: X" | 151 / 323 | 100% / 100% / 100% | 全部由 GSM8K / MATH **训练集**改写（GSM 占约 62%） | 0.19 GB |
| AI-MO/NuminaMath-CoT | 859k | Apache-2.0 | GPT-4o 改写参考解答（据技术报告，card 未写） | ⚠️ 无单独字段，答案在解答的 `\boxed{}` 里 | 354 / 852 | 95% / 100% / 100% | cn_k12 29%，orca_math 21%，synthetic_math 19%，olympiads 18% | 1.23 GB |
| open-r1/OpenR1-Math-220k（default） | 94k | Apache-2.0 | DeepSeek-R1 | ✅ `answer` + Math-Verify 结果 | 4780 / 11625 | 1% / 12% / 42% | olympiads 72%，cn_contest 13%，aops 10% | 2.15 GB |
| nvidia/OpenMathReasoning（cot） | 3.2M | CC-BY-4.0 | DeepSeek-R1 + QwQ-32B | ✅ `expected_answer` | 6834 / 14319 | 1% / 8% / 27% | AoPS 高中竞赛 / 大学数学 | 大 |
| nvidia/Nemotron-SFT-Math-v4 | 545k | CC-BY-4.0 / CC-BY-SA-4.0 | DeepSeek-V4-Pro（High） | ✅ `expected_answer` | 6560 / 41989 | 4% / 16% / 38% | Math StackExchange 60%，AoPS 40% | 大 |
| open-thoughts/OpenThoughts3-1.2M | 1.2M（math 850k） | Apache-2.0 | QwQ-32B | ❌ 无 | 16699 / 16752（大量被截断在 16k） | 0% / 0% / 0% | 竞赛题 | 大 |

GSM8K 训练集自带的人工解答（7473 条，"弱老师"对照组）已在 `data/raw/gsm8k_train.jsonl` 中。

## 关键观察

1. **数据集分成两类，长度差 10–20 倍。**
   - 短 CoT 类（OpenMathInstruct-2、MetaMathQA、NuminaMath-CoT）：p90 < 900 token，和学生模型现在的输出长度（均值 311）同一量级。
   - R1 式长推理类（OpenR1、OpenMathReasoning、Nemotron、OpenThoughts3）：中位数就有 5k–17k token。
2. **长推理数据在本机上基本用不了。** 原因有三：
   - **训练**：18GB 内存下 LoRA 的序列长度实际只能到 2k 左右，而 OpenR1 只有 12% 的样本 ≤2k，
     剩下的这部分又偏向简单题，并不能代表原数据。
   - **评测**：学生学会长推理后，max_tokens 得从 1024 提到 8k 以上。评测时间会从 14 min 涨到几个小时，
     KV cache 也可能撑爆内存。
   - **效果**：已有研究（Li et al. 2025, *Small Models Struggle to Learn from Strong Reasoners*）发现，
     ≤3B 的小模型直接学长 CoT 效果常常不如学短 CoT，存在所谓的 "learnability gap"。
3. **题目难度要和评测集对得上。** GSM8K 是小学应用题；OpenR1、OpenMathReasoning、Nemotron 是竞赛或大学题。
   在 OpenMathInstruct-2 里，从 GSM8K 训练集改写来的题（augmented_gsm8k）在 train_1M 中约占 14%，
   按比例估计有十几万条，完全够抽 500 / 2000 / 10000 的子集。
4. **能不能自动验证：**
   - OpenMathInstruct-2 有 `expected_answer`，可以直接比对。
   - MetaMathQA 的答案在固定的 "The answer is:" 结尾，也能可靠提取。
   - NuminaMath-CoT 的解答本身就是由参考答案改写来的，没有独立的标准答案可以交叉验证。
5. **格式：** OpenMathInstruct-2 的答案 100% 在 `\boxed{}` 里，和我们的评测 prompt 完全一致。
   MetaMathQA 大多以 "The answer is: X" 结尾（只有 31% 带 boxed），转换时要把结尾统一改成 `\boxed{X}`。

## 推荐

**主数据：nvidia/OpenMathInstruct-2（train_1M 子集，0.64 GB）**，主要用 `augmented_gsm8k` + `gsm8k` 部分，
`augmented_math` 留给以后加 MATH 评测时使用。理由：
- 难度和 GSM8K 匹配；
- 长度适合 18GB 内存训练，也适合 1024 token 的评测预算；
- 有标准答案可以做自动验证；
- 格式和我们的评测一致；
- 老师（Llama-3.1-405B）比学生强得多；
- 许可证宽松。

**对照数据（建议一起下载，体量小）：meta-math/MetaMathQA（0.19 GB）。**
它和 OpenMathInstruct-2 的 GSM 部分都是从 **GSM8K 训练集**改写来的，题目分布接近，主要区别在老师：
GPT-3.5 对比 Llama-3.1-405B。再加上 GSM8K 人工解答，就形成了一个难度基本不变、只改老师的干净对照：
**人工 → GPT-3.5 → Llama-405B**。这正好对应阶段 5 "每次只改一个变量" 中的"数据来源 / 老师"这一项。

**暂不下载：** OpenR1-Math-220k 等长推理数据。如果以后想做"详细 vs 精简推理"的实验，可以只取 OpenR1 中 ≤2k token 的样本，
但要注意上面说的难度偏差和评测成本。NuminaMath-CoT 可以作为"GPT-4o、题目难度更宽"的备选。

## 测试集污染风险

- OpenMathInstruct-2 的作者声称对 GSM8K / MATH 测试集做过去污染，并发布了 contamination explorer。
  **我们仍会按计划自己去重**：规范化后精确匹配，再做 n-gram 重叠检查。
- MetaMathQA 的题目改写自训练集，但改写后的题可能和测试题相似，同样要去重。
