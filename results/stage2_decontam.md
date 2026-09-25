# 阶段 2：数据下载、统一格式与测试集去重（2026-09-25）

## 下载与转换

| 数据集 | 下载内容 | 转换后条数 | 老师 | 标准答案（gold） |
|---|---|---|---|---|
| OpenMathInstruct-2 | `train_1M` 子集（0.6 GB） | 1,000,000 | Llama-3.1-405B-Instruct | 100%（`expected_answer`） |
| MetaMathQA | 全量（0.4 GB） | 395,000 | GPT-3.5-Turbo | 72%：AnsAug / Rephrased 通过 `original_question` 回查 GSM8K / MATH 训练集原题得到（回查成功 284,905 / 285,000）；FOBAR / SV 改变了所求量，无 gold |
| NuminaMath-CoT | 全量（1.2 GB） | 859,494 | GPT-4o | 无（解答本身由参考解答改写而来） |
| OpenR1-Math-220k | `default` 配置（2.0 GB） | 93,733 | DeepSeek-R1 | 100%（`answer`），另保留 Math-Verify 结果 |
| GSM8K 训练集人工解答 | 已有 | 7,473 | 人工 | 100% |
| MATH（EleutherAI/hendrycks_math） | train 7,500 / test 5,000 | — | — | test 用于去重和以后的评测；train 用于 MetaMathQA 回查 |

统一字段：`id, question, reasoning, final_answer, gold, source, dataset, teacher`（OpenR1 另有 `math_verify`、`reasoning_complete`）。
代码：`src/convert_datasets.py`。转换条数和 HF 元数据完全一致。

## 去重方法

比对对象：GSM8K test（1,319）+ MATH test（5,000，MATH-500 是它的子集）。代码 `src/decontaminate.py`，配置 `configs/decontam.yaml`。

1. **规范化后精确匹配**：小写化，去掉空白、`$` 和 LaTeX 间距命令，**保留所有数学符号**。
   第一版规范化会把负号和括号也删掉，导致 `sin(-120°)` 被误判成与 `sin 120°` 相同。这个问题已经修复并复核过。
2. **词级 8-gram 覆盖率**：计算一条训练题覆盖了某道测试题多少比例的 8-gram。分母用测试题的 8-gram 数，
   这样训练题再长也不会稀释重合比例。有 207 道测试题不到 8 个词，只能依靠精确匹配。
3. **阈值按测试集分开设**，依据是人工查看 `--scan` 输出的样例：
   - **GSM8K：0.2**。GSM8K 是叙事型应用题，覆盖率 ≥ 0.2 的基本都是"同一故事模板换数字、换人名"，
     例如训练题 "A train travels 270 miles in 3 hours…" 对测试题 "A plane travels 1200 miles in 3 hours…"。
     GSM8K 是主评测集，宁严勿松。
   - **MATH：0.5**。MATH 题里套话很多，比如 "Express your answer as a common fraction"、"Find the value of $x$ that satisfies"，
     阈值取 0.3 时会误删大量不相关的题。0.5–0.7 区间主要是"同一题设、换了问法"的近亲题，删掉更保险。

## 去重结果

| 数据集 | 原始 | 删除 | 删除率 | 其中命中 GSM8K test | 其中命中 MATH test | 去重后 |
|---|---|---|---|---|---|---|
| GSM8K 人工解答 | 7,473 | 4 | 0.05% | 4 | 0 | 7,469 |
| OpenMathInstruct-2 | 1,000,000 | 7,426 | 0.74% | 37 | 7,389 | 992,574 |
| MetaMathQA | 395,000 | 4,650 | 1.18% | 102 | 4,548 | 390,350 |
| NuminaMath-CoT | 859,494 | 3,762 | 0.44% | 13 | 3,749 | 855,732 |
| OpenR1-Math-220k | 93,733 | 148 | 0.16% | 0 | 148 | 93,585 |

- **GSM8K 官方训练集本身就有 4 道题是测试题的模板变体**（覆盖率 0.24–0.67），例如 train-01314 / train-05162 对 test-00602。
  MetaMathQA 和 OMI2 会把每道训练题改写多次，这种重叠也被放大了：MetaMathQA 里为此删掉 102 条，原题只有 4 道。
  **训练时要用去重后的 `data/raw/gsm8k_human.jsonl`，不能用 `gsm8k_train.jsonl`。**
- 命中大多来自 MATH test：OMI2 的 augmented_math 和 MetaMathQA 的 MATH_* 都是由 MATH 训练集改写来的，
  而 MATH 训练集和测试集之间本身就有很多近亲题。
- 独立复核：用另一种规范化方式再查了一遍，去重后的 5 个文件里与测试题完全相同的题都是 0 条。
- 被删样本及命中的测试题都存在 `data/raw/removed/<dataset>.jsonl`，可以抽查。完整统计和样例见 `results/stage2_decontam.json`。

## 去重后可用规模（阶段 3 抽样的池子）

| 数据集 | 条数 | 与 GSM8K 难度接近的部分 |
|---|---|---|
| OpenMathInstruct-2 | 992,574 | augmented_gsm8k 138,520 + gsm8k 14,756 |
| MetaMathQA | 390,350 | GSM_* 共 239,898（AnsAug 79,951 / Rephrased 80,000 带 gold） |
| NuminaMath-CoT | 855,732 | orca_math 153,295（小学应用题）；另有 gsm8k 源少量 |
| OpenR1-Math-220k | 93,585 | 无（竞赛题为主） |
| GSM8K 人工 | 7,469 | 全部 |
