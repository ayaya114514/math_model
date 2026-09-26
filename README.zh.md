# 在笔记本上做数学推理蒸馏

[English](README.md) · 完整实验日志：[EXPERIMENTS.md](EXPERIMENTS.md)

这是一个对照研究，要回答的问题是：**什么样的蒸馏数据真正能让小模型的数学变好**。

学生模型是 Qwen2.5-1.5B-Instruct。我们用 7 种推理数据分别对它做 LoRA 微调，从人工解答到 DeepSeek-R1 都有，
然后在 GSM8K 和 MATH-500 上评测，并用配对显著性检验判断差异是否可信。
全部实验在一台 MacBook Pro（M3 Pro，18 GB）上用 [MLX](https://github.com/ml-explore/mlx) 完成，没有产生任何 API 费用。

## 主要结论

1. **拿更强老师的数据做微调，不等于学到了老师的能力。** 所有外部老师的数据都让学生的推理准确率下降了，
   包括 Llama-3.1-405B、GPT-4o、GPT-3.5、DeepSeek-R1 和人工解答。
   在 GSM8K 上下降 3.5–16.5pp，在 MATH-500 上下降 9–27pp，全部 p < 0.01。
   在 GSM8K 上，三个模型老师之间的差异也不显著。
2. **学生首先学到的是写法。** 学生输出的长度几乎和训练数据一模一样（96 → 99、127 → 133、170 → 168 token）。
   写法离学生自己的风格（约 311 token）越远，伤害越大：
   - 太短的解答会在难题上跳步。
   - 太长的解答（R1 风格）会让学生陷入停不下来的自我检查循环：GSM8K 上 28% 的输出写满了 token 上限，MATH-500 上是 59%。
3. **只有写法接近学生的数据才安全。** 这样的数据有两组：
   - **自蒸馏：** 学生自己采样，只保留答对的解答。
   - **同家族老师：** Qwen2.5-Math-7B，它和学生的词表、chat template 完全相同。

   两组都让 **GSM8K strict 显著提升 3.6–4.0pp**，MATH-500 也没有退步。
   但 lenient 口径显示，这部分提升来自答案格式更规范（`\boxed{}` 率从 95% 升到 99.8%），推理能力并没有提升。
4. **评测上的两个坑：**
   - 只看对格式敏感的指标，会掩盖推理的退化。外部老师的数据在 strict 口径下看起来和基线"持平"，但 lenient 口径下显著变差。
   - 验证集 loss 衡量的是"学得像不像老师"，不代表准确率。

## 结果

所有数据臂都用 2000 条训练数据（OpenR1 是 500 条）、训练 2 个 epoch，超参数完全相同。
括号里是相对未训练基线的变化（pp），`*` 表示 p < 0.05（在同一批题上做精确 McNemar 配对检验）。

| 训练数据（老师） | 训练回答长度 | GSM8K strict | GSM8K lenient | MATH-500 strict | MATH-500 lenient |
|---|---|---|---|---|---|
| 基线（不训练） | — | 70.74 | 74.07 | 54.80 | 55.20 |
| 自蒸馏（学生自己，拒绝采样） | 293 | **74.75 (+4.02\*)** | 74.75 (+0.68) | 53.20 (−1.60) | 53.40 (−1.80) |
| 同家族老师（Qwen2.5-Math-7B，4-bit） | 291 | **74.37 (+3.64\*)** | 74.37 (+0.30) | 54.80 (±0.00) | 54.80 (−0.40) |
| OpenMathInstruct-2（Llama-3.1-405B） | 170 | 70.51 (−0.23) | 70.58 (−3.49\*) | 42.20 (−12.60\*) | 42.80 (−12.40\*) |
| NuminaMath-CoT orca_math（GPT-4o） | 287 | 68.99 (−1.74) | 69.52 (−4.55\*) | 45.00 (−9.80\*) | 45.80 (−9.40\*) |
| MetaMathQA（GPT-3.5） | 127 | 68.46 (−2.27) | 68.46 (−5.61\*) | 37.00 (−17.80\*) | 37.00 (−18.20\*) |
| GSM8K 人工解答 | 96 | 57.54 (−13.19\*) | 57.54 (−16.53\*) | 36.00 (−18.80\*) | 36.20 (−19.00\*) |
| OpenR1-Math（DeepSeek-R1），500 条 | 1489 | 56.33 (−14.40\*)† | 60.05 (−14.03\*)† | 27.40 (−27.40\*) | 32.80 (−22.40\*) |

† OpenR1 在 GSM8K 上用 `max_tokens` 2048 评测，对照的基线也在 2048 下重测过，结果和 1024 时完全相同。

- **strict**：只认最后一个 `\boxed{}` 里的答案。
- **lenient**：没有 `\boxed{}` 时，GSM8K 取全文最后一个数字，MATH 取最后一个表达式。
- MATH-500 基线 54.8%，与 Qwen 官方报告的 55.2% 基本一致，可以用来交叉验证评测流程。
- 每次运行的逐题预测结果（`results/<run>/predictions.jsonl`）和所有对比表都在仓库里，
  见 [`results/stage5/summary.md`](results/stage5/summary.md) 和 [`results/stage7/teacher_math7b.md`](results/stage7/teacher_math7b.md)。

## 方法

| | |
|---|---|
| 学生 | `mlx-community/Qwen2.5-1.5B-Instruct-bf16` |
| 微调 | 用 `mlx_lm.lora`。LoRA rank 16、scale 20、dropout 0.05，加在全部 28 层；学习率 2e-5，前 5% warmup，之后 cosine 衰减到 10%；等效 batch 16，训练 2 个 epoch；loss 只计算回答部分；最大序列长度 1024（OpenR1 为 2048）；峰值内存约 6.7–6.9 GB |
| 评测 | greedy 解码，batch 64。system prompt 是 `Please reason step by step, and put your final answer within \boxed{}.`，训练时用同一句。GSM8K test 共 1319 题，`max_tokens` 1024；MATH-500 用 2048，由 [math-verify](https://github.com/huggingface/Math-Verify) 判断答案是否等价 |
| 统计检验 | 逐题比较对错，与基线做精确 McNemar 检验（双侧） |
| 随机种子 | 全部固定为 42；所有参数都写在 `configs/*.yaml` 里 |

**测试集去重：** 所有训练数据都和 GSM8K、MATH 的测试集比对过，分两步：
1. 文本规范化后做精确匹配。
2. 计算词级 8-gram 的覆盖率：如果一道训练题覆盖了某道 GSM8K 测试题 20% 以上的 8-gram（MATH 测试题的阈值是 50%），就删掉这道训练题。

一共删除 15,990 条，每个数据集删除 0.05%–1.18%。训练代码里还有断言，保证训练数据中不含测试题。
详见 [`results/stage2_decontam.md`](results/stage2_decontam.md)。

**数据臂：** 每个数据臂都经过以下筛选：有标准答案的，只保留答案正确的；只保留恰好有一个一致的 `\boxed{}` 的；去掉过长的。
每个臂单独留出 200 条作为验证集，再用固定种子抽出嵌套的训练子集（500 ⊂ 2000 ⊂ 10000）。

写法接近学生的两个臂，使用同一批 GSM8K 训练题、同样的顺序，2000 题中有 1895 题重合：
- **自蒸馏：** 学生对每题采样 4 次，从答对的里面挑 1 条。
- **7B 老师：** 老师对每题采样 1 次（正确率 95.9%），只保留答对的。

## 仓库结构

```
configs/        所有实验参数（yaml）
src/            数据准备、去重、筛选、训练、评测、对比
results/        每次运行的指标和逐题预测、对比表、训练曲线
EXPERIMENTS.md  按时间顺序的实验日志：配置、结果、结论
data/, adapters/  不入库（可以用脚本重新生成）
```

## 复现

需要 Apple Silicon 和 Python 3.10。

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
PY=.venv/bin/python

# 1. 准备评测集，跑基线
$PY src/prepare_gsm8k.py
$PY -u src/evaluate.py configs/eval_gsm8k_baseline.yaml

# 2. 下载原始训练数据到 data/raw/hf/<name>/，然后依次转换、去重、筛选
hf download nvidia/OpenMathInstruct-2 --repo-type dataset --include "data/train_1M-*" --local-dir data/raw/hf/OpenMathInstruct-2
hf download meta-math/MetaMathQA      --repo-type dataset --include "MetaMathQA-395K.json" --local-dir data/raw/hf/MetaMathQA
hf download AI-MO/NuminaMath-CoT      --repo-type dataset --include "data/train-*" --local-dir data/raw/hf/NuminaMath-CoT
hf download open-r1/OpenR1-Math-220k  --repo-type dataset --include "data/train-*" --local-dir data/raw/hf/OpenR1-Math-220k
hf download EleutherAI/hendrycks_math --repo-type dataset --include "*/*.parquet" --local-dir data/raw/hf/hendrycks_math
$PY -u src/convert_datasets.py
$PY -u src/decontaminate.py configs/decontam.yaml
$PY -u src/filter_sample.py configs/filter_sample.yaml
$PY src/prepare_math500.py

# 3. 训练一个数据臂并评测（所有数据臂的配置见 configs/train_*.yaml）
$PY -u src/train_lora.py configs/train_omi2_gsm_2000.yaml
$PY -u src/evaluate.py configs/eval_gsm8k_baseline.yaml --run-name gsm8k_omi2_gsm_2000 --adapter-path adapters/omi2_gsm_2000
$PY -u src/evaluate.py configs/eval_math500.yaml --run-name math500_omi2_gsm_2000 --adapter-path adapters/omi2_gsm_2000
$PY src/compare.py --ref gsm8k_baseline --metric lenient gsm8k_omi2_gsm_2000

# 自蒸馏和本地老师这两个数据臂：先生成，再构建数据，然后和上面一样训练
$PY -u src/self_generate.py configs/self_rft.yaml generate && $PY -u src/self_generate.py configs/self_rft.yaml build
$PY -u src/self_generate.py configs/teacher_math7b.yaml generate && $PY -u src/self_generate.py configs/teacher_math7b.yaml build
```

M3 Pro 上的大致耗时：

| 步骤 | 耗时 |
|---|---|
| GSM8K 评测 | 约 14 分钟 |
| MATH-500 评测 | 约 15 分钟 |
| 训练 2000 条 × 2 epoch | 约 70–90 分钟 |
| 7B 老师生成 2800 题 | 约 90 分钟 |

评测和生成都支持断点续跑。模型下载到本地之后，可以设置 `HF_HUB_OFFLINE=1`，避免 HF Hub 偶发的网络错误。

## 局限

- 每个配置只跑了一个随机种子，也只测了一个数据规模（2000 条，OpenR1 为 500 条）。
- 只用了一个学生模型。对于更弱的基座模型，能从老师那里学到的东西更多，结论不一定成立。
- MATH-500 只有 500 题，大约只能分辨 ±4pp 以上的差异。
- NuminaMath orca_math 这个臂没有标准答案，训练数据未经验证。
- 只做了序列级 SFT。白盒（logit）蒸馏和强化学习是自然的下一步。

## 数据与模型许可证

GSM8K（MIT）、MATH / hendrycks_math（MIT）、MATH-500、OpenMathInstruct-2（CC-BY-4.0）、
MetaMathQA（MIT）、NuminaMath-CoT（Apache-2.0）、OpenR1-Math-220k（Apache-2.0）、
Qwen2.5-1.5B-Instruct（Apache-2.0）、Qwen2.5-Math-7B-Instruct（Apache-2.0）。
仓库里只存放模型的预测结果和评测指标，不再分发原始数据集或模型权重。
