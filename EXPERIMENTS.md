# 实验日志

每次实验记录：日期、配置文件、关键参数、结果、结论。

## 2026-09-25 阶段 0：环境与测速

- 配置：`configs/stage0_speed.yaml`、`configs/stage0_speed_bs64.yaml`
- 模型：Qwen2.5-1.5B-Instruct（MLX bf16 / 4bit）
- 结果：bf16 逐题 41 tok/s；batch 64 时 273 tok/s，峰值内存 10.5 GB，估算完整 GSM8K test 约 27 min。
  4bit 快约 3 倍，但 `\boxed{}` 遵循率从约 93% 降到约 78%。
- 结论：评测统一用 bf16、batch 64、greedy，max_tokens=1024。详见 `results/stage0_env.md`。

## 2026-09-25 阶段 1：GSM8K 基线

- 配置：`configs/eval_gsm8k_baseline.yaml`（bf16，greedy，max_tokens=1024，batch 64 / chunk 64，
  system prompt = "Please reason step by step, and put your final answer within \boxed{}."）
- 数据：GSM8K 官方 test split，全部 1319 题（`data/raw/gsm8k_test.jsonl`）
- 结果（`results/gsm8k_baseline/`）：

| 指标 | 值 |
|---|---|
| **strict 准确率（只认 `\boxed{}`）** | **70.74%**（933/1319，95% CI ±2.5pp） |
| lenient 准确率（无 boxed 时取最后一个数字） | 74.07% |
| 输出含 `\boxed{}` 比例 | 94.9%（67 题没有 boxed，其中 44 题答案其实正确） |
| 截断（达到 1024 token） | 8 题（0.6%） |
| 平均输出长度 | 311 token |
| 耗时 | 854 s（约 14 min），峰值内存 5.1 GB |

- 可复现性：同一配置跑两次，输出逐字一致（50/50、256/256）。
- 结论：
  - 基线 70.7%，和 Qwen 官方公布的 Qwen2.5-1.5B-Instruct GSM8K 成绩（73.2%）同一水平，
    差异来自 prompt 和评测方式的不同。
  - strict 和 lenient 相差 3.3pp，全部来自"答对了但没写 boxed"。微调后如果这个差距缩小，说明提升里有一部分来自格式。
  - 95% 置信区间约 ±2.5pp。后续对比如果差距小于约 3pp，要用配对检验（同一批题上的 McNemar）判断，不能只看两个数字。
- 踩坑记录：一开始用 chunk 256 时，第 2 个 chunk 发生了 Metal OOM。根因在 mlx-lm 0.31.3 的
  `BatchKVCache.extend()`：新题目插入 batch 时会被左 padding 到当前最长序列的长度，所以 chunk 内持续补题会让 KV cache 长度不断累积（峰值 12.3 GB）。
  改成 chunk_size = batch_size = 64 后，峰值降到 5.1 GB，速度反而快了一倍（attention 不再算 padding）。
  阶段 0 的耗时估算（27 min）是在 continuous batching 下测的，实际用的配置更快。

## 2026-09-25 阶段 2：公开训练数据调研、下载与去重

- 调研：7 个候选数据集，各抽样 500 行（`configs/survey_datasets.yaml`，报告 `results/stage2_survey.md`）。
  R1 式长推理数据（OpenR1 / OpenMathReasoning / Nemotron / OpenThoughts3）的回答中位数有 5k–17k token，
  在 18GB 内存下既难训练也难评测；短 CoT 数据（OMI2 / MetaMathQA / NuminaMath）p90 < 900 token。
- 选定下载：OpenMathInstruct-2（train_1M）、MetaMathQA、NuminaMath-CoT、OpenR1-Math-220k（default），外加 MATH train/test。
- 去重（`configs/decontam.yaml`，报告 `results/stage2_decontam.md`）：精确匹配 + 8-gram 覆盖率，阈值 GSM8K 0.2 / MATH 0.5。
  删除率 0.05%–1.18%，命中主要来自 MATH test。GSM8K 官方训练集自身就有 4 道测试题的模板变体，已删除。
- 结论：训练数据统一从 `data/raw/<dataset>.jsonl`（去重后）读取；
  和 GSM8K 难度匹配的主力池是 OMI2 的 GSM 部分（15.3 万条）和 MetaMathQA 的 GSM 部分（24 万条）。

## 2026-09-25 阶段 3：筛选与抽样

- 配置：`configs/filter_sample.yaml`（seed 42），报告 `results/stage3_filter.md`
- 5 个数据臂：omi2_gsm（Llama-405B）、metamath_gsm（GPT-3.5）、gsm8k_human（人工）、numina_orca（GPT-4o，无法验证）、openr1（R1 长推理）。
  每个臂 valid 200 条，训练子集嵌套 500 ⊂ 2000 ⊂ 10000（人工最多 7000，openr1 最多 2000）。
- 处理：答案验证、统一以 `\boxed{}` 收尾、同题只留一条、长度上限（短 CoT 1024 / openr1 2048）；输出前断言无测试集污染。
- 可用池：omi2_gsm 80,927；metamath_gsm 66,408；gsm8k_human 7,469；numina_orca 150,195；openr1 6,319。
- 注意：各臂的回答长度差很多（96 → 1488 token），比较"老师"时，长度是一个混杂变量；openr1 臂的评测预算需要单独处理。

## 2026-09-25 阶段 4：LoRA 微调（pipeline 跑通 + 首次正式训练）

- 工具：mlx-lm 0.31.3 `mlx_lm.lora`，封装在 `src/train_lora.py` 中（数据转 chat 格式、断言无测试题、按 epoch 计算 iters 和学习率调度、解析日志）。
- 数据格式：`{"messages": [system, user, assistant]}`，system prompt 和评测逐字一致；`mask_prompt: true`。
  已直接解码验证 mask 边界：system + 题目 + `<|im_start|>assistant\n` 不计 loss，回答和结尾的 `<|im_end|>` 计入。
- 超参（`configs/train_omi2_gsm_2000.yaml`）：LoRA 覆盖全部 28 层，rank 16 / scale 20 / dropout 0.05，可训练参数 18.5M（1.2%）；
  AdamW，lr 2e-5，warmup 5% 后 cosine 衰减到 10%；batch 2 × 梯度累积 8 = 等效 batch 16；max_seq_length 1024；开启梯度检查点。
- 内存与速度的取舍（均为实测）：

| 配置 | 峰值内存 | 吞吐 | 结论 |
|---|---|---|---|
| batch 4，无梯度检查点 | OOM | — | 激活值约 8 GB + logits 约 5 GB，超过 14.3 GB |
| batch 4 + 梯度检查点 | 7.8 GB | ~185 tok/s | 不比 batch 2 快（已是计算瓶颈） |
| batch 2，无梯度检查点 | 12.4 GB（48 步内尚未遇到最长样本） | ~260 tok/s | 快 27%，但离上限太近，不用 |
| **batch 2 + 梯度检查点** | **6.8 GB** | **~200 tok/s** | **采用** |

- smoke test（`configs/train_smoke.yaml`，train_500，104 iter）：val loss 0.719 → 0.390；权重保存后能加载，输出和基线完全不同（0/64 相同），说明 LoRA 确实生效。
- 正式训练 `omi2_gsm_2000`（train_2000，2 epoch = 2000 iter / 250 次更新，约 68 min）：
  - val loss：0.719 → 0.339（第 1 个 epoch 结束，iter 1000）→ 0.346（iter 2000）
  - train loss：第 1 个 epoch 约 0.33，第 2 个 epoch 骤降到约 0.24。**第 2 个 epoch 在记忆训练样本，val 不再改善。**
  - 曲线：`results/omi2_gsm_2000/train_curve.png`；权重：`adapters/omi2_gsm_2000/`（含 iter 500/1000/1500/2000 的 checkpoint）
  - 加载检查（64 题，不作为结论）：strict 70.3%（同一批题基线 64.1%），boxed 98%，平均输出 169 token（基线 311）
- 结论：流程全部跑通。第 2 个 epoch 没有带来 val 改善，阶段 5 要对比 iter 1000 和 2000 两个 checkpoint 的真实准确率，
  再决定后续实验用 1 个还是 2 个 epoch。

## 2026-09-25 阶段 5-1：omi2_gsm_2000 全量评测（1 vs 2 epoch）

- 评测：和阶段 1 完全相同（`configs/eval_gsm8k_baseline.yaml` + `--adapter-path`），GSM8K test 1319 题。对比表：`results/stage5/epochs.md`（`src/compare.py`，精确 McNemar 检验）

| 模型 | strict | Δ vs 基线 | 基线错→对 / 基线对→错 | p | lenient | boxed | 平均输出 | 截断 |
|---|---|---|---|---|---|---|---|---|
| 基线 | 70.74% | — | — | — | 74.07% | 94.9% | 311 | 8 |
| omi2_gsm_2000 iter1000（1 epoch） | 67.02% | −3.71pp | 141 / 190 | 0.008 | 67.02% | 98.3% | 170 | 21 |
| omi2_gsm_2000 iter2000（2 epoch） | 70.51% | −0.23pp | 165 / 168 | 0.91 | 70.58% | 99.1% | 168 | 11 |

- 结论：
  - **用 OMI2（Llama-405B 短 CoT）蒸馏 2000 条没有提升准确率**：1 个 epoch 显著变差，2 个 epoch 和基线持平。格式遵循（boxed 95% → 99%）确实变好了，但被推理质量的下降抵消。
  - **验证集 loss 不能代表准确率**：iter1000 的 val loss 更低（0.339 对 0.346），准确率却低 3.5pp。val loss 衡量的是"像不像老师"。
  - **按题目复杂度看**（用基线输出长度分四档）：最简单的一档 78% → 87%，中等偏难的一档 75% → 64–67%。
    每一档的输出长度都差不多砍半（例如 324 → 176 token）。学生学到了老师更简略的写法，难题上的推理步骤不够。
  - 出现了重复循环：iter1000 有 9 条截断是循环（基线 0 条）。
  - iter1000 比 iter2000 差，更可能是因为那时学习率还处在高位、模型处于风格切换的中间状态，而不是 epoch 数本身。后续保持 2 个 epoch。
- 假设：**对一个本身已经很强的 Instruct 学生，推理的详细程度可能比老师的强弱更关键**；比学生自身输出更短的蒸馏数据会伤害难题上的表现。

## 2026-09-25 阶段 5-2：数据来源对照（各 2000 条 × 2 epoch，其余超参完全相同）

- 配置：`configs/train_{gsm8k_human,metamath_gsm,numina_orca}_2000.yaml`（与 `train_omi2_gsm_2000.yaml` 只差数据路径）；评测和阶段 1 相同。
- 对比表：`results/stage5/sources.md`

| 训练数据（老师） | 训练回答均值 | 学生输出均值 | strict | Δ vs 基线 | p（McNemar） | Q1 最简单 | Q2 | Q3 | Q4 最难 |
|---|---|---|---|---|---|---|---|---|---|
| 基线（不训练） | — | 311 | 70.74% | — | — | 78.1% | 77.8% | 74.8% | 52.4% |
| 人工解答 | 96 | 99 | 57.54% | −13.19pp | 4e-19 | 79.0% | 61.1% | 54.4% | 35.8% |
| MetaMathQA（GPT-3.5） | 127 | 133 | 68.46% | −2.27pp | 0.10 | 86.6% | 73.6% | 67.2% | 46.7% |
| OMI2（Llama-3.1-405B） | 170 | 168 | 70.51% | −0.23pp | 0.91 | 86.6% | 75.1% | 66.9% | 53.6% |
| NuminaMath orca（GPT-4o，未验证） | 287 | 202 | 68.99% | −1.74pp | 0.23 | 80.9% | 72.6% | 69.0% | 53.6% |

（Q1–Q4：按基线输出长度把 1319 题分成四档，长度近似代表题目所需步骤数）

- 三个模型老师两两配对检验：Llama 对 GPT-4o p=0.28，Llama 对 GPT-3.5 p=0.13，GPT-4o 对 GPT-3.5 p=0.73，**都不显著**。
- 结论：
  1. **在这个学生身上，4 种公开蒸馏数据各 2000 条都没有提升 GSM8K 准确率**：最好的一组（Llama-405B）和基线持平，人工解答显著变差。
  2. **学生几乎原样学会了老师的回答长度**（96→99、127→133、170→168）。SFT 首先迁移的是写法，而不是能力。
  3. **推理过短会伤害中等和难题**：人工解答只有约 100 token，Q3/Q4 分别掉 20 和 17pp。
     在 130–200 token 区间，长度和准确率没有单调关系；三个模型老师之间的差异在噪声范围内。
  4. 所有微调组都是"简单题变好（格式更规范、更简洁），中等题变差"。整体持平或下降，是这两种效应相互抵消的结果。
- 解读：Qwen2.5-1.5B-Instruct 出厂前已经在大量数学数据上训练过，它自己的 CoT（311 token、结构化）不比这些公开数据差。
  用比它自身写法更简略的数据做 SFT，主要效果是改变写法；要获得提升，蒸馏数据可能需要**比学生自己的推理更好**（更正确、至少同样详细），
  而不只是"来自更强的模型"。
- 局限：每组只做了一次训练（单 seed）；numina_orca 没有答案验证；只测了 2000 条这一个规模。

## 2026-09-25 阶段 5-3：自蒸馏 / 拒绝采样（self-RFT）+ 用 lenient 口径重新审视所有对照

- 配置：`configs/self_rft.yaml`（生成与构建）、`configs/train_self_rft_2000.yaml`（与 omi2 只差数据路径）；代码 `src/self_generate.py`
- 数据：学生对 gsm8k_human 臂的同一批题（valid 200 + train 前 2600 道）每题采样 4 次（temp 0.7 / top_p 0.8 / top_k 20），
  只保留答对且格式合格的解答，每题随机挑 1 条，取前 2000 道（和人工臂 train_2000 的题目重合 1855 道）。
  - 采样正确率 67.8%；pass@4：4/4 对的 964 题，3/4 的 787，2/4 的 532，1/4 的 318，0/4 的 199（这 199 道被排除，因此题目略偏简单）
  - 训练回答均值 293 token（学生自己的写法）；val loss 0.189 → 0.093（本来就是学生自己的分布，所以很低）
- 结果（对比表：`results/stage5/all_strict.md`、`results/stage5/all_lenient.md`）：

| 训练数据 | 平均输出 | strict | Δ strict（p） | lenient | Δ lenient（p） | boxed |
|---|---|---|---|---|---|---|
| 基线 | 311 | 70.74% | — | 74.07% | — | 94.9% |
| **自蒸馏 self-RFT** | 311 | **74.75%** | **+4.02pp（0.0006）** | 74.75% | +0.68pp（0.56） | 99.8% |
| OMI2（Llama-405B） | 168 | 70.51% | −0.23pp（0.91） | 70.58% | **−3.49pp（0.009）** | 99.1% |
| Numina orca（GPT-4o） | 202 | 68.99% | −1.74pp（0.23） | 69.52% | **−4.55pp（0.0005）** | 97.2% |
| MetaMathQA（GPT-3.5） | 133 | 68.46% | −2.27pp（0.10） | 68.46% | **−5.61pp（1e-5）** | 99.8% |
| 人工解答 | 99 | 57.54% | −13.19pp（4e-19） | 57.54% | −16.53pp | 99.5% |

- 结论（**修正阶段 5-2 的说法**）：
  1. **用 lenient 口径（不看格式）看，所有外部老师的数据都显著损害了学生的推理**（−3.5 到 −5.6pp，p < 0.01）。
     阶段 5-2 在 strict 口径下"和基线持平"，是因为 boxed 遵循率从 95% 提升到 97–100% 带来的 +3pp 左右，恰好掩盖了推理上的损失。
     **以后比较微调效果时，必须同时报告 strict 和 lenient。**
  2. **自蒸馏是唯一不损害推理的方案**：lenient +0.7pp（不显著），同时把格式修好（boxed 99.8%），strict +4.0pp 显著。
     它的提升本质上是**格式一致性**，不是推理能力。
  3. 和人工臂几乎同一批题（1855/2000 重合），只是把解答从"人工"换成"学生自己的正确解答"，strict 就从 57.5% 变成 74.8%（+17.2pp，p=2e-32）。
     **对这个学生来说，训练数据的写法是否贴近学生自己的分布，比老师是谁更重要。**
- 局限：单 seed；self-RFT 排除了 199 道 4 次都做错的题，训练题略偏简单；自蒸馏无法教会学生原本不会的东西（lenient 没有显著提升）。

## 2026-09-25 22:4x 暂停（实验 2：OpenR1 长推理对照，进行到第 1 步）

- 已完成：5-1（epoch 对照）、5-2（4 种数据来源）、5-3（自蒸馏 + lenient 复核）。
- 进行中被暂停：**5-4 OpenR1 长推理对照**（DeepSeek-R1，500 条，序列上限 2048，评测 max_tokens 2048）
  - 第 1 步"基线 @2048 重测"完成 128/1319 题：`results/gsm8k_baseline_2048/predictions.jsonl`（评测支持断点续跑，重新运行同一命令即可接着跑）
  - 第 2、3 步（训练、评测）尚未开始；配置已就绪：`configs/eval_gsm8k_2048.yaml`、`configs/train_openr1_500.yaml`
  - 注意：`train_openr1_500.yaml` 用 batch 1 × 累积 16、序列 2048，**峰值内存还没实测**，首次运行时要观察前几十步的 Peak mem
- 恢复命令（依次执行，约 4–4.5 小时）：
  ```
  export PYTHONPATH=src
  .venv/bin/python -u src/evaluate.py configs/eval_gsm8k_2048.yaml
  .venv/bin/python -u src/train_lora.py configs/train_openr1_500.yaml && .venv/bin/python src/plot_curve.py openr1_500
  .venv/bin/python -u src/evaluate.py configs/eval_gsm8k_2048.yaml --run-name gsm8k_openr1_500_2048 --adapter-path adapters/openr1_500
  .venv/bin/python src/compare.py --ref gsm8k_baseline_2048 gsm8k_openr1_500_2048 && .venv/bin/python src/compare.py --ref gsm8k_baseline_2048 --metric lenient gsm8k_openr1_500_2048
  ```
