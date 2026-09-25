# 阶段 0：环境报告（2026-09-25）

## 环境

| 项目 | 值 |
|---|---|
| 机器 | Apple M3 Pro，18GB 统一内存（GPU 推荐工作集上限 14.3 GB） |
| 系统 | macOS 26.6.2（25G83），arm64 |
| Python | 3.10.20（conda-forge，arm64 原生），项目虚拟环境 `.venv` |
| 关键包 | mlx 0.32.2，mlx-lm 0.31.3，datasets 5.0.1，transformers 5.17.0（完整版本锁定见 `requirements.txt`） |
| 学生模型 | `mlx-community/Qwen2.5-1.5B-Instruct-bf16`（2.9 GB）；另下载 `-4bit`（839 MB）做对比 |
| 数据 | `openai/gsm8k` main：train 7473 / test 1319（HF cache 中，阶段 1 再落盘到 `data/raw/`） |

## 生成速度实测

测速题目来自 GSM8K **train** split（seed=42 随机抽取），没有使用 test。
解码方式：greedy，max_tokens=1024，system prompt 为
`Please reason step by step, and put your final answer within \boxed{}.`
平均 prompt 83 token，平均输出约 280 token（最长约 400），全部正常结束，没有被截断。

| 模型 | 模式 | 题数 | 生成 tok/s | 峰值内存 | 含 `\boxed{}` 比例 | 估算 1319 题耗时 |
|---|---|---|---|---|---|---|
| bf16 | 逐题 | 20 | 41 | 3.3 GB | 90% | ~153 min |
| bf16 | batch 8 | 64 | 58 | 5.8 GB | 94% | ~115 min |
| bf16 | batch 16 | 64 | 107 | 6.8 GB | 92% | ~64 min |
| bf16 | batch 32 | 128 | 173 | 7.9 GB | 93% | ~40 min |
| **bf16** | **batch 64** | 128 | **273** | **10.5 GB** | 94% | **~27 min** |
| 4bit | 逐题 | 20 | 123 | 1.2 GB | 75% | ~45 min |
| 4bit | batch 32 | 64 | 483 | 4.2 GB | 78% | ~15 min |

原始数据：`results/stage0_speed.json`、`results/stage0_speed_bs64.json`；
复现方法：`.venv/bin/python src/bench_speed.py configs/stage0_speed.yaml`。

## 结论与建议

1. **评测用 bf16 + batch 64，完整跑 GSM8K test 约 27 分钟，不需要抽子集。**
   （阶段 1 更新：改为 chunk_size = batch_size = 64 后，实测只要 14 分钟，峰值 5.1 GB，原因见 EXPERIMENTS.md。）
   逐题生成时 41 tok/s × 3 GB 权重 ≈ 124 GB/s，已经接近 M3 Pro 的内存带宽上限，
   说明单条生成的瓶颈在读权重。batch 越大，每读一次权重产出的 token 越多，
   所以吞吐量几乎随 batch size 线性增长。
2. **不用 4-bit 做评测。** 4-bit 虽然快 3 倍，但按格式输出 `\boxed{}` 的比例从约 93% 掉到约 78%，
   量化误差会直接拉低准确率，并和"蒸馏数据的效果"混在一起。
3. **评测 batch size 要固定写进配置。** 同一批题在不同 batch size 下，greedy 生成的总 token 数略有差异
   （18847 / 18876 / 18869）。原因是批量矩阵运算的浮点累加顺序不同，得到的结果并不逐位一致。
   为了可复现，所有实验都要用同一个 batch size。
4. bf16 仍有约 6–7% 的输出没有 `\boxed{}`。阶段 1 需要决定答案提取的严格程度（见下方待确认事项）。
5. 后续加入 MATH 时，输出会更长（估计是 GSM8K 的 2–3 倍），评测耗时也会按比例增加，届时可能要评估用 MATH-500 子集。

## 待确认事项（阶段 1 前）

- 答案提取：建议**主指标严格只认 `\boxed{}`**，同时额外记录"没有 boxed 时取最后一个数字"的宽松准确率作为参考。
  这样既能看出模型的格式遵循程度，微调后格式改进带来的提升也能单独看出来。
