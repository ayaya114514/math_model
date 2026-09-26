# Math Reasoning Distillation on a Laptop

[中文版](README.zh.md) · Full experiment log (Chinese): [EXPERIMENTS.md](EXPERIMENTS.md)

A controlled study of **which distillation data actually helps a small math model**. The student is
Qwen2.5-1.5B-Instruct. It was LoRA-fine-tuned on seven kinds of reasoning traces, from human solutions
to DeepSeek-R1, and evaluated on GSM8K and MATH-500 with paired significance tests. Everything runs
locally on a MacBook Pro (M3 Pro, 18 GB) with [MLX](https://github.com/ml-explore/mlx), at zero API cost.

## Key findings

1. **Fine-tuning on data from a stronger teacher is not the same as distilling its skill.** Every
   external-teacher dataset (Llama-3.1-405B, GPT-4o, GPT-3.5, DeepSeek-R1, human solutions) *reduced*
   the student's reasoning accuracy. On GSM8K the loss was 3.5–16.5 pp, and on MATH-500 it was
   9–27 pp (all p < 0.01). On GSM8K, the three model teachers were not significantly different from one
   another.
2. **The student copies the teacher's style first.** Output length tracks the training data almost
   exactly (96 → 99, 127 → 133, 170 → 168 tokens). The further the style is from the student's own
   (about 311 tokens), the larger the damage. Traces that are too short skip steps on hard problems.
   Traces that are too long (R1-style) produce self-checking loops that never terminate: 28% of GSM8K
   outputs and 59% of MATH-500 outputs hit the token limit.
3. **Only in-distribution data is safe.** Two datasets matched the student's style: rejection-sampled
   self-distillation, and traces from a same-family teacher (Qwen2.5-Math-7B, same tokenizer and
   chat template). Both gave a significant **+3.6 to +4.0 pp on GSM8K (strict)** and did no harm on
   MATH-500. The lenient metric shows that the gain comes from answer-format consistency
   (`\boxed{}` rate 95% → 99.8%) rather than better reasoning.
4. **Evaluation pitfalls.** A format-sensitive metric can hide reasoning regressions. Under strict
   scoring, external-teacher models looked "on par" with the baseline, but the lenient metric showed
   significant losses. Validation loss measures imitation of the teacher, not accuracy.

## Results

The student is Qwen2.5-1.5B-Instruct. Each arm uses 2,000 training examples (500 for OpenR1), 2 epochs,
and identical hyper-parameters. Δ is measured against the untrained baseline; `*` means p < 0.05
(exact McNemar test, paired over the same questions).

| Training data (teacher) | Train resp. tokens | GSM8K strict | GSM8K lenient | MATH-500 strict | MATH-500 lenient |
|---|---|---|---|---|---|
| Baseline (no training) | — | 70.74 | 74.07 | 54.80 | 55.20 |
| Self-distillation (student, rejection-sampled) | 293 | **74.75 (+4.02\*)** | 74.75 (+0.68) | 53.20 (−1.60) | 53.40 (−1.80) |
| Same-family teacher (Qwen2.5-Math-7B, 4-bit) | 291 | **74.37 (+3.64\*)** | 74.37 (+0.30) | 54.80 (±0.00) | 54.80 (−0.40) |
| OpenMathInstruct-2 (Llama-3.1-405B) | 170 | 70.51 (−0.23) | 70.58 (−3.49\*) | 42.20 (−12.60\*) | 42.80 (−12.40\*) |
| NuminaMath-CoT orca_math (GPT-4o) | 287 | 68.99 (−1.74) | 69.52 (−4.55\*) | 45.00 (−9.80\*) | 45.80 (−9.40\*) |
| MetaMathQA (GPT-3.5) | 127 | 68.46 (−2.27) | 68.46 (−5.61\*) | 37.00 (−17.80\*) | 37.00 (−18.20\*) |
| GSM8K human solutions | 96 | 57.54 (−13.19\*) | 57.54 (−16.53\*) | 36.00 (−18.80\*) | 36.20 (−19.00\*) |
| OpenR1-Math (DeepSeek-R1), 500 ex. | 1489 | 56.33 (−14.40\*)† | 60.05 (−14.03\*)† | 27.40 (−27.40\*) | 32.80 (−22.40\*) |

† OpenR1 was evaluated on GSM8K with `max_tokens` 2048, against a baseline re-run at 2048. That
baseline scored the same as at 1024.

- **strict**: only the last `\boxed{}` counts. **lenient**: if there is no box, the last number
  (GSM8K) or the last expression (MATH) counts.
- The MATH-500 baseline (54.8%) matches the 55.2% reported by Qwen, which cross-checks the pipeline.
- Per-run predictions (`results/<run>/predictions.jsonl`) and all comparison tables are included.
  See [`results/stage5/summary.md`](results/stage5/summary.md) and
  [`results/stage7/teacher_math7b.md`](results/stage7/teacher_math7b.md).

## Method

| | |
|---|---|
| Student | `mlx-community/Qwen2.5-1.5B-Instruct-bf16` |
| Fine-tuning | `mlx_lm.lora`: rank 16, scale 20, dropout 0.05, all 28 layers; lr 2e-5 with 5% warmup and cosine decay to 10%; effective batch 16; 2 epochs; loss on the response only; max sequence 1024 (2048 for OpenR1); peak memory about 6.7–6.9 GB |
| Evaluation | Greedy decoding, batch 64, system prompt `Please reason step by step, and put your final answer within \boxed{}.` (the same prompt is used in training); GSM8K test (1,319) with `max_tokens` 1024; MATH-500 with 2048, graded by [math-verify](https://github.com/huggingface/Math-Verify) |
| Statistics | Exact two-sided McNemar test on per-question correctness against the baseline |
| Seeds | Fixed seed 42 everywhere; all parameters live in `configs/*.yaml` |

**Test-set decontamination.** Every training source was checked against the GSM8K and MATH test sets.
The check first normalizes text and looks for exact matches, then measures word-level 8-gram
coverage (a training question is dropped if it covers ≥ 20% of a GSM8K test question's 8-grams, or
≥ 50% of a MATH test question's). This removed 15,990 examples (0.05–1.18% per dataset). Training
code also asserts that no test question is present. Details:
[`results/stage2_decontam.md`](results/stage2_decontam.md).

**Data arms.** Each arm was filtered for answer correctness where gold answers exist, a single
consistent `\boxed{}`, and length. Each arm holds out its own 200-example validation split, and nested
training subsets (500 ⊂ 2000 ⊂ 10000) were sampled with a fixed seed. The two in-distribution arms draw
from the same GSM8K training questions in the same order (1,895 of 2,000 questions overlap). The self-distillation arm keeps one correct
answer out of 4 student samples per question; the teacher arm uses one sample from the 7B teacher
(95.9% correct).

## Repository layout

```
configs/        all experiment parameters (yaml)
src/            data prep, decontamination, filtering, training, evaluation, comparison
results/        per-run metrics + predictions, comparison tables, training curves
EXPERIMENTS.md  chronological experiment log with configs, results and conclusions (Chinese)
data/, adapters/  not tracked (regenerated by the scripts)
```

## Reproduce

Requires Apple Silicon and Python 3.10.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
PY=.venv/bin/python

# 1. Evaluation sets and the baseline
$PY src/prepare_gsm8k.py
$PY -u src/evaluate.py configs/eval_gsm8k_baseline.yaml

# 2. Raw training data -> data/raw/hf/<name>/ (then convert, decontaminate, filter)
hf download nvidia/OpenMathInstruct-2 --repo-type dataset --include "data/train_1M-*" --local-dir data/raw/hf/OpenMathInstruct-2
hf download meta-math/MetaMathQA      --repo-type dataset --include "MetaMathQA-395K.json" --local-dir data/raw/hf/MetaMathQA
hf download AI-MO/NuminaMath-CoT      --repo-type dataset --include "data/train-*" --local-dir data/raw/hf/NuminaMath-CoT
hf download open-r1/OpenR1-Math-220k  --repo-type dataset --include "data/train-*" --local-dir data/raw/hf/OpenR1-Math-220k
hf download EleutherAI/hendrycks_math --repo-type dataset --include "*/*.parquet" --local-dir data/raw/hf/hendrycks_math
$PY -u src/convert_datasets.py
$PY -u src/decontaminate.py configs/decontam.yaml
$PY -u src/filter_sample.py configs/filter_sample.yaml
$PY src/prepare_math500.py

# 3. Train one arm and evaluate it (all arms: configs/train_*.yaml)
$PY -u src/train_lora.py configs/train_omi2_gsm_2000.yaml
$PY -u src/evaluate.py configs/eval_gsm8k_baseline.yaml --run-name gsm8k_omi2_gsm_2000 --adapter-path adapters/omi2_gsm_2000
$PY -u src/evaluate.py configs/eval_math500.yaml --run-name math500_omi2_gsm_2000 --adapter-path adapters/omi2_gsm_2000
$PY src/compare.py --ref gsm8k_baseline --metric lenient gsm8k_omi2_gsm_2000

# Self-distillation / local-teacher arms: generate, then build, then train as above
$PY -u src/self_generate.py configs/self_rft.yaml generate && $PY -u src/self_generate.py configs/self_rft.yaml build
$PY -u src/self_generate.py configs/teacher_math7b.yaml generate && $PY -u src/self_generate.py configs/teacher_math7b.yaml build
```

Rough timings on an M3 Pro: GSM8K evaluation about 14 min; MATH-500 about 15 min; training 2,000
examples for 2 epochs about 70–90 min; teacher generation for 2,800 questions about 90 min.
Evaluation and generation resume where they stopped. Once models are cached, setting
`HF_HUB_OFFLINE=1` avoids transient Hub errors.

## Limitations

- Single seed per configuration, and a single data scale (2,000 examples, 500 for OpenR1).
- One student model. The conclusions may not hold for weaker base models, which have more to learn
  from a teacher.
- MATH-500 has 500 questions, so only differences of roughly ±4 pp or more are resolvable.
- The NuminaMath orca_math arm has no gold answers, so its traces are unverified.
- Only sequence-level SFT was tried. White-box (logit) distillation and RL are natural next steps.

## Licenses of the data and models used

GSM8K (MIT), MATH / hendrycks_math (MIT), MATH-500, OpenMathInstruct-2 (CC-BY-4.0),
MetaMathQA (MIT), NuminaMath-CoT (Apache-2.0), OpenR1-Math-220k (Apache-2.0),
Qwen2.5-1.5B-Instruct (Apache-2.0), Qwen2.5-Math-7B-Instruct (Apache-2.0).
Only derived predictions and metrics are stored here; raw datasets and weights are not redistributed.
