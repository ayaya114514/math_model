"""阶段 0：测学生模型的生成速度，并估算完整 GSM8K test 评测耗时。

用法: .venv/bin/python src/bench_speed.py configs/stage0_speed.yaml
"""
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import yaml
from datasets import load_dataset
from mlx_lm import batch_generate, load, stream_generate

RESULTS_DIR = Path("results")


def build_prompt(tokenizer, system_prompt, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def bench_single(model, tokenizer, prompts, max_tokens):
    """逐题生成：得到单条流的 prompt / 生成速度。"""
    rows = []
    for p in prompts:
        last = None
        text = ""
        for r in stream_generate(model, tokenizer, p, max_tokens=max_tokens):
            text += r.text
            last = r
        rows.append({
            "prompt_tokens": last.prompt_tokens,
            "gen_tokens": last.generation_tokens,
            "prompt_tps": last.prompt_tps,
            "gen_tps": last.generation_tps,
            "finish_reason": last.finish_reason,
            "has_boxed": "\\boxed{" in text,
        })
    return rows, text


def bench_batch(model, tokenizer, prompts, max_tokens, batch_size):
    """批量生成：看并行后的总吞吐（评测真正关心的是这个）。"""
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    resp = batch_generate(
        model, tokenizer, prompts, max_tokens=max_tokens,
        completion_batch_size=batch_size, prefill_batch_size=min(batch_size, 8),
    )
    wall = time.perf_counter() - t0
    s = resp.stats
    return {
        "batch_size": batch_size,
        "num_questions": len(prompts),
        "wall_s": wall,
        "gen_tokens": s.generation_tokens,
        "gen_tps": s.generation_tps,
        "prompt_tps": s.prompt_tps,
        "questions_per_s": len(prompts) / wall,
        "peak_memory_gb": s.peak_memory,
        "has_boxed_rate": sum("\\boxed{" in t for t in resp.texts) / len(prompts),
    }


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    random.seed(cfg["seed"])
    mx.random.seed(cfg["seed"])

    ds = load_dataset(cfg["dataset"], cfg["dataset_config"], split=cfg["split"])
    assert cfg["split"] != "test", "测速不得使用测试集"
    n = max(cfg["single"]["num_questions"], cfg["batch"]["num_questions"])
    idx = random.sample(range(len(ds)), n)
    questions = [ds[i]["question"] for i in idx]

    report = {"config": cfg, "models": {}}
    for name, repo in cfg["models"].items():
        print(f"\n=== {name}: {repo}")
        t0 = time.perf_counter()
        model, tokenizer = load(repo)
        load_s = time.perf_counter() - t0
        prompts = [build_prompt(tokenizer, cfg["system_prompt"], q) for q in questions]

        # 预热一次，避免首次编译 kernel 的时间算进去
        list(stream_generate(model, tokenizer, prompts[0], max_tokens=16))

        mx.reset_peak_memory()
        k = cfg["single"]["num_questions"]
        rows, sample_text = bench_single(model, tokenizer, prompts[:k], cfg["max_tokens"])
        single_peak = mx.get_peak_memory() / 1e9
        gen_tokens = [r["gen_tokens"] for r in rows]
        single = {
            "num_questions": k,
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / k,
            "mean_gen_tokens": sum(gen_tokens) / k,
            "max_gen_tokens": max(gen_tokens),
            "mean_gen_tps": sum(r["gen_tps"] for r in rows) / k,
            "mean_prompt_tps": sum(r["prompt_tps"] for r in rows) / k,
            "truncated": sum(r["finish_reason"] == "length" for r in rows),
            "has_boxed_rate": sum(r["has_boxed"] for r in rows) / k,
            "peak_memory_gb": single_peak,
        }
        # 单流估算：每题时间 ≈ 平均生成 token / 生成速度（prefill 很短，忽略）
        single["est_eval_min"] = (cfg["eval_target_size"] * single["mean_gen_tokens"]
                                  / single["mean_gen_tps"] / 60)
        print(json.dumps(single, indent=2))
        print("--- 示例输出（最后一题）---\n" + sample_text[:1500])

        m = cfg["batch"]["num_questions"]
        batches = []
        for bs in cfg["batch"]["batch_sizes"]:
            b = bench_batch(model, tokenizer, prompts[:m], cfg["max_tokens"], bs)
            b["est_eval_min"] = cfg["eval_target_size"] / b["questions_per_s"] / 60
            print(json.dumps(b, indent=2))
            batches.append(b)

        report["models"][name] = {"repo": repo, "load_s": load_s,
                                  "single": single, "batch": batches}
        del model, tokenizer
        mx.clear_cache()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / cfg.get("output", "stage0_speed.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/stage0_speed.yaml")
