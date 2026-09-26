"""统一评测：greedy 批量生成 -> 提取 \\boxed{} -> 与标准答案比对。

用法: .venv/bin/python -u src/evaluate.py configs/eval_gsm8k_baseline.yaml [--limit N] [--run-name NAME] [--adapter-path DIR]
产出: results/<run_name>/predictions.jsonl（逐题）+ metrics.json（汇总）
已完成的题会跳过（断点续跑）；要重跑请换 run_name。
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm import batch_generate, load

from answer import judge, math_judge

JUDGES = {"gsm8k": judge, "math": math_judge}  # 配置里的 task -> 判分函数


def build_prompt(tokenizer, system_prompt, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def summarize(rows, cfg, elapsed_s):
    n = len(rows)
    return {
        "run_name": cfg["run_name"],
        "model": cfg["model"],
        "adapter_path": cfg["adapter_path"],
        "data": cfg["data"],
        "n": n,
        "strict_acc": sum(r["strict_correct"] for r in rows) / n,
        "lenient_acc": sum(r["lenient_correct"] for r in rows) / n,
        "boxed_rate": sum(r["boxed"] is not None for r in rows) / n,
        "truncated_rate": sum(r["gen_tokens"] >= cfg["max_tokens"] for r in rows) / n,
        "mean_gen_tokens": sum(r["gen_tokens"] for r in rows) / n,
        "elapsed_s_this_session": elapsed_s,
        "config": cfg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None, help="覆盖配置里的 limit（调试用）")
    ap.add_argument("--run-name", default=None, help="覆盖配置里的 run_name")
    ap.add_argument("--adapter-path", default=None, help="覆盖配置里的 adapter_path（LoRA 权重目录）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.adapter_path is not None:
        cfg["adapter_path"] = args.adapter_path
    mx.random.seed(cfg["seed"])
    judge_fn = JUDGES[cfg.get("task", "gsm8k")]

    rows = [json.loads(line) for line in open(cfg["data"])]
    assert all(r["split"] == "test" for r in rows), "评测数据必须全部来自 test split"
    if cfg["limit"]:
        rows = rows[: cfg["limit"]]

    out_dir = Path("results") / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = {}
    if pred_path.exists():
        done = {r["id"]: r for r in map(json.loads, open(pred_path))}
    todo = [r for r in rows if r["id"] not in done]
    print(f"共 {len(rows)} 题，已完成 {len(done)}，待评测 {len(todo)}")

    t0 = time.perf_counter()
    if todo:
        model, tokenizer = load(cfg["model"], adapter_path=cfg["adapter_path"])
        with pred_path.open("a") as f:
            for start in range(0, len(todo), cfg["chunk_size"]):
                chunk = todo[start: start + cfg["chunk_size"]]
                prompts = [build_prompt(tokenizer, cfg["system_prompt"], r["question"]) for r in chunk]
                resp = batch_generate(
                    model, tokenizer, prompts, max_tokens=cfg["max_tokens"],
                    completion_batch_size=cfg["batch_size"], prefill_batch_size=8,
                )
                for r, text in zip(chunk, resp.texts):
                    rec = {
                        "id": r["id"],
                        "gold": r["gold"],
                        "response": text,
                        "gen_tokens": len(tokenizer.encode(text)),
                        **judge_fn(text, r["gold"]),
                    }
                    done[r["id"]] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                peak = resp.stats.peak_memory
                del resp
                mx.clear_cache()  # 释放上一 chunk 的缓存 buffer
                n_done = len(done)
                acc = sum(done[i]["strict_correct"] for i in done) / n_done
                print(f"[{n_done}/{len(rows)}] {time.perf_counter() - t0:.0f}s  "
                      f"strict_acc={acc:.4f}  peak_mem={peak:.1f}GB")

    final = [done[r["id"]] for r in rows]
    metrics = summarize(final, cfg, time.perf_counter() - t0)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
