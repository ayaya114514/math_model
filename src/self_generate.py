"""阶段 5-3：自蒸馏 / 拒绝采样（Rejection-sampling Fine-Tuning）。

生成: 学生对 GSM8K 训练题（与 gsm8k_human 臂相同的题目和顺序）每题采样 k 次，写 generations.jsonl（可断点续跑）
构建: 每题从"答对且格式合格"的采样中随机挑 1 条 -> data/filtered/self_rft/{valid,train_N}.jsonl

用法:
  .venv/bin/python -u src/self_generate.py configs/self_rft.yaml generate [--limit N]
  .venv/bin/python -u src/self_generate.py configs/self_rft.yaml build
"""
import collections
import json
import random
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import yaml

from answer import all_boxed, judge
from decontaminate import build_index, load_tests, match


def load_questions(cfg):
    src = Path(cfg["question_source"])
    valid = [json.loads(line) for line in open(src / "valid.jsonl")]
    train = [json.loads(line) for line in open(src / "train_7000.jsonl")][: cfg["n_train_questions"]]
    return [dict(r, split="valid") for r in valid] + [dict(r, split="train") for r in train]


def generate(cfg, limit):
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "generations.jsonl"
    done = {json.loads(line)["id"] for line in open(path)} if path.exists() else set()
    qs = load_questions(cfg)
    if limit:
        qs = qs[:limit]
    todo = [q for q in qs if q["id"] not in done]
    print(f"共 {len(qs)} 题 × {cfg['samples_per_question']} 采样；已完成 {len(done)}，待生成 {len(todo)}")
    if not todo:
        return

    model, tok = load(cfg["model"])
    sampler = make_sampler(temp=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"])
    k = cfg["samples_per_question"]
    t0 = time.perf_counter()
    n_ok = n_all = 0
    with path.open("a") as f:
        for s in range(0, len(todo), cfg["chunk_questions"]):
            chunk = todo[s: s + cfg["chunk_questions"]]
            prompts = []
            for q in chunk:
                msgs = [{"role": "system", "content": cfg["system_prompt"]},
                        {"role": "user", "content": q["question"]}]
                prompts += [tok.apply_chat_template(msgs, add_generation_prompt=True)] * k
            resp = batch_generate(model, tok, prompts, max_tokens=cfg["max_tokens"], sampler=sampler,
                                  completion_batch_size=cfg["batch_size"], prefill_batch_size=8)
            for j, q in enumerate(chunk):
                samples = []
                for text in resp.texts[j * k: (j + 1) * k]:
                    jd = judge(text, q["gold"])
                    samples.append({"response": text, "gen_tokens": len(tok.encode(text)),
                                    "boxed": jd["boxed"], "correct": jd["strict_correct"]})
                    n_ok += jd["strict_correct"]
                    n_all += 1
                f.write(json.dumps({"id": q["id"], "split": q["split"], "question": q["question"],
                                    "gold": q["gold"], "samples": samples}, ensure_ascii=False) + "\n")
            f.flush()
            del resp
            mx.clear_cache()
            print(f"[{s + len(chunk)}/{len(todo)}] {time.perf_counter() - t0:.0f}s  "
                  f"采样正确率 {n_ok / n_all:.3f}")


def build(cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    dcfg = yaml.safe_load(open(cfg["decontam_config"]))
    exact, index, n_grams = build_index(load_tests(dcfg), dcfg["ngram"])
    rng = random.Random(f"{cfg['seed']}-self_rft")
    gens = [json.loads(line) for line in open(Path(cfg["out_dir"]) / "generations.jsonl")]
    order = {q["id"]: i for i, q in enumerate(load_questions(cfg))}
    gens.sort(key=lambda g: order[g["id"]])  # 保持与 gsm8k_human 臂相同的题目顺序

    def prompt_len(q):
        msgs = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": q}]
        return len(tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                              add_special_tokens=False))

    stats = collections.Counter()
    pass_counts = collections.Counter()
    out = {"valid": [], "train": []}
    for g in gens:
        stats["questions"] += 1
        n_correct = sum(s["correct"] for s in g["samples"])
        pass_counts[n_correct] += 1
        ok = []
        for s in g["samples"]:
            if not s["correct"]:
                continue
            boxes = all_boxed(s["response"])
            if len({b.replace(" ", "") for b in boxes if b}) != 1 or s["gen_tokens"] >= cfg["max_tokens"]:
                stats["correct_but_bad_format"] += 1
                continue
            ok.append(s)
        if not ok:
            stats["no_usable_sample"] += 1
            continue
        s = rng.choice(ok)  # 随机挑，不挑最短，避免引入长度偏差
        p_tok, r_tok = prompt_len(g["question"]), len(tok.encode(s["response"])) + 1
        if p_tok + r_tok > cfg["max_total_tokens"]:
            stats["too_long"] += 1
            continue
        how, tid, cov, hit = match(g["question"], exact, index, n_grams, dcfg["ngram"], dcfg["thresholds"])
        assert not hit, f"测试集污染: {g['id']} ~ {tid}"
        out[g["split"]].append({
            "id": g["id"], "question": g["question"], "response": s["response"].strip(),
            "final_answer": s["boxed"], "gold": g["gold"], "verified": True, "source": "gsm8k_train",
            "dataset": "self-RFT", "teacher": "Qwen2.5-1.5B-Instruct (self, rejection-sampled)",
            "prompt_tokens": p_tok, "response_tokens": r_tok,
        })

    n = cfg["train_size"]
    assert len(out["train"]) >= n, f"可用训练题只有 {len(out['train'])} 道，不足 {n}；请增大 n_train_questions"
    d = Path("data/filtered/self_rft")
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in [("valid", out["valid"]), (f"train_{n}", out["train"][:n])]:
        with open(d / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    all_samples = [s for g in gens for s in g["samples"]]
    report = {
        "questions": stats["questions"], "samples": len(all_samples),
        "sample_accuracy": sum(s["correct"] for s in all_samples) / len(all_samples),
        "pass_at_k_distribution": {str(k): v for k, v in sorted(pass_counts.items())},
        "drops": {k: v for k, v in stats.items() if k != "questions"},
        "valid": len(out["valid"]), "train_available": len(out["train"]), "train_written": n,
        "train_response_tokens_mean": statistics.mean(r["response_tokens"] for r in out["train"][:n]),
    }
    Path("results/stage5").mkdir(parents=True, exist_ok=True)
    Path("results/stage5/self_rft_build.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1]))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    {"generate": lambda: generate(cfg, limit), "build": lambda: build(cfg)}[sys.argv[2]]()
