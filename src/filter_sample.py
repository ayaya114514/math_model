"""阶段 3：筛选、清理、题目去重、抽样。

每个数据臂：
  读取 data/raw/<dataset>.jsonl（已去除测试集污染）-> 来源过滤 -> 答案验证 -> 格式清理
  -> 丢弃无明确最终答案 / 多个不同 boxed / 超长的样本 -> 同题只留一条
  -> 留出 valid -> 嵌套抽样训练子集
用法: .venv/bin/python -u src/filter_sample.py configs/filter_sample.yaml [--limit N]
产出: data/filtered/<arm>/{valid,train_<N>}.jsonl, results/stage3_stats.json
"""
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from answer import all_boxed, is_equal, to_number
from decontaminate import build_index, load_tests, match

CALC_RE = re.compile(r"<<[^<>]*>>")
GSM_MARK_RE = re.compile(r"^#### .*$\n?", re.M)
ANSWER_IS_RE = re.compile(r"\n*The answer is:?\s*(.+?)\s*$", re.S)


def clean(arm_dataset, text, final_answer):
    """统一回答格式：以 \\boxed{答案} 收尾。返回清理后的文本。"""
    if arm_dataset == "gsm8k_human":
        text = CALC_RE.sub("", text).strip()
        return f"{text}\nThe answer is $\\boxed{{{final_answer}}}$."
    if arm_dataset == "metamath":
        m = ANSWER_IS_RE.search(text)
        body = text[: m.start()].rstrip() if m else text.rstrip()
        body = GSM_MARK_RE.sub("", body).rstrip()  # 去掉 GSM8K 风格的 "#### 答案" 行
        if "\\boxed" in body:  # MATH 类已有 boxed，去掉重复的结尾
            return body
        return f"{body}\nThe answer is $\\boxed{{{final_answer}}}$."
    return text.strip()


def verify(mode, r):
    """返回 True / False（有标准答案时是否一致）/ None（无法验证）。"""
    if mode == "none":
        return None
    if mode == "math_verify_field":
        return bool(r.get("math_verify")) and bool(r.get("reasoning_complete"))
    if mode == "numeric":
        return is_equal(to_number(r["final_answer"]), to_number(r["gold"]))
    raise ValueError(mode)


def qkey(q):
    return " ".join(q.lower().split())


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    out_root = Path("data/filtered_preview" if limit else "data/filtered")
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    dcfg = yaml.safe_load(open(cfg["decontam_config"]))
    exact, index, n_grams = build_index(load_tests(dcfg), dcfg["ngram"])

    def prompt_len(q):
        msgs = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": q}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return len(tok.encode(text, add_special_tokens=False))

    stats = {}
    for arm, a in cfg["arms"].items():
        print(f"== {arm}")
        drop = collections.Counter()
        kept = []
        for i, line in enumerate(open(f"data/raw/{a['dataset']}.jsonl")):
            if limit and i >= limit:
                break
            r = json.loads(line)
            if a["sources"] and r["source"] not in a["sources"]:
                continue
            drop["seen"] += 1
            if not r["final_answer"] or not r["final_answer"].strip():
                drop["no_final_answer"] += 1
                continue
            ok = verify(a["verify"], r)
            if ok is False:
                drop["answer_mismatch"] += 1
                continue
            text = clean(a["dataset"], r["reasoning"], r["final_answer"])
            answer_part = text.split("</think>")[-1]
            boxes = all_boxed(answer_part)
            if not boxes or any(b is None or not b.strip() for b in boxes):
                drop["no_or_broken_boxed"] += 1
                continue
            if len({b.replace(" ", "") for b in boxes}) > 1:
                drop["multiple_distinct_boxed"] += 1
                continue
            kept.append({"id": r["id"], "question": r["question"], "response": text,
                         "final_answer": boxes[-1], "gold": r["gold"], "verified": ok,
                         "source": r["source"], "dataset": r["dataset"], "teacher": r["teacher"]})

        # 同题只留一条：先按 seed 打乱，再保留每题第一次出现的
        rng = random.Random(f"{cfg['seed']}-{arm}")
        rng.shuffle(kept)
        seen, uniq = set(), []
        for r in kept:
            k = qkey(r["question"])
            if k in seen:
                drop["duplicate_question"] += 1
                continue
            seen.add(k)
            uniq.append(r)

        # 长度：prompt（含 chat 模板）+ 回答 + 结束符
        for r in uniq:
            r["prompt_tokens"] = prompt_len(r["question"])
            r["response_tokens"] = len(tok.encode(r["response"])) + 1
        all_lens = [r["prompt_tokens"] + r["response_tokens"] for r in uniq]
        cap = a.get("max_total_tokens", cfg["max_total_tokens"])
        pool = [r for r in uniq if r["prompt_tokens"] + r["response_tokens"] <= cap]
        drop["too_long"] = len(uniq) - len(pool)

        # 留出 valid，其余按顺序（已打乱）取嵌套前缀
        valid, rest = pool[: cfg["valid_size"]], pool[cfg["valid_size"]:]
        outputs = {"valid": valid}
        for n in a["sizes"]:
            if n <= len(rest):
                outputs[f"train_{n}"] = rest[:n]
            else:
                print(f"  ⚠️ 池子只有 {len(rest)} 条，不足 {n}，跳过")

        # 断言：输出的每一条都不与测试集重合（同一去重标准）
        for name, rows in outputs.items():
            for r in rows:
                how, tid, cov, hit = match(r["question"], exact, index, n_grams, dcfg["ngram"], dcfg["thresholds"])
                assert not hit, f"测试集污染: {arm}/{name} {r['id']} ~ {tid} ({how}, {cov:.2f})"
        assert not ({r["id"] for r in valid} & {r["id"] for r in rest}), "valid 与 train 重叠"

        d = out_root / arm
        d.mkdir(parents=True, exist_ok=True)
        for old in list(d.glob("train_*.jsonl")) + list(d.glob("valid.jsonl")):
            old.unlink()  # 清掉上次的产出，避免残留旧子集
        for name, rows in outputs.items():
            with open(d / f"{name}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        resp = [r["response_tokens"] for r in pool]
        stats[arm] = {
            "drops": dict(drop), "max_total_tokens": cap, "unique_before_len_filter": len(uniq), "pool": len(pool),
            "outputs": {k: len(v) for k, v in outputs.items()},
            "total_tokens_all": {"p50": pct(all_lens, 50), "p90": pct(all_lens, 90), "p99": pct(all_lens, 99),
                                 "max": max(all_lens),
                                 "frac_le_1024": sum(x <= 1024 for x in all_lens) / len(all_lens),
                                 "frac_le_2048": sum(x <= 2048 for x in all_lens) / len(all_lens)},
            "response_tokens_pool": {"mean": statistics.mean(resp), "p50": pct(resp, 50), "p90": pct(resp, 90),
                                     "p99": pct(resp, 99), "max": max(resp)},
        }
        s = stats[arm]
        print(f"  drops={s['drops']}\n  unique={len(uniq)} pool(<= {cap})={len(pool)} "
              f"outputs={s['outputs']}\n  total_tokens(去长度过滤前)={s['total_tokens_all']}\n"
              f"  response_tokens(池内)={s['response_tokens_pool']}")

    out = Path("results") / ("stage3_stats_preview.json" if limit else "stage3_stats.json")
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
