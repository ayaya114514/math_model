"""阶段 2 调研：通过 HF datasets-server 抽样（不下载完整数据集），统计各候选数据集的
回答长度（Qwen tokenizer）、来源分布、答案字段情况。

用法: .venv/bin/python -u src/survey_datasets.py configs/survey_datasets.yaml
产出: results/stage2_survey.json
"""
import collections
import json
import random
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from transformers import AutoTokenizer

API = "https://datasets-server.huggingface.co"


def get_json(url, retries=6):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "math-distill-survey"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # 网络抖动 / 限流时重试
            if i == retries - 1:
                raise
            print(f"  retry {i + 1}: {e}")
            time.sleep(15 * (i + 1))


def fetch_rows(ds, config, split, n_total, offsets, per_offset):
    rows = []
    for off in offsets:
        q = urllib.parse.urlencode({"dataset": ds, "config": config, "split": split,
                                    "offset": off, "length": per_offset})
        rows += [r["row"] for r in get_json(f"{API}/rows?{q}")["rows"]]
    return rows


def last_assistant(msgs, role_key="role", content_key="content", assistant=("assistant", "gpt")):
    for m in reversed(msgs):
        if m.get(role_key) in assistant:
            return m.get(content_key) or ""
    return ""


# 每个数据集怎么取出 (回答文本, 来源, 是否带标准答案, 额外信息)
def extract(name, row):
    if name == "openr1":
        text = last_assistant(row["messages"])
        ok = row.get("correctness_math_verify") or []
        return text, row.get("source"), bool(row.get("answer")), {"verified_any": any(ok)}
    if name == "openthoughts3":
        text = last_assistant(row["conversations"], "from", "value")
        return text, f'{row.get("domain")}/{row.get("source")}', False, {}
    if name == "numina_cot":
        return row["solution"], row.get("source"), False, {}
    if name == "omi2":
        return row["generated_solution"], row.get("problem_source"), bool(row.get("expected_answer")), {}
    if name in ("nemotron_v4", "omr"):
        text = row.get("generated_solution") or last_assistant(row.get("messages") or [])
        extra = {}
        if "messages" in row:
            m = [x for x in row["messages"] if x.get("role") == "assistant"]
            extra["has_reasoning_field"] = bool(m and m[-1].get("reasoning_content"))
            if extra["has_reasoning_field"]:
                text = m[-1]["reasoning_content"] + "\n" + (m[-1].get("content") or "")
        return text, row.get("problem_source") or row.get("source"), bool(row.get("expected_answer")), extra
    if name == "metamath":
        return row["response"], row.get("type"), False, {}
    raise ValueError(name)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main(cfg_path, only=None):
    cfg = yaml.safe_load(open(cfg_path))
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    out = Path("results/stage2_survey.json")
    report = json.loads(out.read_text()) if (only and out.exists()) else {}
    for c in cfg["datasets"]:
        name = c["name"]
        if only and name not in only:
            continue
        rng = random.Random(f'{cfg["seed"]}-{name}')  # 每个数据集独立的抽样种子，单独重跑结果不变
        print(f"== {name}: {c['repo']} [{c['config']}/{c['split']}]")
        n_rows = c["num_rows"]
        offsets = sorted(rng.sample(range(0, n_rows - cfg["per_offset"]), cfg["num_offsets"]))
        rows = fetch_rows(c["repo"], c["config"], c["split"], n_rows, offsets, cfg["per_offset"])
        lens, sources, has_gold, boxed, think, extras = [], collections.Counter(), 0, 0, 0, collections.Counter()
        example = None
        for row in rows:
            text, src, gold, extra = extract(name, row)
            if c.get("domain_filter") and not str(src).startswith(c["domain_filter"]):
                continue
            lens.append(len(tok.encode(text)))
            sources[src] += 1
            has_gold += gold
            boxed += "\\boxed" in text
            think += "<think>" in text
            for k, v in extra.items():
                extras[k] += bool(v)
            example = example or text
        n = len(lens)
        report[name] = {
            "repo": c["repo"], "config": c["config"], "split": c["split"], "sampled": n,
            "tokens_mean": statistics.mean(lens), "tokens_p50": pct(lens, 50),
            "tokens_p90": pct(lens, 90), "tokens_max": max(lens),
            "frac_le_1024": sum(x <= 1024 for x in lens) / n,
            "frac_le_2048": sum(x <= 2048 for x in lens) / n,
            "frac_le_4096": sum(x <= 4096 for x in lens) / n,
            "has_gold_field": has_gold / n, "boxed_rate": boxed / n, "think_tag_rate": think / n,
            "extra": {k: v / n for k, v in extras.items()},
            "top_sources": sources.most_common(8),
            "example_head": example[:600],
        }
        r = report[name]
        print(f"  n={n} mean={r['tokens_mean']:.0f} p50={r['tokens_p50']} p90={r['tokens_p90']} "
              f"max={r['tokens_max']} <=1k={r['frac_le_1024']:.2f} <=2k={r['frac_le_2048']:.2f} "
              f"boxed={r['boxed_rate']:.2f} gold={r['has_gold_field']:.2f} think={r['think_tag_rate']:.2f} "
              f"extra={r['extra']}")
        print(f"  sources={r['top_sources']}")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"写入 {out}")


if __name__ == "__main__":
    # 用法: survey_datasets.py <config> [--only name1,name2]（被限流时只补跑部分数据集）
    args = sys.argv[1:]
    only = args[args.index("--only") + 1].split(",") if "--only" in args else None
    main(args[0] if args and args[0] != "--only" else "configs/survey_datasets.yaml", only)
