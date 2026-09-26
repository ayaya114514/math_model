"""从本地 MATH test（5000 题）中取出标准子集 MATH-500：data/raw/math500.jsonl

MATH-500（HuggingFaceH4/MATH-500，来自 PRM800K 的划分）是论文和榜单常用的子集，便于和官方分数对照。
按题目文本对齐到 data/raw/math_test.jsonl，沿用本地 id；gold 用 MATH-500 自带的 answer 字段。
用法: .venv/bin/python src/prepare_math500.py
"""
import json
import re
from pathlib import Path

from datasets import load_dataset

SRC = Path("data/raw/math_test.jsonl")
OUT = Path("data/raw/math500.jsonl")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    local = {norm(r["question"]): r for r in map(json.loads, open(SRC))}
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows, gold_diff = [], 0
    for ex in ds:
        r = local.get(norm(ex["problem"]))
        assert r is not None, f"MATH-500 题目在本地 test 中找不到：{ex['unique_id']}"
        assert r["split"] == "test"
        gold_diff += r["gold"] != ex["answer"]
        rows.append({**r, "gold": ex["answer"], "subject": ex["subject"], "level": ex["level"],
                     "unique_id": ex["unique_id"]})
    assert len(rows) == 500 and len({r["id"] for r in rows}) == 500
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} 题 -> {OUT}（本地 gold 与 MATH-500 answer 字面不同的 {gold_diff} 条，已用后者）")


if __name__ == "__main__":
    main()
