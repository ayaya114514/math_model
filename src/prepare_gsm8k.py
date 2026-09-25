"""把 GSM8K 下载并转成统一 jsonl：data/raw/gsm8k_{train,test}.jsonl

每条字段：id, question, solution（人工解答，已去掉 #### 行）, gold（标准答案字符串）, source, split
用法: .venv/bin/python src/prepare_gsm8k.py
"""
import json
from pathlib import Path

from datasets import load_dataset

from answer import gsm8k_gold

OUT_DIR = Path("data/raw")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")
    for split in ["train", "test"]:
        out = OUT_DIR / f"gsm8k_{split}.jsonl"
        with out.open("w") as f:
            for i, ex in enumerate(ds[split]):
                solution = ex["answer"].split("####")[0].strip()
                row = {
                    "id": f"gsm8k-{split}-{i:05d}",
                    "question": ex["question"],
                    "solution": solution,
                    "gold": gsm8k_gold(ex["answer"]),
                    "source": "gsm8k",
                    "split": split,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(ds[split])} 条 -> {out}")


if __name__ == "__main__":
    main()
