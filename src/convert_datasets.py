"""阶段 2：把下载的公开数据集转成统一 jsonl（只转换格式，不筛选；筛选在阶段 3）。

统一字段: id, question, reasoning（老师完整回答）, final_answer（从回答中提取的答案，提取不到为 null）,
         gold（标准答案，没有为 null）, source（子来源）, dataset, teacher
另外生成 MATH 的 train / test（test 用于去重和以后评测，train 用于给 MetaMathQA 回查标准答案）。

用法: .venv/bin/python -u src/convert_datasets.py [--limit N]
产出: data/raw/converted/<dataset>.jsonl, data/raw/math_{train,test}.jsonl
"""
import glob
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

from answer import extract_boxed

HF = Path("data/raw/hf")
OUT = Path("data/raw/converted")


def norm_q(s):
    """回查原题时用的轻度规范化：只合并空白。"""
    return " ".join(s.split())


def iter_parquet(pattern, columns=None):
    for f in sorted(glob.glob(str(HF / pattern))):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=columns):
            yield from batch.to_pylist()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"  {path}: {n} 条")
    return n


def limited(it, limit):
    for i, x in enumerate(it):
        if limit and i >= limit:
            return
        yield x


# ---------- MATH（hendrycks）----------
def convert_math(split, limit):
    def gen():
        for i, ex in enumerate(iter_parquet(f"hendrycks_math/*/{split}-*.parquet")):
            yield {
                "id": f"math-{split}-{i:05d}",
                "question": ex["problem"],
                "solution": ex["solution"],
                "gold": extract_boxed(ex["solution"]),
                "source": ex["type"],
                "level": ex["level"],
                "split": split,
            }
    return write_jsonl(Path(f"data/raw/math_{split}.jsonl"), limited(gen(), limit))


# ---------- 各训练数据集 ----------
def convert_omi2(limit):
    def gen():
        for i, ex in enumerate(iter_parquet("OpenMathInstruct-2/data/train_1M-*.parquet")):
            text = ex["generated_solution"]
            yield {
                "id": f"omi2-{i:07d}", "question": ex["problem"], "reasoning": text,
                "final_answer": extract_boxed(text), "gold": ex["expected_answer"],
                "source": ex["problem_source"], "dataset": "OpenMathInstruct-2",
                "teacher": "Llama-3.1-405B-Instruct",
            }
    return write_jsonl(OUT / "omi2.jsonl", limited(gen(), limit))


def convert_openr1(limit):
    def gen():
        for i, ex in enumerate(iter_parquet("OpenR1-Math-220k/data/train-*.parquet")):
            text = next(m["content"] for m in reversed(ex["messages"]) if m["role"] == "assistant")
            # messages 里的回答对应 generations 中的哪一条，以及它是否通过 Math-Verify
            gens = ex["generations"] or []
            idx = gens.index(text) if text in gens else None
            mv = ex["correctness_math_verify"] or []
            after_think = text.split("</think>")[-1]
            yield {
                "id": f"openr1-{i:06d}", "question": ex["problem"], "reasoning": text,
                "final_answer": extract_boxed(after_think), "gold": ex["answer"],
                "source": ex["source"], "dataset": "OpenR1-Math-220k", "teacher": "DeepSeek-R1",
                "math_verify": mv[idx] if idx is not None and idx < len(mv) else None,
                "reasoning_complete": "</think>" in text,
            }
    return write_jsonl(OUT / "openr1.jsonl", limited(gen(), limit))


def convert_numina(limit):
    def gen():
        for i, ex in enumerate(iter_parquet("NuminaMath-CoT/data/train-*.parquet")):
            text = ex["solution"]
            yield {
                "id": f"numina-{i:06d}", "question": ex["problem"], "reasoning": text,
                "final_answer": extract_boxed(text),
                "gold": None,  # 解答由参考解答改写而来，没有独立的标准答案
                "source": ex["source"], "dataset": "NuminaMath-CoT", "teacher": "GPT-4o",
            }
    return write_jsonl(OUT / "numina.jsonl", limited(gen(), limit))


ANSWER_IS = re.compile(r"The answer is:?\s*(.+?)\s*$", re.S)


def convert_metamath(limit):
    # 回查表：GSM8K / MATH 训练集原题 -> 标准答案
    gold_of = {}
    for path in ["data/raw/gsm8k_train.jsonl", "data/raw/math_train.jsonl"]:
        for r in map(json.loads, open(path)):
            gold_of[norm_q(r["question"])] = r["gold"]
    data = json.load(open(HF / "MetaMathQA/MetaMathQA-395K.json"))
    stats = {"lookup_ok": 0, "lookup_miss": 0}

    def gen():
        for i, ex in enumerate(data):
            text = ex["response"]
            m = ANSWER_IS.search(text)
            gold = None
            # AnsAug（原题重新作答）和 Rephrased（改写问法）答案与原题相同；FOBAR/SV 改变了所求量
            if ex["type"].endswith(("_AnsAug", "_Rephrased")):
                gold = gold_of.get(norm_q(ex["original_question"]))
                stats["lookup_ok" if gold is not None else "lookup_miss"] += 1
            yield {
                "id": f"metamath-{i:06d}", "question": ex["query"], "reasoning": text,
                "final_answer": m.group(1).strip() if m else None, "gold": gold,
                "source": ex["type"], "dataset": "MetaMathQA", "teacher": "GPT-3.5-Turbo",
            }
    n = write_jsonl(OUT / "metamath.jsonl", limited(gen(), limit))
    print(f"  MetaMathQA 原题回查: {stats}")
    return n


def convert_gsm8k_human(limit):
    def gen():
        for r in map(json.loads, open("data/raw/gsm8k_train.jsonl")):
            yield {
                "id": r["id"], "question": r["question"], "reasoning": r["solution"],
                "final_answer": r["gold"], "gold": r["gold"], "source": "gsm8k_train",
                "dataset": "GSM8K-human", "teacher": "human",
            }
    return write_jsonl(OUT / "gsm8k_human.jsonl", limited(gen(), limit))


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    global OUT
    if limit:
        OUT = Path("data/raw/converted_preview")  # 小规模预览，不覆盖正式产出
    print("MATH:")
    convert_math("test", None)
    convert_math("train", None)
    for fn in [convert_gsm8k_human, convert_omi2, convert_metamath, convert_numina, convert_openr1]:
        print(fn.__name__)
        fn(limit)


if __name__ == "__main__":
    main()
