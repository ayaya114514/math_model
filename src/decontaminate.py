"""阶段 2：训练数据与 GSM8K / MATH 测试集去重。

1) 规范化后精确匹配（小写、去 LaTeX 间距命令、去空白和 $；保留数学符号，避免 sin(-120°) 与 sin 120° 误判相同）
2) 词级 n-gram 重叠：某条训练题覆盖了某道测试题 ≥ threshold 比例的 n-gram，就判为污染
   （分母用测试题的 n-gram 数：训练题再长也不会稀释重合比例）
   阈值按测试集分别设置（GSM8K 叙事题更严，MATH 套话多更宽），见 configs/decontam.yaml

用法:
  .venv/bin/python -u src/decontaminate.py configs/decontam.yaml            # 正式去重
  .venv/bin/python -u src/decontaminate.py configs/decontam.yaml --scan     # 只统计各阈值命中数 + 样例，不写数据
产出: data/raw/<dataset>.jsonl（去重后）, data/raw/removed/<dataset>.jsonl（被删样本及命中的测试题）,
      results/stage2_decontam.json
"""
import collections
import json
import re
import sys
from pathlib import Path

import yaml

LATEX_SPACING = re.compile(r"\\[,;:! ]|\\quad|\\qquad|\\left|\\right|\\displaystyle")
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(s):
    """n-gram 用：只保留字母数字词。"""
    s = LATEX_SPACING.sub(" ", s.lower())
    return NON_ALNUM.sub(" ", s).strip()


def normalize_exact(s):
    """精确匹配用：保留全部数学符号，只去掉空白、$ 和结尾标点。"""
    s = LATEX_SPACING.sub("", s.lower()).replace("$", "")
    return re.sub(r"\s+", "", s).rstrip(".?!")


def ngrams(words, n):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_tests(cfg):
    tests = []
    for path in cfg["test_sets"]:
        for r in map(json.loads, open(path)):
            assert r["split"] == "test", path
            tests.append({"id": r["id"], "question": r["question"]})
    return tests


def build_index(tests, n):
    exact = {}
    index = collections.defaultdict(list)
    n_grams = {}
    for t in tests:
        exact.setdefault(normalize_exact(t["question"]), t["id"])
        grams = ngrams(normalize(t["question"]).split(), n)
        n_grams[t["id"]] = len(grams)
        for g in grams:
            index[g].append(t["id"])
    return exact, index, n_grams


def test_set(tid):
    return tid.split("-")[0]  # gsm8k-test-00001 -> gsm8k


def match(question, exact, index, n_grams, n, thresholds):
    """返回 (匹配方式, 测试题 id, 覆盖率, 是否污染)。
    选"覆盖率减去该测试集阈值"最大的那道测试题，超过阈值即为污染。"""
    key = normalize_exact(question)
    if key in exact:
        return "exact", exact[key], 1.0, True
    hits = collections.Counter()
    for g in ngrams(normalize(question).split(), n):
        for tid in index.get(g, ()):
            hits[tid] += 1
    if not hits:
        return None, None, 0.0, False
    tid, c = max(hits.items(), key=lambda kv: kv[1] / n_grams[kv[0]] - thresholds[test_set(kv[0])])
    cov = c / n_grams[tid]
    return "ngram", tid, cov, cov >= thresholds[test_set(tid)]


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    scan = "--scan" in sys.argv
    n, thr = cfg["ngram"], cfg["thresholds"]
    tests = load_tests(cfg)
    test_q = {t["id"]: t["question"] for t in tests}
    exact, index, n_grams = build_index(tests, n)
    print(f"测试题 {len(tests)} 道（n={n}，thresholds={thr}）；其中少于 {n} 个词、只能精确匹配的有 "
          f"{sum(v == 0 for v in n_grams.values())} 道")

    report = {"ngram": n, "thresholds": thr, "test_sets": cfg["test_sets"], "datasets": {}}
    for name in cfg["datasets"]:
        src = Path(cfg["input_dir"]) / f"{name}.jsonl"
        bins = collections.Counter()
        by_test = collections.Counter()
        examples = collections.defaultdict(list)
        kept = removed = total = 0
        out = rem = None
        if not scan:
            Path("data/raw/removed").mkdir(parents=True, exist_ok=True)
            out = open(f"data/raw/{name}.jsonl", "w")
            rem = open(f"data/raw/removed/{name}.jsonl", "w")
        for line in open(src):
            r = json.loads(line)
            total += 1
            how, tid, cov, hit = match(r["question"], exact, index, n_grams, n, thr)
            for b in (0.2, 0.3, 0.5, 0.8):
                if how == "exact" or cov >= b:
                    bins[f">={b}"] += 1
                    if len(examples[b]) < cfg["examples_per_bin"] and (how == "exact" or cov < b + 0.2):
                        examples[b].append({"cov": round(cov, 3), "how": how, "train": r["question"][:300],
                                            "test_id": tid, "test": test_q[tid][:300]})
            bins["exact"] += how == "exact"
            if hit:
                removed += 1
                by_test[test_set(tid)] += 1
                if rem:
                    rem.write(json.dumps({**r, "contam": {"how": how, "test_id": tid, "coverage": cov,
                                                          "test_question": test_q[tid]}},
                                         ensure_ascii=False) + "\n")
            else:
                kept += 1
                if out:
                    out.write(line)
        if out:
            out.close()
            rem.close()
        report["datasets"][name] = {"total": total, "removed": removed, "kept": kept,
                                    "removed_rate": removed / total, "bins": dict(bins),
                                    "removed_by_test_set": dict(by_test),
                                    "examples": {str(k): v for k, v in examples.items()}}
        print(f"{name}: total={total} removed={removed} ({removed / total:.3%}) bins={dict(bins)} "
              f"by_test={dict(by_test)}")
    out_path = Path("results") / ("stage2_decontam_scan.json" if scan else "stage2_decontam.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"写入 {out_path}")


if __name__ == "__main__":
    main()
