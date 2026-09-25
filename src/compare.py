"""对比多个评测 run：准确率 + 与参照 run 的配对 McNemar 检验。

用法: .venv/bin/python src/compare.py --ref gsm8k_baseline run1 run2 ... [--out results/xxx.md]
读取 results/<run>/predictions.jsonl（strict_correct 字段），输出 markdown 表格。
"""
import argparse
import json
import math
from pathlib import Path


def load(run):
    path = Path("results") / run / "predictions.jsonl"
    return {r["id"]: r for r in map(json.loads, open(path))}


def mcnemar_exact(b, c):
    """精确 McNemar（双侧）：b = 参照错/新对，c = 参照对/新错。在 H0 下 b ~ Binomial(b+c, 0.5)。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="gsm8k_baseline")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="strict", choices=["strict", "lenient"],
                    help="Δ 和配对检验用哪个口径：strict（只认 boxed）/ lenient（无 boxed 取最后一个数字）")
    args = ap.parse_args()

    ref = load(args.ref)
    key = f"{args.metric}_correct"
    rows = []
    for run in [args.ref] + args.runs:
        pred = load(run)
        assert pred.keys() == ref.keys(), f"{run} 的题目集合与参照不同"
        max_tokens = json.loads((Path("results") / run / "metrics.json").read_text())["config"]["max_tokens"]
        n = len(pred)
        acc = sum(r[key] for r in pred.values()) / n
        row = {
            "run": run, "n": n, "strict": sum(r["strict_correct"] for r in pred.values()) / n,
            "lenient": sum(r["lenient_correct"] for r in pred.values()) / n,
            "boxed": sum(r["boxed"] is not None for r in pred.values()) / n,
            "tokens": sum(r["gen_tokens"] for r in pred.values()) / n,
            "truncated": sum(r["gen_tokens"] >= max_tokens for r in pred.values()),
            "max_tokens": max_tokens,
        }
        if run != args.ref:
            b = sum((not ref[i][key]) and pred[i][key] for i in ref)
            c = sum(ref[i][key] and not pred[i][key] for i in ref)
            row.update({"delta": acc - sum(r[key] for r in ref.values()) / n,
                        "gain": b, "loss": c, "p": mcnemar_exact(b, c)})
        rows.append(row)

    lines = [
        f"参照：`{args.ref}`；Δ 与配对检验（精确 McNemar，双侧）的口径：**{args.metric}**。",
        "",
        "| run | strict | Δ vs 参照 | 参照错→对 | 参照对→错 | p 值 | lenient | boxed | 平均输出 token | 截断（≥max_tokens） |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = (f"{r['delta'] * 100:+.2f}pp | {r['gain']} | {r['loss']} | {r['p']:.2g}"
             if "delta" in r else "— | — | — | —")
        lines.append(f"| {r['run']} | {r['strict'] * 100:.2f}% | {d} | {r['lenient'] * 100:.2f}% | "
                     f"{r['boxed'] * 100:.1f}% | {r['tokens']:.0f} | {r['truncated']} (@{r['max_tokens']}) |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
