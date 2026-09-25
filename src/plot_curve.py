"""画训练曲线：adapters/<run>/curve.json -> results/<run>/train_curve.png

用法: .venv/bin/python src/plot_curve.py <run_name> [<run_name> ...]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 参考调色板（dataviz skill，已用 validate_palette.js 验证 light 模式通过）
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRAIN, VAL = "#2a78d6", "#eb6834"


def plot(run):
    curve = json.loads((Path("adapters") / run / "curve.json").read_text())
    tr, va = curve["train"], curve["val"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot([p["iter"] for p in tr], [p["loss"] for p in tr], color=TRAIN, lw=2, label="train loss")
    ax.plot([p["iter"] for p in va], [p["loss"] for p in va], color=VAL, lw=2, marker="o", ms=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="val loss")

    # 直接标注每条线的末端值（文字用墨色，不用系列色）
    for pts, name in [(tr, "train"), (va, "val")]:
        if pts:
            x, y = pts[-1]["iter"], pts[-1]["loss"]
            ax.annotate(f"{name} {y:.3f}", (x, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)
    if va:
        ax.annotate(f"val {va[0]['loss']:.3f}", (va[0]["iter"], va[0]["loss"]), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color=INK_2)

    ax.set_title(f"{run}: LoRA training loss", color=INK, fontsize=12, loc="left")
    ax.set_xlabel("iteration (micro-batch)", color=MUTED, fontsize=10)
    ax.set_ylabel("loss (response tokens only)", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    out = Path("results") / run / "train_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"写入 {out}")


if __name__ == "__main__":
    for run in sys.argv[1:]:
        plot(run)
