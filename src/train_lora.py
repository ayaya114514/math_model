"""阶段 4：LoRA 微调（封装 mlx_lm.lora）。

1) data/filtered 的数据 -> mlx-lm chat 格式 {"messages": [system, user, assistant]}（system prompt 与评测一致）
2) 断言训练 / 验证数据不含测试题
3) 按 epochs 计算 iters 和 cosine 学习率调度，生成 mlx-lm 配置并调用 mlx_lm.lora
4) 解析日志 -> adapters/<run>/curve.json

用法: .venv/bin/python -u src/train_lora.py configs/train_<run>.yaml
产出: data/lora/<run>/{train,valid}.jsonl, adapters/<run>/（LoRA 权重、mlx 配置、train.log、curve.json）
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

from decontaminate import build_index, load_tests, match


def to_chat(row, system_prompt):
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["response"]},
    ]}


def assert_clean(rows, dcfg, exact, index, n_grams, name):
    for r in rows:
        how, tid, cov, hit = match(r["question"], exact, index, n_grams, dcfg["ngram"], dcfg["thresholds"])
        assert not hit, f"{name}: {r['id']} 与测试题 {tid} 重合（{how}, {cov:.2f}）"


def parse_log(text):
    train = [{"iter": int(m[1]), "loss": float(m[2]), "lr": float(m[3]), "peak_mem_gb": float(m[4])}
             for m in re.finditer(r"Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+-]+),.*?Peak mem ([\d.]+) GB", text)]
    val = [{"iter": int(m[1]), "loss": float(m[2])}
           for m in re.finditer(r"Iter (\d+): Val loss ([\d.]+)", text)]
    return {"train": train, "val": val}


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    run = cfg["run_name"]
    data_dir = Path("data/lora") / run
    adapter_dir = Path("adapters") / run
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train = [json.loads(line) for line in open(cfg["train_file"])]
    valid = [json.loads(line) for line in open(cfg["valid_file"])]
    assert not ({r["id"] for r in train} & {r["id"] for r in valid}), "train 与 valid 重叠"
    dcfg = yaml.safe_load(open(cfg["decontam_config"]))
    exact, index, n_grams = build_index(load_tests(dcfg), dcfg["ngram"])
    assert_clean(train, dcfg, exact, index, n_grams, "train")
    assert_clean(valid, dcfg, exact, index, n_grams, "valid")

    for name, rows in [("train", train), ("valid", valid)]:
        with open(data_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(to_chat(r, cfg["system_prompt"]), ensure_ascii=False) + "\n")

    m = dict(cfg["mlx"])  # 原样传给 mlx_lm.lora 的参数
    bs, accum = m["batch_size"], m["grad_accumulation_steps"]
    # mlx 按 batch 迭代，丢弃不满一个 batch 的尾部；iters 按 micro-batch 计数
    iters = cfg.get("iters") or math.ceil(cfg["epochs"] * len(train) / bs)
    iters = math.ceil(iters / accum) * accum  # 对齐到整数次参数更新
    updates = iters // accum
    warmup = max(1, round(cfg["warmup_ratio"] * updates))
    lr = m["learning_rate"]
    m.update({
        "model": cfg["base_model"], "train": True, "data": str(data_dir),
        "adapter_path": str(adapter_dir), "iters": iters, "seed": cfg["seed"],
        # 调度按参数更新次数计步（optimizer.update 才推进）
        "lr_schedule": {"name": "cosine_decay", "warmup": warmup, "warmup_init": 0.0,
                        "arguments": [lr, updates - warmup, lr * cfg["min_lr_ratio"]]},
    })
    mlx_cfg = adapter_dir / "mlx_lora_config.yaml"
    mlx_cfg.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False))
    print(f"[{run}] train={len(train)} valid={len(valid)} iters={iters} "
          f"(≈{iters * bs / len(train):.2f} epoch, {updates} 次更新, warmup {warmup})")

    log_path = adapter_dir / "train.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "mlx_lm", "lora", "-c", str(mlx_cfg)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if re.search(r"Iter \d+:|Error|Traceback|Saved|Trainable", line):
                print(line.rstrip())
        proc.wait()
    curve = parse_log(log_path.read_text())
    (adapter_dir / "curve.json").write_text(json.dumps(curve, indent=2))
    assert proc.returncode == 0, f"mlx_lm.lora 退出码 {proc.returncode}，见 {log_path}"
    print(f"完成：{adapter_dir}（train 点 {len(curve['train'])}，val 点 {len(curve['val'])}）")


if __name__ == "__main__":
    main()
