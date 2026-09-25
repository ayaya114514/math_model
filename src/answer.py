"""答案提取与比对。

- 模型答案：取回答中最后一个 \\boxed{...}（支持嵌套花括号）
- GSM8K 标准答案：#### 之后的数字
- 严格指标（strict）：只认 \\boxed{}；宽松指标（lenient）：没有 boxed 时取全文最后一个数字
"""
import re

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_boxed(text):
    """返回最后一个 \\boxed{...} 的内容；没有则返回 None。"""
    start = text.rfind("\\boxed")
    if start == -1:
        return None
    return _boxed_at(text, start)


def all_boxed(text):
    """按出现顺序返回所有 \\boxed{...} 的内容（不闭合的为 None）。"""
    out, pos = [], text.find("\\boxed")
    while pos != -1:
        out.append(_boxed_at(text, pos))
        pos = text.find("\\boxed", pos + 6)
    return out


def _boxed_at(text, start):
    """解析从 start 处开始的 \\boxed{...}，支持嵌套花括号。"""
    i = text.find("{", start)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None  # 花括号不闭合（通常是输出被截断）


def _clean(s):
    """去掉 LaTeX 修饰、单位、千分位逗号，把 \\frac{a}{b} 变成 a/b。"""
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(?:text|mbox|mathrm)\{([^{}]*)\}", r" \1 ", s)  # 保留内容，如 \text{ dollars}
    s = s.replace("{,}", ",")
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = re.sub(r"(\d),(?=\d{3}\b)", r"\1", s)  # 1,000 -> 1000
    s = s.replace("(", "").replace(")", "")
    return s


def to_number(s):
    """把答案字符串解析成 float；有多个数字时取最后一个（如 "2 \\times 3 = 6"）。解析失败返回 None。"""
    if s is None:
        return None
    nums = NUM_RE.findall(_clean(s))
    if not nums:
        return None
    last = nums[-1]
    if "/" in last:
        a, b = last.split("/")
        return float(a) / float(b) if float(b) != 0 else None
    return float(last)


def gsm8k_gold(answer_field):
    """GSM8K 原始 answer 字段 -> 标准答案字符串（#### 之后，去逗号）。"""
    assert "####" in answer_field, answer_field
    return answer_field.split("####")[-1].strip().replace(",", "")


def is_equal(pred, gold, tol=1e-4):
    return pred is not None and gold is not None and abs(pred - gold) <= tol * max(1.0, abs(gold))


def judge(response, gold_str):
    """返回单条评测结果：strict / lenient 是否正确，以及提取到的内容。"""
    gold = float(gold_str)
    boxed = extract_boxed(response)
    strict_pred = to_number(boxed)
    lenient_pred = strict_pred if boxed is not None else to_number(response)
    return {
        "boxed": boxed,
        "pred": strict_pred,
        "lenient_pred": lenient_pred,
        "strict_correct": is_equal(strict_pred, gold),
        "lenient_correct": is_equal(lenient_pred, gold),
    }
