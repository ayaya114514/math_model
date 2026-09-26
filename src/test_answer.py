"""answer.py 的单元测试。用法: .venv/bin/python src/test_answer.py"""
from answer import all_boxed, extract_boxed, gsm8k_gold, judge, math_judge, to_number

# extract_boxed
assert extract_boxed(r"so \boxed{18}.") == "18"
assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"                  # 取最后一个
assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"           # 嵌套花括号
assert extract_boxed("no answer 42") is None
assert extract_boxed(r"\boxed{12") is None                                # 截断

# all_boxed
assert all_boxed(r"\boxed{1} and \boxed{\frac{2}{3}} end") == ["1", r"\frac{2}{3}"]
assert all_boxed("none") == []
assert all_boxed(r"\boxed{1} \boxed{2") == ["1", None]

# to_number
cases = {
    "18": 18, "-3": -3, "1,000": 1000, "12,345,678": 12345678, r"\$18": 18, "$18.50": 18.5,
    r"5\%": 5, r"18 \text{ dollars}": 18, r"\text{18}": 18, r"\frac{3}{4}": 0.75,
    r"\dfrac{10}{4}": 2.5, "x = 7": 7, r"2 \times 3 = 6": 6, "3.0": 3, "abc": None,
    r"1{,}000": 1000,
}
for s, want in cases.items():
    got = to_number(s)
    assert got == want, (s, got, want)

# gsm8k_gold
assert gsm8k_gold("blah\n#### 1,234") == "1234"
assert gsm8k_gold("#### -5") == "-5"

# judge
r = judge(r"... the answer is \boxed{72}.", "72")
assert r["strict_correct"] and r["lenient_correct"]
r = judge("... so she has 72 clips.", "72")
assert not r["strict_correct"] and r["lenient_correct"]
r = judge(r"\boxed{70} ... wait 72", "72")
assert not r["strict_correct"] and not r["lenient_correct"]              # 有 boxed 时宽松也只认 boxed
r = judge(r"\boxed{72.0}", "72")
assert r["strict_correct"]

# math_judge（MATH，math-verify 判等价）
r = math_judge(r"... so the answer is \boxed{\dfrac{1}{2}}.", r"\frac{1}{2}")
assert r["strict_correct"] and r["lenient_correct"]
assert math_judge(r"\boxed{(3, \frac{\pi}{2})}", r"\left( 3, \frac{\pi}{2} \right)")["strict_correct"]
assert math_judge(r"\boxed{10\sqrt2}", r"10\sqrt{2}")["strict_correct"]
assert math_judge(r"\boxed{C}", r"\text{(C)}")["strict_correct"]
assert math_judge(r"\boxed{[-2, 7]}", "[-2,7]")["strict_correct"]
assert not math_judge(r"\boxed{3}", "2")["lenient_correct"]
r = math_judge("Therefore the answer is $5$.", "5")
assert not r["strict_correct"] and r["lenient_correct"]                   # 无 boxed：宽松取最后一个表达式
r = math_judge(r"\boxed{4} ... actually 5", "5")
assert not r["strict_correct"] and not r["lenient_correct"]              # 有 boxed 时宽松也只认 boxed
assert not math_judge(r"\boxed{12", "12")["strict_correct"]               # 截断的 boxed
print("test_answer: all passed")
