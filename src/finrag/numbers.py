# 数字处理：保留原文另存规范值，用Decimal按明确倍率转换。
# 数值词项与单元格共用规范化规则；这里不纠正OCR、不猜财务口径、不换汇。
import re
import unicodedata
from decimal import Decimal, InvalidOperation

# 长单位写在短单位前：百万元/千元/万元要先于“元”匹配，避免提前取倍率1。
SCALES = {"亿元": Decimal("1e8"), "百万元": Decimal("1e6"), "千元": Decimal("1e3"),
          "万元": Decimal("1e4"), "元": Decimal(1), "billion": Decimal("1e9"),
          "million": Decimal("1e6"), "thousand": Decimal("1e3")}


# 返回（基础维度,倍率），如USD million→(USD,1e6)，percent→(ratio,0.01)。
# 维度供计算检查，倍率供原数字规范化；已规范化的value不能再乘一遍。
def unit_info(unit):
    u = unicodedata.normalize("NFKC", unit).lower()
    if "%" in u or "percent" in u or "百分" in u:
        return "ratio", Decimal("0.01")
    currency = "RMB" if any(x in u for x in ("rmb", "cny", "人民币", "元")) else "USD" if any(x in u for x in ("usd", "$", "dollar", "美元")) else "unknown"
    # 美元的“元”不能判为人民币。
    if "美元" in u:
        currency = "USD"
    for label, scale in SCALES.items():
        if label in u:
            return currency, scale
    if u in ("ratio", "count", "dimensionless"):
        return "dimensionless", Decimal(1)
    return currency, Decimal(1)


# 输入原数字文字与上下文单位，输出Decimal字符串或None。
# 先做NFKC/千分位/括号负号，再检查纯数字语法；合法类型不保证原图读对。
def parse_number(text, unit="unknown"):
    raw = unicodedata.normalize("NFKC", str(text)).strip().replace("−", "-").replace("，", ",")
    raw = raw.replace("$", "").replace("¥", "").strip()
    if re.fullmatch(r"\(\s*[\d,.]+\s*\)", raw):
        raw = "-" + raw[1:-1].strip()
    # 格子自带%优先于外部金额单位，25%规范成0.25。
    # 这只是明确百分号处理，未实现任意混合单位的财务语义解析。
    own_unit = "%" if raw.endswith("%") else unit
    raw = raw.rstrip("%").strip().replace(",", "")
    # G00、夹字或日期不会被猜成数字；返回None让下游显式处理缺值。
    # 数字字符串解析成功仍可能是合法的错数，如600识别成601。
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw):
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    _, scale = unit_info(own_unit)
    return str(value * scale)


# 扫描文字里的数字/紧邻单位生成可匹配词项；独立年份生成year_2024。
# 100万元和1000000元可共享num_1E+6_RMB，但其普通词项仍不同。
# 远处表头的单位不自动传入这里，无显式单位的数字按dimensionless处理。
def numeric_tokens(text):
    result = []
    for m in re.finditer(r"(?:\([\d,]+(?:\.\d+)?\)|[+-]?\d[\d,]*(?:\.\d+)?)(?:%|亿元|百万元|万元|千元|元)?", unicodedata.normalize("NFKC", text)):
        raw = m.group()
        if re.fullmatch(r"(?:19|20)\d{2}", raw):
            result.append("year_" + raw)
            continue
        unit = next((u for u in ("%", "亿元", "百万元", "万元", "千元", "元") if raw.endswith(u)), "dimensionless")
        stripped = raw[:-len(unit)] if unit != "dimensionless" else raw
        value = parse_number(stripped, unit)
        if value is not None:
            dim, _ = unit_info(unit)
            result.append("num_" + str(Decimal(value).normalize()) + "_" + dim)
    return result
