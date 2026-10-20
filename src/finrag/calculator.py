# 本地数值工具：表达式文本→受限语法树→带单位的Decimal结果。
# 解释AST而非eval；表达式合法仍不替代原始数字与科目的核验。
"""Decimal AST解释器。无eval/exec；独立spawn worker，不是OS沙箱。"""
import ast
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from .numbers import unit_info


class CalculationError(ValueError):
    pass


@dataclass
# 每个中间结果同时带数值value和维度dim，例如USD或无量纲。
# 只传数字会失去“美元能否与人民币相加”的检查依据。
class Quantity:
    value: Decimal
    dim: str


def _check(value):
    if not value.is_finite() or abs(value) > Decimal("1e30"):
        raise CalculationError("数值非有限或超过1e30")
    return value


# bindings里的金额已经规范为元/美元，此处不重复乘million/千元倍率。
# 规模、节点种类、递归深度和数值范围都有界，防止很短输入触发无界计算。
def interpret(expression, bindings):
    if len(expression) > 512 or len(bindings) > 16:
        raise CalculationError("表达式/绑定规模超限")
    try:
        # mode="eval"仅表示按单个表达式解析语法，并没有执行Python eval。
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError):
        raise CalculationError("表达式语法无效") from None
    if len(list(ast.walk(tree))) > 128:
        raise CalculationError("AST节点超过128")
    # AST节点白名单先排除属性访问、下标、列表、幂等语法。
    # Call允许进入树不代表任意函数可执行；下面另查函数名与参数。
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub, ast.Call)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise CalculationError("禁止AST节点：" + type(node).__name__)

    def walk(node, depth=0):
        if depth > 16:
            raise CalculationError("AST深度超过16")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise CalculationError("仅允许十进制数字常量")
            # 从原表达式截取十进制字符串再构造Decimal，避免先把0.1转成浮点近似值。
            # ast.Constant中的0.1可能已是float，故不用Decimal(node.value)，取源字符"0.1"。
            literal = ast.get_source_segment(expression, node)
            if len(literal) > 64 or any(x in literal.lower() for x in ("e", "x", "_", "j")):
                raise CalculationError("数字常量格式/长度受限")
            return Quantity(_check(Decimal(literal)), "dimensionless")
        # 变量只从绑定取值，并要求期间/单位明确。
        # 期间非空是格式条件，不会判断new与old是否真的对应问题所要求的两年。
        if isinstance(node, ast.Name):
            if node.id not in bindings:
                raise CalculationError("变量未绑定到已检索证据")
            b = bindings[node.id]
            if b.get("value") is None or not b.get("period"):
                raise CalculationError("数值缺失或期间不明")
            dim, _ = unit_info(b["unit"])
            if dim == "unknown":
                raise CalculationError("单位不明")
            return Quantity(_check(Decimal(b["value"])), "dimensionless" if dim == "ratio" else dim)
        if isinstance(node, ast.UnaryOp):
            q = walk(node.operand, depth + 1)
            return Quantity(_check(-q.value if isinstance(node.op, ast.USub) else q.value), q.dim)
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left, depth + 1), walk(node.right, depth + 1)
            # 加减要求同维度；乘法至少一侧无量纲；同维度相除得到无量纲。
            # 例如(new-old)/old*100可执行，但USD+RMB或金额乘金额被拒。
            # 同维度并不保证财务口径相同，合并/母公司金额还须上游选择正确。
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if a.dim != b.dim:
                    raise CalculationError("加减单位不兼容；财务常量须证据绑定")
                value = a.value + b.value if isinstance(node.op, ast.Add) else a.value - b.value
                dim = a.dim
            elif isinstance(node.op, ast.Mult):
                if a.dim != "dimensionless" and b.dim != "dimensionless":
                    raise CalculationError("不支持金额乘金额")
                # new*1000000语法与维度都合法，但可能重复倍率；量纲规则不检查财务意图。
                value, dim = a.value * b.value, b.dim if a.dim == "dimensionless" else a.dim
            else:
                if b.value == 0:
                    raise CalculationError("除零")
                if b.dim not in ("dimensionless", a.dim):
                    raise CalculationError("除法单位不兼容")
                value = a.value / b.value
                dim = "dimensionless" if a.dim == b.dim else a.dim
            return Quantity(_check(value), dim)
        # 函数仅abs/min/max/round；拒绝关键字参数和任意属性调用。
        # round使用Decimal量化与四舍五入，不依赖Python浮点round的二进制结果。
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ("abs", "min", "max", "round") or node.keywords:
                raise CalculationError("函数不在白名单或含关键字参数")
            if not 1 <= len(node.args) <= 16:
                raise CalculationError("函数参数数目超限")
            args = [walk(a, depth + 1) for a in node.args]
            name = node.func.id
            if name == "abs":
                if len(args) != 1:
                    raise CalculationError("abs须一个参数")
                return Quantity(abs(args[0].value), args[0].dim)
            if name == "round":
                if len(args) != 2 or args[1].dim != "dimensionless" or args[1].value != args[1].value.to_integral_value() or not 0 <= args[1].value <= 10:
                    raise CalculationError("round须0至10整数小数位")
                return Quantity(args[0].value.quantize(Decimal(1).scaleb(-int(args[1].value)), rounding=ROUND_HALF_UP), args[0].dim)
            if len({a.dim for a in args}) != 1:
                raise CalculationError("min/max单位不兼容")
            return Quantity((min if name == "min" else max)(a.value for a in args), args[0].dim)
        raise CalculationError("未知节点")

    # 局部Decimal精度50只影响当前解释过程，不改整个程序的全局数值设置。
    # 50位是有效数字精度，不是固定50位小数；返回rounding字段不代表所有运算强制四舍五入。
    # 只有上面的显式round分支指定ROUND_HALF_UP，普通除法服从Decimal当前上下文。
    with localcontext() as ctx:
        ctx.prec = 50
        try:
            result = walk(tree)
        except (InvalidOperation, OverflowError):
            raise CalculationError("Decimal数值运算失败") from None
    return {"value": str(result.value), "unit": result.dim, "precision": 50, "rounding": "ROUND_HALF_UP"}


def _worker(expression, bindings, output):
    try:
        output.put({"ok": True, **interpret(expression, bindings)})
    except Exception as e:
        output.put({"ok": False, "error": str(e) if isinstance(e, CalculationError) else type(e).__name__})


# 独立spawn进程执行，超时terminate，清理时必要则kill。
# 这提供可终止的资源边界；没有配置操作系统访问隔离，不能称为OS沙箱。
# 启动worker的时间也从timeout扣除；Windows调用者需有主入口保护。
def calculate(expression, bindings, timeout=2):
    ctx = mp.get_context("spawn")
    output = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker, args=(expression, bindings, output), daemon=True)
    start = time.monotonic()
    proc.start()
    try:
        proc.join(max(0, timeout - (time.monotonic() - start)))
        if proc.is_alive():
            proc.terminate()
            proc.join(1)
            return {"ok": False, "error": "计算worker超时"}
        try:
            return output.get(timeout=0.2)
        except queue.Empty:
            return {"ok": False, "error": "计算worker未返回结果"}
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()
        output.close()
        output.join_thread()
        proc.close()
