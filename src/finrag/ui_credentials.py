"""仅供运行时调用的密钥检查/保存；不返回密钥，不记录输入或异常对象。"""
import re
from dotenv import dotenv_values
from filelock import FileLock
from .paths import ROOT


def key_available(root=ROOT):
    path = root / ".secrets/deepseek.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=10):
        if not path.exists():
            path.write_text("DEEPSEEK_API_KEY=\n", encoding="utf-8")
        return bool((dotenv_values(path).get("DEEPSEEK_API_KEY") or "").strip())


def save_key(value, root=ROOT):
    # 单行白名单防止换行注入dotenv；不在错误中包含输入。
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{8,256}", value.strip()):
        raise ValueError("密钥格式不正确，请检查是否完整，且没有空格或换行。")
    path = root / ".secrets/deepseek.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=10):
        path.write_text("DEEPSEEK_API_KEY=" + value.strip() + "\n", encoding="utf-8")
