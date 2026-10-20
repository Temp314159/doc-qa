# 本地持久化辅助：JSON存整对象，JSONL每行一条；这里只序列化，不校验业务事实。
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ensure_ascii=False让中文可读；default=str用于Decimal等对象，类型校验仍由schema负责。
def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in items), encoding="utf-8")


# 文件哈希验证同一份字节，不验证来源可信或金额真实。
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 相同输入组合得到相同20位标识，方便关联父表/行/证据；ID不是内容可信度分数。
def stable_id(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]
