"""工作台只读适配与显式提交边界；导入本模块不加载模型或密钥。"""
import datetime as dt
import hashlib
import json
import re
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

from filelock import FileLock
from pydantic import ValidationError

from .paths import ROOT
from .schema import RunResult, Evidence

SOURCE_LABELS = {"original": "原始财报", "reconstructed": "按真实数据重排的材料", "synthetic": "合成测试材料"}
FEEDBACK = ("helpful", "recheck", "unanswered")
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,160}\Z")


def load_json(path):
    if path.stat().st_size > 8_000_000:
        raise ValueError("记录过大")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_asset(value, kind, root=ROOT):
    """只接受登记的资料目录，resolve同时阻止..和符号链接越界。"""
    if kind not in {"image", "pdf"} or not isinstance(value, (str, Path)) or not str(value):
        return None
    try:
        path = (root / value).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    folders = [root / "data/derived/figures"] if kind == "image" else [root / "data/raw", root / "data/derived"]
    suffixes = {".png", ".jpg", ".jpeg"} if kind == "image" else {".pdf"}
    if path.suffix.lower() not in suffixes or not any(path.is_relative_to(p.resolve()) for p in folders):
        return None
    return path if path.is_file() else None


def ledger_events(root=ROOT):
    path = root / "runs/cost_ledger.jsonl"
    if not path.exists():
        return []
    events = []
    # 不初始化Ledger，不产生锁或费用事件；尾部半行可安全忽略。
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            if isinstance(e, dict):
                events.append(e)
        except (ValueError, TypeError):
            continue
    return events


def cost_summary(run_id=None, root=ROOT):
    latest = {}
    for e in ledger_events(root):
        if run_id is None or e.get("run_id") == run_id:
            latest[e.get("request_id")] = e
    total = Decimal(0)
    pending = 0
    for e in latest.values():
        try:
            amount = Decimal(e["cost_cny"])
            if amount.is_finite() and amount >= 0:
                total += amount
        except (KeyError, InvalidOperation, TypeError):
            continue
        pending += e.get("event") != "settle" or e.get("state") == "usage_missing_reserved"
    return {"amount": str(total), "pending": pending, "requests": len(latest)}


def validate_history(payload, events):
    """真实标识要求schema、显式模式、响应计数与同run账本来源互相吻合。"""
    try:
        if not isinstance(payload, dict):
            raise ValueError("记录须为对象")
        RunResult.model_validate({k: v for k, v in payload.items() if k in RunResult.model_fields}, strict=True)
        for evidence in payload.get("evidence", []):
            Evidence.model_validate({**evidence, "provenance": evidence.get("provenance") or []})
        if not SAFE_ID.fullmatch(payload["run_id"]):
            raise ValueError("记录编号无效")
    except (ValidationError, ValueError, TypeError, KeyError):
        return "invalid", None
    if payload.get("mode") == "fake_model_behavior":
        return "fake", None
    if payload.get("mode") != "real_api":
        return "unknown", None
    rid = payload["run_id"]
    reserved = {e.get("request_id") for e in events if e.get("run_id") == rid and e.get("mode") == "real_api" and e.get("event") == "reserve"}
    settled = {e.get("request_id"): e for e in events if e.get("run_id") == rid and e.get("mode") == "real_api" and e.get("event") == "settle" and e.get("request_id") in reserved and e.get("model")}
    responses = payload.get("api_responses")
    if type(responses) is not int or responses <= 0 or responses != len(settled) or payload["api_calls"] != len(reserved):
        return "unknown", None
    # 时间只来自账本中的显式ISO日期，不使用文件mtime或run_id猜测。
    times = []
    for e in settled.values():
        try:
            value = dt.datetime.fromisoformat(e["time"])
            if value.tzinfo:
                times.append(value)
        except (ValueError, KeyError, TypeError):
            pass
    return "real", min(times).isoformat() if times else None


def histories(root=ROOT):
    events = ledger_events(root)
    entries = []
    for path in sorted((root / "runs/answers").glob("*.json")):
        if not path.resolve().is_relative_to((root / "runs/answers").resolve()):
            continue
        try:
            payload = load_json(path)
            category, recorded_at = validate_history(payload, events)
            if category == "invalid":
                continue
            if payload["run_id"] != path.stem:
                category = "unknown"
            entries.append({"result": payload, "category": category, "recorded_at": recorded_at})
        except (ValueError, TypeError, OSError, KeyError):
            continue
    return entries


def filter_history(entries, query="", status="全部", category="real", since=None):
    return [e for e in entries if e["category"] == category
            and query.casefold() in e["result"].get("question", "").casefold()
            and (status == "全部" or e["result"]["status"] == status)
            and (since is None or (e["recorded_at"] and dt.datetime.fromisoformat(e["recorded_at"]).date() >= since))]


class Catalog:
    """索引元数据，不创建Retriever；金额格通过evidence_ids查行。"""
    def __init__(self, root=ROOT):
        self.root = root
        folder = root / "data/indexes/mixed"
        self.views = load_json(folder / "views.json")
        self.evidence = load_json(folder / "evidence.json")
        self.figures = load_json(folder / "figures.json")
        self.docs = {}
        for path in (root / "data/parsed").glob("*_mixed.json"):
            d = load_json(path)
            self.docs[d["doc_id"]] = {k: d.get(k) for k in ("doc_id", "company", "year", "source_type", "original_path", "working_path")}
        self.rows = {}
        for view in self.views:
            if view["kind"] == "row":
                for eid in view["evidence_ids"]:
                    self.rows.setdefault(eid, view)

    def source(self, evidence):
        doc = self.docs.get(evidence.get("doc_id"), {})
        provenance = evidence.get("provenance") or []
        return {"evidence": evidence, "document": doc,
                "name": " · ".join(str(doc[k]) for k in ("company", "year") if doc.get(k)) or "文档名称未记录",
                "source_type": SOURCE_LABELS.get(doc.get("source_type"), "来源类型未记录"),
                "pages": sorted({p["original_page"] for p in provenance if isinstance(p.get("original_page"), int)}),
                "row": self.rows.get(evidence.get("evidence_id"))}

    def cited(self, result):
        # 使用该次快照，不拿当前索引补造缺失的历史证据。
        by_id = {e.get("evidence_id"): e for e in result.get("evidence", []) if isinstance(e, dict)}
        return [self.source(by_id[eid]) if eid in by_id else {"missing": eid} for eid in result.get("citations", [])]

    def visual_observations(self, result):
        found = []
        for event in result.get("trace", []):
            ob = event.get("observation") or {}
            if not isinstance(ob, dict):
                continue
            fid = ob.get("figure_id")
            if (event.get("state") == "OBSERVE" and event.get("tool") == "inspect_figure"
                    and not event.get("error") and ob.get("mode") == "real_api"
                    and isinstance(ob.get("observation"), str) and ob["observation"].strip()
                    and ob.get("response_model") and isinstance(ob.get("usage"), dict)
                    and fid in self.figures and result.get("mode") == "real_api" and result.get("api_responses", 0) > 0):
                found.append({"figure": self.figures[fid], "observation": ob["observation"], "details": ob})
        return found

    def page_preview(self, evidence, page):
        doc = self.docs.get(evidence.get("doc_id"), {})
        pdf = safe_asset(doc.get("original_path"), "pdf", self.root)
        if pdf is None or type(page) is not int or page < 1:
            return None
        import fitz
        stamp = f"{pdf}:{pdf.stat().st_mtime_ns}:{page}"
        cache = self.root / ".cache/ui_pages" / (hashlib.sha256(stamp.encode()).hexdigest() + ".png")
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not cache.exists():
                with fitz.open(pdf) as document:
                    if page > len(document):
                        return None
                    document[page - 1].get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(cache)
            return cache
        except (OSError, RuntimeError, ValueError):
            return None


def calculation_rows(calculation):
    rows = []
    inputs = calculation.get("inputs") if isinstance(calculation.get("inputs"), dict) else {}
    bindings = calculation.get("bindings") if isinstance(calculation.get("bindings"), dict) else {}
    for variable, eid in bindings.items():
        e = inputs.get(variable)
        if not isinstance(e, dict) or e.get("evidence_id") != eid:
            rows.append({"变量": variable, "项目与原始金额": "绑定来源缺失", "资料编号": eid})
            continue
        # content保留原报表金额；value是核心计算器规范到基础货币后的值，不误标为million。
        rows.append({"变量": variable, "项目与原始金额": e.get("content", "未记录"),
                     "单位": e.get("unit", "未记录"), "期间": e.get("period") or "未记录",
                     "原页": "、".join(str(p["original_page"]) for p in e.get("provenance", [])),
                     "资料编号": eid})
    return rows


def status_message(result):
    from .ui_text import STATUS, STATUS_REASONS, STOP_REASONS
    status, reason = result.get("status"), result.get("termination_reason", "")
    if status == "DONE":
        return STATUS[status][:2]
    if status == "INSUFFICIENT_EVIDENCE":
        return STATUS[status][:2]
    if reason in STATUS_REASONS:
        return STATUS_REASONS[reason]
    if status == "GUARD_STOP":
        return STATUS[status][0], STOP_REASONS.get(reason, "处理达到上限") + "。" + STATUS[status][1]
    return STATUS.get(status, STATUS["ERROR"])[:2]


def redact(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[图片传输内容已隐藏]", text)
    text = re.sub(r"(?i)(Bearer\s+|sk-)[A-Za-z0-9_.-]+", "[认证信息已隐藏]", text)
    text = re.sub(r'(?i)(api_key|authorization|DEEPSEEK_API_KEY)(\\?"?\s*[:=]\s*\\?"?)[^"\s,}]+', r'\1: [已隐藏]', text)
    return text


def save_feedback(run_id, kind, note="", source="history", root=ROOT):
    if not SAFE_ID.fullmatch(run_id) or kind not in FEEDBACK or source not in {"history", "live"}:
        raise ValueError("反馈参数无效")
    if not isinstance(note, str) or len(note) > 500:
        raise ValueError("备注最多500字")
    if not (root / "runs/answers" / (run_id + ".json")).is_file():
        raise ValueError("回答记录不存在")
    note = note.strip()
    identity = hashlib.sha256(json.dumps([run_id, kind, note, source], ensure_ascii=False).encode()).hexdigest()
    path = root / "runs/ui_feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        old = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        for line in old:
            try:
                if json.loads(line).get("feedback_id") == identity:
                    return False
            except ValueError:
                continue
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"feedback_id": identity, "run_id": run_id, "type": kind, "note": note,
                                "time": dt.datetime.now(dt.timezone.utc).isoformat(), "source": source}, ensure_ascii=False) + "\n")
    return True


def search_local(question, retriever):
    if not question.strip() or len(question) > 1024:
        raise ValueError("请输入1至1024字的问题")
    return retriever.search(question.strip(), top_k=5)


def ask_live(question, retriever, run_id=None):
    if not question.strip() or len(question) > 1024:
        raise ValueError("请输入1至1024字的问题")
    from .agent import Agent
    return Agent(retriever).run(question.strip(), run_id=run_id or "ui_" + uuid.uuid4().hex)
