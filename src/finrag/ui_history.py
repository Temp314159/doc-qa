"""仅本机的提问历史；提交先落盘，再进行任何检索或问答。"""
import datetime as dt
import json
import os
import time
import uuid
from filelock import FileLock
from .paths import ROOT
from .ui_service import redact


class HistoryStore:
    """项目内JSON历史的读写入口；GUI和测试共用同一格式与文件锁。"""

    def __init__(self, root=ROOT):
        self.path = root / ".local/history.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=10)
        # 首次使用只创建空容器，不从样例或答案目录推断用户的个人历史。
        with self.lock:
            if not self.path.exists():
                self._write([])  # 新安装始终空历史，不自动导入仓库中的答案。

    def _read(self):
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("entries"), list):
            raise ValueError("历史记录格式不支持")
        return payload["entries"]

    def _write(self, entries):
        # 临时文件与目标文件位于同一目录，os.replace可避免崩溃留下半份JSON。
        temp = self.path.with_name("history." + uuid.uuid4().hex + ".tmp")
        try:
            temp.write_text(json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)

    def list(self, query=""):
        """按提问时间倒序读取，可选按问题文字做不区分大小写的包含匹配。"""
        with self.lock:
            entries = self._read()
        return sorted((e for e in entries if query.casefold() in e["question"].casefold()),
                      key=lambda e: e.get("asked_at") or "", reverse=True)

    def get(self, entry_id):
        return next((e for e in self.list() if e["id"] == entry_id), None)

    def begin(self, question, kind):
        """校验问题后立即落盘；条目ID同时作为界面任务和回答记录的稳定编号。"""
        question = question.strip()
        if not question or len(question) > 1024 or kind not in {"ask", "search"}:
            raise ValueError("请输入1至1024字的问题")
        entry = {"id": "ui_" + uuid.uuid4().hex, "question": question, "kind": kind,
                 "asked_at": dt.datetime.now(dt.timezone.utc).isoformat(), "state": "queued"}
        with self.lock:
            entries = self._read()
            entries.append(entry)
            self._write(entries)
        return entry

    def update(self, entry_id, **fields):
        if not set(fields) <= {"state", "result", "hits", "error_type"}:
            raise ValueError("历史字段无效")
        # 防止模型输出或反馈中的认证样式进入本地历史；此处只清理待写数据，不接触密钥文件。
        fields = json.loads(redact(fields))
        with self.lock:
            entries = self._read()
            item = next((e for e in entries if e["id"] == entry_id), None)
            if item is None:
                raise ValueError("历史记录不存在")
            item.update(fields)
            self._write(entries)

    def claim(self, entry_id):
        """一次性领取任务，刷新/展开/并发重跑不得重复请求。"""
        with self.lock:
            entries = self._read()
            item = next((e for e in entries if e["id"] == entry_id), None)
            if item is None or item["state"] != "queued":
                return False
            item["state"] = "running"
            self._write(entries)
            return True

    def import_verified(self, entries):
        """显式一次性迁移本机旧历史；正常页面不调用此方法。"""
        with self.lock:
            local = self._read()
            ids = {e["id"] for e in local}
            count = 0
            for entry in entries:
                result = entry["result"]
                if entry["category"] != "real" or result["run_id"] in ids:
                    continue
                local.append({"id": result["run_id"], "question": result["question"], "kind": "ask",
                              "asked_at": entry["recorded_at"], "state": "completed", "result": json.loads(redact(result)),
                              "time_source": "legacy_response_time"})
                ids.add(result["run_id"])
                count += 1
            self._write(local)
        return count


def search_history(store, query, session_state, cooldown_seconds=0.35, now=None):
    """在内存快照上筛选问题，短时间内不重复读取完整历史文件。"""
    moment = time.monotonic() if now is None else now
    modified = store.path.stat().st_mtime_ns
    cached_items = session_state.get("_history_search_items")
    refreshed_at = session_state.get("_history_search_refreshed_at", 0)
    cached_modified = session_state.get("_history_search_modified_at")

    # 文件改动立即刷新；输入连续更新时仅复用本机快照，至少间隔一小段时间再读盘。
    if cached_items is None or modified != cached_modified or moment - refreshed_at >= cooldown_seconds:
        cached_items = store.list()
        session_state["_history_search_items"] = cached_items
        session_state["_history_search_refreshed_at"] = moment
        session_state["_history_search_modified_at"] = modified

    # 输入过滤只在内存快照上执行；冷却期降低连续键入造成的磁盘读取次数。
    needle = query.casefold().strip()
    if not needle:
        return cached_items
    return [item for item in cached_items if needle in item["question"].casefold()]


def friendly_time(value):
    """将带时区的ISO时间按当前界面语言显示到分钟；没有可信时区时不补造时间。"""
    from . import ui_text
    if not value:
        return ui_text.TEXT["question_time_missing"]
    try:
        moment = dt.datetime.fromisoformat(value)
        if moment.tzinfo is None:
            return ui_text.TEXT["question_time_missing"]
        local = moment.astimezone(dt.timezone(dt.timedelta(hours=8)))
        pattern = "%Y-%m-%d %H:%M" if ui_text.CURRENT_LANGUAGE == "en" else "%Y年%m月%d日 %H:%M"
        return local.strftime(pattern)
    except (ValueError, TypeError):
        return ui_text.TEXT["question_time_missing"]
