# 预算层：调用前保守预留，响应后按usage结算；失败无usage时不擅自释放。
# 账本是项目的预算占用记录，不是平台账单，网络失败也可能显示预留金额。
"""预留先于请求；不完整usage保留预留费用，永不当零。"""
import datetime as dt
import json
import uuid
from decimal import Decimal
from filelock import FileLock
from .paths import ROOT
from .store import read_json


class BudgetError(RuntimeError):
    pass


class Ledger:
    # 允许测试传隔离路径；正常运行使用项目JSONL账本与普通价格配置。
    # 文件锁把“检查剩余额度→写预留”串起来，避免两个前端同时花同一余额。
    def __init__(self, path=None, config=None):
        self.path = path or ROOT / "runs/cost_ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or read_json(ROOT / "configs/runtime.json")
        self.lock = FileLock(str(self.path) + ".lock", timeout=5)

    def _events(self):
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x]

    @staticmethod
    # 每个request_id只取最新事件金额：reserve后settle替换金额，不能把两笔相加。
    # 未结算请求仍占其预留；run_id过滤得到本轮占用，默认得到全项目占用。
    def _totals(events, run_id=None):
        requests = {}
        for e in events:
            if run_id is None or e["run_id"] == run_id:
                requests[e["request_id"]] = Decimal(e["cost_cny"])
        return sum(requests.values(), Decimal(0))

    def _append(self, event):
        event["time"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def total(self, run_id=None):
        with self.lock:
            return self._totals(self._events(), run_id)

    # 用输入/输出上界和峰值价格算本次最坏费用，先检查总预算，再检查本轮请求次数。
    # 计数按reserve事件，所以失败与重试都占请求额度；不是只数成功模型回答。
    def reserve(self, run_id, input_upper, output_upper, mode="real_api"):
        c = self.config
        cost = (Decimal(input_upper) * Decimal(c["input_miss_per_million"]) +
                Decimal(output_upper) * Decimal(c["output_per_million"])) / Decimal(1000000)
        if input_upper <= 0 or output_upper <= 0:
            raise BudgetError("未知token上界，禁止请求")
        with self.lock:
            events = self._events()
            # 锁内读总额、比较本次上界、追加reserve，避免并发各自看见同一余额。
            if self._totals(events) + cost > Decimal(c["budget_cny"]):
                raise BudgetError("预请求预算不足")
            if sum(e["event"] == "reserve" and e["run_id"] == run_id for e in events) >= c["max_api_calls"]:
                raise BudgetError("单轮API请求数达到上限")
            request_id = uuid.uuid4().hex
            self._append({"event": "reserve", "request_id": request_id, "run_id": run_id,
                "mode": mode, "model": c["model"], "input_upper": input_upper,
                "output_upper": output_upper, "cost_cny": str(cost), "price_date": c["price_date"]})
        return request_id, cost

    # 有usage才按命中/未命中输入与输出分别估价；缺usage继续用reserved。
    # 请求失败但无usage时，账本金额仍是预算预留，不代表平台实扣。
    def settle(self, request_id, run_id, usage, response_model, reserved):
        if not usage or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
            cost = reserved
            # 无usage不表示请求确定免费；保留占用是预算规则，不能说成平台已扣费。
            state = "usage_missing_reserved"
        else:
            c = self.config
            hit = usage.get("prompt_cache_hit_tokens", 0) or 0
            miss = usage["prompt_tokens"] - hit
            cost = (Decimal(miss) * Decimal(c["input_miss_per_million"]) +
                    Decimal(hit) * Decimal(c["input_hit_per_million"]) +
                    Decimal(usage["completion_tokens"]) * Decimal(c["output_per_million"])) / Decimal(1000000)
            state = "settled_peak_price_upper"
        with self.lock:
            self._append({"event": "settle", "request_id": request_id, "run_id": run_id,
                "mode": "real_api", "model": response_model, "usage": usage,
                "cost_cny": str(cost), "state": state, "price_date": self.config["price_date"]})
        return cost
