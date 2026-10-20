# 外部模型边界：只在运行时加载认证，统一经过Ledger再发文本/工具/视觉请求。
# 只在运行时构造客户端；静态检查不得实例化客户端或访问密钥文件。
import base64
import json
import time
from dotenv import dotenv_values
from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
from .paths import ROOT
from .store import read_json
from .budget import Ledger


class CapabilityError(RuntimeError):
    pass


class NetworkAccessError(CapabilityError):
    pass


class DeepSeek:
    # 普通配置与秘密分开；SDK禁止内置重试，所有尝试在chat中显式计数/预留。
    def __init__(self, run_id, config=None):
        self.config = config or read_json(ROOT / "configs/runtime.json")
        # 唯一密钥读取位置。任何调用者都不得输出此局部变量/对象。
        key = dotenv_values(ROOT / ".secrets/deepseek.env").get("DEEPSEEK_API_KEY")
        if not key:
            raise CapabilityError("项目密钥缺失")
        self.client = OpenAI(api_key=key, base_url=self.config["base_url"], max_retries=0)
        self.ledger = Ledger(config=self.config)
        self.run_id = run_id
        self.calls = 0
        self.last_usage = None
        self.responses = 0

    # 模型列表只证明接口可连通及标识可见，不能证明工具或图像能力。
    # 异常只给类型，避免输出可能含认证细节的完整请求或响应。
    def models(self):
        try:
            return [m.id for m in self.client.models.list(timeout=20).data]
        except Exception as e:
            raise CapabilityError("模型列表失败：" + type(e).__name__) from None

    # messages可能带此前工具观察；tools是可提议工具的schema。
    # 输入以UTF-8字节数加余量保守估计，图片另加上界，不把base64当普通文本token。
    def chat(self, messages, tools=None, deadline=None, max_tokens=None, images=0, tool_choice=None):
        c = self.config
        # data URL 被独立计数，不把完整base64字符串作为文字token计费。
        safe_messages = []
        for message in messages:
            copy = dict(message)
            if isinstance(copy.get("content"), list):
                copy["content"] = [p for p in copy["content"] if p.get("type") != "image_url"]
            safe_messages.append(copy)
        # 这是字节数与余量的保守上界，不是tokenizer实测。
        # 超上界检查先于reserve，决策步数与实际请求数可能不同。
        input_upper = len(json.dumps({"messages": safe_messages, "tools": tools}, ensure_ascii=False).encode("utf-8")) + 2048 + images * c["image_token_upper_bound"]
        if input_upper > 32000:
            raise CapabilityError("上下文保守上界超限，须减少证据")
        output_upper = max_tokens or c["max_output_tokens"]
        # 最多两次尝试，且同受全局deadline限制；每次单独reserve并加calls。
        # 不是一次失败后无限重试，也不能把上一笔未确认费用当作已退款。
        for attempt in range(2):
            remaining = deadline - time.monotonic() if deadline else c["network_timeout"]
            if remaining <= 0:
                raise TimeoutError("全局deadline耗尽")
            request_id, reserved = self.ledger.reserve(self.run_id, input_upper, output_upper)
            self.calls += 1
            kwargs = {"model": c["model"], "messages": messages, "max_tokens": output_upper,
                "timeout": min(c["network_timeout"], remaining),
                "extra_body": {"thinking": {"type": "disabled"}}}
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            try:
                response = self.client.chat.completions.create(**kwargs)
                # 只有收到API返回才计responses；usage缺失时账本仍按保守预留。
                self.responses += 1
                usage = response.usage.model_dump() if response.usage else None
                self.last_usage = usage
                self.ledger.settle(request_id, self.run_id, usage, response.model, reserved)
                return response
            except APIConnectionError as exc:
                cause=exc
                blocked=False
                for _ in range(8):
                    if getattr(cause,"winerror",None)==10013 or getattr(cause,"errno",None)==10013:
                        blocked=True;break
                    cause=getattr(cause,"__cause__",None) or getattr(cause,"__context__",None)
                    if cause is None:break
                # Windows 10013表示访问被权限阻止，停止整批；再次重试同权限没有意义。
                if blocked:
                    raise NetworkAccessError("沙箱网络权限阻塞，未得到API响应；停止整批") from None
                if attempt == 0:
                    continue
                raise NetworkAccessError("网络连接失败，未得到API响应；停止整批并保留预留") from None
            except APITimeoutError:
                if attempt == 0:continue
                raise TimeoutError("API请求超时；保留费用预留") from None
            except APIStatusError as exc:
                if exc.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                    continue
                raise CapabilityError("API状态 " + str(exc.status_code) + "；未输出可能含敏感信息的错误正文") from None
        raise CapabilityError("请求失败")

    # 真正发送裁剪像素的入口，不靠图标题猜图内事实；这会额外调用一次chat。
    # base64仅在请求内存中构造，日志不要保存其完整内容。
    def inspect(self, crop, question, deadline=None):
        # base64只是图片字节的传输编码；当前文本Qwen索引并未编码这些像素。
        # 裁剪既可来自PDF位图也可来自页面区域渲染，后者包含矢量图。
        data = base64.b64encode(crop.read_bytes()).decode("ascii")
        response = self.chat([{"role": "user", "content": [
            {"type": "text", "text": "只观察这张图回答，读不到就明确说明。" + question},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}]}],
            images=1, deadline=deadline)
        return {"observation": response.choices[0].message.content, "mode": "real_api",
                "response_model": response.model, "usage": self.last_usage}
