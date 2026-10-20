# 项目路径与当前进程环境：导入时把模型/库缓存固定到项目内，不写系统持久变量。
import os
from pathlib import Path

# 以源码所在位置求根目录，而非依赖启动PowerShell的当前目录。
ROOT = Path(__file__).resolve().parents[2]


def configure_local():
    """只改变当前进程和子进程；所有新增缓存留在项目。"""
    paths = {
        "HF_HOME": ROOT / ".cache/huggingface",
        "HF_HUB_CACHE": ROOT / ".cache/huggingface/hub",
        "TORCH_HOME": ROOT / ".cache/torch",
        "XDG_CACHE_HOME": ROOT / ".cache",
        "PIP_CACHE_DIR": ROOT / ".cache/pip",
        "UV_CACHE_DIR": ROOT / ".cache/uv",
        "UV_PYTHON_INSTALL_DIR": ROOT / ".runtime/python",
        "EASYOCR_MODULE_PATH": ROOT / "models/easyocr",
        "TEMP": ROOT / ".cache/tmp", "TMP": ROOT / ".cache/tmp",
        "MPLCONFIGDIR": ROOT / ".cache/matplotlib",
    }
    # 先创建目录再设置进程os.environ；子进程继承，其他独立终端不受影响。
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return ROOT


configure_local()
