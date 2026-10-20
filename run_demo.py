"""请用项目.venv解释器运行。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
try:
    from finrag.demo_process import main
except ImportError:
    print("启动依赖不完整。请按README恢复项目.venv，启动器不会自动安装。")
    raise SystemExit(1)
if __name__ == "__main__":
    raise SystemExit(main())
