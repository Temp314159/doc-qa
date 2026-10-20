"""仅管理当前项目、身份可核验的Streamlit子进程。"""
import argparse
import ctypes
import importlib.util
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
import uuid
import webbrowser
from pathlib import Path
import psutil
from filelock import FileLock
from .paths import ROOT

STATE = ROOT / "runs/ui"
PYTHON = ROOT / ".venv/Scripts/python.exe"
ENTRY = ROOT / "streamlit_app.py"


def same_path(a, b):
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def command_for(port, instance):
    return [str(PYTHON), "-m", "streamlit", "run", str(ENTRY), "--server.address", "127.0.0.1",
            "--server.port", str(port), "--server.headless", "true", "--browser.gatherUsageStats", "false",
            "--", "--workbench-instance", instance]


def owned_process(record):
    """创建时间、解释器、入口、实例token全部匹配。"""
    try:
        if not same_path(record["project"], ROOT) or not same_path(record["entry"], ENTRY):
            return None
        p = psutil.Process(record["pid"])
        if abs(p.create_time() - record["created_at"]) > 0.001:
            return None
        cmd, expected = p.cmdline(), command_for(record["port"], record["instance"])
        if len(cmd) != len(expected) or not same_path(cmd[0], expected[0]) or cmd[1:] != expected[1:]:
            return None
        if not same_path(p.exe(), record["executable"]):
            return None
        return p
    except (psutil.Error, OSError, KeyError, TypeError, ValueError):
        return None


def health(port):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as r:
            return r.status == 200 and r.read(32).strip() == b"ok"
    except (OSError, ValueError):
        return False


def port_free(port):
    with socket.socket() as s:
        if os.name == "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class ChildJob:
    """Windows内核Job：控制台关闭或父进程强退时，内核清理子进程。"""
    def __init__(self, pid):
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("per_process", ctypes.c_int64), ("per_job", ctypes.c_int64), ("flags", w.DWORD),
                        ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t), ("active", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        k.CreateJobObjectW.restype = w.HANDLE
        k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        k.OpenProcess.restype = w.HANDLE
        k.CloseHandle.argtypes = [w.HANDLE]
        self.kernel = k
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("无法创建窗口退出清理对象")
        limits = Extended()
        limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE
        process = k.OpenProcess(0x0100 | 0x0001, False, pid)
        try:
            if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise OSError("无法配置窗口退出清理")
            if not process or not k.AssignProcessToJobObject(self.handle, process):
                raise OSError("无法登记窗口退出清理")
        except Exception:
            self.close()
            raise
        finally:
            if process:
                k.CloseHandle(process)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def read_record(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def stop_record(path):
    record = read_record(path)
    process = owned_process(record)
    if process:
        try:
            process.terminate()
            process.wait(timeout=8)
        except psutil.TimeoutExpired:
            if owned_process(record):
                process.kill()
                process.wait(timeout=3)
        except psutil.NoSuchProcess:
            pass
    path.unlink(missing_ok=True)
    return bool(process)


def open_page(port, enabled):
    if enabled:
        try:
            if webbrowser.open(f"http://127.0.0.1:{port}"):
                print("页面已打开。使用结束后，双击‘关闭工作台’，或在这个窗口按 Ctrl+C。只关闭浏览器，后台服务还会继续运行。", flush=True)
                return
        except OSError:
            pass
        print("浏览器未能自动打开，请手动访问：", flush=True)
    print(f"工作台地址：http://127.0.0.1:{port}；使用结束请运行关闭工作台。", flush=True)


def launch(port=8503, open_browser=True, timeout=35):
    if not PYTHON.is_file() or not ENTRY.is_file() or importlib.util.find_spec("streamlit") is None:
        print("启动条件缺失：请确认项目.venv、Streamlit和页面文件完整；启动器不会自动安装。", flush=True)
        return 1
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / f"server_{port}.json"
    child = job = None
    instance = uuid.uuid4().hex
    try:
        with FileLock(str(STATE / "lifecycle.lock"), timeout=40):
            if path.exists():
                record = read_record(path)
                if owned_process(record):
                    if health(port):
                        print("工作台已在运行，继续打开已有页面。", flush=True)
                        open_page(port, open_browser)
                        return 0
                    print("已有工作台尚未就绪；请先运行关闭工作台，再重新启动。", flush=True)
                    return 1
                path.unlink(missing_ok=True)
            if not port_free(port):
                print(f"{port}已被其他程序使用，未结束该程序。请用 --port 8504 换一个入口。", flush=True)
                return 1
            logpath = STATE / f"server_{port}_{instance}.log"
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            with logpath.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command_for(port, instance), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            job = ChildJob(child.pid)
            process = psutil.Process(child.pid)
            record = {"pid": child.pid, "created_at": process.create_time(), "port": port, "project": str(ROOT),
                      "entry": str(ENTRY), "instance": instance, "executable": process.exe(), "launcher_pid": os.getpid()}
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            deadline = time.monotonic() + timeout
            while not health(port):
                if child.poll() is not None:
                    print(f"页面服务提前退出，日志：{logpath.relative_to(ROOT)}", flush=True)
                    return 1
                if time.monotonic() >= deadline:
                    print(f"等待页面就绪超时，已清理本次服务。日志：{logpath.relative_to(ROOT)}", flush=True)
                    return 1
                time.sleep(0.25)
            if not owned_process(record):
                raise RuntimeError("新服务进程归属校验失败")
            open_page(port, open_browser)
        while child.poll() is None:
            time.sleep(0.3)
        return 0
    except KeyboardInterrupt:
        print("正在关闭本次工作台……", flush=True)
        return 0
    except Exception as exc:
        print(f"启动未完成（{type(exc).__name__}）。请查看runs/ui日志或使用关闭入口。", flush=True)
        return 1
    finally:
        if job:
            job.close()
        if child:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=8)
            with FileLock(str(STATE / "lifecycle.lock"), timeout=10):
                if read_record(path).get("instance") == instance:
                    path.unlink(missing_ok=True)


def stop(port=None):
    STATE.mkdir(parents=True, exist_ok=True)
    with FileLock(str(STATE / "lifecycle.lock"), timeout=40):
        paths = [STATE / f"server_{port}.json"] if port else list(STATE.glob("server_*.json"))
        count = sum(stop_record(p) for p in paths if p.is_file())
    print("工作台已关闭。" if count else "工作台已关闭；未发现需要停止的自有服务。", flush=True)
    return 0


def main(stopping=False):
    if os.name == "nt":
        # 某些父控制台/进程组会继承忽略Ctrl+C标志，启动器必须显式恢复。
        ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    parser = argparse.ArgumentParser(description="报告与文档工作台启停")
    parser.add_argument("--port", type=int, default=None if stopping else 8503)
    if not stopping:
        parser.add_argument("--no-browser", action="store_true", help="仅启动服务，供本地验收")
    args = parser.parse_args()
    if args.port is not None and not 1024 <= args.port <= 65535:
        parser.error("端口须为1024至65535")
    return stop(args.port) if stopping else launch(args.port, not args.no_browser)
