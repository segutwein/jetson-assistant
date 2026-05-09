"""Process management — llama-server lifecycle and PID tracking."""

import os
import sys
import time
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import httpx

STATE_DIR = Path.home() / ".jetson-assistant"
LLAMA_PID_FILE = STATE_DIR / "llama-server.pid"
LLAMA_LOG_FILE = STATE_DIR / "llama-server.log"

_LLAMA_SEARCH_PATHS = [
    Path.home() / "llama.cpp/build/bin/llama-server",
    Path("/usr/local/bin/llama-server"),
    Path("/usr/bin/llama-server"),
]


def find_llama_server() -> Optional[Path]:
    for p in _LLAMA_SEARCH_PATHS:
        if p.exists() and os.access(p, os.X_OK):
            return p
    found = shutil.which("llama-server")
    return Path(found) if found else None


def find_gguf_models() -> list[Path]:
    search_dirs = [
        Path.home() / "models",
        Path.home() / ".cache/huggingface",
        Path.cwd(),
    ]
    models = []
    for d in search_dirs:
        if d.exists():
            models.extend(sorted(d.rglob("*.gguf")))
    seen = set()
    return [m for m in models if not (m in seen or seen.add(m))]


def _save_pid(pid_file: Path, pid: int):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def is_llama_server_running() -> bool:
    pid = read_pid(LLAMA_PID_FILE)
    if pid and is_process_running(pid):
        return True
    # also check via HTTP in case PID file is stale
    try:
        r = httpx.get("http://127.0.0.1:8080/v1/models", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_llama_server(timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get("http://127.0.0.1:8080/v1/models", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_llama_server(model_path: Path, port: int = 8080, ctx: int = 4096) -> Optional[int]:
    llama = find_llama_server()
    if not llama:
        return None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(llama),
        "-m", str(model_path),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", "99",
        "-c", str(ctx),
    ]
    with open(LLAMA_LOG_FILE, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=log,
            start_new_session=True,
        )
    _save_pid(LLAMA_PID_FILE, proc.pid)
    return proc.pid


def stop_llama_server():
    pid = read_pid(LLAMA_PID_FILE)
    if pid and is_process_running(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not is_process_running(pid):
                break
            time.sleep(0.5)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if LLAMA_PID_FILE.exists():
        LLAMA_PID_FILE.unlink()


def get_llama_model_name() -> Optional[str]:
    try:
        r = httpx.get("http://127.0.0.1:8080/v1/models", timeout=2.0)
        models = r.json().get("data", [])
        return models[0].get("id") if models else None
    except Exception:
        return None
