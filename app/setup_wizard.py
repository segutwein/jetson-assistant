"""First-time setup wizard — build llama.cpp and download a model."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

LLAMA_DIR = Path.home() / "llama.cpp"
MODELS_DIR = Path.home() / "models"

RECOMMENDED_MODELS = [
    {
        "name": "Gemma 4 E4B Q4_K_M",
        "repo": "unsloth/gemma-4-E4B-it-GGUF",
        "filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        "size": "~4.6 GB",
        "description": "Recommended — Google Gemma 4, modern and fast",
        "license_url": None,
    },
]


# ── Prerequisite checks ────────────────────────────────────────────

def check_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def check_cuda() -> Optional[str]:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    # JetPack may install nvcc in versioned or unversioned paths
    candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def check_cuda_libs() -> Optional[str]:
    """Check for CUDA libraries even if nvcc is missing (runtime-only install)."""
    candidates = [
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists() or list(p.parent.glob("libcudart.so*") if p.parent.exists() else []):
            return str(p.parent)
    return None


# required: build fails without these
# optional: warning only, build may still succeed
_REQUIRED = ["git", "cmake", "make"]
_OPTIONAL = ["nvcc"]  # cmake can sometimes find CUDA without nvcc in PATH


def check_prerequisites() -> dict:
    results = {
        "git":   (check_tool("git"),  True),
        "cmake": (check_tool("cmake"), True),
        "make":  (check_tool("make"),  True),
        "nvcc":  (check_cuda(),        False),   # optional
        "pip3":  (check_tool("pip3") or check_tool("pip"), True),
    }
    return results


# ── llama.cpp ──────────────────────────────────────────────────────

def llama_server_path() -> Optional[Path]:
    p = LLAMA_DIR / "build/bin/llama-server"
    return p if p.exists() else None


def clone_llama_cpp() -> bool:
    if LLAMA_DIR.exists():
        return True
    rc = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ggml-org/llama.cpp", str(LLAMA_DIR)],
    ).returncode
    return rc == 0


def build_llama_cpp() -> bool:
    env_path = "/usr/local/cuda/bin"
    import os
    env = os.environ.copy()
    env["PATH"] = env_path + ":" + env.get("PATH", "")
    env["CUDA_HOME"] = "/usr/local/cuda"

    cmake_configure = subprocess.run(
        [
            "cmake", "-B", "build",
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=87",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=LLAMA_DIR,
        env=env,
    )
    if cmake_configure.returncode != 0:
        return False

    nproc = subprocess.run(
        ["nproc"], capture_output=True, text=True
    ).stdout.strip() or "4"

    cmake_build = subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "-j", nproc],
        cwd=LLAMA_DIR,
        env=env,
    )
    return cmake_build.returncode == 0


# ── Model download ─────────────────────────────────────────────────

def _ensure_huggingface_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        rc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"],
        ).returncode
        return rc == 0


def _hf_cmd() -> Optional[str]:
    """Return the HuggingFace CLI binary: `hf` (new) or `huggingface-cli` (old)."""
    return shutil.which("hf") or shutil.which("huggingface-cli")


def check_hf_login() -> bool:
    """Return True if a HuggingFace token is saved locally."""
    cmd = _hf_cmd()
    if cmd:
        name = Path(cmd).name
        check_args = [cmd, "auth", "token"] if name == "hf" else [cmd, "whoami"]
        r = subprocess.run(check_args, capture_output=True, text=True)
        return r.returncode == 0 and "Not logged in" not in r.stdout
    # Python fallback
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token() is not None
    except Exception:
        return False


def hf_login() -> bool:
    """Run HuggingFace login interactively. Return True on success."""
    cmd = _hf_cmd()
    if cmd:
        name = Path(cmd).name
        login_args = [cmd, "auth", "login"] if name == "hf" else [cmd, "login"]
        return subprocess.run(login_args).returncode == 0
    # Python fallback
    return subprocess.run(
        [sys.executable, "-c", "from huggingface_hub import login; login()"]
    ).returncode == 0


class DownloadAuthError(Exception):
    """Raised when a model download fails due to missing authentication (HTTP 401/403)."""


def download_model(repo: str, filename: str) -> Optional[Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists():
        return dest

    cmd = _hf_cmd()
    if cmd:
        name = Path(cmd).name
        if name == "hf":
            args = [cmd, "download", repo, "--include", filename, "--local-dir", str(MODELS_DIR)]
        else:
            args = [cmd, "download", repo, filename, "--local-dir", str(MODELS_DIR)]
        # Let stdout/stderr through so the hf CLI progress bar is visible.
        # Capture only stderr separately to detect auth errors on failure.
        r = subprocess.run(args, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0 and dest.exists():
            return dest
        err = r.stderr or ""
        if "401" in err or "403" in err or "not logged in" in err.lower() or "token" in err.lower():
            raise DownloadAuthError(repo)
        if err.strip():
            print(err.strip())
        return None

    # Python fallback via huggingface_hub
    if not _ensure_huggingface_hub():
        return None
    from huggingface_hub import hf_hub_download, HfFolder
    token = HfFolder.get_token()
    try:
        path = hf_hub_download(
            repo_id=repo, filename=filename,
            local_dir=str(MODELS_DIR), token=token,
        )
        return Path(path)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            raise DownloadAuthError(repo) from e
        print(f"Download failed: {e}")
        return None


# ── venv ───────────────────────────────────────────────────────────

def setup_venv(project_dir: Path) -> bool:
    venv_dir = project_dir / "venv"
    if venv_dir.exists():
        return True
    rc = subprocess.run(
        ["python3.10", "-m", "venv", str(venv_dir)]
    ).returncode
    if rc != 0:
        return False
    pip = venv_dir / "bin/pip"
    rc = subprocess.run(
        [str(pip), "install", "--upgrade", "pip", "wheel"]
    ).returncode
    if rc != 0:
        return False
    req = project_dir / "requirements.txt"
    rc = subprocess.run(
        [str(pip), "install", "-r", str(req)]
    ).returncode
    return rc == 0
