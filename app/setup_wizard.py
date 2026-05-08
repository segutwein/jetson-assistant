# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        "name": "Gemma 3 4B  Q4_K_M",
        "repo": "bartowski/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
        "size": "~2.7 GB",
        "description": "Recommended — good balance of quality and speed",
    },
    {
        "name": "Qwen3 4B   Q4_K_M",
        "repo": "bartowski/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q4_K_M.gguf",
        "size": "~2.6 GB",
        "description": "Strong reasoning, good for Q&A",
    },
    {
        "name": "Gemma 3 1B  Q8_0",
        "repo": "ggml-org/gemma-3-1b-it-GGUF",
        "filename": "gemma-3-1b-it-Q8_0.gguf",
        "size": "~1.3 GB",
        "description": "Fastest, lower quality — good for testing",
    },
]


# ── Prerequisite checks ────────────────────────────────────────────

def check_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def check_cuda() -> Optional[str]:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    # JetPack puts it here
    fallback = Path("/usr/local/cuda/bin/nvcc")
    return str(fallback) if fallback.exists() else None


def check_prerequisites() -> dict:
    return {
        "git":   check_tool("git"),
        "cmake": check_tool("cmake"),
        "make":  check_tool("make"),
        "nvcc":  check_cuda(),
        "pip3":  check_tool("pip3") or check_tool("pip"),
    }


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


def download_model(repo: str, filename: str) -> Optional[Path]:
    if not _ensure_huggingface_hub():
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists():
        return dest

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(MODELS_DIR),
        )
        return Path(path)
    except Exception as e:
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
