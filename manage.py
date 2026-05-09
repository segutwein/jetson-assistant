#!/usr/bin/env python3
"""
Jetson Voice Assistant — management CLI

Commands:
  setup     First-time setup: build llama.cpp and download a model
  start     Pick a model and start the assistant
  stop      Stop llama-server and voice chat
  status    Show what is running and memory usage
  optimize  Apply memory optimizations
  test      Test individual components (--llm, --stt, --tts, --vad, --all)
"""

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

from app.manager import (
    find_llama_server, find_gguf_models,
    start_llama_server, stop_llama_server,
    is_llama_server_running, wait_for_llama_server,
    read_pid, is_process_running, get_llama_model_name,
    LLAMA_PID_FILE, LLAMA_LOG_FILE,
)
from app.optimize import build_plan, apply_optimizations, restore_optimizations, load_state
from app.monitor import get_system_stats, format_stats
from app.setup_wizard import (
    check_prerequisites, llama_server_path, clone_llama_cpp, build_llama_cpp,
    download_model, setup_venv, check_hf_login, hf_login, DownloadAuthError,
    ctranslate2_has_cuda, install_ctranslate2_cuda,
    whisper_model_cached, download_whisper_model,
    LLAMA_DIR, MODELS_DIR, RECOMMENDED_MODELS,
)

console = Console()
app = typer.Typer(
    name="jetson-assistant",
    help="Manage the Jetson Voice Assistant",
    add_completion=False,
    pretty_exceptions_enable=False,
)


# ── setup ─────────────────────────────────────────────────────────

@app.command()
def setup(
    skip_llama: bool = typer.Option(False, "--skip-llama", help="Skip building llama.cpp"),
    skip_model: bool = typer.Option(False, "--skip-model", help="Skip model download"),
    skip_venv: bool = typer.Option(False, "--skip-venv", help="Skip Python venv setup"),
):
    """First-time setup: build llama.cpp, download a model, set up Python environment."""
    console.print(Panel.fit(
        "[bold cyan]Jetson Voice Assistant — Setup[/bold cyan]\n"
        "[dim]This will build llama.cpp (~15 min) and download a model[/dim]",
        border_style="cyan",
    ))

    project_dir = Path(__file__).parent

    # ── Step 1: Prerequisites ──────────────────────────────────
    console.print("\n[bold]Step 1/7 — Checking prerequisites[/bold]")
    prereqs = check_prerequisites()
    missing_required = []
    missing_optional = []
    for name, (path, required) in prereqs.items():
        if path:
            console.print(f"  [green]✓[/green] {name:8}  [dim]{path}[/dim]")
        elif required:
            console.print(f"  [red]✗[/red] {name:8}  not found")
            missing_required.append(name)
        else:
            console.print(f"  [yellow]![/yellow] {name:8}  not found in PATH [dim](optional — cmake may find CUDA anyway)[/dim]")
            missing_optional.append(name)

    if missing_optional:
        console.print(
            f"\n  [yellow]Note:[/yellow] nvcc not found. "
            "If the build fails, install the CUDA compiler toolkit:\n"
            "  [dim]sudo apt-get install cuda-toolkit-12-6[/dim]\n"
            "  or add it to PATH: [dim]export PATH=/usr/local/cuda/bin:$PATH[/dim]"
        )

    if missing_required:
        console.print("\n[red]Missing required tools. Install them first:[/red]")
        fixes = ["sudo apt-get install -y cmake build-essential git portaudio19-dev"]
        if "python3-venv" in missing_required:
            fixes.append("sudo apt install python3.10-venv")
        for fix in fixes:
            console.print(f"  [dim]{fix}[/dim]")
        raise typer.Exit(1)

    # ── Step 2: Build llama.cpp ────────────────────────────────
    console.print("\n[bold]Step 2/7 — llama.cpp[/bold]")

    if skip_llama:
        console.print("  [dim]Skipped (--skip-llama)[/dim]")
    elif llama_server_path():
        console.print(f"  [green]✓ Already built[/green]  [dim]{llama_server_path()}[/dim]")
    else:
        if not Confirm.ask(
            f"  Build llama.cpp into {LLAMA_DIR}? (~15 min)", default=True
        ):
            console.print("  [yellow]Skipped.[/yellow]")
        else:
            console.print("  Cloning llama.cpp...", end=" ")
            if not clone_llama_cpp():
                console.print("[red]failed[/red]")
                raise typer.Exit(1)
            console.print("[green]done[/green]")

            console.print(
                "  Building with CUDA (ARCH=87)... "
                "[dim]this takes ~15 minutes[/dim]"
            )
            console.print()
            if not build_llama_cpp():
                console.print("\n[red]✗ Build failed.[/red]")
                console.print("  Check output above for errors.")
                console.print("  Common fix: [dim]export PATH=/usr/local/cuda/bin:$PATH[/dim]")
                raise typer.Exit(1)
            console.print(f"\n  [green]✓ Built:[/green] [dim]{llama_server_path()}[/dim]")

    # ── Step 3: Download model ─────────────────────────────────
    console.print("\n[bold]Step 3/7 — Model[/bold]")
    console.print(
        "  [dim]A free HuggingFace account is required to download models.\n"
        "  If not logged in yet: [bold]hf auth login[/bold]  "
        "(token at huggingface.co/settings/tokens)[/dim]"
    )

    if skip_model:
        console.print("  [dim]Skipped (--skip-model)[/dim]")
    else:
        existing = list(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.exists() else []
        if existing:
            console.print(f"  [green]✓ Models found in {MODELS_DIR}:[/green]")
            for m in existing:
                console.print(f"    [dim]{m.name}  ({m.stat().st_size/1e9:.1f} GB)[/dim]")
            if not Confirm.ask("  Download an additional model?", default=False):
                pass
            else:
                _model_download_dialog()
        else:
            console.print(f"  No models found in {MODELS_DIR}.")
            _model_download_dialog()

    # ── Step 4: Python venv ────────────────────────────────────
    console.print("\n[bold]Step 4/7 — Python environment[/bold]")

    if skip_venv:
        console.print("  [dim]Skipped (--skip-venv)[/dim]")
    elif (project_dir / "venv").exists():
        console.print(f"  [green]✓ venv already exists[/green]  [dim]{project_dir}/venv[/dim]")
    else:
        if not Confirm.ask("  Create Python venv and install dependencies?", default=True):
            console.print("  [yellow]Skipped.[/yellow]")
        else:
            console.print("  Creating venv and installing packages...")
            if setup_venv(project_dir):
                console.print("  [green]✓ venv ready[/green]")
            else:
                console.print("  [red]✗ venv setup failed.[/red]")
                console.print(
                    "  If you see 'ensurepip is not available', install the missing package:\n"
                    "  [dim]sudo apt install python3.10-venv[/dim]\n"
                    "  Then re-run: [dim]./jetson-assistant setup --skip-llama --skip-model[/dim]"
                )

    # ── Step 5: CTranslate2 CUDA ──────────────────────────────
    console.print("\n[bold]Step 5/7 — CTranslate2 (STT GPU acceleration)[/bold]")

    venv_dir = project_dir / "venv"
    if ctranslate2_has_cuda(venv_dir):
        console.print("  [green]✓ CTranslate2 already has CUDA support[/green]")
    elif not venv_dir.exists():
        console.print("  [dim]Skipped — venv not ready yet[/dim]")
    else:
        console.print(
            "  The standard PyPI wheel has no CUDA support on Jetson.\n"
            "  We'll first try a pre-built CUDA wheel from jetson-ai-lab.dev,\n"
            "  then fall back to building from source (~20 min) if needed."
        )
        if Confirm.ask("  Install CTranslate2 with CUDA now?", default=True):
            if install_ctranslate2_cuda(venv_dir):
                console.print("  [green]✓ CTranslate2 with CUDA ready[/green]")
            else:
                console.print(
                    "  [red]✗ CUDA install failed — STT will fall back to CPU.[/red]\n"
                    "  You can retry later: [dim]./jetson-assistant setup --skip-llama "
                    "--skip-model --skip-venv[/dim]"
                )
        else:
            console.print("  [dim]Skipped — STT will run on CPU.[/dim]")

    # ── Step 6: TTS voice models ───────────────────────────────
    console.print("\n[bold]Step 6/7 — TTS voice models[/bold]")

    from app.tts import _download_kokoro_models_if_missing, VOICES_DIR
    model_file = VOICES_DIR / "kokoro-v1.0.onnx"
    voices_file = VOICES_DIR / "voices-v1.0.bin"

    if model_file.exists() and voices_file.exists():
        console.print(f"  [green]✓ Kokoro models already downloaded[/green]  [dim]{VOICES_DIR}[/dim]")
    else:
        missing = []
        if not model_file.exists():
            missing.append("kokoro-v1.0.onnx (~311 MB)")
        if not voices_file.exists():
            missing.append("voices-v1.0.bin (~30 MB)")
        console.print(f"  Missing: [dim]{', '.join(missing)}[/dim]")
        if Confirm.ask("  Download Kokoro TTS models now?", default=True):
            console.print("  Downloading...")
            if _download_kokoro_models_if_missing():
                console.print("  [green]✓ Kokoro models ready[/green]")
            else:
                console.print("  [yellow]⚠ Download failed — will retry on first use[/yellow]")
        else:
            console.print("  [dim]Skipped — will download on first use.[/dim]")

    # ── Step 6: Whisper STT model ──────────────────────────────
    console.print("\n[bold]Step 7/7 — STT model (Whisper)[/bold]")

    from app.config import Config
    stt_model = Config.load().stt.model
    if whisper_model_cached(stt_model):
        console.print(f"  [green]✓ faster-whisper/{stt_model} already cached[/green]")
    else:
        console.print(f"  Model: [dim]Systran/faster-whisper-{stt_model}[/dim]")
        if Confirm.ask("  Download Whisper model now?", default=True):
            console.print("  Downloading...")
            if download_whisper_model(stt_model):
                console.print(f"  [green]✓ faster-whisper/{stt_model} ready[/green]")
            else:
                console.print("  [yellow]⚠ Download failed — will retry on first use[/yellow]")
        else:
            console.print("  [dim]Skipped — will download on first use.[/dim]")

    # ── Done ───────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold green]Setup complete![/bold green]\n"
        "Run [cyan]./jetson-assistant start[/cyan] to launch the assistant.",
        border_style="green",
    ))


def _model_download_dialog():
    console.print("\n  [bold]Recommended models:[/bold]")
    for i, m in enumerate(RECOMMENDED_MODELS, 1):
        license_tag = (
            f"  [yellow][accept license first][/yellow]" if m.get("license_url") else ""
        )
        console.print(
            f"  [cyan]{i}[/cyan]  {m['name']}  {m['size']}{license_tag}\n"
            f"      [dim]{m['description']}[/dim]"
        )
    console.print(f"  [cyan]{len(RECOMMENDED_MODELS)+1}[/cyan]  Skip")
    console.print()

    choice = Prompt.ask(
        "  Select",
        choices=[str(i) for i in range(1, len(RECOMMENDED_MODELS) + 2)],
        default="1",
    )
    idx = int(choice) - 1
    if idx >= len(RECOMMENDED_MODELS):
        console.print("  [dim]Skipped.[/dim]")
        return

    m = RECOMMENDED_MODELS[idx]

    if m.get("license_url"):
        console.print(
            f"\n  [yellow]License required.[/yellow] Accept at:\n"
            f"  [dim]{m['license_url']}[/dim]\n"
        )

    # HuggingFace now requires a token for all downloads
    if not check_hf_login():
        console.print(
            "  [yellow]HuggingFace login required[/yellow] "
            "(needed for all model downloads).\n"
        )
        if Confirm.ask("  Log in now?", default=True):
            if not hf_login():
                console.print("  [red]Login failed. Skipping download.[/red]")
                console.print(
                    "  Run [dim]hf auth login[/dim] manually, then re-run:\n"
                    "  [dim]./jetson-assistant setup --skip-llama --skip-venv[/dim]"
                )
                return
        else:
            console.print("  [dim]Skipped.[/dim]")
            return

    console.print(f"  Downloading [green]{m['filename']}[/green] ({m['size']})...")
    try:
        path = download_model(m["repo"], m["filename"])
    except DownloadAuthError:
        console.print("  [red]✗ Download failed: authentication error (HTTP 401).[/red]")
        console.print(
            "\n  To fix:\n"
            "  1. Run [dim]hf auth login[/dim] and enter your token\n"
            + (
                f"  2. Accept the license at [dim]{m['license_url']}[/dim]\n"
                if m.get("license_url") else ""
            ) +
            "  Then re-run: [dim]./jetson-assistant setup --skip-llama --skip-venv[/dim]"
        )
        return

    if path:
        console.print(f"  [green]✓ Saved to {path}[/green]")
    else:
        console.print("  [red]✗ Download failed.[/red]")
        console.print(
            f"  Try manually: [dim]hf download {m['repo']} "
            f"--include '{m['filename']}' --local-dir ~/models[/dim]"
        )


# ── start ─────────────────────────────────────────────────────────

@app.command()
def start(
    model: str = typer.Option(None, "--model", "-m", help="Path to GGUF model file"),
    port: int = typer.Option(8080, "--port", help="llama-server port"),
    ctx: int = typer.Option(4096, "--ctx", help="Context window size"),
    keep_server: bool = typer.Option(False, "--keep-server", "-k",
                                     help="Keep llama-server running after voice chat exits"),
    server_only: bool = typer.Option(False, "--server-only",
                                     help="Start llama-server only, skip voice chat"),
):
    """Start the voice assistant: pick a model, launch llama-server, start voice chat."""
    console.print(Panel.fit(
        "[bold cyan]Jetson Voice Assistant[/bold cyan]",
        border_style="cyan",
    ))

    # ── Check llama-server binary ──────────────────────────────
    llama_bin = find_llama_server()
    if not llama_bin:
        console.print("[red]✗ llama-server not found.[/red]")
        console.print("  Build it first — see SETUP.md Part 2.")
        raise typer.Exit(1)

    # ── Model selection ────────────────────────────────────────
    model_path = Path(model) if model else None

    if model_path is None:
        models = find_gguf_models()
        if not models:
            console.print("[red]✗ No .gguf models found.[/red]")
            console.print("  Download a model first, e.g.:")
            console.print("  [dim]huggingface-cli download bartowski/gemma-3-4b-it-GGUF "
                          "--include 'gemma-3-4b-it-Q4_K_M.gguf' --local-dir ~/models[/dim]")
            raise typer.Exit(1)

        console.print("\n[bold]Available models:[/bold]")
        for i, m in enumerate(models, 1):
            size_gb = m.stat().st_size / 1e9
            console.print(f"  [cyan]{i}[/cyan]  {m.name}  [dim]({size_gb:.1f} GB)[/dim]")
            console.print(f"      [dim]{m.parent}[/dim]")

        console.print()
        choice = Prompt.ask(
            "Select model",
            choices=[str(i) for i in range(1, len(models) + 1)],
            default="1",
        )
        model_path = models[int(choice) - 1]

    if not model_path.exists():
        console.print(f"[red]✗ Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n  Model: [green]{model_path.name}[/green]")
    console.print(f"  Context: {ctx} tokens  |  Port: {port}")

    # ── Check if server already running ───────────────────────
    if is_llama_server_running():
        running_model = get_llama_model_name()
        console.print(f"\n[yellow]llama-server already running[/yellow]"
                      + (f" ({running_model})" if running_model else ""))
        if not Confirm.ask("Stop it and start fresh?", default=False):
            console.print("  Using existing server.")
        else:
            console.print("  Stopping existing server...", end=" ")
            stop_llama_server()
            console.print("[green]done[/green]")
            _launch_server(model_path, port, ctx)
    else:
        _launch_server(model_path, port, ctx)

    # ── Start voice chat ───────────────────────────────────────
    if server_only:
        console.print("\n  [green]llama-server is running.[/green]  "
                      "[dim](--server-only: skipping voice chat)[/dim]")
        console.print("  Stop with: [dim]./jetson-assistant stop[/dim]")
        return

    console.print("\n[bold green]Starting voice chat...[/bold green]\n")
    voice_chat = Path(__file__).parent / "run_voice_chat.py"
    try:
        subprocess.run([sys.executable, str(voice_chat)])
    except KeyboardInterrupt:
        pass

    # ── Cleanup ────────────────────────────────────────────────
    if not keep_server:
        console.print("\n  Stopping llama-server...", end=" ")
        stop_llama_server()
        console.print("[green]done[/green]")
    else:
        console.print("\n  [dim]llama-server still running (--keep-server)[/dim]")


def _launch_server(model_path: Path, port: int, ctx: int):
    console.print("\n  Starting llama-server...", end=" ")
    pid = start_llama_server(model_path, port=port, ctx=ctx)
    if not pid:
        console.print("[red]failed[/red]")
        console.print(f"  Binary: {find_llama_server()}")
        raise typer.Exit(1)
    console.print(f"[dim]pid {pid}[/dim]")
    console.print(f"  Waiting for server (log: {LLAMA_LOG_FILE})...", end=" ")
    if not wait_for_llama_server(timeout=120):
        console.print("[red]timeout[/red]")
        console.print(f"  Check logs: [dim]tail -f {LLAMA_LOG_FILE}[/dim]")
        stop_llama_server()
        raise typer.Exit(1)
    console.print("[green]ready[/green]")


# ── stop ──────────────────────────────────────────────────────────

@app.command()
def stop():
    """Stop llama-server and any running voice chat."""
    stopped = []

    if is_llama_server_running():
        console.print("  Stopping llama-server...", end=" ")
        stop_llama_server()
        console.print("[green]done[/green]")
        stopped.append("llama-server")
    else:
        console.print("  [dim]llama-server not running[/dim]")

    # kill any lingering voice chat processes
    result = subprocess.run(
        ["pkill", "-f", "run_voice_chat.py"],
        capture_output=True,
    )
    if result.returncode == 0:
        console.print("  Stopped voice chat process.")
        stopped.append("voice-chat")

    if not stopped:
        console.print("  Nothing was running.")


# ── status ────────────────────────────────────────────────────────

@app.command()
def status():
    """Show running processes and system memory."""
    console.print()

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    # llama-server
    if is_llama_server_running():
        model = get_llama_model_name() or "unknown"
        pid = read_pid(LLAMA_PID_FILE)
        table.add_row("llama-server", "[green]running[/green]",
                      f"pid {pid}  model: {model}")
    else:
        table.add_row("llama-server", "[red]stopped[/red]", "")

    # voice chat
    result = subprocess.run(
        ["pgrep", "-f", "run_voice_chat.py"], capture_output=True
    )
    if result.returncode == 0:
        pids = result.stdout.decode().strip()
        table.add_row("voice-chat", "[green]running[/green]", f"pid {pids}")
    else:
        table.add_row("voice-chat", "[dim]stopped[/dim]", "")

    # memory
    stats = get_system_stats()
    table.add_row("system", "[cyan]info[/cyan]", format_stats(stats))

    # optimization state
    state = load_state()
    if state:
        table.add_row("optimized", "[yellow]yes[/yellow]",
                      "optimize --restore to undo")
    else:
        table.add_row("optimized", "[dim]no[/dim]", "optimize to apply")

    console.print(table)
    console.print()


# ── optimize ──────────────────────────────────────────────────────

@app.command()
def optimize(
    restore: bool = typer.Option(False, "--restore", help="Restore system to pre-optimization state"),
    status: bool = typer.Option(False, "--status", help="Show current optimization state"),
    all_: bool = typer.Option(False, "--all", help="Apply all optimizations without prompting"),
):
    """Manage memory optimizations for better LLM performance on Jetson.

    Default: step-by-step dialog — confirm each optimization individually.
    --all:     apply every optimization without prompting.
    --restore: revert all applied optimizations.
    --status:  show what is currently applied.
    """
    if restore:
        _optimize_restore()
    elif status:
        _optimize_status()
    else:
        _optimize_apply(skip_prompts=all_)


def _optimize_status():
    state = load_state()
    console.print()
    if not state:
        console.print("  [dim]No optimizations applied.[/dim]")
        console.print("  Run [bold]optimize[/bold] to apply.")
        return

    applied = (
        state.get("applied", []) +
        state.get("services_disabled", []) +
        state.get("zram_disabled", [])
    )
    console.print(f"  [yellow]Optimized[/yellow] — {len(applied)} change(s) active:\n")
    for item in applied:
        console.print(f"    [dim]• {item}[/dim]")
    console.print(f"\n  Run [bold]optimize --restore[/bold] to revert.")
    console.print()


def _optimize_apply(skip_prompts: bool = False):
    console.print(Panel.fit(
        "[bold yellow]Memory Optimization[/bold yellow]\n"
        "[dim]Safe, reversible system tuning for Jetson Orin Nano[/dim]",
        border_style="yellow",
    ))

    if load_state():
        console.print("[yellow]Optimizations already applied.[/yellow]")
        console.print("Run [bold]optimize --restore[/bold] first to reapply.")
        raise typer.Exit()

    console.print("\n[bold]Analyzing system...[/bold]")
    plan = build_plan()

    def _ask(question: str, default: bool) -> bool:
        return True if skip_prompts else Confirm.ask(question, default=default)

    # Build a filtered plan based on user choices
    approved = {
        "target": False,
        "services_to_disable": [],
        "zram": False,
        "jetson_clocks": False,
    }

    console.print()

    # ── GUI / target ──────────────────────────────────────────
    row = plan["target"]
    if row["change"]:
        console.print(f"  [bold]Disable desktop GUI[/bold] (switch to multi-user.target)"
                      f"  [dim]~{row['savings_mb']} MB[/dim]")
        approved["target"] = _ask("  Apply?", default=True)
    else:
        console.print("  [dim]GUI already disabled (multi-user.target) — skipping[/dim]")

    # ── Services ──────────────────────────────────────────────
    services = plan["services"]["to_disable"]
    if services:
        console.print()
        console.print("  [bold]Disable unused services:[/bold]")
        for svc, meta in services.items():
            console.print(f"    [cyan]{svc}[/cyan]  [dim]{meta['description']}[/dim]")
            if _ask(f"    Disable {svc}?", default=meta["default_on"]):
                approved["services_to_disable"].append(svc)
    else:
        console.print("\n  [dim]No unused services found — skipping[/dim]")

    # ── zram ─────────────────────────────────────────────────
    row = plan["zram"]
    if row["active"]:
        console.print()
        console.print(f"  [bold]Disable zram compressed swap[/bold]"
                      f"  [dim]~{row['savings_mb']} MB  (use NVMe swap instead)[/dim]")
        approved["zram"] = _ask("  Apply?", default=True)
    else:
        console.print("  [dim]zram not active — skipping[/dim]")

    # ── jetson_clocks ─────────────────────────────────────────
    row = plan["jetson_clocks"]
    if row["available"]:
        console.print()
        console.print("  [bold]Set all CPU/GPU clocks to maximum[/bold]  [dim](jetson_clocks)[/dim]")
        approved["jetson_clocks"] = _ask("  Apply?", default=True)
    else:
        console.print("  [dim]jetson_clocks not found — skipping[/dim]")

    # ── Nothing selected ─────────────────────────────────────
    if (not approved["target"] and not approved["services_to_disable"]
            and not approved["zram"] and not approved["jetson_clocks"]):
        console.print("\n[yellow]Nothing selected — no changes made.[/yellow]")
        raise typer.Exit()

    # ── Filter plan and apply ─────────────────────────────────
    filtered_plan = {
        "target": {**plan["target"], "change": approved["target"]},
        "services": {**plan["services"],
                     "to_disable": {s: plan["services"]["to_disable"][s]
                                    for s in approved["services_to_disable"]}},
        "zram":   {**plan["zram"], "active": approved["zram"],
                   "to_disable": plan["zram"]["to_disable"] if approved["zram"] else {}},
        "jetson_clocks": {**plan["jetson_clocks"],
                          "available": approved["jetson_clocks"]},
    }

    console.print("\n[bold]Applying...[/bold]")
    state = apply_optimizations(filtered_plan)

    n = (
        len(state.get("services_disabled", [])) +
        len(state.get("zram_disabled", [])) +
        len(state.get("applied", []))
    )
    console.print(f"\n[green]✓ Done — {n} changes applied.[/green]")
    console.print("  [dim]A reboot is required to fully apply all changes.[/dim]")
    console.print("  Run [bold]optimize --restore[/bold] to undo.\n")

    if _ask("Reboot now to fully apply all changes?", default=False):
        subprocess.run(["sudo", "reboot"])


def _optimize_restore():
    state = load_state()
    if not state:
        console.print("[yellow]No saved optimization state found. Nothing to restore.[/yellow]")
        raise typer.Exit()

    console.print(Panel.fit(
        "[bold]Restore System[/bold]\n[dim]Reverts all applied optimizations[/dim]",
        border_style="cyan",
    ))

    applied = (
        state.get("applied", []) +
        state.get("services_disabled", []) +
        state.get("zram_disabled", [])
    )
    console.print(f"\n  Will revert: {len(applied)} change(s)")
    for item in applied:
        console.print(f"    [dim]• {item}[/dim]")
    console.print()

    if not Confirm.ask("Restore? (requires sudo)", default=False):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit()

    console.print("\n[bold]Restoring...[/bold]")
    restored = restore_optimizations(state)

    for item in restored:
        console.print(f"  [green]✓[/green] {item}")

    console.print(f"\n[green]✓ Done — {len(restored)} items restored.[/green]\n")

    if Confirm.ask("Reboot now to fully apply?", default=False):
        subprocess.run(["sudo", "reboot"])


# ── test ──────────────────────────────────────────────────────────

@app.command()
def test(
    llm: bool = typer.Option(False, "--llm", help="Test LLM (requires llama-server running)"),
    stt: bool = typer.Option(False, "--stt", help="Test speech-to-text (records 3 seconds)"),
    tts: bool = typer.Option(False, "--tts", help="Test text-to-speech (plays a sentence)"),
    vad: bool = typer.Option(False, "--vad", help="Test VAD (shows mic activity for 5 seconds)"),
    all_: bool = typer.Option(False, "--all", help="Run all component tests"),
):
    """Test individual pipeline components."""
    if not any([llm, stt, tts, vad, all_]):
        console.print("Specify at least one flag: --llm  --stt  --tts  --vad  --all")
        raise typer.Exit(1)

    from app.config import Config
    cfg = Config.load()

    if llm or all_:
        _test_llm(cfg)
    if stt or all_:
        _test_stt(cfg)
    if tts or all_:
        _test_tts(cfg)
    if vad or all_:
        _test_vad(cfg)


def _test_llm(cfg):
    console.print("\n[bold cyan]── LLM test ──[/bold cyan]")

    server_started = False
    if not is_llama_server_running():
        models = find_gguf_models()
        if not models:
            console.print("  [red]✗ No .gguf models found in ~/models.[/red]")
            console.print("  Run [dim]./jetson-assistant setup[/dim] first.")
            return
        model_path = models[0]
        console.print(f"  Starting llama-server  [dim]{model_path.name}[/dim]...")
        pid = start_llama_server(model_path)
        if not pid or not wait_for_llama_server(timeout=120):
            console.print("  [red]✗ llama-server failed to start.[/red]")
            stop_llama_server()
            return
        console.print("  [green]server ready[/green]")
        server_started = True

    import time
    from app.llm import LLM
    llm = LLM(
        model=cfg.llm.model,
        base_url=cfg.llm.base_url,
        backend=cfg.llm.backend,
        max_tokens=64,
        temperature=cfg.llm.temperature,
    )
    console.print("  Connecting...", end=" ")
    if not llm.load():
        console.print("[red]failed[/red]")
        if server_started:
            stop_llama_server()
        return
    console.print(f"[green]ok[/green]  model: [dim]{llm.model}[/dim]")

    prompt = "Say hello in exactly one short sentence."
    console.print(f"  Prompt: [dim]{prompt}[/dim]")
    console.print("  Response: ", end="")
    t0 = time.time()
    for chunk, meta in llm.generate_stream(prompt):
        if chunk:
            console.print(chunk, end="", highlight=False)
    elapsed = time.time() - t0
    console.print(f"\n  [green]✓[/green] [dim]{elapsed:.1f}s[/dim]")
    llm.unload()

    if server_started:
        console.print("  Stopping llama-server...", end=" ")
        stop_llama_server()
        console.print("[dim]done[/dim]")


def _test_stt(cfg):
    console.print("\n[bold cyan]── STT test ──[/bold cyan]")
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        console.print("  [red]✗ sounddevice not installed.[/red]")
        return

    from app.stt import STT
    stt = STT(
        model=cfg.stt.model,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
        language=cfg.stt.language,
        beam_size=cfg.stt.beam_size,
    )
    console.print("  Loading Whisper model...", end=" ")
    if not stt.load():
        console.print("[red]failed[/red]")
        return
    console.print("[green]ok[/green]")

    duration = 3
    sample_rate = 16000
    console.print(f"  Recording {duration}s — [bold]speak now...[/bold]")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.squeeze()

    console.print("  Transcribing...", end=" ")
    import time
    t0 = time.time()
    result = stt.transcribe(audio, sample_rate=sample_rate)
    elapsed = time.time() - t0
    text = result.get("text", "").strip()
    console.print(f"[green]ok[/green]  ({elapsed:.2f}s)")
    console.print(f"  Transcript: [bold]{text or '[silence]'}[/bold]")


def _test_tts(cfg):
    console.print("\n[bold cyan]── TTS test ──[/bold cyan]")
    from app.tts import create_tts
    import numpy as np
    import os

    tts = create_tts(
        voice=cfg.tts.voice,
        speed=cfg.tts.speed,
        lang=cfg.tts.lang,
    )
    console.print("  Loading Kokoro TTS...", end=" ")
    # Suppress the harmless ONNX GPU discovery warning on Jetson
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    if not tts.load():
        console.print("[red]failed[/red]")
        return
    console.print("[green]ok[/green]")

    sentence = "Hello! The Jetson voice assistant is up and running."
    console.print(f"  Synthesizing: [dim]{sentence}[/dim]")
    import time
    t0 = time.time()
    result = tts.synthesize(sentence)
    elapsed = time.time() - t0

    if result.get("error"):
        console.print(f"  [red]✗ {result['error']}[/red]")
        tts.unload()
        return

    audio = result.get("audio")
    sample_rate = result.get("sample_rate", 24000)
    console.print(f"  Synthesized in [dim]{elapsed:.2f}s[/dim] — playing...")

    try:
        import sounddevice as sd
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        console.print("  [green]✓ Playback complete[/green]")
    except Exception as e:
        console.print(f"  [yellow]⚠ Playback failed: {e}[/yellow]")
        console.print("  (audio was synthesized successfully)")

    tts.unload()


def _test_vad(cfg):
    console.print("\n[bold cyan]── VAD test ──[/bold cyan]")
    try:
        import sounddevice as sd
    except ImportError:
        console.print("  [red]✗ sounddevice not installed.[/red]")
        return

    from app.pipeline import load_silero
    console.print("  Loading Silero VAD...", end=" ")
    # Suppress the harmless ONNX GPU-discovery warning emitted at C++ init time
    import os as _os
    _devnull = _os.open(_os.devnull, _os.O_WRONLY)
    _saved = _os.dup(2)
    _os.dup2(_devnull, 2)
    _os.close(_devnull)
    try:
        vad = load_silero()
    finally:
        _os.dup2(_saved, 2)
        _os.close(_saved)
    if vad is None:
        console.print("[red]failed[/red]")
        return
    console.print("[green]ok[/green]")

    sample_rate = 16000
    # Silero requires exactly 512 samples at 16kHz → 32ms
    chunk_ms = 32
    chunk_size = int(sample_rate * chunk_ms / 1000)
    duration = 5

    console.print(f"  Listening for {duration}s — [bold]speak or stay silent...[/bold]\n")

    import time
    t_end = time.time() + duration
    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="int16", blocksize=chunk_size) as stream:
        while time.time() < t_end:
            raw, _ = stream.read(chunk_size)
            score = vad(raw.tobytes())
            bar_len = int(score * 30)
            bar = "[green]" + "█" * bar_len + "[/green]" + "░" * (30 - bar_len)
            label = "[bold green] SPEECH[/bold green]" if score > cfg.vad.silero_threshold else "[dim] silence[/dim]"
            console.print(f"  {bar} {score:.2f}{label}", end="\r", highlight=False)

    console.print("\n  [green]✓ VAD test complete[/green]")


if __name__ == "__main__":
    app()
