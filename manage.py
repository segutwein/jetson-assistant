#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Jetson Voice Assistant — management CLI

Commands:
  setup     First-time setup: build llama.cpp and download a model
  start     Pick a model and start the assistant
  stop      Stop llama-server and voice chat
  status    Show what is running and memory usage
  optimize  Apply memory optimizations
  restore   Undo memory optimizations
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
    download_model, setup_venv, LLAMA_DIR, MODELS_DIR, RECOMMENDED_MODELS,
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
    console.print("\n[bold]Step 1/4 — Checking prerequisites[/bold]")
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
        console.print("  [dim]sudo apt-get install -y cmake build-essential git[/dim]")
        raise typer.Exit(1)

    # ── Step 2: Build llama.cpp ────────────────────────────────
    console.print("\n[bold]Step 2/4 — llama.cpp[/bold]")

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
    console.print("\n[bold]Step 3/4 — Model[/bold]")

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
    console.print("\n[bold]Step 4/4 — Python environment[/bold]")

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
                console.print("  [yellow]⚠ venv setup failed — check output above[/yellow]")

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
        console.print(
            f"  [cyan]{i}[/cyan]  {m['name']}  {m['size']}\n"
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
    console.print(f"  Downloading [green]{m['filename']}[/green] ({m['size']})...")
    path = download_model(m["repo"], m["filename"])
    if path:
        console.print(f"  [green]✓ Saved to {path}[/green]")
    else:
        console.print("  [red]✗ Download failed.[/red]")
        console.print("  Try manually: [dim]huggingface-cli download "
                      f"{m['repo']} --include '{m['filename']}' "
                      f"--local-dir ~/models[/dim]")


# ── start ─────────────────────────────────────────────────────────

@app.command()
def start(
    model: str = typer.Option(None, "--model", "-m", help="Path to GGUF model file"),
    port: int = typer.Option(8080, "--port", help="llama-server port"),
    ctx: int = typer.Option(4096, "--ctx", help="Context window size"),
    keep_server: bool = typer.Option(False, "--keep-server", "-k",
                                     help="Keep llama-server running after voice chat exits"),
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
):
    """Manage memory optimizations for better LLM performance on Jetson.

    Default: analyze and apply optimizations.
    --restore: revert all applied optimizations.
    --status:  show what is currently applied.
    """
    if restore:
        _optimize_restore()
    elif status:
        _optimize_status()
    else:
        _optimize_apply()


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


def _optimize_apply():
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

    table = Table(box=box.SIMPLE, padding=(0, 2))
    table.add_column("Optimization")
    table.add_column("Est. savings", justify="right")
    table.add_column("Will change")

    row = plan["target"]
    table.add_row(
        row["description"],
        f"~{row['savings_mb']} MB" if row["change"] else "—",
        "[green]yes[/green]" if row["change"] else "[dim]already set[/dim]",
    )

    row = plan["services"]
    services = list(row["to_disable"].keys())
    table.add_row(
        row["description"],
        f"~{row['savings_mb']} MB" if services else "—",
        "\n".join(services) if services else "[dim]none to disable[/dim]",
    )

    row = plan["zram"]
    table.add_row(
        row["description"],
        f"~{row['savings_mb']} MB" if row["active"] else "—",
        "[green]yes[/green]" if row["active"] else "[dim]not active[/dim]",
    )

    row = plan["jetson_clocks"]
    table.add_row(
        row["description"],
        "performance",
        "[green]yes[/green]" if row["available"] else "[dim]not found[/dim]",
    )

    console.print(table)

    total = sum(
        v["savings_mb"]
        for v in [plan["target"], plan["services"], plan["zram"]]
        if v.get("change") or v.get("active") or v.get("to_disable")
    )
    console.print(f"\n  Estimated total savings: [bold green]~{total} MB[/bold green]")
    console.print("  [dim]A reboot is required to fully apply all changes.[/dim]\n")

    if not Confirm.ask("Apply optimizations? (requires sudo)", default=False):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit()

    console.print("\n[bold]Applying...[/bold]")
    state = apply_optimizations(plan)

    n = (
        len(state.get("services_disabled", [])) +
        len(state.get("zram_disabled", [])) +
        len(state.get("applied", []))
    )
    console.print(f"\n[green]✓ Done — {n} changes applied.[/green]")
    console.print("  Run [bold]optimize --restore[/bold] to undo.\n")

    if Confirm.ask("Reboot now to fully apply all changes?", default=False):
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


if __name__ == "__main__":
    app()
