"""Component tests for the voice pipeline."""

import os
import time

from rich.console import Console

console = Console()


def test_llm(cfg, start_server_fn, stop_server_fn, is_running_fn, wait_fn, find_models_fn):
    console.print("\n[bold cyan]── LLM test ──[/bold cyan]")

    from app.llm import LLM

    server_started = False
    if not is_running_fn():
        models = find_models_fn()
        if not models:
            console.print("  [red]✗ No .gguf models found in ~/models.[/red]")
            console.print("  Run [dim]./jetson-assistant setup[/dim] first.")
            return
        model_path = models[0]
        console.print(f"  Starting llama-server  [dim]{model_path.name}[/dim]...")
        pid = start_server_fn(model_path)
        if not pid or not wait_fn(timeout=120):
            console.print("  [red]✗ llama-server failed to start.[/red]")
            stop_server_fn()
            return
        console.print("  [green]server ready[/green]")
        server_started = True

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
            stop_server_fn()
        return
    console.print(f"[green]ok[/green]  model: [dim]{llm.model}[/dim]")

    prompt = "Say hello in exactly one short sentence."
    console.print(f"  Prompt: [dim]{prompt}[/dim]")
    console.print("  Response: ", end="")
    t0 = time.time()
    for chunk, _ in llm.generate_stream(prompt):
        if chunk:
            console.print(chunk, end="", highlight=False)
    elapsed = time.time() - t0
    console.print(f"\n  [green]✓[/green] [dim]{elapsed:.1f}s[/dim]")
    llm.unload()

    if server_started:
        console.print("  Stopping llama-server...", end=" ")
        stop_server_fn()
        console.print("[dim]done[/dim]")


def test_stt(cfg):
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
    t0 = time.time()
    result = stt.transcribe(audio, sample_rate=sample_rate)
    elapsed = time.time() - t0
    text = result.get("text", "").strip()
    console.print(f"[green]ok[/green]  ({elapsed:.2f}s)")
    console.print(f"  Transcript: [bold]{text or '[silence]'}[/bold]")
    stt.unload()


def test_tts(cfg):
    console.print("\n[bold cyan]── TTS test ──[/bold cyan]")
    from app.tts import create_tts

    tts = create_tts(voice=cfg.tts.voice, speed=cfg.tts.speed, lang=cfg.tts.lang)
    console.print("  Loading Kokoro TTS...", end=" ")
    if not tts.load():
        console.print("[red]failed[/red]")
        return
    console.print("[green]ok[/green]")

    sentence = "Hello! The Jetson voice assistant is up and running."
    console.print(f"  Synthesizing: [dim]{sentence}[/dim]")
    t0 = time.time()
    result = tts.synthesize(sentence)
    elapsed = time.time() - t0

    if result.get("error"):
        console.print(f"  [red]✗ {result['error']}[/red]")
        tts.unload()
        return

    console.print(f"  Synthesized in [dim]{elapsed:.2f}s[/dim] — playing...")
    try:
        import sounddevice as sd
        sd.play(result["audio"], samplerate=result.get("sample_rate", 24000))
        sd.wait()
        console.print("  [green]✓ Playback complete[/green]")
    except Exception as e:
        console.print(f"  [yellow]⚠ Playback failed: {e}[/yellow]")
        console.print("  (audio was synthesized successfully)")

    tts.unload()


def test_vad(cfg):
    console.print("\n[bold cyan]── VAD test ──[/bold cyan]")
    try:
        import sounddevice as sd
    except ImportError:
        console.print("  [red]✗ sounddevice not installed.[/red]")
        return

    from app.pipeline import load_silero

    console.print("  Loading Silero VAD...", end=" ")
    # Suppress the harmless ONNX GPU-discovery warning at C++ init time
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _saved = os.dup(2)
    os.dup2(_devnull, 2)
    os.close(_devnull)
    try:
        vad = load_silero()
    finally:
        os.dup2(_saved, 2)
        os.close(_saved)

    if vad is None:
        console.print("[red]failed[/red]")
        return
    console.print("[green]ok[/green]")

    sample_rate = 16000
    chunk_size  = int(sample_rate * 32 / 1000)  # 512 samples — Silero requires exactly 32ms
    duration    = 5

    console.print(f"  Listening for {duration}s — [bold]speak or stay silent...[/bold]\n")

    t_end = time.time() + duration
    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="int16", blocksize=chunk_size) as stream:
        while time.time() < t_end:
            raw, _ = stream.read(chunk_size)
            score   = vad(raw.tobytes())
            bar_len = int(score * 30)
            bar     = "[green]" + "█" * bar_len + "[/green]" + "░" * (30 - bar_len)
            label   = "[bold green] SPEECH[/bold green]" if score > cfg.vad.silero_threshold else "[dim] silence[/dim]"
            console.print(f"  {bar} {score:.2f}{label}", end="\r", highlight=False)

    console.print("\n  [green]✓ VAD test complete[/green]")
