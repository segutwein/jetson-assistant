# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory optimizations for Jetson — safe, reversible system tuning."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

STATE_DIR = Path.home() / ".jetson-assistant"
OPT_STATE_FILE = STATE_DIR / "optimization-state.json"

# Services that are safe to disable on a headless voice assistant
_OPTIONAL_SERVICES = [
    "bluetooth.service",
    "avahi-daemon.service",
    "cups.service",
    "cups-browsed.service",
    "ModemManager.service",
    "snapd.service",
    "apt-daily.service",
    "apt-daily-upgrade.service",
    "unattended-upgrades.service",
]

_ZRAM_SERVICES = [
    "zram-config.service",
    "zramswap.service",
]

JETSON_CLOCKS_STORE = STATE_DIR / "jetson_clocks_backup.conf"


# ── Helpers ────────────────────────────────────────────────────────

def _run(cmd: list[str], sudo: bool = False) -> tuple[int, str]:
    if sudo:
        cmd = ["sudo"] + cmd
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def _service_exists(name: str) -> bool:
    rc, _ = _run(["systemctl", "list-unit-files", name])
    return rc == 0 and name in _


def _service_enabled(name: str) -> bool:
    rc, out = _run(["systemctl", "is-enabled", name])
    return rc == 0 and out.strip() in ("enabled", "static")


def _service_active(name: str) -> bool:
    rc, _ = _run(["systemctl", "is-active", name])
    return rc == 0


def _get_default_target() -> str:
    _, out = _run(["systemctl", "get-default"])
    return out.strip()


def _has_jetson_clocks() -> bool:
    return shutil.which("jetson_clocks") is not None


# ── State persistence ──────────────────────────────────────────────

def save_state(state: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OPT_STATE_FILE.write_text(json.dumps(state, indent=2))


def load_state() -> Optional[dict]:
    if not OPT_STATE_FILE.exists():
        return None
    try:
        return json.loads(OPT_STATE_FILE.read_text())
    except Exception:
        return None


# ── Optimization plan ──────────────────────────────────────────────

def build_plan() -> dict:
    """Inspect the system and return a plan of what can be optimized."""
    plan = {}

    current_target = _get_default_target()
    plan["target"] = {
        "current": current_target,
        "change": current_target != "multi-user.target",
        "savings_mb": 150,
        "description": "Disable desktop/GUI (switch to multi-user.target)",
    }

    services = {}
    for svc in _OPTIONAL_SERVICES:
        if _service_exists(svc) and _service_enabled(svc):
            services[svc] = True
    plan["services"] = {
        "to_disable": services,
        "savings_mb": len(services) * 5,
        "description": f"Disable {len(services)} unnecessary services",
    }

    zram = {}
    for svc in _ZRAM_SERVICES:
        if _service_exists(svc) and _service_active(svc):
            zram[svc] = True
    plan["zram"] = {
        "to_disable": zram,
        "active": bool(zram),
        "savings_mb": 100,
        "description": "Disable zram compressed swap (use NVMe swap instead)",
    }

    plan["jetson_clocks"] = {
        "available": _has_jetson_clocks(),
        "savings_mb": 0,
        "description": "Set all CPU/GPU clocks to maximum (jetson_clocks)",
    }

    return plan


# ── Apply ──────────────────────────────────────────────────────────

def apply_optimizations(plan: dict) -> dict:
    """Apply the plan and return saved state for restore."""
    state = {"applied": [], "target_before": _get_default_target(), "services_disabled": [], "zram_disabled": []}

    if plan["target"]["change"]:
        rc, out = _run(["systemctl", "set-default", "multi-user.target"], sudo=True)
        if rc == 0:
            state["applied"].append("target")

    for svc in plan["services"]["to_disable"]:
        rc, _ = _run(["systemctl", "disable", "--now", svc], sudo=True)
        if rc == 0:
            state["services_disabled"].append(svc)

    for svc in plan["zram"]["to_disable"]:
        rc, _ = _run(["systemctl", "disable", "--now", svc], sudo=True)
        if rc == 0:
            state["zram_disabled"].append(svc)

    if plan["jetson_clocks"]["available"]:
        _run(["jetson_clocks", "--store", str(JETSON_CLOCKS_STORE)], sudo=True)
        rc, _ = _run(["jetson_clocks"], sudo=True)
        if rc == 0:
            state["applied"].append("jetson_clocks")

    save_state(state)
    return state


# ── Restore ────────────────────────────────────────────────────────

def restore_optimizations(state: dict) -> list[str]:
    """Revert optimizations from saved state. Returns list of restored items."""
    restored = []

    if "target" in state.get("applied", []):
        target = state.get("target_before", "graphical.target")
        rc, _ = _run(["systemctl", "set-default", target], sudo=True)
        if rc == 0:
            restored.append(f"Default target → {target}")

    for svc in state.get("services_disabled", []):
        rc, _ = _run(["systemctl", "enable", "--now", svc], sudo=True)
        if rc == 0:
            restored.append(f"Re-enabled {svc}")

    for svc in state.get("zram_disabled", []):
        rc, _ = _run(["systemctl", "enable", "--now", svc], sudo=True)
        if rc == 0:
            restored.append(f"Re-enabled {svc}")

    if "jetson_clocks" in state.get("applied", []) and JETSON_CLOCKS_STORE.exists():
        rc, _ = _run(["jetson_clocks", "--restore", str(JETSON_CLOCKS_STORE)], sudo=True)
        if rc == 0:
            restored.append("Restored jetson_clocks settings")

    if OPT_STATE_FILE.exists():
        OPT_STATE_FILE.unlink()

    return restored
