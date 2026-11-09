"""Minimal .env loader (no dependency).

Rules: variables already set in the environment always win; empty values are skipped, so a
blank ANTHROPIC_API_KEY copied from .env.example cannot shadow an `ant auth login` profile.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path) -> list[str]:
    """Load KEY=VALUE lines from `path` into os.environ. Returns the names of the keys it set."""
    if not path.exists():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():  # utf-8-sig: tolerate a Notepad BOM
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
