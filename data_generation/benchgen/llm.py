"""LLM access shared by every stage.

`LLM.call(task, user, schema, tag)` returns a validated pydantic object or None. Every
response is cached on disk under a hash of (backend, model, effort, prompt, schema), so a
re-run replays finished work for free and resumes where it stopped. Every call -- cached or
not, successful or not -- is appended to generation_log.jsonl with token usage and cost.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from . import mock
from .config import Config
from .prompts import SYSTEM
from .textutil import stable_hash

M = TypeVar("M", bound=BaseModel)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Removed from the Claude Code subprocess environment: if either is present, Claude Code bills the
# API instead of the logged-in subscription. The pipeline's .env loader may have set them.
CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
CLAUDE_CODE_TIMEOUT = 1200  # seconds per call


class LLMFatalError(RuntimeError):
    """A failure that retrying cannot fix: bad credentials, unknown model, malformed request."""


class GenerationLog:
    def __init__(self, path: Path, prices: dict[str, float]):
        self.path = path
        self.prices = prices
        self._lock = threading.Lock()
        self.totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.validation_path = path.parent / "validation_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

    def cost(self, usage: dict) -> float:
        if "notional_cost_usd" in usage:  # Claude Code reports its own API-equivalent figure
            return usage["notional_cost_usd"]
        p = self.prices
        return (usage.get("input_tokens", 0) * p["input"]
                + usage.get("output_tokens", 0) * p["output"]
                + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
                + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]) / 1e6

    def record(self, *, task: str, tag: str, ok: bool, cached: bool, reason: str | None = None,
               usage: dict | None = None, stop_reason: str | None = None) -> None:
        usage = usage or {}
        cost = 0.0 if cached else self.cost(usage)
        event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "tag": tag, "ok": ok,
                 "cached": cached, "reason": reason, "stop_reason": stop_reason,
                 "usage": usage, "cost_usd": round(cost, 6)}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            t = self.totals[task]
            t["calls"] += 1
            t["cached"] += int(cached)
            t["failed"] += int(not ok)
            t["cost_usd"] += cost
            for k, v in usage.items():
                if isinstance(v, (int, float)):
                    t[k] += v

    def record_validation(self, *, task: str, tag: str, failures: dict[str, list[str]]) -> None:
        """Why model output was rejected by a validator (the model call itself succeeded)."""
        if not failures:
            return
        event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "tag": tag,
                 "n_failed": len(failures), "failures": failures}
        with self._lock:
            with self.validation_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        with self._lock:
            return {task: dict(v) for task, v in self.totals.items()}


class LLM:
    def __init__(self, cfg: Config, log: GenerationLog):
        self.cfg = cfg
        self.log = log
        self.backend = cfg.llm.backend
        self.cache_dir = cfg.llm.cache_dir / self.backend
        # The SDK retries 408/409/429/5xx and connection errors itself; raise the ceiling for long runs.
        self.client = anthropic.Anthropic(max_retries=6) if self.backend == "claude" else None
        if self.backend == "claude_code":
            self.claude_exe = shutil.which("claude")
            if not self.claude_exe:
                raise LLMFatalError("the `claude` CLI is not on PATH: install Claude Code, or use backend: claude")
            self.subprocess_env = {k: v for k, v in os.environ.items() if k not in CREDENTIAL_VARS}
            # Run from an empty directory so no project CLAUDE.md or memory is pulled into the context.
            self.neutral_cwd = tempfile.mkdtemp(prefix="benchgen_cc_")

    def call(self, task: str, user: str, schema: type[M], tag: str) -> M | None:
        llm = self.cfg.llm
        key = stable_hash([self.backend, llm.model, llm.effort, llm.fallbacks, SYSTEM, task, user,
                           schema.model_json_schema()])
        path = self.cache_dir / key[:2] / f"{key}.json"

        if path.exists():
            try:
                obj = schema.model_validate(json.loads(path.read_text(encoding="utf-8"))["output"])
                self.log.record(task=task, tag=tag, ok=True, cached=True)
                return obj
            except (ValidationError, KeyError, json.JSONDecodeError):
                pass  # stale or corrupt cache entry: regenerate

        if self.backend == "mock":
            payload, usage, stop = mock.mock_response(task, user), {}, "end_turn"
        elif self.backend == "claude_code":
            payload, usage, stop = self._claude_code(user, schema)
        else:
            payload, usage, stop = self._claude(user, schema)

        if payload is None:
            self.log.record(task=task, tag=tag, ok=False, cached=False, reason=stop, usage=usage, stop_reason=stop)
            return None
        try:
            obj = schema.model_validate(payload)
        except ValidationError as e:
            self.log.record(task=task, tag=tag, ok=False, cached=False, reason=f"schema: {e.errors()[:2]}",
                            usage=usage, stop_reason=stop)
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"output": obj.model_dump(), "usage": usage, "stop_reason": stop},
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        self.log.record(task=task, tag=tag, ok=True, cached=False, usage=usage, stop_reason=stop)
        return obj

    def _claude(self, user: str, schema: type[BaseModel]) -> tuple[dict | None, dict, str]:
        llm = self.cfg.llm
        kwargs = dict(
            model=llm.model,
            max_tokens=llm.max_tokens,
            # One stable system prompt for every call keeps the cacheable prefix identical.
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": llm.effort,
                "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema.model_json_schema())},
            },
        )
        if llm.fallbacks:
            # Server-side refusal fallbacks, "default" routing. Passed via extra_body because the
            # installed SDK predates a typed `fallbacks` parameter.
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["extra_body"] = {"fallbacks": "default"}

        try:
            resp = self.client.beta.messages.create(**kwargs)
        except TypeError as e:
            # Raised by the SDK before any request is sent when no credential source resolves.
            if "authentication method" in str(e):
                raise LLMFatalError("no credentials found: put ANTHROPIC_API_KEY in codebase/data_generation/.env, "
                                    "export it in your shell, or run `ant auth login`") from e
            raise
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise LLMFatalError(f"credentials rejected ({e.status_code}): set ANTHROPIC_API_KEY or run `ant auth login`") from e
        except anthropic.NotFoundError as e:
            raise LLMFatalError(f"model {llm.model!r} not found or not available to this organization") from e
        except anthropic.BadRequestError as e:
            message = str(e.message)
            if "credit balance" in message.lower():
                raise LLMFatalError("the API account has no credits: add credits at console.anthropic.com "
                                    "-> Plans & Billing, then re-run (nothing was generated or billed)") from e
            hint = " -- re-run with --no-fallbacks" if llm.fallbacks and "anthropic-beta" in message else ""
            raise LLMFatalError(f"request rejected: {message}{hint}") from e
        except anthropic.RateLimitError:
            return None, {}, "rate_limited_after_retries"
        except anthropic.APIStatusError as e:
            return None, {}, f"api_error_{e.status_code}"
        except anthropic.APIConnectionError:
            return None, {}, "connection_error"

        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens or 0,
            "output_tokens": u.output_tokens or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return None, usage, f"refusal:{category}"
        if resp.stop_reason == "max_tokens":
            return None, usage, "max_tokens"
        text = next((b.text for b in resp.content if b.type == "text"), None)
        try:
            return json.loads(text), usage, resp.stop_reason
        except (TypeError, json.JSONDecodeError):
            return None, usage, "bad_json"

    def _claude_code(self, user: str, schema: type[BaseModel]) -> tuple[dict | None, dict, str]:
        """One headless Claude Code call (`claude -p`), billed to the logged-in subscription."""
        llm = self.cfg.llm
        cmd = [
            self.claude_exe, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(anthropic.transform_schema(schema.model_json_schema())),
            "--system-prompt", SYSTEM,   # replaces Claude Code's default agent prompt
            "--tools", "",               # pure generation: no file, shell, or web tools
            "--model", llm.model,
            "--effort", llm.effort,
            "--no-session-persistence",
            # Claude Code otherwise loads MCP connectors and the skill list into every call: ~30k
            # tokens of context per call in the first subscription trial, none of it needed here.
            "--strict-mcp-config",
            "--disable-slash-commands",
        ]
        try:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                                  env=self.subprocess_env, cwd=self.neutral_cwd, timeout=CLAUDE_CODE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None, {}, "timeout"

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            self._raise_if_fatal((proc.stderr or proc.stdout).strip())
            return None, {}, f"cli_error: {(proc.stderr or proc.stdout).strip()[:200]}"

        u = envelope.get("usage") or {}
        usage = {k: int(u.get(k) or 0) for k in
                 ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        usage["notional_cost_usd"] = float(envelope.get("total_cost_usd") or 0.0)

        if envelope.get("is_error"):
            message = str(envelope.get("result") or envelope.get("subtype") or "unknown error")
            self._raise_if_fatal(message)
            return None, usage, f"cli_error: {message[:200]}"
        stop = envelope.get("stop_reason") or "end_turn"
        if stop in ("refusal", "max_tokens"):
            return None, usage, stop
        payload = envelope.get("structured_output")
        if payload is None:
            try:
                payload = json.loads(envelope.get("result") or "")
            except json.JSONDecodeError:
                return None, usage, "bad_json"
        return payload, usage, stop

    @staticmethod
    def _raise_if_fatal(message: str) -> None:
        low = message.lower()
        if "usage limit" in low or "limit reached" in low or "limit will reset" in low:
            raise LLMFatalError(f"subscription usage limit reached ({message[:200]}). Re-run after it resets; "
                                "finished work is cached and will not be regenerated.")
        if "/login" in low or "not logged in" in low or "invalid api key" in low or "authentication" in low:
            raise LLMFatalError("Claude Code is not logged in to a subscription: run `claude`, then /login")
