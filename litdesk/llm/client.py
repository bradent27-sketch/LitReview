"""LLM client wrapper with two provider backends. Every function in this
package treats a missing SDK, missing credentials, a missing `claude` CLI,
or a failed request as "skip this," never a hard failure — per spec: "Make
the whole LLM layer skippable with a flag."

"api": the Anthropic SDK, billed per token against ANTHROPIC_API_KEY.
"claude_code": shells out to `claude -p` per request, billed against your
Claude Code subscription instead of a metered key. ClaudeCodeClient exposes
the same `.messages.create(...) -> response.content[i].text` shape as
anthropic.Anthropic(), so summarize.py's existing try/except around every
`.create()` call already handles claude_code failures identically to API
errors — nothing downstream needs to know which provider it's talking to.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess

from litdesk.config import Config

logger = logging.getLogger("litdesk.llm")


def get_client(cfg: Config):
    """Returns an object exposing `.messages.create(model=, messages=)`, or
    None if unavailable for any reason. Never raises."""
    if cfg.llm.provider == "claude_code":
        return _get_claude_code_client(cfg)
    if cfg.llm.provider != "api":
        logger.warning(
            "llm.provider=%r isn't implemented (use 'api' or 'claude_code') — skipping LLM calls this run",
            cfg.llm.provider,
        )
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("`anthropic` package not installed — run `uv pip install -e '.[llm]'` to enable summaries")
        return None
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 - optional feature, must never break the digest run
        logger.warning("Could not create Anthropic client (no credentials?): %s", exc)
        return None


def _get_claude_code_client(cfg: Config) -> ClaudeCodeClient | None:
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        logger.warning("llm.provider='claude_code' but no `claude` CLI found on PATH — skipping LLM calls this run")
        return None
    return ClaudeCodeClient(claude_bin, cfg.llm.claude_code_timeout_seconds)


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ClaudeCodeResponse:
    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class _ClaudeCodeMessages:
    def __init__(self, claude_bin: str, timeout_seconds: int):
        self._claude_bin = claude_bin
        self._timeout = timeout_seconds

    def create(self, *, model: str, messages: list[dict], **_ignored):
        prompt = messages[-1]["content"]

        # No documented flag fully disables tool use in `-p` mode. TLDR/
        # synthesis prompts are pure text-summarization requests with
        # nothing to act on, so this is a low-risk gap, not one worth
        # engineering around blind.
        result = subprocess.run(
            [self._claude_bin, "-p", prompt, "--model", model, "--output-format", "json"],
            capture_output=True, text=True, timeout=self._timeout,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"claude -p produced non-JSON output (exit {result.returncode}): {result.stderr[:300]}"
            ) from exc

        if result.returncode != 0 or payload.get("is_error"):
            raise RuntimeError(f"claude -p failed: {payload.get('result') or result.stderr[:300]}")

        text = payload.get("result")
        if not text:
            raise RuntimeError("claude -p returned an empty result")
        return _ClaudeCodeResponse(text)


class ClaudeCodeClient:
    """Drop-in stand-in for anthropic.Anthropic() that bills through the
    `claude` CLI's headless `-p` mode (your Claude Code subscription)
    instead of a metered API key. One subprocess per `.messages.create()`
    call — there's no persistent connection to hold open."""

    def __init__(self, claude_bin: str, timeout_seconds: int):
        self.messages = _ClaudeCodeMessages(claude_bin, timeout_seconds)
