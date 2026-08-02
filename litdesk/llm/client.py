"""Thin Anthropic API client wrapper. Every function in this package treats
a missing SDK, a missing key, or a failed request as "skip this," never a
hard failure — per spec: "Make the whole LLM layer skippable with a flag."
"""

from __future__ import annotations

import logging

from litdesk.config import Config

logger = logging.getLogger("litdesk.llm")


def get_client(cfg: Config):
    """Returns an anthropic.Anthropic() client, or None if unavailable for
    any reason (SDK not installed, no credentials, unimplemented provider).
    Never raises."""
    if cfg.llm.provider != "api":
        logger.warning(
            "llm.provider=%r isn't implemented (only 'api' is) — skipping LLM calls this run",
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
