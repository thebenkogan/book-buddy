"""book-buddy: spoiler-safe reading companion.

This package owns the spoiler guarantee. The rule enforced by the tests is
that **only** :func:`src.bookbuddy.book.text_up_to` may slice ``Book.text``
into anything that reaches a prompt. Everything else in the codebase reads
book text exclusively through that function.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Default model: free, 1M context. Override with ``BOOKBUDDY_MODEL``.
DEFAULT_MODEL = "stealth/space-bunny-alpha"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

__all__ = [
    "DEFAULT_MODEL",
    "OPENROUTER_URL",
    "get_api_key",
    "get_model",
    "cache_dir",
    "chat",
    "ChatResult",
]


class ChatResult(Dict[str, Any]):
    """Plain dict result of a chat completion (content + token usage)."""


def get_model() -> str:
    """Return the model id to use for all LLM calls."""
    return os.environ.get("BOOKBUDDY_MODEL") or DEFAULT_MODEL


def get_api_key() -> str:
    """Return the OpenRouter API key, loading ``~/.hermes/.env`` if needed.

    The key is never logged and never returned to a caller-visible message.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = os.path.expanduser("~/.hermes/.env")
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip("'\"")
                    os.environ["OPENROUTER_API_KEY"] = value
                    return value
    except OSError:
        logger.warning("no OpenRouter key in env or ~/.hermes/.env")
    raise RuntimeError("OPENROUTER_API_KEY is not set")


def cache_dir() -> str:
    """Return (and create) the on-disk cache directory."""
    path = os.environ.get("BOOKBUDDY_CACHE_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cache",
    )
    os.makedirs(path, exist_ok=True)
    return path


def _load_dotenv() -> None:
    """Best-effort load of ``OPENROUTER_API_KEY`` from ``~/.hermes/.env``."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a hard dep
        return
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)


def chat(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 180.0,
    reasoning_effort: Optional[str] = None,
) -> ChatResult:
    """One OpenRouter chat completion.

    Returns ``{"content": str, "tokens_in": int, "tokens_out": int,
    "model": str}``. Raises on any transport or protocol error after logging
    it, per AGENTS.md.

    ``reasoning_effort`` matters more than it looks: the default free model is a
    reasoning model, and on a 117-section labelling job it spent a 16k token
    budget entirely on hidden reasoning and returned an EMPTY content string.
    Pass ``"low"`` for any long-output structured call.
    """
    import httpx

    _load_dotenv()
    api_key = get_api_key()
    model_id = model or get_model()
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    try:
        response = httpx.post(
            OPENROUTER_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("OpenRouter chat call failed (model=%s)", model_id)
        raise
    try:
        message = data["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError):
        logger.exception("unexpected OpenRouter response shape: %s", list(data))
        raise
    if not content:
        logger.warning(
            "OpenRouter returned EMPTY content for %s "
            "(finish_reason=%s, completion_tokens=%s); the model spent its "
            "budget on reasoning tokens. Retry with a larger max_tokens or a "
            "lower reasoning_effort.",
            model_id,
            data["choices"][0].get("finish_reason"),
            (data.get("usage") or {}).get("completion_tokens"),
        )
    usage = data.get("usage") or {}
    return ChatResult(
        content=content or "",
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        model=model_id,
    )


def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON object/array out of a model response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.rsplit("```", 1)[0]
    start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    end = text.rfind(closer)
    if end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
