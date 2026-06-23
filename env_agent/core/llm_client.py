"""Thin LLM client for the env agent.

Reuses SPIKE's existing LLM proxy configuration (``agent/conf/gemini_config.json``)
which exposes an OpenAI-compatible endpoint via a LiteLLM proxy.  The client:

  * loads model / base_url / api-key-env from a SPIKE-style json config,
  * reads the key from the process env or ``env/.env``,
  * exposes ``chat`` and ``chat_json`` helpers,
  * degrades gracefully: if no key / no network / ``openai`` missing, it runs in
    ``offline`` mode and raises a clear error only when a call is attempted,
    so the rest of the pipeline (analysis, retrieval) still works.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional


class LLMUnavailable(RuntimeError):
    """Raised when an LLM call is attempted but no backend is configured."""


def _load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class LLMClient:
    def __init__(
        self,
        config_path: str | os.PathLike,
        *,
        dotenv_path: Optional[str | os.PathLike] = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.model: str = cfg.get("comp_model", "gpt-4o-mini")
        self.base_url: Optional[str] = cfg.get("base_url")
        self.key_var: str = cfg.get("key_var", "OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # resolve key: env first, then .env file
        self.api_key = os.environ.get(self.key_var, "")
        if not self.api_key and dotenv_path:
            self.api_key = _load_dotenv(Path(dotenv_path)).get(self.key_var, "")
        # litellm proxies usually accept any non-empty token
        self._client = None
        self._init_error: Optional[str] = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            self._init_error = f"openai package not importable: {e}"
            return
        if not self.base_url and not self.api_key:
            self._init_error = (
                f"no base_url and no api key (${self.key_var}); set the key or "
                "configure a base_url proxy"
            )
            return
        try:
            self._client = OpenAI(
                api_key=self.api_key or "sk-noauth",
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except Exception as e:  # pragma: no cover
            self._init_error = f"failed to construct OpenAI client: {e}"

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self._client is None:
            raise LLMUnavailable(self._init_error or "LLM client not initialised")
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
    ) -> Any:
        """Chat and parse a JSON object/array out of the reply (robust to fences)."""
        content = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return extract_json(content)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM reply."""
    text = text.strip()
    # strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # fall back to the first balanced { } or [ ] block
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    raise ValueError(f"could not parse JSON from LLM reply:\n{text[:500]}")
