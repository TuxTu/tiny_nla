"""Completion provider backends for API labeling.

Stage 2 calls an external LLM to produce natural-language explanations of
source text. `CompletionProvider` is the pluggable interface: submit a batch
of fully-formed prompts, get back a batch of completions (or None for prompts
that exhausted retries). Concurrency, retries, rate limits, and auth are all
the provider's problem.

Swap via `--provider` CLI flag (deepseek / anthropic).
"""

import asyncio
import os
from abc import ABC, abstractmethod

import httpx


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Returns `prompts[i] -> completion[i]` (or None for prompts that exhausted
    retries). None returns are per-prompt gave-up signals — the caller drops
    those rows.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class DeepSeekProvider(CompletionProvider):
    """DeepSeek API via OpenAI-compatible /v1/chat/completions endpoint.

    Uses httpx.AsyncClient for HTTP transport — no `openai` SDK dependency.
    Semaphore-based concurrency with exponential backoff on 429/5xx.
    """

    _BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        max_tokens: int = 400,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY env var "
                "or pass api_key= to the constructor."
            )

    async def _one(
        self, sem: asyncio.Semaphore, client: httpx.AsyncClient, prompt: str
    ) -> str | None:
        async with sem:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "thinking": {"type": "disabled"},
                        },
                    )
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else 2 ** attempt
                        await asyncio.sleep(wait)
                        continue
                    if resp.status_code >= 500:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    if not content:
                        return None
                    return content
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500 or e.response.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
                except (
                    httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError,
                    httpx.ProxyError,
                ):
                    await asyncio.sleep(2 ** attempt)
                    continue
            return None

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None]:
            sem = asyncio.Semaphore(self.concurrency)
            async with httpx.AsyncClient(
                base_url=self._BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0),
                proxy=None,  # ignore HTTP_PROXY / HTTPS_PROXY env vars
            ) as client:
                raw = await asyncio.gather(
                    *(self._one(sem, client, p) for p in prompts),
                    return_exceptions=True,
                )
                out: list[str | None] = []
                for r in raw:
                    if isinstance(r, str):
                        out.append(r)
                    elif isinstance(r, BaseException):
                        print(f"  [DeepSeekProvider] request failed: {r}")
                        out.append(None)
                    else:
                        out.append(r)  # includes None
                return out

        return asyncio.run(_run())


class AnthropicProvider(CompletionProvider):
    """Anthropic Messages API with bounded async concurrency.

    Uses the official `anthropic` SDK for transport. Per-prompt failures after
    exhausting retries return None — caller drops those rows.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 400,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self._TOLERATED = (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
        )

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        if resp.stop_reason == "refusal":
            return None
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r}"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion"
        return text

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None]:
            sem = asyncio.Semaphore(self.concurrency)
            raw = await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )
            out: list[str | None] = []
            n_failed = 0
            n_refused = 0
            for i, r in enumerate(raw):
                if isinstance(r, str):
                    out.append(r)
                elif r is None:
                    n_refused += 1
                    out.append(None)
                elif isinstance(r, self._TOLERATED):
                    n_failed += 1
                    out.append(None)
                elif isinstance(r, BaseException):
                    raise r
                else:
                    raise AssertionError(
                        f"gather returned unexpected type at [{i}]: {type(r).__name__}"
                    )
            if n_failed or n_refused:
                print(
                    f"  [AnthropicProvider] dropped {n_refused} refused"
                    f" + {n_failed} retry-exhausted of {len(prompts)}"
                )
            return out

        return asyncio.run(_run())


def resolve_provider(name: str, kwargs: dict | None = None) -> CompletionProvider:
    """Resolve a provider by name: 'deepseek' or 'anthropic'.

    Additional kwargs are forwarded to the provider constructor.
    """
    kwargs = kwargs or {}
    if name == "deepseek":
        return DeepSeekProvider(**kwargs)
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    raise ValueError(
        f"unknown provider: {name!r} (expected 'deepseek' or 'anthropic')"
    )
