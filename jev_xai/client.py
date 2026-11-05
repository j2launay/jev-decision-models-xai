"""Async client for Ollama's `/v1/systemone` decision endpoint, plus a blocking facade for callers
(e.g. `shap.KernelExplainer`) that need a synchronous batch function."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from typing import Any, Coroutine, TypeVar

import httpx

from .cache import ResponseCache, payload_key
from .schemas import ChoiceQuestion, DecisionResult, NoulQuestion, ScoreQuestion, validate_response

QuestionSet = dict[str, ChoiceQuestion | NoulQuestion | ScoreQuestion]
T = TypeVar("T")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


LEGACY_DIGESTS: dict[str, tuple[str, ...]] = {
    "tev1:0.8b": ("8d11b3146b7f3f4f4d5e9a64665ab2bdaf8b46e4ec42a5880b60a716ae50e3fb",),
}


class SystemOneError(RuntimeError):
    pass


class SystemOneClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        concurrency: int = 4,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        keep_alive: str = "30m",
        cache: ResponseCache | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.keep_alive = keep_alive
        self.cache = cache
        self.model_digest: str | None = None
        self._http: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = None

    async def __aenter__(self) -> "SystemOneClient":
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_s,
            limits=httpx.Limits(max_connections=self.concurrency * 2),
        )
        self._sem = asyncio.Semaphore(self.concurrency)
        self.model_digest = await self._resolve_digest()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def _resolve_digest(self) -> str:
        assert self._http is not None
        resp = await self._http.get("/api/tags")
        resp.raise_for_status()
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        for entry in resp.json().get("models", []):
            if entry.get("name") == wanted or entry.get("model") == wanted:
                return entry["digest"]
        raise SystemOneError(f"model {self.model!r} is not installed in Ollama")

    def _payload(self, state: Any, questions: QuestionSet) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": state,
            "questions": {name: q.model_dump(exclude_none=True) for name, q in questions.items()},
        }

    async def decide(self, state: Any, questions: QuestionSet, use_cache: bool = True) -> DecisionResult:
        assert self._http is not None and self._sem is not None and self.model_digest is not None
        payload = self._payload(state, questions)
        key = payload_key(self.model_digest, payload)
        if use_cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is None:
                for alt_digest in LEGACY_DIGESTS.get(self.model, ()):
                    hit = self.cache.get(payload_key(alt_digest, payload))
                    if hit is not None:
                        break
            if hit is not None:
                raw, latency = hit
                return validate_response(questions, raw, latency, cached=True)

        body = {**payload, "keep_alive": self.keep_alive}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            async with self._sem:
                start = time.perf_counter()
                try:
                    resp = await self._http.post("/v1/systemone", json=body)
                except httpx.TransportError as exc:
                    last_error = exc
                    resp = None
                latency_ms = (time.perf_counter() - start) * 1000.0
            if resp is not None:
                if resp.status_code == 200:
                    raw = resp.json()
                    result = validate_response(questions, raw, latency_ms)
                    if self.cache is not None:
                        self.cache.put(key, self.model_digest, raw, latency_ms)
                    return result
                if resp.status_code not in RETRYABLE_STATUS:
                    raise SystemOneError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                last_error = SystemOneError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            await asyncio.sleep(0.5 * 2**attempt)
        raise SystemOneError(f"request failed after {self.max_retries + 1} attempts") from last_error

    async def decide_batch(
        self, states: Sequence[Any], questions: QuestionSet, use_cache: bool = True
    ) -> list[DecisionResult]:
        return list(await asyncio.gather(*(self.decide(s, questions, use_cache) for s in states)))


class BlockingSystemOne:
    """Runs a `SystemOneClient` on a private event loop thread and exposes synchronous calls."""

    def __init__(self, client: SystemOneClient):
        self.client = client
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._run(client.__aenter__())

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def model_digest(self) -> str:
        assert self.client.model_digest is not None
        return self.client.model_digest

    def decide(self, state: Any, questions: QuestionSet, use_cache: bool = True) -> DecisionResult:
        return self._run(self.client.decide(state, questions, use_cache))

    def decide_batch(
        self, states: Sequence[Any], questions: QuestionSet, use_cache: bool = True
    ) -> list[DecisionResult]:
        return self._run(self.client.decide_batch(states, questions, use_cache))

    def close(self) -> None:
        self._run(self.client.__aexit__(None, None, None))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self) -> "BlockingSystemOne":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
