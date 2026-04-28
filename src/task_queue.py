"""
GodDev 3.0 — Task Queue & File Lock Registry
=============================================

Solves two critical production problems:

1. FILE CONFLICT: Multiple squads assigning different agents to write the
   SAME file (e.g. index.html). Last writer wins → broken app.
   Solution: Global file lock registry. First squad to claim a file owns it.
   All others are redirected to complementary files.

2. THUNDERING HERD: All N workers fire simultaneously → N × 429s.
   Solution: Token-bucket rate limiter per provider + staggered dispatch.
   Workers are admitted into a per-role async semaphore with a minimum
   interval between acquisitions (token refill rate).
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from typing import Optional


# ── File Lock Registry ────────────────────────────────────────────────────────

class FileLockRegistry:
    """
    Thread-safe registry that prevents two workers from writing the same file.

    Usage:
        registry = get_file_registry()
        if registry.claim(job_id, "/path/to/index.html", "frontend-worker-1"):
            # OK — this worker owns this file
            ...
        else:
            owner = registry.get_owner(job_id, "/path/to/index.html")
            # Skip — another worker already claimed it
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # job_id → {file_path: worker_id}
        self._claims: dict[str, dict[str, str]] = defaultdict(dict)

    def claim(self, job_id: str, file_path: str, worker_id: str) -> bool:
        """Try to claim file_path for worker_id. Returns True if claimed."""
        normalized = file_path.rstrip("/").lower()
        with self._lock:
            job_claims = self._claims[job_id]
            if normalized in job_claims:
                return False  # Already claimed by another worker
            job_claims[normalized] = worker_id
            return True

    def get_owner(self, job_id: str, file_path: str) -> Optional[str]:
        normalized = file_path.rstrip("/").lower()
        with self._lock:
            return self._claims[job_id].get(normalized)

    def is_claimed(self, job_id: str, file_path: str) -> bool:
        normalized = file_path.rstrip("/").lower()
        with self._lock:
            return normalized in self._claims[job_id]

    def release(self, job_id: str, file_path: str) -> None:
        """Release a file lock (e.g., after a failed write)."""
        normalized = file_path.rstrip("/").lower()
        with self._lock:
            self._claims[job_id].pop(normalized, None)

    def clear_job(self, job_id: str) -> None:
        with self._lock:
            self._claims.pop(job_id, None)

    def snapshot(self, job_id: str) -> dict[str, str]:
        with self._lock:
            return dict(self._claims.get(job_id, {}))


# ── Token Bucket Rate Limiter ──────────────────────────────────────────────────

class TokenBucket:
    """
    Classic token-bucket: refills at `rate` tokens/second up to `capacity`.
    Consumers call `acquire(tokens=1)` which blocks until a token is available.

    This prevents the thundering herd: workers share a per-role bucket, so
    they naturally space out their API calls without explicit sleep() hacks.
    """

    def __init__(self, rate: float, capacity: int) -> None:
        """
        rate     — tokens added per second (= max sustained RPS)
        capacity — max burst size
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Blocks asynchronously until `tokens` are available in the bucket, then consumes them.

        This method handles token refill based on the configured rate and capacity.
        It will wait if insufficient tokens are available, preventing bursts beyond capacity
        and ensuring calls are spaced according to the rate.

        Args:
            tokens: The number of tokens to acquire. Defaults to 1.0.

        Returns:
            None. The method only returns once the tokens have been successfully acquired.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Need to wait — calculate how long
                deficit = tokens - self._tokens
                wait = deficit / self.rate
                await asyncio.sleep(wait)


# ── Per-role rate limiter map ─────────────────────────────────────────────────
# Conservative defaults — LiteLLM router handles fallbacks when these are hit.
# rate = sustainable requests/second (burst=capacity at start, then sustained)
#
# Claude Opus:  ~2 RPM free / ~60 RPM paid → use 0.5 RPS (1 per 2s)
# GPT-4o:       ~500 RPM paid              → use 2 RPS
# Gemini Flash: ~15 RPM free / 1000 paid   → use 0.3 RPS (1 per 3s)
# DeepSeek:     generous                   → use 3 RPS

_ROLE_BUCKETS: dict[str, TokenBucket] = {
    # Claude Opus 4.7: paid account, ~1000 RPM available
    # But we still want to stagger — use 0.8 RPS (1 per 1.25s)
    "architect":  TokenBucket(rate=0.8, capacity=3),   # Claude primary

    # GPT-4.1: very generous limits, ~10k RPM paid → 3 RPS sustained
    "backend":    TokenBucket(rate=3.0, capacity=8),   # GPT-4.1 primary

    # Gemini 2.5 Flash: use sparingly due to 0-byte issues; go slow
    "frontend":   TokenBucket(rate=0.5, capacity=2),   # Gemini primary (cautious)

    # DeepSeek V3: very generous API
    "devops":     TokenBucket(rate=3.0, capacity=8),   # DeepSeek primary
    "qa":         TokenBucket(rate=3.0, capacity=8),   # DeepSeek primary

    # Critic: GPT-4.1 primary + Claude for security. 3 critics run parallel.
    # max_concurrency=6 at graph level means they don't all fire simultaneously.
    "critic":     TokenBucket(rate=2.0, capacity=5),

    # Sequential roles (one at a time by design)
    "cto":        TokenBucket(rate=0.8, capacity=1),   # Claude sequential
    "integrator": TokenBucket(rate=0.8, capacity=1),   # Claude sequential

    # Squad lead: GPT-4.1, typically 3-5 parallel at once
    "squad_lead": TokenBucket(rate=2.0, capacity=5),

    # Self-improvement workers: Claude/OpenAI, up to 6 tasks
    "self-improvement": TokenBucket(rate=1.0, capacity=4),

    # Meta CTO (self-improvement strategist): sequential, modest rate
    "meta_cto":   TokenBucket(rate=0.8, capacity=2),

    # ── Cheap pools (DeepSeek + Gemini Flash) ── generous RPS, large bursts.
    # Used when complexity == "trivial". Multiple deployments behind each role.
    "architect_cheap":  TokenBucket(rate=4.0, capacity=12),
    "backend_cheap":    TokenBucket(rate=4.0, capacity=12),
    "frontend_cheap":   TokenBucket(rate=4.0, capacity=12),
    "devops_cheap":     TokenBucket(rate=4.0, capacity=12),
    "qa_cheap":         TokenBucket(rate=4.0, capacity=12),
    "integrator_cheap": TokenBucket(rate=4.0, capacity=12),
    "squad_lead_cheap": TokenBucket(rate=4.0, capacity=12),
    "critic_cheap":     TokenBucket(rate=4.0, capacity=12),
    "meta_cto_cheap":   TokenBucket(rate=4.0, capacity=12),
    "cto_cheap":        TokenBucket(rate=4.0, capacity=12),

    # Compressor: cheap, runs inline before expensive calls.
    "compressor":       TokenBucket(rate=5.0, capacity=15),
}


async def acquire_slot(role: str) -> None:
    """Acquire a rate-limit token for the given role before making an LLM call."""
    bucket = _ROLE_BUCKETS.get(role, _ROLE_BUCKETS["backend"])
    await bucket.acquire()


# ── Singletons ────────────────────────────────────────────────────────────────

_registry: Optional[FileLockRegistry] = None
_registry_lock = threading.Lock()


def get_file_registry() -> FileLockRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = FileLockRegistry()
    return _registry