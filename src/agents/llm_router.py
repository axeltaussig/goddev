"""
GodDev 3.0 — Unified LLM Router  (v2 — 2025 State-of-the-Art)
===============================================================

Models updated to latest 2025 state-of-the-art:
  Claude Opus 4.7     — best reasoning + coding (Anthropic flagship, Apr 2025)
  GPT-4.1             — best OpenAI coding model, 54% better than GPT-4o on SWE-bench
  Gemini 2.5 Flash    — Google's best price/performance (2.0 Flash is DEPRECATED)
  DeepSeek-chat (V3.2)— best open-weight coding, 128K context, near-GPT4 at 1/10 cost

Role → Primary → Fallback 1 → Fallback 2
─────────────────────────────────────────
architect  → Claude Opus 4.7  → GPT-4.1      → DeepSeek V3
backend    → GPT-4.1          → Claude Opus   → DeepSeek V3
frontend   → Gemini 2.5 Flash → GPT-4.1      → DeepSeek V3
devops     → DeepSeek V3      → GPT-4.1      (cheap, reliable)
qa         → DeepSeek V3      → GPT-4.1      → Claude Opus
critic     → GPT-4.1          → DeepSeek V3  (structured JSON output)
cto        → Claude Opus 4.7  → GPT-4.1      (sequential, strategic)
integrator → Claude Opus 4.7  → GPT-4.1      (synthesis, analysis)
squad_lead → GPT-4.1          → DeepSeek V3  (reliable JSON)

Rate limit budgets (conservative, paid tier):
  Claude Opus 4.7  : 2 parallel / 60s cooldown  (expensive, ~$15/MTok)
  GPT-4.1          : 6 parallel / 30s cooldown  (~$2/MTok)
  Gemini 2.5 Flash : 3 parallel / 60s cooldown  (tier-dependent)
  DeepSeek V3.2    : 8 parallel / 20s cooldown  (very generous, ~$0.27/MTok)
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import litellm
from litellm import Router

# Suppress verbose litellm logging
litellm.suppress_debug_info = True
litellm.set_verbose = False

# ── Provider health state (for UI status indicators) ─────────────────────────
# Updated by _track_health() on each call outcome
_provider_health: dict[str, dict] = {
    "claude":    {"status": "unknown", "last_ok": 0.0, "last_error": "", "calls": 0},
    "openai":    {"status": "unknown", "last_ok": 0.0, "last_error": "", "calls": 0},
    "gemini":    {"status": "unknown", "last_ok": 0.0, "last_error": "", "calls": 0},
    "deepseek":  {"status": "unknown", "last_ok": 0.0, "last_error": "", "calls": 0},
}

def _provider_from_model(model: str) -> str:
    if "anthropic" in model or "claude" in model:
        return "claude"
    if "gemini" in model:
        return "gemini"
    if "deepseek" in model:
        return "deepseek"
    return "openai"

def _track_health(model: str, success: bool, error: str = "") -> None:
    p = _provider_from_model(model)
    h = _provider_health.get(p, {})
    h["calls"] = h.get("calls", 0) + 1
    if success:
        h["status"] = "ok"
        h["last_ok"] = time.time()
        h["last_error"] = ""
    else:
        h["last_error"] = error[:120]
        h["status"] = "error" if "rate" not in error.lower() else "rate_limited"
    _provider_health[p] = h


def _build_router() -> Router:
    """Build LiteLLM Router — 2025 model strings, proper fallbacks, circuit breakers."""

    anthropic_key   = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key      = os.getenv("OPENAI_API_KEY", "")
    deepseek_key    = os.getenv("DEEPSEEK_API_KEY", "")
    gemini_key      = os.getenv("GOOGLE_API_KEY", "")
    anthropic_model = os.getenv("ANTHROPIC_MODEL",  "claude-opus-4-7")
    openai_model    = os.getenv("OPENAI_MODEL",     "gpt-4.1")
    deepseek_model  = os.getenv("DEEPSEEK_MODEL",   "deepseek-chat")
    gemini_model    = os.getenv("GEMINI_MODEL",     "gemini-2.5-flash")
    deepseek_base   = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # Mark providers as configured
    if anthropic_key:
        _provider_health["claude"]["status"]  = "configured"
    if openai_key:
        _provider_health["openai"]["status"]  = "configured"
    if gemini_key:
        _provider_health["gemini"]["status"]  = "configured"
    if deepseek_key:
        _provider_health["deepseek"]["status"] = "configured"

    def claude(order: int, parallel: int, tokens: int = 16000, desc: str = "") -> dict:
        return {
            "litellm_params": {
                "model": f"anthropic/{anthropic_model}",
                "api_key": anthropic_key,
                "max_tokens": tokens,
            },
            "model_info": {
                "order": order, "max_parallel_requests": parallel,
                "provider": "claude", "description": desc or f"Claude Opus 4.7 (order={order})",
            },
        }

    def openai(order: int, parallel: int, tokens: int = 16000, desc: str = "") -> dict:
        return {
            "litellm_params": {
                "model": f"openai/{openai_model}",
                "api_key": openai_key,
                "max_tokens": tokens,
                "temperature": 0,
            },
            "model_info": {
                "order": order, "max_parallel_requests": parallel,
                "provider": "openai", "description": desc or f"GPT-4.1 (order={order})",
            },
        }

    def gemini(order: int, parallel: int, tokens: int = 8192, desc: str = "") -> dict:
        return {
            "litellm_params": {
                "model": f"gemini/{gemini_model}",
                "api_key": gemini_key,
                "max_tokens": tokens,
            },
            "model_info": {
                "order": order, "max_parallel_requests": parallel,
                "provider": "gemini", "description": desc or f"Gemini 2.5 Flash (order={order})",
            },
        }

    def deepseek(order: int, parallel: int, tokens: int = 8192, desc: str = "") -> dict:
        return {
            "litellm_params": {
                "model": f"openai/{deepseek_model}",
                "api_key": deepseek_key,
                "api_base": deepseek_base,
                "max_tokens": tokens,
                "temperature": 0,
            },
            "model_info": {
                "order": order, "max_parallel_requests": parallel,
                "provider": "deepseek", "description": desc or f"DeepSeek V3.2 (order={order})",
            },
        }

    roles: dict[str, list[dict]] = {}

    # ── architect: Claude → GPT-4.1 ──────────────────────────────────────────
    # Claude excels at architecture, interfaces, security analysis
    if anthropic_key:
        roles.setdefault("architect", []).append({"model_name": "architect", **claude(1, 2, desc="Claude Opus 4.7 — architecture, interfaces, security")})
    if openai_key:
        roles.setdefault("architect", []).append({"model_name": "architect", **openai(2, 5, desc="GPT-4.1 — architect fallback")})

    # ── backend: GPT-4.1 → Claude → DeepSeek ─────────────────────────────────
    # GPT-4.1: 54% better at SWE-bench vs GPT-4o, 1M token context
    if openai_key:
        roles.setdefault("backend", []).append({"model_name": "backend", **openai(1, 5, desc="GPT-4.1 — APIs, server code, business logic")})
    if anthropic_key:
        roles.setdefault("backend", []).append({"model_name": "backend", **claude(2, 2, desc="Claude Opus 4.7 — backend fallback")})
    if deepseek_key:
        roles.setdefault("backend", []).append({"model_name": "backend", **deepseek(3, 7, desc="DeepSeek V3.2 — backend last resort")})

    # ── frontend: GPT-4.1 → Claude → DeepSeek ────────────────────────────────
    if openai_key:
        roles.setdefault("frontend", []).append({"model_name": "frontend", **openai(1, 4, desc="GPT-4.1 — HTML/CSS/JS, canvas, animations")})
    if anthropic_key:
        roles.setdefault("frontend", []).append({"model_name": "frontend", **claude(2, 2, desc="Claude Opus 4.7 — frontend fallback")})
    if deepseek_key:
        roles.setdefault("frontend", []).append({"model_name": "frontend", **deepseek(3, 7, desc="DeepSeek V3.2 — frontend last resort")})

    # ── devops: DeepSeek → GPT-4.1 ───────────────────────────────────────────
    # DeepSeek V3.2: near-GPT4 at 1/10th cost, great for Dockerfiles/CI/CD
    if deepseek_key:
        roles.setdefault("devops", []).append({"model_name": "devops", **deepseek(1, 7, tokens=4096, desc="DeepSeek V3.2 — Dockerfiles, CI/CD, nginx, shell")})
    if openai_key:
        roles.setdefault("devops", []).append({"model_name": "devops", **openai(2, 5, tokens=4096, desc="GPT-4.1 — devops fallback")})

    # ── qa: Gemini → DeepSeek → GPT-4.1 ─────────────────────────────────────
    if gemini_key:
        roles.setdefault("qa", []).append({"model_name": "qa", **gemini(1, 3, desc="Gemini — test suites, assertions")})
    if deepseek_key:
        roles.setdefault("qa", []).append({"model_name": "qa", **deepseek(2, 7, desc="DeepSeek V3.2 — tests, algorithms, edge cases fallback")})
    if openai_key:
        roles.setdefault("qa", []).append({"model_name": "qa", **openai(3, 5, desc="GPT-4.1 — QA last resort")})

    # ── critic: GPT-4.1 → DeepSeek ───────────────────────────────────────────
    # GPT-4.1 is highly reliable for structured JSON output
    if openai_key:
        roles.setdefault("critic", []).append({"model_name": "critic", **openai(1, 5, tokens=4096, desc="GPT-4.1 — structured critic verdicts")})
    if deepseek_key:
        roles.setdefault("critic", []).append({"model_name": "critic", **deepseek(2, 7, tokens=4096, desc="DeepSeek V3.2 — critic fallback")})

    # ── cto: Claude → GPT-4.1 ────────────────────────────────────────────────
    if anthropic_key:
        roles.setdefault("cto", []).append({"model_name": "cto", **claude(1, 2, desc="Claude Opus 4.7 — CTO strategic planning, vision")})
    if openai_key:
        roles.setdefault("cto", []).append({"model_name": "cto", **openai(2, 4, desc="GPT-4.1 — CTO fallback")})

    # ── integrator: Gemini → Claude → GPT-4.1 ──────────────────────────────
    if gemini_key:
        roles.setdefault("integrator", []).append({"model_name": "integrator", **gemini(1, 3, desc="Gemini — 1M token context integration analysis and synthesis")})
    if anthropic_key:
        roles.setdefault("integrator", []).append({"model_name": "integrator", **claude(2, 2, desc="Claude Opus 4.7 — integrator fallback")})
    if openai_key:
        roles.setdefault("integrator", []).append({"model_name": "integrator", **openai(3, 4, desc="GPT-4.1 — integrator last resort")})

    # ── squad_lead: GPT-4.1 → DeepSeek ──────────────────────────────────────
    if openai_key:
        roles.setdefault("squad_lead", []).append({"model_name": "squad_lead", **openai(1, 5, tokens=8192, desc="GPT-4.1 — squad plan JSON output")})
    if deepseek_key:
        roles.setdefault("squad_lead", []).append({"model_name": "squad_lead", **deepseek(2, 7, tokens=8192, desc="DeepSeek V3.2 — squad lead fallback")})

    model_list = [item for items in roles.values() for item in items]
    if not model_list:
        raise RuntimeError("No API keys configured — set at least OPENAI_API_KEY")

    router = Router(
        model_list=model_list,
        num_retries=3,
        retry_after=5,
        allowed_fails=2,        # cool down after 2 consecutive fails
        cooldown_time=60,       # 60s cooldown before retrying failed deployment
        routing_strategy="least-busy",
        timeout=120,
        set_verbose=False,
    )
    return router


# ── Singleton ─────────────────────────────────────────────────────────────────
_router: Router | None = None

def get_router() -> Router:
    global _router
    if _router is None:
        _router = _build_router()
    return _router


async def acall(role: str, messages: list[dict], **kwargs) -> str:
    """
    Async LLM call with automatic fallback, circuit breaker, and health tracking.
    Any 429/overload triggers LiteLLM's built-in cooldown + retry to next deployment.
    """
    router = get_router()
    model_used = role  # track which model actually handled the call

    try:
        response = await router.acompletion(model=role, messages=messages, **kwargs)
        content = response.choices[0].message.content or ""

        # Track which actual model responded (for health panel)
        try:
            model_used = response.model or role
        except Exception:
            pass
        _track_health(model_used, success=True)
        return content.strip()

    except Exception as exc:
        _track_health(model_used, success=False, error=str(exc))
        raise


def get_model_status() -> list[dict]:
    """Return deployment list for the /models API endpoint."""
    router = get_router()
    statuses = []
    for m in router.model_list:
        params = m.get("litellm_params", {})
        info   = m.get("model_info", {})
        statuses.append({
            "role":        m.get("model_name", "?"),
            "model":       params.get("model", "?"),
            "order":       info.get("order", 1),
            "parallel":    info.get("max_parallel_requests", "?"),
            "provider":    info.get("provider", "unknown"),
            "description": info.get("description", ""),
        })
    return statuses


def get_provider_health() -> dict:
    """Return live health state for all 4 providers — used by /health/models."""
    now = time.time()
    result = {}
    for provider, h in _provider_health.items():
        status = h.get("status", "unknown")
        last_ok = h.get("last_ok", 0.0)
        age_ok = now - last_ok if last_ok > 0 else None
        result[provider] = {
            "status":      status,               # ok | error | rate_limited | configured | unknown
            "last_ok_ago": round(age_ok, 1) if age_ok is not None else None,
            "last_error":  h.get("last_error", ""),
            "total_calls": h.get("calls", 0),
        }
    return result
