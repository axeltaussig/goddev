"""
GodDev 3.5 — Unified LLM Router — Multi-Tier Workforce Architecture
=====================================================================

Each role is a "swarm" with multiple tiers:
  ● Lead  — highest quality, strategic thinking
  ↪ Worker — high volume, cost-effective execution
  ↪ Fallback — safety net for edge cases

22 deployments across 4 providers with auto-fallback and circuit breakers.

Model capabilities:
  Claude Opus 4.7  — Literary logic, lowest hallucination, expert codebase logic
  GPT-4.1          — Best structural adherence, JSON schemas, backend logic
  Gemini 2.5 Flash — Massive 2M token context, visual synthesis
  DeepSeek Chat    — Hyper-logic, 20-50x cheaper than OpenAI
  Gemini 1.5 Pro  — Legacy 2M context, research & history
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import Router
from .cost_tracker import get_tracker
from . import engine_telemetry

# ── Model Profiles: cost ($/1M tok) + power score (1-10) per deployment ─────
# Power score is a holistic intelligence rating used for routing decisions.
# Power-per-dollar = power / blended_cost — drives the "cost/power" intelligence.
MODEL_PROFILES: dict[str, dict[str, float]] = {
    # provider/model_id      cost_in  cost_out  power  speed   (Apr 2026 verified)
    # ── Anthropic ─────────────────────────────────────────────
    "claude-opus-4-7":       {"cost_in": 15.00, "cost_out": 75.00, "power": 10, "speed": 2},
    "claude-opus-4-5":       {"cost_in": 15.00, "cost_out": 75.00, "power": 10, "speed": 2},
    "claude-sonnet-4-5":     {"cost_in":  3.00, "cost_out": 15.00, "power":  9, "speed": 5},
    "claude-3-5-sonnet":     {"cost_in":  3.00, "cost_out": 15.00, "power":  9, "speed": 5},
    "claude-haiku-4-5":      {"cost_in":  0.80, "cost_out":  4.00, "power":  7, "speed": 8},
    "claude-3-5-haiku":      {"cost_in":  0.80, "cost_out":  4.00, "power":  7, "speed": 8},
    # ── OpenAI ────────────────────────────────────────────────
    "gpt-4.1-nano":          {"cost_in":  0.10, "cost_out":  0.40, "power":  6, "speed": 9},
    "gpt-4.1-mini":          {"cost_in":  0.40, "cost_out":  1.60, "power":  7, "speed": 8},
    "gpt-4.1":               {"cost_in":  2.00, "cost_out":  8.00, "power":  9, "speed": 4},
    "gpt-4o-mini":           {"cost_in":  0.15, "cost_out":  0.60, "power":  6, "speed": 9},
    "gpt-4o":                {"cost_in":  2.50, "cost_out": 10.00, "power":  8, "speed": 6},
    # ── Google ────────────────────────────────────────────────
    "gemini-2.5-pro":        {"cost_in":  1.25, "cost_out": 10.00, "power":  9, "speed": 5},
    "gemini-2.5-flash":      {"cost_in":  0.075,"cost_out":  0.30, "power":  7, "speed": 10},
    "gemini-1.5-pro":        {"cost_in":  1.25, "cost_out":  5.00, "power":  8, "speed": 5},
    # ── DeepSeek ──────────────────────────────────────────────
    "deepseek-coder":        {"cost_in":  0.14, "cost_out":  0.28, "power":  7, "speed": 8},
    "deepseek-chat":         {"cost_in":  0.27, "cost_out":  1.10, "power":  7, "speed": 7},
}
_DEFAULT_PROFILE = {"cost_in": 3.0, "cost_out": 15.0, "power": 7, "speed": 5}


def get_profile(model: str) -> dict[str, float]:
    m = (model or "").lower()
    for key, prof in MODEL_PROFILES.items():
        if key in m:
            return prof
    return _DEFAULT_PROFILE


def power_per_dollar(model: str, in_out_ratio: float = 4.0) -> float:
    """Higher = better value. Assumes typical 4:1 input:output ratio for code gen."""
    p = get_profile(model)
    blended = (p["cost_in"] * in_out_ratio + p["cost_out"]) / (in_out_ratio + 1)
    return p["power"] / max(blended, 0.001)

# Suppress verbose litellm logging
litellm.suppress_debug_info = True
litellm.set_verbose = False

# ── Singleton router ─────────────────────────────────────────────────────────
_router: Router | None = None

# ── Provider health state (for UI status indicators) ─────────────────────────
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
    """Build the multi-tier workforce router — 22 deployments across 4 providers."""

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

    def gemini_flash(order: int, parallel: int, tokens: int = 8192, desc: str = "") -> dict:
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

    def gemini_pro(order: int, parallel: int, tokens: int = 8192, desc: str = "") -> dict:
        """Gemini 1.5 Pro — 2M context window for research & history"""
        return {
            "litellm_params": {
                "model": f"gemini/gemini-1.5-pro",
                "api_key": gemini_key,
                "max_tokens": tokens,
            },
            "model_info": {
                "order": order, "max_parallel_requests": parallel,
                "provider": "gemini", "description": desc or f"Gemini 1.5 Pro (order={order})",
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
                "provider": "deepseek", "description": desc or f"DeepSeek Chat (order={order})",
            },
        }

    def deepseek_coder(order: int, parallel: int, tokens: int = 8192, desc: str = "") -> dict:
        """DeepSeek-Coder: code-specialist, $0.14/$0.28 — ~50% cheaper than DeepSeek-Chat for code."""
        return {
            "litellm_params": {
                "model": "openai/deepseek-coder",
                "api_key": deepseek_key,
                "api_base": deepseek_base,
                "max_tokens": tokens,
                "temperature": 0,
            },
            "model_info": {
                "order": order, "max_parallel_requests": parallel,
                "provider": "deepseek",
                "description": desc or f"DeepSeek Coder (order={order})",
            },
        }

    def openai_mini(order: int, parallel: int, tokens: int = 8192, desc: str = "") -> dict:
        """GPT-4o-mini: $0.15/$0.60 — 17× cheaper than GPT-4.1, great for structured / orchestration."""
        return {
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": openai_key,
                "max_tokens": tokens,
                "temperature": 0,
            },
            "model_info": {
                "order": order, "max_parallel_requests": parallel,
                "provider": "openai",
                "description": desc or f"GPT-4o-mini (order={order})",
            },
        }

    roles: dict[str, list[dict]] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # PARALLEL TASK FORCES — Multi-engine primary pools per role.
    # ═══════════════════════════════════════════════════════════════════════════
    # Every flagship role gets 3-4 DIFFERENT engines at order=1 (primary).
    # LiteLLM's least-busy strategy round-robins across them, creating a
    # self-assembling task force that adapts to load + provider health.
    # Swarm telemetry tracks pheromone scores per (role, affinity, engine)
    # so the system learns which engine works best for each task type.
    #
    # Design principles:
    #   ⚡ DIVERSITY — different engines bring different strengths
    #   ⚡ CLAUDE-SMART — Claude is the brain, not the hands. Pre-compile context
    #     with cheap models, then let Claude make strategic decisions.
    #     Claude at order=1 for strategic roles, order=2 for code generation.
    #   ⚡ SPEED — Gemini Flash (10ms TTFT) in every pool
    #   ⚡ CODE-SPECIALIST — DeepSeek Coder in every code-producing pool

    # ═══ Architect (The Visionary Swarm) ══════════════════════════════════════
    # Claude: strategic decisions (system design). GPT-4.1: code gen workhorse.
    # Claude is order=1 for STRATEGIC calls only (handled by complexity routing).
    # For standard/trivial architect work, GPT-4.1 and Gemini handle it.
    if anthropic_key:
        roles.setdefault("architect", []).append({"model_name": "architect", **claude(1, 2, desc="⚡ Claude Opus — strategic architecture vision (pre-compiled context)")})
    if openai_key:
        roles.setdefault("architect", []).append({"model_name": "architect", **openai(1, 3, desc="⚡ GPT-4.1 — code generation + system design patterns")})
    if gemini_key:
        roles.setdefault("architect", []).append({"model_name": "architect", **gemini_flash(1, 4, desc="⚡ Gemini Flash — fast scaffolding + config generation")})
    if deepseek_key:
        roles.setdefault("architect", []).append({"model_name": "architect", **deepseek_coder(1, 5, desc="⚡ DeepSeek Coder — code specialist")})

    # ═══ Backend (The Logic Factory) ══════════════════════════════════════════
    # GPT-4.1 + DeepSeek Coder + Gemini Flash — all primary, pure throughput.
    if openai_key:
        roles.setdefault("backend", []).append({"model_name": "backend", **openai(1, 3, desc="⚡ GPT-4.1 — algorithm design + data architecture")})
    if deepseek_key:
        roles.setdefault("backend", []).append({"model_name": "backend", **deepseek_coder(1, 6, desc="⚡ DeepSeek Coder — CRUD + algorithm implementation")})
    if gemini_key:
        roles.setdefault("backend", []).append({"model_name": "backend", **gemini_flash(1, 4, desc="⚡ Gemini Flash — fast iteration + boilerplate")})
    if deepseek_key:
        roles.setdefault("backend", []).append({"model_name": "backend", **deepseek(2, 5, desc="↪ DeepSeek Chat — high-volume fallback")})

    # ═══ Frontend (The Visual Swarm) ══════════════════════════════════════════
    # Gemini Flash dominates (fastest for HTML/CSS/JS). GPT-4.1 for React logic.
    if gemini_key:
        roles.setdefault("frontend", []).append({"model_name": "frontend", **gemini_flash(1, 5, desc="⚡ Gemini Flash — HTML/CSS/JS rapid iteration")})
    if openai_key:
        roles.setdefault("frontend", []).append({"model_name": "frontend", **openai(1, 3, desc="⚡ GPT-4.1 — React/complex UI logic")})
    if deepseek_key:
        roles.setdefault("frontend", []).append({"model_name": "frontend", **deepseek_coder(1, 5, desc="⚡ DeepSeek Coder — JS/TS component generation")})
    if anthropic_key:
        roles.setdefault("frontend", []).append({"model_name": "frontend", **claude(2, 2, desc="↪ Claude Opus — complex UI architecture fallback")})

    # ═══ DevOps (The Infrastructure Hive) ═════════════════════════════════════
    # DeepSeek excels at shells/Docker/CI. Gemini Flash for configs.
    if deepseek_key:
        roles.setdefault("devops", []).append({"model_name": "devops", **deepseek(1, 5, tokens=4096, desc="⚡ DeepSeek Chat — shell scripts + Dockerfiles")})
    if deepseek_key:
        roles.setdefault("devops", []).append({"model_name": "devops", **deepseek_coder(1, 5, tokens=4096, desc="⚡ DeepSeek Coder — pipelines + infra code")})
    if gemini_key:
        roles.setdefault("devops", []).append({"model_name": "devops", **gemini_flash(1, 4, tokens=4096, desc="⚡ Gemini Flash — config generation + YAML")})
    if openai_key:
        roles.setdefault("devops", []).append({"model_name": "devops", **openai(2, 3, tokens=4096, desc="↪ GPT-4.1 — security hardening fallback")})

    # ═══ QA (The Adversarial Sandbox) ═════════════════════════════════════════
    # GPT-4.1 for test strategy + DeepSeek Coder for test code volume.
    if openai_key:
        roles.setdefault("qa", []).append({"model_name": "qa", **openai(1, 3, desc="⚡ GPT-4.1 — test strategy + edge cases")})
    if deepseek_key:
        roles.setdefault("qa", []).append({"model_name": "qa", **deepseek_coder(1, 6, desc="⚡ DeepSeek Coder — unit test code generation")})
    if gemini_key:
        roles.setdefault("qa", []).append({"model_name": "qa", **gemini_flash(1, 4, desc="⚡ Gemini Flash — fixture + config generation")})
    if deepseek_key:
        roles.setdefault("qa", []).append({"model_name": "qa", **deepseek(2, 5, desc="↪ DeepSeek Chat — bulk coverage fallback")})

    # ═══ Critic (The Review Board) ════════════════════════════════════════════
    # GPT-4.1 + Gemini Flash as primary. No Claude ($75/M output is absurd for review).
    if openai_key:
        roles.setdefault("critic", []).append({"model_name": "critic", **openai(1, 3, tokens=4096, desc="⚡ GPT-4.1 — security + code review ($2/$8)")})
    if gemini_key:
        roles.setdefault("critic", []).append({"model_name": "critic", **gemini_flash(1, 4, tokens=4096, desc="⚡ Gemini Flash — fast perf review ($0.075/$0.30)")})
    if deepseek_key:
        roles.setdefault("critic", []).append({"model_name": "critic", **deepseek(2, 4, tokens=4096, desc="↪ DeepSeek Chat — fallback reviewer")})

    # ═══ CTO (The Orchestrator) ═══════════════════════════════════════════════
    # Claude is the BRAIN here — strategic blueprint decisions. But context is
    # always pre-compiled by compressor before Claude sees it.
    # GPT-4.1 handles standard calls; Claude for strategic only.
    if anthropic_key:
        roles.setdefault("cto", []).append({"model_name": "cto", **claude(1, 2, desc="⚡ Claude Opus — strategic blueprint brain (pre-compiled context)")})
    if openai_key:
        roles.setdefault("cto", []).append({"model_name": "cto", **openai(1, 3, desc="⚡ GPT-4.1 — blueprint generation + decisions ($2/$8)")})
    if gemini_key:
        roles.setdefault("cto", []).append({"model_name": "cto", **gemini_flash(1, 4, desc="⚡ Gemini Flash — fast blueprint iteration ($0.075/$0.30)")})

    # ═══ Squad Lead (The Traffic Controller) ══════════════════════════════════
    if openai_key:
        roles.setdefault("squad_lead", []).append({"model_name": "squad_lead", **openai(1, 3, tokens=8192, desc="⚡ GPT-4.1 — task allocation + orchestration")})
    if gemini_key:
        roles.setdefault("squad_lead", []).append({"model_name": "squad_lead", **gemini_flash(1, 4, tokens=8192, desc="⚡ Gemini Flash — fast task decomposition")})
    if openai_key:
        roles.setdefault("squad_lead", []).append({"model_name": "squad_lead", **openai_mini(1, 6, tokens=8192, desc="⚡ GPT-4o-mini — cheap structured allocation")})
    if deepseek_key:
        roles.setdefault("squad_lead", []).append({"model_name": "squad_lead", **deepseek(2, 5, tokens=8192, desc="↪ DeepSeek Chat — fallback")})

    # ═══ Integrator (The Glue) ════════════════════════════════════════════════
    if gemini_key:
        roles.setdefault("integrator", []).append({"model_name": "integrator", **gemini_flash(1, 5, desc="⚡ Gemini Flash — low-latency code stitching")})
    if openai_key:
        roles.setdefault("integrator", []).append({"model_name": "integrator", **openai_mini(1, 6, desc="⚡ GPT-4o-mini — structured stitching ($0.15/$0.60)")})
    if deepseek_key:
        roles.setdefault("integrator", []).append({"model_name": "integrator", **deepseek(1, 4, desc="⚡ DeepSeek Chat — formatting + linting")})
    if openai_key:
        roles.setdefault("integrator", []).append({"model_name": "integrator", **openai(2, 3, desc="↪ GPT-4.1 — complex integration fallback")})

    # ═══ Meta CTO (Self-Improvement Strategist) ══════════════════════════════
    # Claude's intelligence is NEEDED here — it analyses the entire codebase.
    # Context is pre-compiled by compressor. GPT-4.1 as co-primary.
    if anthropic_key:
        roles.setdefault("meta_cto", []).append({"model_name": "meta_cto", **claude(1, 2, desc="⚡ Claude Opus — self-improvement brain (pre-compiled context)")})
    if openai_key:
        roles.setdefault("meta_cto", []).append({"model_name": "meta_cto", **openai(1, 3, desc="⚡ GPT-4.1 — self-improvement strategy ($2/$8)")})
    if gemini_key:
        roles.setdefault("meta_cto", []).append({"model_name": "meta_cto", **gemini_flash(1, 4, desc="⚡ Gemini Flash — fast source analysis ($0.075/$0.30)")})

    # ═══════════════════════════════════════════════════════════════════════════
    # CHEAP POOLS — ALL engines at order=1 for maximum diversity & throughput.
    # ═══════════════════════════════════════════════════════════════════════════
    CODE_HEAVY_ROLES   = {"architect", "backend", "frontend", "qa", "devops"}
    ORCHESTRATION_ROLES = {"squad_lead", "integrator", "critic", "meta_cto", "cto"}

    CHEAP_ROLES = ("architect", "backend", "frontend", "devops", "qa",
                   "integrator", "squad_lead", "critic", "meta_cto", "cto")
    for r in CHEAP_ROLES:
        cheap_name = f"{r}_cheap"

        # ⚡ Gemini Flash — universal cheap primary
        if gemini_key:
            roles.setdefault(cheap_name, []).append({
                "model_name": cheap_name,
                **gemini_flash(1, 8, desc=f"⚡ Gemini Flash — cheap {r}")
            })

        # ⚡ DeepSeek Coder — code specialist (code-heavy roles)
        if deepseek_key and r in CODE_HEAVY_ROLES:
            roles.setdefault(cheap_name, []).append({
                "model_name": cheap_name,
                **deepseek_coder(1, 8, desc=f"⚡ DeepSeek Coder — code {r}")
            })

        # ⚡ GPT-4o-mini — structured output (orchestration roles)
        if openai_key and r in ORCHESTRATION_ROLES:
            roles.setdefault(cheap_name, []).append({
                "model_name": cheap_name,
                **openai_mini(1, 8, desc=f"⚡ GPT-4o-mini — structured {r}")
            })

        # ⚡ DeepSeek Chat — volume (all roles, primary tier)
        if deepseek_key:
            roles.setdefault(cheap_name, []).append({
                "model_name": cheap_name,
                **deepseek(1, 6, desc=f"⚡ DeepSeek Chat — volume {r}")
            })

    # ═══ Polish Team — DeepSeek micro-fixers + Gemini coordinator ════════════
    # DeepSeek specialists fix syntax/imports/contracts per-file.
    # Gemini Flash coordinator merges fixes. All cheap, all parallel.
    if deepseek_key:
        roles.setdefault("polisher", []).append({"model_name": "polisher", **deepseek_coder(1, 8, tokens=8192, desc="⚡ DeepSeek Coder — polish specialist")})
    if deepseek_key:
        roles.setdefault("polisher", []).append({"model_name": "polisher", **deepseek(1, 8, tokens=8192, desc="⚡ DeepSeek Chat — polish specialist")})
    if gemini_key:
        roles.setdefault("polisher", []).append({"model_name": "polisher", **gemini_flash(2, 6, tokens=8192, desc="↪ Gemini Flash — polish fallback")})

    if gemini_key:
        roles.setdefault("polish_lead", []).append({"model_name": "polish_lead", **gemini_flash(1, 6, tokens=8192, desc="⚡ Gemini Flash — polish coordinator")})
    if deepseek_key:
        roles.setdefault("polish_lead", []).append({"model_name": "polish_lead", **deepseek(2, 6, tokens=8192, desc="↪ DeepSeek Chat — polish coordinator fallback")})

    # ═══ Smooth AF Team — 3-Tier Quality Pyramid ════════════════════════════
    # Tier 1: DeepSeek workforce — does ALL heavy analysis
    if deepseek_key:
        roles.setdefault("smooth_worker", []).append({"model_name": "smooth_worker", **deepseek_coder(1, 10, tokens=2048, desc="⚡ DeepSeek Coder — Smooth AF analyst")})
    if deepseek_key:
        roles.setdefault("smooth_worker", []).append({"model_name": "smooth_worker", **deepseek(1, 10, tokens=2048, desc="⚡ DeepSeek Chat — Smooth AF analyst")})
    if gemini_key:
        roles.setdefault("smooth_worker", []).append({"model_name": "smooth_worker", **gemini_flash(2, 8, tokens=2048, desc="↪ Gemini Flash — Smooth AF fallback")})

    # Tier 2a: Gemini synthesizer — compresses reports for Claude
    if gemini_key:
        roles.setdefault("smooth_synth", []).append({"model_name": "smooth_synth", **gemini_flash(1, 6, tokens=1024, desc="⚡ Gemini Flash — Smooth AF synthesizer")})
    if deepseek_key:
        roles.setdefault("smooth_synth", []).append({"model_name": "smooth_synth", **deepseek(2, 6, tokens=1024, desc="↪ DeepSeek — Smooth AF synth fallback")})

    # Tier 2b: OpenAI validator — cross-validates synthesis
    if openai_key:
        roles.setdefault("smooth_validator", []).append({"model_name": "smooth_validator", **openai_mini(1, 6, tokens=1024, desc="⚡ GPT-4o-mini — Smooth AF validator")})
    if gemini_key:
        roles.setdefault("smooth_validator", []).append({"model_name": "smooth_validator", **gemini_flash(2, 6, tokens=1024, desc="↪ Gemini Flash — Smooth AF validator fallback")})

    # Tier 3: Claude director — sees ~200 words, makes strategic decision
    if anthropic_key:
        roles.setdefault("smooth_director", []).append({"model_name": "smooth_director", **claude(1, 2, tokens=512, desc="⚡ Claude Opus — Smooth AF director (minimal tokens)")})
    if openai_key:
        roles.setdefault("smooth_director", []).append({"model_name": "smooth_director", **openai(2, 4, tokens=512, desc="↪ GPT-4.1 — Smooth AF director fallback")})

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPRESSOR — cheap summarizer for context compression before flagships.
    # ═══════════════════════════════════════════════════════════════════════════
    if gemini_key:
        roles.setdefault("compressor", []).append({
            "model_name": "compressor",
            **gemini_flash(1, 8, tokens=2048, desc="⚡ Compressor — Gemini Flash context summarizer")
        })
    if deepseek_key:
        roles.setdefault("compressor", []).append({
            "model_name": "compressor",
            **deepseek(1, 8, tokens=2048, desc="⚡ Compressor — DeepSeek Chat context summarizer")
        })
    if openai_key:
        roles.setdefault("compressor", []).append({
            "model_name": "compressor",
            **openai_mini(2, 6, tokens=2048, desc="↪ Compressor fallback — GPT-4o-mini")
        })

    model_list = [item for items in roles.values() for item in items]
    if not model_list:
        raise RuntimeError("No API keys configured — set at least OPENAI_API_KEY")

    router = Router(
        model_list=model_list,
        num_retries=3,
        retry_after=5,
        allowed_fails=2,
        cooldown_time=60,
        routing_strategy="least-busy",
        timeout=120,
        set_verbose=False,
    )
    return router


# ── Smart Routing Policy ─────────────────────────────────────────────────────
# Roles that ALWAYS need top-tier strategic models (no cheap pool override).
# CTO designs user apps and is the most strategic of all — keep flagship-only.
STRATEGIC_ROLES = {"cto"}

# Roles that have a `{role}_cheap` pool of cheap engines (Gemini Flash + DeepSeek).
HAS_CHEAP_POOL = {
    "architect", "backend", "frontend", "devops", "qa",
    "integrator", "squad_lead", "critic", "meta_cto", "cto",
}

# Compress context above this many input characters when going to flagship.
COMPRESS_THRESHOLD_CHARS = 3000
# Trivial messages (very short or simple prompts) auto-routed to cheap pool.
TRIVIAL_THRESHOLD_CHARS  = 500

# ── Job-level intelligence: budget pacing + adaptive failure tracking ────────
# These drive the "incredibly intelligent resource distribution":
#  - When a job is burning budget too fast → auto-downgrade complexity tier.
#  - When a deployment fails repeatedly on a job → exclude it for the rest.
#  - When a job is well under budget → allow flagship usage freely.

DEFAULT_JOB_BUDGET_USD = 0.80
# Burn-rate thresholds (fraction of budget consumed) — tightened for snappier
# response: the router reacts BEFORE the next big call, not after.
BURN_DOWNGRADE_STANDARD_TO_TRIVIAL  = 0.25  # >25% spent → standard becomes trivial
BURN_DOWNGRADE_STRATEGIC_TO_STANDARD = 0.35  # >35% spent → strategic becomes standard
BURN_HARD_CAP                        = 0.85  # >85% spent → ALL traffic goes cheap

# Predictive pacing: estimate THIS call's cost and downgrade if it would push
# the budget past the hard cap. Cheap call against a strategic chain often costs
# >$0.30 alone — predictive pacing prevents single-call blowouts.
PREDICTIVE_PACING = True


@dataclass
class JobIntel:
    job_id: str
    target_budget_usd: float = DEFAULT_JOB_BUDGET_USD
    decisions: list[dict] = field(default_factory=list)  # routing audit trail
    deployment_failures: dict[str, int] = field(default_factory=dict)


_intel: dict[str, JobIntel] = {}
_intel_lock = threading.Lock()


def get_intel(job_id: str) -> JobIntel:
    with _intel_lock:
        if job_id not in _intel:
            _intel[job_id] = JobIntel(job_id=job_id)
        return _intel[job_id]


def set_job_budget(job_id: str, budget_usd: float) -> None:
    """API for the /build endpoint to register a per-job budget cap."""
    intel = get_intel(job_id)
    intel.target_budget_usd = max(0.05, float(budget_usd))


def get_job_intel_summary(job_id: str) -> dict:
    """Return the routing audit trail for /build/{id}/intel."""
    intel = _intel.get(job_id)
    if not intel:
        return {"job_id": job_id, "decisions": [], "budget_usd": None}
    spent = get_tracker(job_id).total_cost() if job_id else 0.0
    return {
        "job_id": job_id,
        "budget_usd": intel.target_budget_usd,
        "spent_usd": round(spent, 6),
        "burn_ratio": round(spent / intel.target_budget_usd, 3),
        "deployment_failures": dict(intel.deployment_failures),
        "decisions": list(intel.decisions[-50:]),  # last 50 calls
    }


def _record_decision(job_id: str | None, role: str, target_role: str,
                     complexity: str, downgraded_from: str | None,
                     burn_ratio: float | None) -> None:
    if not job_id:
        return
    try:
        intel = get_intel(job_id)
        intel.decisions.append({
            "ts": round(time.time(), 1),
            "role": role,
            "dispatched_to": target_role,
            "complexity": complexity,
            "downgraded_from": downgraded_from,
            "burn_ratio": burn_ratio,
        })
    except Exception:
        pass


def get_router() -> Router:
    global _router
    if _router is None:
        _router = _build_router()
    return _router


def _msg_chars(messages: list[dict]) -> int:
    return sum(len((m.get("content") or "")) for m in messages)


def _classify_complexity(role: str, messages: list[dict], hint: str | None) -> str:
    """
    Returns one of: 'trivial' | 'standard' | 'strategic'.
    Caller hint always wins. Otherwise infer from role + payload size.
    """
    if hint in {"trivial", "standard", "strategic"}:
        return hint
    if role in STRATEGIC_ROLES:
        return "strategic"
    chars = _msg_chars(messages)
    if chars < TRIVIAL_THRESHOLD_CHARS:
        return "trivial"
    return "standard"


def _has_deployment(model_name: str) -> bool:
    try:
        router = get_router()
        return any(m.get("model_name") == model_name for m in router.model_list)
    except Exception:
        return False


async def _compress_user_context(messages: list[dict], job_id: str | None) -> list[dict]:
    """
    Summarize bulky portions of the user message via a CHEAP model (Gemini Flash
    / DeepSeek) before sending to an expensive engine. Preserves the directive
    head and tail; compresses the dependency-files / context middle.
    """
    if not _has_deployment("compressor"):
        return messages
    user_idx = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    if user_idx is None:
        return messages
    body = messages[user_idx].get("content", "") or ""
    if len(body) <= COMPRESS_THRESHOLD_CHARS:
        return messages

    head = body[:1200]
    tail = body[-1200:]
    middle = body[1200:-1200]
    if len(middle) < 1500:
        return messages

    prompt = [
        {"role": "system", "content":
            "You compress technical context for downstream LLMs. "
            "Summarize the input into <=600 words preserving EVERY: file path, "
            "API name, function/class signature, type, error message, and "
            "explicit decision. Use terse bullet points. No prose, no fluff."},
        {"role": "user", "content": middle[:30000]},
    ]
    router = get_router()
    try:
        resp = await router.acompletion(
            model="compressor", messages=prompt, max_tokens=1200,
            metadata={"job_id": job_id} if job_id else None,
        )
        compressed = (resp.choices[0].message.content or "").strip()
        if not compressed:
            return messages
        # Track compressor cost too
        if job_id and hasattr(resp, "usage") and resp.usage:
            try:
                get_tracker(job_id).record(
                    agent_role="compressor",
                    model=getattr(resp, "model", "compressor"),
                    input_tokens=getattr(resp.usage, "prompt_tokens", 0),
                    output_tokens=getattr(resp.usage, "completion_tokens", 0),
                )
            except Exception:
                pass
    except Exception as exc:
        print(f"[GodDev Router] compressor failed, sending raw context: {exc}")
        return messages

    new_body = (
        f"{head}\n\n"
        f"## [CONTEXT COMPRESSED — original {len(body):,} chars summarized below]\n"
        f"{compressed}\n\n"
        f"{tail}"
    )
    new_messages = list(messages)
    new_messages[user_idx] = {**messages[user_idx], "content": new_body}
    return new_messages


async def acall(
    role: str,
    messages: list[dict],
    *,
    complexity: str | None = None,
    compress: bool = True,
    affinity: str = "default",
    **kwargs,
) -> str:
    """
    Smart cost-aware LLM dispatch.

    Routing rules:
      - complexity='trivial'   → `{role}_cheap` pool (DeepSeek + Gemini Flash)
      - complexity='strategic' → flagship pool (Claude / GPT-4.1) + context
                                 pre-compressed via `compressor` if large
      - complexity='standard'  → flagship pool, no compression
      - complexity=None        → auto-classified by payload size and role

    Multi-tier workforce: every flagship pool falls back through Lead → Worker
    → Fallback automatically (handled by LiteLLM Router on 429s/errors).
    """
    router = get_router()
    job_id = kwargs.get("metadata", {}).get("job_id")

    initial_complexity = _classify_complexity(role, messages, complexity)
    chosen_complexity = initial_complexity
    downgraded_from: str | None = None
    burn_ratio: float | None = None

    # ── Budget pacing intelligence ─────────────────────────────────────────
    # If we're burning the budget too fast, auto-downgrade. This is the core
    # "cost/power-driven" decision: invest expensive calls only when we have
    # the budget headroom for them.
    if job_id:
        try:
            intel = get_intel(job_id)
            spent = get_tracker(job_id).total_cost()
            budget = max(intel.target_budget_usd, 0.01)
            burn_ratio = round(spent / budget, 3)

            # Predictive pacing: estimate worst-case cost of THIS call and
            # downgrade pre-emptively if it would blow the budget. This stops
            # single high-output Claude/GPT-4.1 calls from blowing the budget
            # AFTER they happen.
            if PREDICTIVE_PACING and chosen_complexity in {"standard", "strategic"}:
                input_chars = _msg_chars(messages)
                # Crude tokens estimate: ~4 chars/token input, assume 4000 max output.
                est_in_tk = input_chars / 4
                est_out_tk = 4000
                # Use the most-expensive model in the role pool as the upper bound
                worst_model = "claude-opus-4-7" if chosen_complexity == "strategic" else "gpt-4.1"
                wp = get_profile(worst_model)
                est_cost = (est_in_tk * wp["cost_in"] + est_out_tk * wp["cost_out"]) / 1_000_000
                projected = spent + est_cost
                projected_ratio = projected / budget
                if projected_ratio >= BURN_HARD_CAP and chosen_complexity != "trivial":
                    downgraded_from = chosen_complexity
                    chosen_complexity = "trivial"
                elif (projected_ratio >= BURN_DOWNGRADE_STRATEGIC_TO_STANDARD
                      and chosen_complexity == "strategic"):
                    downgraded_from = "strategic"
                    chosen_complexity = "standard"

            if burn_ratio >= BURN_HARD_CAP:
                # Out of budget — force everything cheap
                if chosen_complexity != "trivial":
                    downgraded_from = chosen_complexity
                    chosen_complexity = "trivial"
            elif (burn_ratio >= BURN_DOWNGRADE_STRATEGIC_TO_STANDARD
                  and chosen_complexity == "strategic"):
                downgraded_from = "strategic"
                chosen_complexity = "standard"
            elif (burn_ratio >= BURN_DOWNGRADE_STANDARD_TO_TRIVIAL
                  and chosen_complexity == "standard"
                  and role in HAS_CHEAP_POOL):
                downgraded_from = "standard"
                chosen_complexity = "trivial"
        except Exception:
            pass

    # ── Pool selection ─────────────────────────────────────────────────────
    target_role = role
    if chosen_complexity == "trivial" and role in HAS_CHEAP_POOL:
        cheap_name = f"{role}_cheap"
        if _has_deployment(cheap_name):
            target_role = cheap_name

    # Compress context before hitting expensive engines.
    # For Claude-heavy pools (cto, architect, meta_cto), compress more aggressively
    # because Claude is $15/$75 per 1M tokens — every saved token matters.
    _CLAUDE_POOLS = {"cto", "architect", "meta_cto"}
    compress_threshold = 3000 if role in _CLAUDE_POOLS else COMPRESS_THRESHOLD_CHARS
    if (compress
            and target_role == role
            and chosen_complexity in {"standard", "strategic"}
            and _msg_chars(messages) > compress_threshold):
        messages = await _compress_user_context(messages, job_id)

    _record_decision(job_id, role, target_role, chosen_complexity,
                     downgraded_from, burn_ratio)

    # ── FLUID SWARM CLOSED-LOOP: pheromone-weighted deployment ordering ────
    # When telemetry shows a strong winner for this (role, affinity), boost
    # that deployment's `order` so LiteLLM's least-busy strategy prefers it.
    # Mutation is in-place but order is integer-only and cheap to compute.
    # Falls back silently if telemetry has insufficient samples.
    swarm_pick: str | None = None
    try:
        deployments = [d for d in (router.model_list or [])
                       if d.get("model_name") == target_role]
        if len(deployments) >= 2:
            candidates = [
                (d.get("litellm_params") or {}).get("model", "").split("/", 1)[-1]
                for d in deployments
            ]
            best_id, _breakdown = engine_telemetry.best_engine(
                role, affinity, candidates, min_calls=5,
            )
            if best_id:
                swarm_pick = best_id
    except Exception:
        pass

    model_used = target_role
    t_start = time.time()
    in_tk = 0
    out_tk = 0
    call_cost = 0.0
    try:
        # Stamp swarm pick onto metadata for observability (cost tracker logs it)
        if swarm_pick:
            kwargs.setdefault("metadata", {})["swarm_pick"] = swarm_pick
        response = await router.acompletion(model=target_role, messages=messages, **kwargs)
        content = response.choices[0].message.content or ""

        try:
            model_used = getattr(response, "model", target_role)
        except Exception:
            model_used = target_role

        if hasattr(response, "usage") and response.usage:
            in_tk = getattr(response.usage, "prompt_tokens", 0)
            out_tk = getattr(response.usage, "completion_tokens", 0)
            if job_id:
                call_cost = get_tracker(job_id).record(
                    agent_role=role,
                    model=model_used,
                    input_tokens=in_tk,
                    output_tokens=out_tk,
                )

        _track_health(model_used, success=True)
        # Swarm telemetry: pheromone trail strengthening
        engine_telemetry.record(
            role=role, affinity=affinity, engine=str(model_used),
            in_tokens=in_tk, out_tokens=out_tk, cost=call_cost,
            latency_ms=int((time.time() - t_start) * 1000),
            success=True, job_id=job_id or "",
        )
        return content.strip()

    except Exception as exc:
        _track_health(model_used, success=False, error=str(exc))
        if job_id:
            try:
                intel = get_intel(job_id)
                key = str(model_used)
                intel.deployment_failures[key] = intel.deployment_failures.get(key, 0) + 1
            except Exception:
                pass
        # Swarm telemetry: pheromone decay on failure
        engine_telemetry.record(
            role=role, affinity=affinity, engine=str(model_used),
            in_tokens=in_tk, out_tokens=out_tk, cost=call_cost,
            latency_ms=int((time.time() - t_start) * 1000),
            success=False, error=str(exc), job_id=job_id or "",
        )
        raise


async def acall_structured(
    role: str,
    messages: list[dict],
    schema: type,
    *,
    complexity: str | None = None,
    compress: bool = True,
    affinity: str = "default",
    max_retries: int = 2,
    **kwargs,
):
    """
    Like ``acall`` but returns a parsed Pydantic instance.

    Brings ``with_structured_output``-style flows (CTO, Meta CTO, Self-Improver,
    Self-Assessor, critics) into the cost/power-aware router so they benefit
    from cheap-pool routing, context compression, budget pacing, and
    adaptive failure tracking — all the intelligence the worker pipeline gets.

    The schema is injected as JSON Schema in the system prompt; the model
    output is stripped of markdown fences and validated. On parse failure,
    retries with explicit error feedback, escalating complexity each retry.
    """
    import json
    schema_json = schema.model_json_schema() if hasattr(schema, "model_json_schema") else {}
    fmt_instruction = (
        "\n\n## STRICT OUTPUT FORMAT\n"
        "Return ONLY a JSON object that validates against the schema below. "
        "No prose, no markdown fences, no preamble.\n\n"
        f"```json-schema\n{json.dumps(schema_json, indent=2)[:6000]}\n```"
    )

    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = (msgs[0].get("content") or "") + fmt_instruction
    else:
        msgs.insert(0, {"role": "system", "content": fmt_instruction})

    last_err: Exception | None = None
    cur_complexity = complexity
    for attempt in range(max_retries + 1):
        raw = await acall(role, msgs, complexity=cur_complexity,
                          compress=compress, affinity=affinity, **kwargs)
        cleaned = _strip_json(raw)
        try:
            return schema.model_validate_json(cleaned)
        except Exception as exc:
            last_err = exc
            # Escalate to flagship + add error feedback to next attempt
            cur_complexity = "strategic"
            msgs.append({"role": "assistant", "content": raw})
            msgs.append({"role": "user", "content":
                f"Your previous output FAILED schema validation:\n{exc}\n"
                "Output the corrected JSON only — no fences, no commentary."})
    raise ValueError(f"acall_structured: schema validation failed after {max_retries+1} attempts: {last_err}")


def _strip_json(text: str) -> str:
    """Strip markdown fences and locate the outermost JSON object."""
    s = (text or "").strip()
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        # remove trailing fence
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    # If the model wrapped JSON in prose, locate first { ... last }
    if not s.lstrip().startswith("{"):
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            s = s[i:j+1]
    return s.strip()


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
            "status":      status,
            "last_ok_ago": round(age_ok, 1) if age_ok is not None else None,
            "last_error":  h.get("last_error", ""),
            "total_calls": h.get("calls", 0),
        }
    return result
