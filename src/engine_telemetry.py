"""
GodDev 3.0 — Engine Performance Telemetry & Swarm Map (Stigmergy Layer)
========================================================================

Every router call leaves a "pheromone trail": (role, affinity, engine, cost,
latency, success/failure). The aggregated map drives Fluid Swarm Intelligence:

  - **Reinforcement**: engines that succeed cheaply on (role, affinity) win
    more dispatches via cost-weighted success ratio scoring.
  - **Cool-down**: engines with 3+ consecutive failures get a 60-min penalty
    so the swarm self-heals around bad providers automatically.
  - **Cross-job memory**: the JSONL persists, so future jobs benefit from
    every past call — like ant colonies remembering food trails.

Architecture: append-only JSONL on disk + in-memory aggregate. Lock-protected
writes so concurrent worker threads don't corrupt the trail.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_TELEMETRY_PATH = Path(os.getenv("MEMORY_DIR", "/opt/goddev/memory")) / "engine_telemetry.jsonl"
try:
    _TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

_lock = threading.Lock()
_swarm_map: dict[tuple[str, str, str], dict] = {}
_loaded = False
_FAILURE_COOLDOWN_S = 3600  # 1h ban after 3 consecutive failures
_MIN_CALLS_FOR_PREFERENCE = 3


def _load_existing() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not _TELEMETRY_PATH.exists():
        return
    try:
        with open(_TELEMETRY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    _ingest(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass


def _ingest(rec: dict) -> None:
    key = (
        rec.get("role") or "",
        rec.get("affinity") or "default",
        rec.get("engine") or "",
    )
    s = _swarm_map.setdefault(key, {
        "calls": 0, "successes": 0, "failures": 0,
        "total_cost": 0.0, "total_latency_ms": 0,
        "total_in_tokens": 0, "total_out_tokens": 0,
        "last_call": 0.0, "consecutive_failures": 0,
    })
    s["calls"] += 1
    if rec.get("success", True):
        s["successes"] += 1
        s["consecutive_failures"] = 0
    else:
        s["failures"] += 1
        s["consecutive_failures"] += 1
    s["total_cost"] += float(rec.get("cost", 0.0))
    s["total_latency_ms"] += int(rec.get("latency_ms", 0))
    s["total_in_tokens"] += int(rec.get("in_tokens", 0))
    s["total_out_tokens"] += int(rec.get("out_tokens", 0))
    s["last_call"] = max(s["last_call"], float(rec.get("ts", time.time())))


def record(
    role: str,
    affinity: str,
    engine: str,
    *,
    in_tokens: int = 0,
    out_tokens: int = 0,
    cost: float = 0.0,
    latency_ms: int = 0,
    success: bool = True,
    error: str = "",
    job_id: str = "",
) -> None:
    """Record one router call to the swarm trail (memory + disk JSONL)."""
    _load_existing()
    rec = {
        "ts": time.time(),
        "role": role,
        "affinity": affinity or "default",
        "engine": engine,
        "in_tokens": int(in_tokens),
        "out_tokens": int(out_tokens),
        "cost": round(float(cost), 6),
        "latency_ms": int(latency_ms),
        "success": bool(success),
        "error": (error or "")[:200],
        "job_id": job_id,
    }
    with _lock:
        _ingest(rec)
        try:
            with open(_TELEMETRY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass


def is_engine_excluded(role: str, engine: str) -> bool:
    """True if engine is in cool-down (3+ consecutive failures within 1h)."""
    _load_existing()
    now = time.time()
    for (r, _aff, eng), s in _swarm_map.items():
        if r != role or eng != engine:
            continue
        if (s["consecutive_failures"] >= 3
                and now - s["last_call"] < _FAILURE_COOLDOWN_S):
            return True
    return False


def best_engine(
    role: str,
    affinity: str,
    candidate_engines: list[str],
    min_calls: int = _MIN_CALLS_FOR_PREFERENCE,
) -> tuple[str | None, dict]:
    """
    Return (best_engine_id, score_breakdown) for this (role, affinity).

    Score = success_rate / avg_cost_per_call — pheromone strength.
    Excludes engines in cool-down and engines without enough sample data.
    Returns (None, {}) if no clear winner — caller should fall back to
    LiteLLM least-busy routing.
    """
    _load_existing()
    now = time.time()
    best_id: str | None = None
    best_score = -1.0
    breakdown: dict[str, dict] = {}

    for engine in candidate_engines:
        # Try affinity-specific first, then fall back to "default" affinity
        s = _swarm_map.get((role, affinity, engine))
        if not s or s["calls"] < min_calls:
            s = _swarm_map.get((role, "default", engine))
        if not s or s["calls"] < min_calls:
            continue
        # Cool-down check
        if (s["consecutive_failures"] >= 3
                and now - s["last_call"] < _FAILURE_COOLDOWN_S):
            breakdown[engine] = {"score": -1, "reason": "cooldown"}
            continue
        success_rate = s["successes"] / max(s["calls"], 1)
        avg_cost = s["total_cost"] / max(s["calls"], 1)
        score = success_rate / max(avg_cost, 0.0001)
        breakdown[engine] = {
            "score": round(score, 2),
            "calls": s["calls"],
            "success_rate": round(success_rate, 3),
            "avg_cost": round(avg_cost, 6),
        }
        if score > best_score:
            best_id = engine
            best_score = score
    return best_id, breakdown


def swarm_summary() -> dict:
    """Aggregated swarm map for the /swarm endpoint."""
    _load_existing()
    by_engine: list[dict] = []
    by_role: dict[str, dict] = {}
    now = time.time()
    for (role, affinity, engine), s in _swarm_map.items():
        c = s["calls"]
        if c == 0:
            continue
        success_rate = s["successes"] / c
        avg_cost = s["total_cost"] / c
        score = success_rate / max(avg_cost, 0.0001)
        by_engine.append({
            "role": role,
            "affinity": affinity,
            "engine": engine,
            "calls": c,
            "success_rate": round(success_rate, 3),
            "avg_cost_usd": round(avg_cost, 6),
            "avg_latency_ms": int(s["total_latency_ms"] / c),
            "consecutive_failures": s["consecutive_failures"],
            "total_cost_usd": round(s["total_cost"], 4),
            "pheromone_score": round(score, 2),
            "in_cooldown": (
                s["consecutive_failures"] >= 3
                and now - s["last_call"] < _FAILURE_COOLDOWN_S
            ),
            "last_call_ago_s": int(now - s["last_call"]) if s["last_call"] else None,
        })
        # Per-role rollup
        r = by_role.setdefault(role, {
            "calls": 0, "total_cost": 0.0, "engines": set(),
        })
        r["calls"] += c
        r["total_cost"] += s["total_cost"]
        r["engines"].add(engine)

    by_engine.sort(key=lambda x: (x["role"], x["affinity"], -x["pheromone_score"]))
    role_summary = [
        {
            "role": r,
            "calls": v["calls"],
            "total_cost_usd": round(v["total_cost"], 4),
            "engine_count": len(v["engines"]),
            "engines": sorted(v["engines"]),
        }
        for r, v in sorted(by_role.items())
    ]
    return {
        "total_calls": sum(v["calls"] for v in by_role.values()),
        "total_cost_usd": round(sum(v["total_cost"] for v in by_role.values()), 4),
        "by_role": role_summary,
        "by_engine": by_engine,
    }
