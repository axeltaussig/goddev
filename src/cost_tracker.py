"""
GodDev 3.0 — Token Cost Tracker
===================================

Tracks token usage and USD cost across all agent calls in a run.
Uses model-specific pricing (April 2026 rates).

Thread-safe: workers run in parallel so all updates use a lock.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

# ─── Pricing Table (USD per 1M tokens, April 2026) ────────────────────────────

PRICING: dict[str, dict[str, float]] = {
    # model_id → {input, output} per 1M tokens — verified Apr 2026.
    # Order matters: more specific keys first (substring match in _get_price).
    # ── Anthropic ─────────────────────────────────────────────────
    "claude-opus-4-7":     {"input": 15.00, "output": 75.00},
    "claude-opus-4-5":     {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5":   {"input":  3.00, "output": 15.00},
    "claude-3-5-sonnet":   {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":    {"input":  0.80, "output":  4.00},  # 2026 cheap tier
    "claude-3-5-haiku":    {"input":  0.80, "output":  4.00},
    "claude-3-haiku":      {"input":  0.25, "output":  1.25},  # legacy
    # ── OpenAI ────────────────────────────────────────────────────
    "gpt-4.1-mini":        {"input":  0.40, "output":  1.60},  # 2026 mid
    "gpt-4.1-nano":        {"input":  0.10, "output":  0.40},  # 2026 cheap
    "gpt-4.1":             {"input":  2.00, "output":  8.00},  # 2026 flagship code
    "gpt-4o-mini":         {"input":  0.15, "output":  0.60},
    "gpt-4o":              {"input":  2.50, "output": 10.00},
    "gpt-4-turbo":         {"input": 10.00, "output": 30.00},  # legacy
    # ── Google ────────────────────────────────────────────────────
    "gemini-2.5-pro":      {"input":  1.25, "output": 10.00},
    "gemini-2.5-flash":    {"input":  0.075,"output":  0.30},
    "gemini-2.0-flash":    {"input":  0.10, "output":  0.40},
    "gemini-1.5-pro":      {"input":  1.25, "output":  5.00},  # legacy long-context
    # ── DeepSeek ──────────────────────────────────────────────────
    "deepseek-coder":      {"input":  0.14, "output":  0.28},
    "deepseek-chat":       {"input":  0.27, "output":  1.10},
    "deepseek-v3":         {"input":  0.27, "output":  1.10},
}

_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}  # fallback


def _get_price(model: str) -> dict[str, float]:
    model_lower = model.lower()
    for key, price in PRICING.items():
        if key in model_lower:
            return price
    return _DEFAULT_PRICING


# ─── Per-call record ─────────────────────────────────────────────────────────


@dataclass
class CallRecord:
    agent_role: str        # cto | squad_leader | worker | code_critic | security_critic | perf_critic | integrator | reflector | self_improver
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


# ─── Cost Tracker (thread-safe singleton per run) ─────────────────────────────


@dataclass
class CostTracker:
    _records: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        agent_role: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record a model call. Returns the cost for this call in USD."""
        price = _get_price(model)
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
        record = CallRecord(
            agent_role=agent_role,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        with self._lock:
            self._records.append(record)
        return cost

    def total_cost(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    def total_tokens(self) -> dict[str, int]:
        with self._lock:
            return {
                "input": sum(r.input_tokens for r in self._records),
                "output": sum(r.output_tokens for r in self._records),
            }

    def breakdown_by_agent(self) -> dict[str, dict]:
        """Per-(agent, model) rollup so heterogeneous routing is visible.

        Key format: '{agent_role}:{model}' so a critic that ran 2× on Gemini
        Flash + 1× on Claude shows TWO rows (no more first-write-wins
        misattribution).
        """
        with self._lock:
            breakdown: dict[str, dict] = {}
            for r in self._records:
                key = f"{r.agent_role}::{r.model}"
                if key not in breakdown:
                    breakdown[key] = {
                        "agent_role": r.agent_role,
                        "model": r.model,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "calls": 0,
                    }
                d = breakdown[key]
                d["input_tokens"] += r.input_tokens
                d["output_tokens"] += r.output_tokens
                d["cost_usd"] += r.cost_usd
                d["calls"] += 1
            return breakdown

    def summary(self) -> dict:
        breakdown = self.breakdown_by_agent()
        tokens = self.total_tokens()
        return {
            "total_cost_usd": round(self.total_cost(), 6),
            "total_input_tokens": tokens["input"],
            "total_output_tokens": tokens["output"],
            "total_tokens": tokens["input"] + tokens["output"],
            "by_agent": {
                role: {**d, "cost_usd": round(d["cost_usd"], 6)}
                for role, d in breakdown.items()
            },
        }

    def to_markdown_table(self) -> str:
        breakdown = self.breakdown_by_agent()
        lines = [
            "| Agent | Model | Calls | Input | Output | Cost USD |",
            "|-------|-------|-------|-------|--------|----------|",
        ]
        for _key, d in sorted(breakdown.items(),
                              key=lambda kv: (-kv[1]["cost_usd"],
                                              kv[1]["agent_role"])):
            lines.append(
                f"| {d['agent_role']} | {d['model']} | {d['calls']} "
                f"| {d['input_tokens']:,} | {d['output_tokens']:,} "
                f"| ${d['cost_usd']:.4f} |"
            )
        total = self.total_cost()
        lines.append(f"| **TOTAL** | — | — | — | — | **${total:.4f}** |")
        return "\n".join(lines)


# ─── LangChain callback for automatic token capture ──────────────────────────

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class CostCallback(BaseCallbackHandler):
    """
    Attach to any LangChain LLM to automatically capture token usage dynamically.
    """

    def __init__(self, tracker: CostTracker):
        self.tracker = tracker

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            # Dynamically resolve model explicitly assigned via kwargs during invoke
            model = kwargs.get("invocation_params", {}).get("model") or kwargs.get("invocation_params", {}).get("model_name") or "unknown-model"
            # Attempt to pull tags injected by LangGraph node context, fallback to "agent"
            tags = kwargs.get("tags", [])
            role = tags[0] if tags else "graph_node"

            # Parse LangChain UsageMetadata natively
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(gen, "message", None)
                    if meta:
                        usage = getattr(meta, "usage_metadata", None)
                        if usage:
                            self.tracker.record(
                                agent_role=role,
                                model=model,
                                input_tokens=usage.get("input_tokens", 0),
                                output_tokens=usage.get("output_tokens", 0),
                            )
                            return

            # Fallback for older structured outputs
            usage = (response.llm_output or {}).get("usage", {})
            if usage:
                self.tracker.record(
                    agent_role=role,
                    model=model,
                    input_tokens=usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                    output_tokens=usage.get("completion_tokens", usage.get("output_tokens", 0)),
                )
        except Exception as exc:
            print(f"[CostCallback] Failed to track: {exc}")


# ─── Global tracker registry (per job_id) ─────────────────────────────────────

_trackers: dict[str, CostTracker] = {}
_trackers_lock = threading.Lock()


def get_tracker(job_id: str) -> CostTracker:
    with _trackers_lock:
        if job_id not in _trackers:
            _trackers[job_id] = CostTracker()
        return _trackers[job_id]


def remove_tracker(job_id: str) -> Optional[CostTracker]:
    with _trackers_lock:
        return _trackers.pop(job_id, None)
