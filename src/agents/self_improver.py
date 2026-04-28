"""
GodDev 3.0 — Self-Improvement Agent
=======================================

The most important node in the system. After each build, it:

1. Reads ALL past reflections from memory/reflections.md
2. Reads the current runtime_config.json
3. Uses Claude to analyse patterns across runs and propose improvements
4. Writes an updated runtime_config.json (agents load this next run)
5. Writes memory/improvements.md (log of every change made)

What it can improve:
  - Agent prompt additions (learned patterns that reduce critic rejections)
  - Score thresholds (tighten or loosen quality gates based on history)
  - Agent routing (e.g. "JS files → openai outperforms deepseek by 2pts")
  - Verification strategies (new verification commands that catch bugs)
  - Squad sizing (max files per squad)

This creates a genuine flywheel: each run teaches the system, and the next
run benefits from those lessons. Over ~10 runs the system meaningfully improves.

Research basis:
  - Self-Refine (Madaan et al. 2023): iterative self-feedback improves LLM output
  - Reflexion (Shinn et al. 2023): verbal reinforcement learning via reflection
  - Constitutional AI: self-critique and revision loops
  - AlphaCode 2: code generation improves with execution feedback
"""
from __future__ import annotations

import asyncio
import json
import os
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from ..config import load_runtime_config, save_runtime_config, _prune_prompt_additions
from ..contracts import validate_improvement_plan
from ..llm_router import acall_structured
from ..state import GodDevState
from ..task_queue import acquire_slot

# ─── Improvement Plan schema ──────────────────────────────────────────────────


class ImprovementPlan(BaseModel):
    """Claude's structured analysis and proposed config changes."""

    summary: str = Field(description="2-3 sentence summary of what was learned")
    reasoning: str = Field(description="Detailed explanation of patterns found and why these changes help")

    # Prompt additions (appended to system prompts)
    cto_prompt_additions: str = Field(
        default="",
        description="Lessons to add to CTO system prompt. E.g. 'Always specify error handling in milestone deliverables'",
    )
    worker_prompt_additions: str = Field(
        default="",
        description="Lessons to add to all Worker system prompts. E.g. 'Always add input validation to API endpoints'",
    )

    # Threshold changes
    new_critic_score_threshold: float = Field(
        default=7.0, ge=4.0, le=9.5,
        description="Recommended critic approval threshold (0-10). Lower = more lenient, higher = stricter",
    )

    # Agent routing updates (task_type → preferred agent)
    routing_updates: dict[str, str] = Field(
        default_factory=dict,
        description="Task type → agent updates based on performance data. Keys must match preferred_agents keys",
    )

    # Config changes
    recommended_max_files_per_squad: int = Field(
        default=8, ge=3, le=12,
        description="Recommended max FileTasks per squad (tune based on timeout frequency)",
    )

    # Priority signals for next run
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Top 3 areas to focus on improving in next run",
    )


# ─── System prompt ────────────────────────────────────────────────────────────

_SELF_IMPROVER_SYSTEM = """You are GodDev's Self-Improvement Engine — the meta-cognitive layer that analyses past runs and makes the system smarter.

## Your Mission
Study the execution history and improve the system's configuration so the NEXT run performs better.

## What You Have Access To
- Reflections from past runs (verification rates, critic scores, timing, patterns)
- Current runtime configuration (thresholds, agent routing, prompt additions)
- Cost tracking data per agent

## Analysis Framework (think step by step)

### 1. Quality Signals
- If avg_critic_score < 7.0: prompts need to be more specific about requirements
- If verification_pass_rate < 0.7: workers aren't testing enough / verification cmds are wrong
- If critic_rounds consistently = 2: critics are too strict OR work quality is consistently low
- If integration_passed = false repeatedly: files aren't connecting to each other properly

### 2. Agent Performance
- Compare agent-level verification rates: if one agent's files fail more → route that task type away
- Identify which file types (Python vs JS vs config) have the highest failure rates
- Check if certain squad domains produce more issues than others

### 3. Prompt Improvements (most impactful)
- Missing patterns in worker output → add to worker_prompt_additions
- CTO blueprints missing key requirements → add to cto_prompt_additions
- Be SPECIFIC: "Always include X" not vague advice

### 4. Threshold Calibration
- If critic always force-approves: threshold is too high → lower it slightly
- If integration consistently fails: threshold is too low → raise it
- If `avg_critic_score` is consistently below 7.0, and especially if `integration_passed` is false, incrementally raise `new_critic_score_threshold` by 0.5 points, aiming for a target of at least 7.0 for initial approval.

## Rules
- Only propose changes supported by actual data from the reflections
- Be conservative: small, targeted improvements beat large sweeps
- If there's only 1 run, don't change thresholds — insufficient data
- If everything is working well (avg_score > 8.0, verif_rate > 0.9), minimal changes

## ⚠️ Incremental Excellence (CRITICAL)
A small improvement that SHIPS is infinitely better than an ambitious plan that breaks things.
- Change thresholds by at most ±0.5 per run — large jumps destabilize the system
- Prompt additions should be SHORT and SPECIFIC — max 1-2 sentences per addition
- Never replace existing prompt additions — only APPEND short, targeted lessons
- If the last run was decent (score > 6.0), make MINIMAL changes — don't over-correct
- Routing updates should only change ONE agent at a time, backed by data
- When in doubt, do NOTHING — stability beats optimization

## Integration Contracts (NEVER BREAK)
- critic_score_threshold must stay between 4.0 and 9.5
- max_files_per_squad must stay between 3 and 12
- routing_updates agents must be one of: gemini, openai, deepseek, claude
- prompt additions must not exceed 500 chars per field

Return a valid JSON ImprovementPlan. No prose outside the JSON."""


# ─── Helper functions ─────────────────────────────────────────────────────────

def _extract_recent_runs(reflections_text: str, n: int = 10) -> str:
    """Extracts the content of the last N runs from the reflections text.
    Assumes reflections are separated by '## Run:'."""
    runs = reflections_text.split("## Run:")
    # The first part might be empty or preamble. Filter out empty strings.
    actual_runs = [f"## Run:{run.strip()}" for run in runs if run.strip()]
    return "\n\n".join(actual_runs[-n:])


# ─── Node ─────────────────────────────────────────────────────────────────────


def self_improver_node(state: GodDevState) -> dict:
    """
    Analyse all past run reflections and update runtime_config.json.

    OPT-IN: only runs when explicitly enabled, since each call costs ~$0.13 on
    the meta_cto (Claude) flagship pool. Default behavior is to skip on regular
    /build calls — the dedicated /self-improve flow always runs it.

    Triggers:
      - state.metadata.enable_self_improvement = True   (explicit flag)
      - state.metadata.mode == "self_improvement"        (set by /self-improve)
      - reflections >= 5 AND mod-of-5 == 0               (every 5th run, opportunistic)
    """
    md = state.get("metadata") or {}
    enabled = md.get("enable_self_improvement") is True
    is_self_improve_mode = md.get("mode") == "self_improvement"

    memory_dir = Path(os.getenv("MEMORY_DIR", "/opt/goddev/memory"))
    reflections_path = memory_dir / "reflections.md"
    improvements_path = memory_dir / "improvements.md"

    # Read reflection history
    reflections_text = ""
    if reflections_path.exists():
        reflections_text = reflections_path.read_text(encoding="utf-8")

    # Count runs
    run_count = reflections_text.count("## Run:")

    if run_count < 1:
        msg = "🧠 **Self-Improver**: No reflection history yet — skipping analysis (need ≥1 run)"
        return {"messages": [AIMessage(content=msg)]}

    # ── Skip-by-default for regular builds ──────────────────────────────────
    # Only run when caller opts in explicitly, when we're in self-improvement
    # mode, or opportunistically every 5th completed run.
    opportunistic = (run_count >= 5 and run_count % 5 == 0)
    if not (enabled or is_self_improve_mode or opportunistic):
        msg = (f"🧠 **Self-Improver**: skipping (opt-in only). "
               f"Pass `enable_self_improvement=true` in /build to run "
               f"(currently {run_count} reflections; next opportunistic run at "
               f"{((run_count // 5) + 1) * 5}).")
        return {"messages": [AIMessage(content=msg)]}

    # Load current config
    current_config = load_runtime_config()
    current_config_json = json.dumps(current_config, indent=2)

    # Extract recent reflections (last 10 runs to keep context manageable)
    recent_reflections = _extract_recent_runs(reflections_text, n=10)

    # ── Cost/power-aware analysis via the router ─────────────────────────────
    # Self-improvement analysis is mostly summarization + pattern recognition →
    # the cheap pool (Gemini Flash + DeepSeek) handles this excellently. Only
    # escalates to flagship (Claude/GPT-4.1) on schema-validation retry.
    job_id = (state.get("metadata") or {}).get("job_id")
    analysis_messages = [
        {"role": "system", "content": _SELF_IMPROVER_SYSTEM},
        {"role": "user",   "content":
            f"""Analyse these GodDev execution reflections and propose improvements.

**Total runs analysed**: {run_count}
**Recent runs** (last {min(run_count, 10)}):

{recent_reflections}

**Current runtime_config.json**:
```json
{current_config_json}
```

Propose improvements to make the NEXT run perform better."""},
    ]

    async def _run_self_improver() -> ImprovementPlan:
        await acquire_slot("meta_cto")
        # standard complexity → flagship pool but with auto-cheap-pool downgrade
        # if budget is tight; budget-pacing logic in the router handles this.
        return await acall_structured(
            "meta_cto", analysis_messages, ImprovementPlan,
            complexity="standard",
            compress=True,
            metadata={"job_id": job_id} if job_id else None,
        )

    try:
        plan = asyncio.run(_run_self_improver())
    except RuntimeError:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            plan = pool.submit(asyncio.run, _run_self_improver()).result()

    # ── Contract validation: clamp plan to safe incremental ranges ──────────
    _plan_dict = plan.model_dump() if hasattr(plan, 'model_dump') else dict(plan)
    _safe_plan, _corrections = validate_improvement_plan(_plan_dict, current_config)
    if _corrections:
        print(f"  [CONTRACT] Self-improver plan corrections: {'; '.join(_corrections)}")
        # Apply corrections back to plan object
        plan.new_critic_score_threshold = _safe_plan["new_critic_score_threshold"]
        plan.recommended_max_files_per_squad = _safe_plan["recommended_max_files_per_squad"]
        plan.routing_updates = _safe_plan.get("routing_updates", {})
        if "cto_prompt_additions" in _safe_plan:
            plan.cto_prompt_additions = _safe_plan["cto_prompt_additions"]
        if "worker_prompt_additions" in _safe_plan:
            plan.worker_prompt_additions = _safe_plan["worker_prompt_additions"]

    # ── Apply the improvement plan to config ────────────────────────────────────
    new_config = current_config.copy()
    ac = new_config["agent_config"]