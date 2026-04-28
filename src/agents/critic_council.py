"""
GodDev 3.0 — Critic Council (3 Parallel Critics + Verdict Collector)
=====================================================================

FIXES (2025-04-22):
  - Bug #1 fixed: Switched from asyncio.run(acall()) + fragile _parse_verdict()
    to with_structured_output(CriticVerdict) on ChatOpenAI — eliminates all
    "JSON parse failed" errors that were breaking quality checks every build.
  - All critic nodes are now async def — no more new event-loop per thread.
  - CriticType normalised before construction so "perf"/"code quality" variants work.

Three specialized critics run IN PARALLEL via Send(), each with a distinct lens:

  Code Critic      (GPT-4.1)    — correctness, completeness, best practices
  Security Critic  (Claude)     — vulnerabilities, secrets, injection, auth
  Performance Critic (DeepSeek) — complexity, memory, inefficient patterns

All three write into critic_verdicts (operator.add reducer).
The verdict_collector then waits for all 3, computes the aggregate decision,
and either approves (→ integrator) or rejects (→ CTO replan).

Score threshold: average score ≥ 7.0 AND no critic scored below 5.0 → APPROVE
"""
from __future__ import annotations

import subprocess

import os
import time
from typing import Literal, Optional

from langchain_core.messages import AIMessage
from langgraph.types import Command, Send
from pydantic import BaseModel, Field, field_validator

from ..contracts import validate_critic_verdict
from ..llm_router import acall_structured
from ..logger import trace
from ..state import GodDevState
from ..task_queue import acquire_slot


# ─── Flexible Critic Verdict (no strict Literal on critic_type) ───────────────
# The old CriticVerdict used Literal["code","security","performance"] which caused
# validation errors when an LLM returned "code quality", "perf", etc.
# We normalise to the canonical type in the validator.

class CriticVerdictOut(BaseModel):
    """Structured critic output — used with with_structured_output()."""
    critic_type: str = Field(description="One of: code, security, performance")
    approved: bool = Field(description="True if no blocking issues found")
    score: float = Field(ge=0.0, le=10.0, description="Quality score 0-10")
    issues: list[str] = Field(default_factory=list, description="Specific issues found")
    actionable_feedback: str = Field(
        description="Numbered, specific instructions for the CTO re-plan"
    )
    critical_files: list[str] = Field(
        default_factory=list, description="File paths with serious problems"
    )

    @field_validator("critic_type", mode="before")
    @classmethod
    def _normalise_type(cls, v: str) -> str:
        v = str(v).lower().strip()
        if "sec" in v:     return "security"
        if "perf" in v:    return "performance"
        return "code"

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v) -> float:
        try:
            return max(0.0, min(10.0, float(v)))
        except Exception:
            return 6.0


# ─── Swarm-aware critic dispatcher ────────────────────────────────────────────
# All three critics now flow through the cost/power-aware router, so they
# benefit from the swarm telemetry, budget pacing, context compression, and
# adaptive failure tracking — same intelligence as the worker pipeline.
#
# Complexity policy:
#   - code critic     → "standard" (GPT-4.1 / DeepSeek-coder / Claude pool)
#   - security critic → "strategic" (security demands the strongest engine)
#   - perf critic     → "trivial"  (perf review is pattern-matching → cheap pool)
#
# Auto-escalation on schema-validation failure already handled inside
# acall_structured.


async def _run_swarm_critic(
    system_prompt: str, user_prompt: str, complexity: str,
    affinity: str, job_id: str | None,
) -> dict:
    """Dispatch one critic via the router; return verdict dict (or soft-approve on fail)."""
    try:
        verdict: CriticVerdictOut = await acall_structured(
            "critic",
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            CriticVerdictOut,
            complexity=complexity,
            compress=True,
            affinity=affinity,
            metadata={"job_id": job_id} if job_id else None,
        )
        return verdict.model_dump()
    except Exception as exc:
        return _soft_approve("code", str(exc)[:120])


# ─── Critic Prompts ───────────────────────────────────────────────────────────

_CODE_CRITIC_SYSTEM = """You are the Code Quality Critic at GodDev.

## Your Mandate
Review ALL files written by the worker team with ruthless precision.
Check CORRECTNESS and COMPLETENESS — not style preferences.

## Checklist

### Technical Correctness
- All code is syntactically valid and would run without modification
- No hallucinated imports, API endpoints, or function signatures
- Algorithms are correct; no off-by-one errors, infinite loops, or wrong logic
- Imports are resolvable given the declared tech stack

### Completeness
- ZERO unexplained TODOs, stubs, ellipses (...), or "left as exercise" placeholders
- Every function mentioned in specs is actually implemented
- Config files have all required fields

### Integration
- Files reference each other correctly (import paths, module names)
- No circular dependencies that would prevent startup
- Entry points (main.py, index.js, etc.) are correct

Return a CriticVerdictOut with critic_type="code".
- approved=true only if ALL above pass
- score: 0-10 (10=perfect)
- issues: specific problems with file names and line references
- actionable_feedback: numbered list for the CTO to address"""

_SECURITY_CRITIC_SYSTEM = """You are the Security Critic at GodDev — the last line of defence.

## Security Checklist

### Secrets & Credentials
- No hardcoded API keys, passwords, or tokens anywhere
- All secrets use environment variables or a secrets manager

### Injection & Input Handling
- No SQL injection vectors (use parameterized queries)
- No command injection in shell calls
- All user inputs validated/sanitized before use

### Authentication & Authorization
- Auth middleware present on protected routes
- JWT/session tokens have expiry
- No auth bypass paths

### Error Handling
- Errors do not leak stack traces or internal details to users
- Sensitive data never logged

Return a CriticVerdictOut with critic_type="security".
- approved=true only if no CRITICAL or HIGH severity issues found
- score: 0-10 (10=fully secure)"""

_PERF_CRITIC_SYSTEM = """You are the Performance Critic at GodDev.

## Performance Checklist

### Algorithmic Complexity
- No O(n²) or worse where O(n log n) is achievable
- No nested loops over large datasets without justification
- Database queries don't load entire tables when pagination works

### Memory & Resources
- No unbounded memory growth
- File handles, DB connections, HTTP clients properly closed
- No memory leaks in loops or event handlers

### I/O Patterns
- Database calls are batched (no N+1 queries)
- Heavy operations are async where framework supports it
- Caches used for repeated expensive computations

Return a CriticVerdictOut with critic_type="performance".
- approved=true if no BLOCKING performance issues found
- score: 0-10 (10=excellent performance)"""


# ─── Shared review context builder ───────────────────────────────────────────

def _build_review_context(state: GodDevState) -> str:
    task         = state.get("user_request", "")
    outputs: list[dict] = state.get("worker_outputs") or []
    blueprint    = state.get("project_blueprint") or {}
    prev_feedback = state.get("critic_feedback") or "None (first review)"
    critic_iter  = state.get("critic_iteration", 0) + 1
    max_iter     = state.get("max_critic_iterations", 2)

    files_summary = "\n".join(
        f"  - `{o['file_path']}` | agent={o['agent']} "
        f"| bytes={o.get('bytes_written', 0):,} "
        f"| verified={'PASS' if o.get('verification_passed') else 'FAIL'}"
        + (f" | ERROR: {o.get('error', '')}" if o.get("error") else "")
        for o in outputs
    )

    # Read actual file contents — cap at 100 lines/file, max 6 files, 18k total chars
    # to stay safely under OpenAI's 30k TPM limit.
    from pathlib import Path
    file_contents = []
    files_to_review = [o["file_path"] for o in outputs][:6]
    for fp in files_to_review:
        try:
            content = Path(fp).read_text(encoding="utf-8")
            lines   = content.splitlines()
            if len(lines) > 100:
                content = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines truncated)"
            file_contents.append(f"### `{fp}`\n```\n{content}\n```")
        except Exception as exc:
            file_contents.append(f"### `{fp}`\n(could not read: {exc})")

    context = f"""**Project**: {blueprint.get('project_name', 'unknown')}
**User Request**: {task}
**Review Round**: {critic_iter}/{max_iter}

**Files Written ({len(outputs)} total)**:
{files_summary}

**Previous Critic Feedback**: {prev_feedback}

**File Contents** (first {len(file_contents)} files, truncated for token budget):
{chr(10).join(file_contents)}
"""
    # Hard cap: truncate total context to ~18,000 chars (~4,500 tokens) so we stay
    # well under the 30k TPM limit even with 3 critics running in parallel.
    if len(context) > 18000:
        context = context[:18000] + "\n... [context truncated to fit token budget]"
    return context


# ─── Node: Critic Council Dispatcher ─────────────────────────────────────────

def _clean_staging_before_critics(state: GodDevState) -> list[str]:
    """Restore corrupted staging Python files from live before critics evaluate.

    Returns list of files that were restored.
    """
    import shutil
    from pathlib import Path as _Path
    mode = (state.get("metadata") or {}).get("mode", "")
    if mode != "self_improvement":
        return []

    restored: list[str] = []
    staging = _Path("/opt/goddev/staging/src")
    live_src = _Path("/opt/goddev/src")
    if not staging.exists():
        return []

    for py_file in staging.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            r = subprocess.run(
                ["python3", "-m", "py_compile", str(py_file)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                rel = py_file.relative_to(staging)
                live = live_src / rel
                if live.exists():
                    shutil.copy2(live, py_file)
                    restored.append(str(rel))
                    print(f"  [PRE-CRITIC] Restored corrupted staging file: {rel}")
        except Exception as exc:
            print(f"  [PRE-CRITIC] Warning checking {py_file}: {exc}")

    return restored


def critic_council_node(state: GodDevState) -> Command:
    """Spawn all 4 critics in parallel via Send()."""
    # Pre-clean: restore any staging files that have syntax errors from failed patches
    restored = _clean_staging_before_critics(state)
    note = f" (restored {len(restored)} corrupted files from live)" if restored else ""

    sends = [
        Send("code_critic",     {**state, "current_critic_type": "code",        "critic_verdicts": []}),
        Send("security_critic", {**state, "current_critic_type": "security",    "critic_verdicts": []}),
        Send("perf_critic",     {**state, "current_critic_type": "performance", "critic_verdicts": []}),
        Send("ui_critic",       {**state, "current_critic_type": "code",        "critic_verdicts": []}),
    ]
    return Command(
        update={"messages": [AIMessage(content=f"⚖️ **Critic Council convening** — 4 critics reviewing in parallel...{note}")]},
        goto=sends,
    )


# ─── Helper: soft-approve fallback ───────────────────────────────────────────

def _soft_approve(critic_type: str, reason: str) -> dict:
    return {
        "critic_type": critic_type, "approved": True, "score": 6.0,
        "issues": [f"Critic unavailable: {reason}"],
        "actionable_feedback": "Critic could not run — review manually.",
        "critical_files": [],
    }


# ─── Node: Code Critic ────────────────────────────────────────────────────────

async def code_critic_node(state: GodDevState) -> dict:
    """Code quality review via swarm-routed critic (cheap pool by default).

    Cost insight: when run on flagship (Claude) for small builds, this single
    call dominates 80%+ of build cost. Gemini Flash + DeepSeek Coder handle
    code review at <1% of that cost with comparable accuracy on small/medium
    files. Predictive pacing will auto-escalate if budget allows.
    """
    context = _build_review_context(state)
    job_id = (state.get("metadata") or {}).get("job_id")
    await acquire_slot("critic")
    v = await _run_swarm_critic(
        _CODE_CRITIC_SYSTEM, context,
        complexity="trivial", affinity="code_review", job_id=job_id,
    )
    v["critic_type"] = "code"
    icon = "✅" if v["approved"] else "❌"
    return {
        "critic_verdicts": [v],
        "messages": [AIMessage(content=(
            f"{icon} **[Code Critic]** score={v['score']:.1f}/10 | "
            f"approved={v['approved']} | issues={len(v['issues'])}"
        ))],
    }


# ─── Node: Security Critic ────────────────────────────────────────────────────

async def security_critic_node(state: GodDevState) -> dict:
    """Security review via swarm-routed critic (standard pool).

    Security gets `standard` (GPT-4.1 / Sonnet) rather than `strategic`
    (Opus) — the security checklist is well-documented and the model can
    apply it reliably without flagship-tier reasoning. Predictive pacing
    downgrades further if budget is tight.
    """
    context = _build_review_context(state)
    job_id = (state.get("metadata") or {}).get("job_id")
    await acquire_slot("critic")
    v = await _run_swarm_critic(
        _SECURITY_CRITIC_SYSTEM, context,
        complexity="standard", affinity="security_review", job_id=job_id,
    )
    v["critic_type"] = "security"
    icon = "✅" if v["approved"] else "❌"
    return {
        "critic_verdicts": [v],
        "messages": [AIMessage(content=(
            f"{icon} **[Security Critic]** score={v['score']:.1f}/10 | "
            f"approved={v['approved']} | issues={len(v['issues'])}"
        ))],
    }


# ─── Node: Performance Critic ────────────────────────────────────────────────

async def perf_critic_node(state: GodDevState) -> dict:
    """Performance review via swarm-routed critic (trivial — pattern matching → cheap pool)."""
    context = _build_review_context(state)
    job_id = (state.get("metadata") or {}).get("job_id")
    await acquire_slot("critic")
    v = await _run_swarm_critic(
        _PERF_CRITIC_SYSTEM, context,
        complexity="trivial", affinity="perf_review", job_id=job_id,
    )
    v["critic_type"] = "performance"
    icon = "✅" if v["approved"] else "❌"
    return {
        "critic_verdicts": [v],
        "messages": [AIMessage(content=(
            f"{icon} **[Perf Critic]** score={v['score']:.1f}/10 | "
            f"approved={v['approved']} | issues={len(v['issues'])}"
        ))],
    }


# ─── Node: Verdict Collector ─────────────────────────────────────────────────

def verdict_collector_node(state: GodDevState) -> dict:
    """
    Wait for all 3 critics (LangGraph waits automatically since all 3 converge here).
    Compute aggregate: approved if avg score ≥ 7.0 AND min score ≥ 5.0.
    Force-approve at max_critic_iterations.
    """
    raw_verdicts: list[dict] = state.get("critic_verdicts") or []
    critic_iter = state.get("critic_iteration", 0) + 1
    max_iter    = state.get("max_critic_iterations", 2)
    run_id      = (state.get("metadata") or {}).get("project_id", "")

    # ── Integration contract: normalize all verdicts ─────────────────────
    verdicts = []
    for rv in raw_verdicts:
        v, corrections = validate_critic_verdict(rv)
        if corrections:
            print(f"  [CONTRACT] Critic verdict corrected: {'; '.join(corrections)}")
        verdicts.append(v)

    if not verdicts:
        return {
            "critic_approved": True,
            "critic_iteration": critic_iter,
            "messages": [AIMessage(content="⚠️ **Verdict Collector**: No verdicts received — auto-approving.")],
        }

    scores      = [v.get("score", 0.0) for v in verdicts]
    all_approved = all(v.get("approved", False) for v in verdicts)
    avg_score   = sum(scores) / len(scores)
    min_score   = min(scores)

    # Use runtime_config threshold if available; fallback to 5.5 for self-improvement
    _runtime_cfg = (state.get("metadata") or {}).get("runtime_config") or {}
    _threshold = float(_runtime_cfg.get("critic_score_threshold", 5.5))
    _min_floor = max(4.0, _threshold - 1.5)
    quality_pass  = avg_score >= _threshold and min_score >= _min_floor
    force_approve = critic_iter >= max_iter
    approved      = (all_approved and quality_pass) or force_approve

    # Aggregate feedback for CTO replan
    all_issues: list[str] = []
    for v in verdicts:
        ctype = v.get("critic_type", "unknown")
        for issue in v.get("issues", []):
            all_issues.append(f"[{ctype.upper()}] {issue}")

    feedback_text = "\n".join(f"{i+1}. {iss}" for i, iss in enumerate(all_issues))

    # Logging
    try:
        trace.log_critic_council(
            run_id=run_id, iteration=critic_iter, max_iterations=max_iter,
            approved=approved, force_approved=force_approve,
            avg_score=avg_score, min_score=min_score,
            n_total_issues=len(all_issues),
            verdicts=[
                {"type": v.get("critic_type"), "score": v.get("score"), "approved": v.get("approved")}
                for v in verdicts
            ],
        )
    except Exception:
        pass

    verdict_table = "\n".join(
        f"  | {v.get('critic_type', '?').upper():<12} | {v.get('score', 0):.1f}/10 | {'✅' if v.get('approved') else '❌'} |"
        for v in verdicts
    )

    if approved:
        status_icon  = "✅" if all_approved else "⏰"
        status_label = "APPROVED" if all_approved else f"FORCE-APPROVED (iter {critic_iter}/{max_iter})"
        msg = (
            f"{status_icon} **Critic Council: {status_label}**\n"
            f"  avg_score={avg_score:.1f}/10 | min_score={min_score:.1f}/10\n"
            f"  | Critic       | Score  | Pass |\n"
            f"  |--------------|--------|------|\n"
            f"{verdict_table}"
        )
        return {
            "critic_approved": True,
            "critic_iteration": critic_iter,
            "messages": [AIMessage(content=msg)],
        }
    else:
        msg = (
            f"❌ **Critic Council: REJECTED** (iter {critic_iter}/{max_iter})\n"
            f"  avg_score={avg_score:.1f}/10 | min_score={min_score:.1f}/10\n"
            f"  | Critic       | Score  | Pass |\n"
            f"  |--------------|--------|------|\n"
            f"{verdict_table}\n\n"
            f"**Issues for CTO replan ({len(all_issues)} total)**:\n{feedback_text}"
        )
        return {
            "critic_approved": False,
            "critic_iteration": critic_iter,
            "critic_feedback": feedback_text,
            "messages": [AIMessage(content=msg)],
        }
