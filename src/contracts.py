"""
GodDev 3.0 — Integration Contracts
====================================

Centralized validation functions for inter-node data contracts.
Every node boundary validates its inputs/outputs to catch issues
EARLY — before they cascade into expensive critic rejections.

Contract violations are logged and auto-corrected where possible,
rather than causing hard failures (graceful degradation).

Contracts validated:
  meta_cto → worker:        validate_blueprint_task()
  worker → polish_team:     validate_worker_output()
  critic → verdict_collector: validate_critic_verdict()
  self_improver → config:   validate_improvement_plan()
"""
from __future__ import annotations

from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

VALID_IMPROVEMENT_TYPES = frozenset({
    "prompt_enhancement", "new_tool", "logic_fix",
    "new_node", "perf_improvement", "schema_upgrade",
})

VALID_AGENTS = frozenset({"openai", "claude", "deepseek", "gemini"})

VALID_CRITIC_TYPES = frozenset({"code", "security", "performance"})

# Safe threshold bounds for self-improver
CRITIC_THRESHOLD_MIN = 4.0
CRITIC_THRESHOLD_MAX = 9.5
THRESHOLD_MAX_DELTA = 1.0  # max change per run

MAX_FILES_MIN = 3
MAX_FILES_MAX = 12

PROMPT_ADDITION_MAX_CHARS = 500


# ── Meta CTO → Worker Contract ──────────────────────────────────────────────

def validate_blueprint_task(task: dict, staging_dir: str) -> tuple[bool, str, dict]:
    """
    Validate a SelfImprovementTask before dispatching to a worker.

    Returns:
        (valid, reason, corrected_task)
        - valid: True if the task can proceed (possibly after auto-correction)
        - reason: human-readable explanation of any issues
        - corrected_task: task dict with auto-fixes applied
    """
    corrected = dict(task)
    issues: list[str] = []

    # 1. staging_file must be non-empty and a real path
    sf = (task.get("staging_file") or "").strip()
    if not sf or sf in (".", "/"):
        return False, f"empty staging_file: '{sf}'", corrected

    if not sf.startswith("/opt/goddev/staging/"):
        raw = (task.get("target_file") or "").lstrip("/")
        if raw.startswith("src/"):
            raw = raw[4:]
        corrected["staging_file"] = f"{staging_dir}/{raw}"
        issues.append(f"auto-fixed staging_file → {corrected['staging_file']}")

    # 2. target_file must be non-empty
    tf = (task.get("target_file") or "").strip()
    if not tf or tf in (".", "/"):
        return False, "empty target_file", corrected

    # 3. improvement_type must be valid
    itype = (task.get("improvement_type") or "").strip()
    if itype not in VALID_IMPROVEMENT_TYPES:
        corrected["improvement_type"] = "logic_fix"
        issues.append(f"invalid improvement_type '{itype}' → logic_fix")

    # 4. agent must be valid
    agent = (task.get("agent") or "").strip().lower()
    if agent not in VALID_AGENTS:
        corrected["agent"] = "deepseek"
        issues.append(f"invalid agent '{agent}' → deepseek")

    # 5. specific_changes must be substantive
    changes = (task.get("specific_changes") or "").strip()
    if len(changes) < 20:
        return False, f"specific_changes too vague ({len(changes)} chars)", corrected

    reason = "; ".join(issues) if issues else "OK"
    return True, reason, corrected


# ── Worker → Polish Team / Critics Contract ─────────────────────────────────

def validate_worker_output(output: dict) -> tuple[bool, list[str]]:
    """
    Validate a worker output dict meets minimum contract requirements.

    Returns:
        (valid, issues) — valid=True means output is acceptable for downstream nodes.
    """
    issues: list[str] = []

    fp = (output.get("file_path") or "").strip()
    if not fp:
        issues.append("missing file_path")
        return False, issues

    bw = output.get("bytes_written", 0)
    if bw < 50:
        issues.append(f"bytes_written={bw} (min 50)")

    if fp and not Path(fp).exists():
        issues.append(f"file not on disk: {fp}")

    if not output.get("task_id"):
        issues.append("missing task_id")

    return len(issues) == 0, issues


# ── Critic → Verdict Collector Contract ─────────────────────────────────────

def validate_critic_verdict(verdict: dict) -> tuple[dict, list[str]]:
    """
    Validate and normalize a critic verdict dict.

    Returns:
        (normalized_verdict, corrections) — corrections list is empty if no fixes needed.
    """
    v = dict(verdict)
    corrections: list[str] = []

    # score must be numeric 0-10
    score = v.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 5.0
        corrections.append(f"score not numeric → 5.0")
    v["score"] = max(0.0, min(10.0, score))
    if v["score"] != score:
        corrections.append(f"score clamped: {score} → {v['score']}")

    # approved must be bool
    if not isinstance(v.get("approved"), bool):
        v["approved"] = v["score"] >= 7.0
        corrections.append(f"approved inferred from score: {v['approved']}")

    # critic_type must be valid
    ct = str(v.get("critic_type", "code")).lower().strip()
    if "sec" in ct:
        ct = "security"
    elif "perf" in ct:
        ct = "performance"
    else:
        ct = "code"
    if ct != v.get("critic_type"):
        corrections.append(f"critic_type normalized: {v.get('critic_type')} → {ct}")
    v["critic_type"] = ct

    # issues must be a list
    if not isinstance(v.get("issues"), list):
        v["issues"] = []
        corrections.append("issues defaulted to []")

    # actionable_feedback must be a string
    if not isinstance(v.get("actionable_feedback"), str):
        v["actionable_feedback"] = ""
        corrections.append("actionable_feedback defaulted to ''")

    # critical_files must be a list
    if not isinstance(v.get("critical_files"), list):
        v["critical_files"] = []
        corrections.append("critical_files defaulted to []")

    return v, corrections


# ── Self-Improver → Config Contract ─────────────────────────────────────────

def validate_improvement_plan(plan_dict: dict, current_config: dict) -> tuple[dict, list[str]]:
    """
    Validate and clamp an ImprovementPlan before writing to runtime_config.
    Enforces INCREMENTAL changes — no big jumps that destabilize the system.

    Returns:
        (safe_plan, corrections) — safe_plan has all values clamped to safe ranges.
    """
    p = dict(plan_dict)
    corrections: list[str] = []
    current_ac = current_config.get("agent_config", {})

    # ── Critic score threshold: clamp + max ±1.0 per run ────────────────
    threshold = p.get("new_critic_score_threshold", 7.0)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 7.0
        corrections.append("threshold not numeric → 7.0")

    threshold = max(CRITIC_THRESHOLD_MIN, min(CRITIC_THRESHOLD_MAX, threshold))
    current_threshold = float(current_ac.get("critic_score_threshold", 7.0))
    delta = threshold - current_threshold
    if abs(delta) > THRESHOLD_MAX_DELTA:
        threshold = current_threshold + (THRESHOLD_MAX_DELTA if delta > 0 else -THRESHOLD_MAX_DELTA)
        corrections.append(
            f"threshold change too large ({delta:+.1f}) → clamped to {threshold:.1f} (max ±{THRESHOLD_MAX_DELTA})"
        )
    p["new_critic_score_threshold"] = round(threshold, 1)

    # ── Max files per squad: clamp ──────────────────────────────────────
    mf = p.get("recommended_max_files_per_squad", 8)
    try:
        mf = int(mf)
    except (TypeError, ValueError):
        mf = 8
    clamped = max(MAX_FILES_MIN, min(MAX_FILES_MAX, mf))
    if clamped != mf:
        corrections.append(f"max_files clamped: {mf} → {clamped}")
    p["recommended_max_files_per_squad"] = clamped

    # ── Routing updates: only valid agents ──────────────────────────────
    routing = p.get("routing_updates", {})
    if isinstance(routing, dict):
        bad_keys = [k for k, v in routing.items() if v not in VALID_AGENTS]
        for k in bad_keys:
            corrections.append(f"removed invalid routing: {k}→{routing[k]}")
            del routing[k]
        p["routing_updates"] = routing

    # ── Prompt additions: cap length ────────────────────────────────────
    for key in ("cto_prompt_additions", "worker_prompt_additions"):
        val = p.get(key, "")
        if isinstance(val, str) and len(val) > PROMPT_ADDITION_MAX_CHARS:
            p[key] = val[:PROMPT_ADDITION_MAX_CHARS]
            corrections.append(f"{key} truncated: {len(val)} → {PROMPT_ADDITION_MAX_CHARS}")

    return p, corrections
