"""
GodDev 3.0 — Reflection Agent
================================

The Reflector runs after every completed build. It analyses the execution
trace and writes structured improvement signals to memory/reflections.md.

Over time this file becomes the "institutional memory" of GodDev — patterns
that worked, patterns that failed, agent performance benchmarks.

The Reflector does NOT modify prompts directly. Instead it writes structured
markdown that the CTO prompt can optionally ingest in a future run.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage

from ..state import GodDevState


def reflector_node(state: GodDevState) -> dict:
    """Analyse the completed run and write reflection signals to disk.

    This function collects various metrics and metadata from the GodDevState
    after a complete development cycle (build + critic iterations). It
    computes performance signals, identifies patterns, and records this
    information into a structured markdown file (`reflections.md`) for
    future self-improvement and analysis.

    Args:
        state: The current GodDevState, containing all accumulated data
               from the ongoing development run.

    Returns:
        A dictionary containing:
        - "reflection_notes": The markdown string of the generated reflection entry.
        - "messages": A list containing an AIMessage with a summary of the reflection.
    """
    # Retrieve iteration context for logging early
    critic_iter = state.get("critic_iteration", 1)
    # Assumes 'max_critic_iterations' is available in state or defaults to 1
    max_runs = state.get("max_critic_iterations", 1)
    print(f'[Reflector] Analysing run {critic_iter} of {max_runs}')

    blueprint = state.get("project_blueprint") or {}
    worker_outputs: list[dict] = state.get("worker_outputs") or []
    critic_verdicts: list[dict] = state.get("critic_verdicts") or []
    squad_plans: list[dict] = state.get("squad_plans") or []
    integration_result: dict = state.get("integration_result") or {}
    metadata: dict = state.get("metadata") or {}

    project_name = state.get("project_name", "unknown")
    project_id = metadata.get("project_id", "unknown")
    complexity = metadata.get("complexity", "unknown")
    n_squads = metadata.get("n_squads", 0)
    n_files = metadata.get("n_files", 0)
    # critic_iter is already retrieved above

    # ── Compute signals ───────────────────────────────────────────────────────
    total_bytes = sum(o.get("bytes_written", 0) for o in worker_outputs)
    verif_pass = sum(1 for o in worker_outputs if o.get("verification_passed"))
    verif_fail = len(worker_outputs) - verif_pass
    verif_rate = verif_pass / len(worker_outputs) if worker_outputs else 0.0

    agent_counts: dict[str, int] = {}
    agent_bytes: dict[str, int] = {}
    agent_verif_rate: dict[str, list[int]] = {}  # [passed, total]
    for o in worker_outputs:
        a = o.get("agent", "unknown")
        agent_counts[a] = agent_counts.get(a, 0) + 1
        agent_bytes[a] = agent_bytes.get(a, 0) + o.get("bytes_written", 0)
        # Track verification by agent
        if a not in agent_verif_rate:
            agent_verif_rate[a] = [0, 0]
        agent_verif_rate[a][1] += 1
        if o.get("verification_passed"):
            agent_verif_rate[a][0] += 1

    critic_scores = {
        v.get("critic_type", "unknown"): v.get("score", 0.0)
        for v in critic_verdicts
    }
    avg_critic_score = (
        sum(critic_scores.values()) / len(critic_scores) if critic_scores else 0.0
    )

    integration_passed = integration_result.get("passed", False)
    files_missing = integration_result.get("files_missing", [])

    # ── Build reflection entry ────────────────────────────────────────────────
    utc_now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    agent_table = "\n".join(
        f"| {a} | {agent_counts[a]} files | {agent_bytes.get(a, 0):,} bytes | {agent_verif_rate[a][0]}/{agent_verif_rate[a][1]} verif |"
        for a in sorted(agent_counts)
    )
    critic_table = "\n".join(
        f"| {ctype} | {score:.1f}/10 |"
        for ctype, score in critic_scores.items()
    )

    # Identify patterns
    patterns: list[str] = []
    if verif_rate >= 0.9:
        patterns.append(f"HIGH_VERIF_RATE: {verif_rate:.0%} files passed syntax check — prompts working well")
    elif verif_rate < 0.5:
        patterns.append(f"LOW_VERIF_RATE: {verif_rate:.0%} — worker prompts need stronger verification instructions")

    if critic_iter == 1 and avg_critic_score >= 7.5:
        patterns.append("FIRST_PASS_QUALITY: Critics approved on round 1 with high scores — squad planning effective")
    elif critic_iter >= 2:
        patterns.append(f"MULTI_ROUND: Needed {critic_iter} critic rounds — consider stronger CTO blueprint requirements")

    if files_missing:
        patterns.append(f"MISSING_FILES: {len(files_missing)} files from blueprint not found — squad leader missed tasks")

    if not patterns:
        patterns.append("NOMINAL_RUN: No significant patterns detected")

    reflection_entry = f"""## Run: {utc_now} | project_id=`{project_id}`

### Project Metadata
| field | value |
|-------|-------|
| project_name | `{project_name}` |
| complexity | `{complexity}` |
| n_squads | {n_squads} |
| n_blueprint_files | {n_files} |
| n_files_written | {len(worker_outputs)} |
| total_bytes | {total_bytes:,} |

### Agent Performance (Efficiency Matters!)
| agent | files | bytes | verif_rate |
|-------|-------|-------|-----------|
{agent_table or "_(no data)_"}

**Note**: Cost/effectiveness tracking — agents with high verification rates and low token waste are preferred for cost-optimization in self-improvement.

### Quality Signals
| metric | value |
|--------|-------|
| verification_pass_rate | {verif_rate:.1%} |
| verification_pass | {verif_pass} |
| verification_fail | {verif_fail} |
| critic_rounds | {critic_iter} |
| avg_critic_score | {avg_critic_score:.1f}/10 |
| integration_passed | {integration_passed} |
| files_missing | {len(files_missing)} |

### Critic Scores
| critic | score |
|--------|-------|
{critic_table or "_(no data)_"}

### Patterns Detected
{chr(10).join(f"- {p}" for p in patterns)}

### Missing Files
{chr(10).join(f"- `{f}`" for f in files_missing) if files_missing else "_(none)_"}

---

"""

    # ── Write to memory file ──────────────────────────────────────────────────
    memory_dir = Path(os.getenv("MEMORY_DIR", "/opt/goddev/memory"))
    memory_dir.mkdir(parents=True, exist_ok=True)
    reflections_path = memory_dir / "reflections.md"

    try:
        # Write header if new or empty file
        if not reflections_path.exists() or reflections_path.stat().st_size == 0:
            reflections_path.write_text(
                "# GodDev 3.0 — Reflection Memory\n\n"
                "> Accumulated learning across all project builds.\n"
                "> Payload-agnostic: no task content, only structural signals.\n\n---\n\n",
                encoding="utf-8",
            )

        with open(reflections_path, "a", encoding="utf-8") as f:
            f.write(reflection_entry)
    except IOError as e:
        print(f"Error writing reflection to {reflections_path}: {e}")
        # Optionally, log to a different error stream or store in state for later reporting
        # For now, just print and continue, as the reflection itself is still in 'reflection_notes'

    # ── Final message ─────────────────────────────────────────────────────────
    msg = (
        f"🧠 **Reflector**: Run analysed and logged to `{reflections_path}`\n"
        f"- Verification rate: {verif_rate:.1%}\n"
        f"- Avg critic score: {avg_critic_score:.1f}/10\n"
        f"- Patterns: {'; '.join(patterns[:2])}"
    )

    print(f'[Reflector] Run complete - verif_rate={verif_rate:.0%} avg_score={avg_critic_score:.1f}/10')

    return {
        "reflection_notes": reflection_entry,
        "messages": [AIMessage(content=msg)],
    }