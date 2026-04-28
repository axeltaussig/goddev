"""
GodDev 3.0 — Meta CTO (Self-Improvement Mode)
===============================================

The Meta CTO is GodDev's self-improvement brain. Unlike the regular CTO
which designs user apps, the Meta CTO:

1. Reads EVERY source file in /opt/goddev/src/ (its own code)
2. Reads execution_trace.md (how it has been performing)
3. Reads memory/reflections.md (accumulated quality signals)
4. Reads memory/improvements.md (what has already been improved)
5. Reads memory/runtime_config.json (current configuration)

Then produces a MetaBlueprint: a concrete plan to improve specific files,
targeting real weaknesses identified in the performance data.

The improvements target the staging directory (/opt/goddev/staging/src/)
so they NEVER touch live code until the Self-Deployer validates and deploys.

Research basis:
  - AlphaCode 2: self-play and iterative refinement
  - Voyager (Wang et al. 2023): skill library self-improvement via code execution
  - OpenAI o1: chain-of-thought self-correction
  - Recursive Self-Improvement: each iteration should target the highest-leverage bottleneck
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from ..logger import trace
from ..state import GodDevState

# ─── Schema ───────────────────────────────────────────────────────────────────


class SelfImprovementTask(BaseModel):
    task_id: str
    target_file: str = Field(description="Relative path in src/ to improve, e.g. 'agents/worker.py'")
    staging_file: str = Field(description="Absolute staging path to write to")
    improvement_type: str = Field(description="prompt_enhancement | new_tool | logic_fix | new_node | perf_improvement | schema_upgrade")
    specific_changes: str = Field(description="Exact, specific changes to make in this file (not vague)")
    agent: str = Field(description="openai | claude | deepseek | gemini")
    verification_cmd: str = Field(default="", description="Verification command after writing")
    expected_impact: str = Field(description="What metric this improvement targets")


class MetaBlueprint(BaseModel):
    improvement_id: str = Field(description="kebab-case ID e.g. self-improve-run-003")
    analysis_summary: str = Field(description="What the data shows about current system weaknesses")
    highest_leverage_bottleneck: str = Field(description="The single biggest weakness to fix this run")
    tasks: list[SelfImprovementTask] = Field(description="Max 6 improvement tasks, ordered by priority")
    expected_outcomes: list[str] = Field(description="Measurable improvements expected after this run")
    staging_dir: str = Field(description="Absolute path to staging directory")


# ─── System Prompt ────────────────────────────────────────────────────────────

_META_CTO_SYSTEM = """You are GodDev's Meta CTO — the architect of the system's own evolution.

## Your Mission
You have just read GodDev's own source code, performance logs, and memory.
Your job: plan SPECIFIC, HIGH-IMPACT improvements to GodDev's code that will make the system measurably better.

## Analysis Framework

### What to look for in the data:
1. **Low verification pass rates** → Workers need better prompts or auto-fix logic
2. **Critic rejection patterns** → Specific gaps in agent prompts that cause repeated failures
3. **High wall times** → Parallelism not being leveraged, or timeout issues
4. **Low critic scores in specific types** → Code, Security, or Performance consistently low
5. **Missing files after builds** → Squad leaders not generating complete plans
6. **Repeated critic feedback themes** → Prompts not encoding the right constraints

### Improvement Types (pick the right one per task):
| type | what it changes | when to use |
|------|----------------|-------------|
| **prompt_enhancement** | System prompt in an agent | Agent consistently underperforms on specific task types |
| **new_tool** | Adds a tool to dev_tools.py | Agents failing because they lack a capability (e.g. no linter) |
| **logic_fix** | Bug or logic error in agent code | A specific bug identified in traces |
| **new_node** | New graph node | A missing capability that would fill a gap |
| **perf_improvement** | Speed/efficiency improvement | High wall times, unnecessary sequential ops |
| **schema_upgrade** | State or Pydantic schema | Missing fields causing data loss between nodes |

## Rules for MetaBlueprint
- STAGING DIR: always use `/opt/goddev/staging/src/` as the base for all staging files
- MAXIMUM 6 tasks — focus on the highest-leverage changes only
- Be BRUTALLY specific: "Add ESLint syntax check to run_shell_command whitelist" not "improve verification"
- Do NOT plan changes to: `.env`, `venv/`, `deploy/`, `static/index.html` (those are infrastructure)
- Each task must address a SPECIFIC observed weakness, not a vague idea

## Staging Output
All improved files go to `/opt/goddev/staging/src/` mirroring the live `src/` structure.
Example: to improve `src/agents/worker.py` → write to `/opt/goddev/staging/src/agents/worker.py`

Return a valid JSON MetaBlueprint. No prose outside the JSON."""


# ─── LLM ─────────────────────────────────────────────────────────────────────


def _get_meta_cto_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
        max_tokens=16000,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


# ─── Source reader ────────────────────────────────────────────────────────────


def _read_goddev_source() -> str:
    """Read all Python source files from /opt/goddev/src/ for the Meta CTO to analyse."""
    src_dir = Path("/opt/goddev/src")
    if not src_dir.exists():
        return "(source directory not found)"

    parts: list[str] = []
    for ext in ("*.py", "*.html", "*.css", "*.js"):
        for py_file in sorted(src_dir.parent.rglob(ext)):
            if "__pycache__" in str(py_file) or ".venv" in str(py_file) or "node_modules" in str(py_file):
                continue
            # Cap frontend files to avoid absolutely massive files
            rel = py_file.relative_to(src_dir.parent)
            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.splitlines()
                limit = 500 if ext == "*.py" else 800
                if len(lines) > limit:
                    content = "\n".join(lines[:limit]) + f"\n... ({len(lines) - limit} more lines)"
                parts.append(f"### /{rel}\n```{ext.replace('*.', '')}\n{content}\n```")
            except Exception as exc:
                parts.append(f"### /{rel}\n(error reading: {exc})")

    return "\n\n".join(parts)


def _read_source_file(target_file: str) -> str:
    """
    Read the LIVE source file so the worker sees the actual current code
    to improve — not just an abstract instruction.
    """
    # Strip leading slashes and 'src/' prefix variations
    rel = target_file.lstrip("/")
    if rel.startswith("src/"):
        rel = rel[4:]

    candidates = [
        Path("/opt/goddev/src") / rel,
        Path("/opt/goddev") / rel,
        Path("/opt/goddev/src/agents") / rel.split("/")[-1],
    ]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return "(file not found — write from scratch based on the specific_changes description)"


def _read_performance_data() -> str:
    """Read all performance and memory data."""
    parts: list[str] = []

    log_dir = Path(os.getenv("LOG_DIR", "/opt/goddev/logs"))
    memory_dir = Path(os.getenv("MEMORY_DIR", "/opt/goddev/memory"))

    for path, label in [
        (log_dir / "execution_trace.md", "Execution Trace"),
        (memory_dir / "reflections.md", "Reflections"),
        (memory_dir / "improvements.md", "Previous Improvements"),
        (memory_dir / "runtime_config.json", "Current Runtime Config"),
    ]:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if len(content) > 3000:
                content = content[-3000:]  # last 3000 chars = most recent
            parts.append(f"## {label}\n{content}")
        else:
            parts.append(f"## {label}\n(file not yet created)")

    return "\n\n---\n\n".join(parts)


# ─── Node ─────────────────────────────────────────────────────────────────────


def meta_cto_node(state: GodDevState) -> Command:
    """
    Meta CTO reads GodDev's own source + performance data
    and plans specific improvements to staging/.
    """
    llm = _get_meta_cto_llm().with_structured_output(MetaBlueprint)

    staging_dir = "/opt/goddev/staging/src"
    Path(staging_dir).mkdir(parents=True, exist_ok=True)

    critic_feedback: str = state.get("critic_feedback") or ""
    critic_iter: int = state.get("critic_iteration", 0)

    print("[Meta CTO] Reading GodDev source code...")
    source_code = _read_goddev_source()
    perf_data = _read_performance_data()

    plan_prompt = f"""Analyse GodDev 3.0 and plan improvements.

## Current Source Code
{source_code}

---

## Performance & Memory Data
{perf_data}"""

    if critic_feedback:
        plan_prompt += (
            f"\n\n---\n**CRITIC REJECTED PREVIOUS SELF-BUILD #{critic_iter}.**\n"
            f"Issues: {critic_feedback}\nFix all issues in your new plan."
        )

    # ── Vision Capture ────────────────────────────────────────────────────────
    import sys
    sys.path.append("/opt/goddev")
    try:
        from src.vision_capture import capture_frontend_screenshot_base64
        b64 = capture_frontend_screenshot_base64()
    except Exception:
        b64 = None

    human_content = [{"type": "text", "text": plan_prompt}]
    if b64:
        human_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
        print("[Meta CTO] Successfully attached GodDev UI Screenshot for visual reflection.")

    blueprint: MetaBlueprint = llm.invoke([
        SystemMessage(content=_META_CTO_SYSTEM),
        HumanMessage(content=human_content),
    ])

    # Ensure staging dir is set correctly
    blueprint.staging_dir = staging_dir

    # Fix staging file paths if the model used wrong base
    for task in blueprint.tasks:
        if not task.staging_file.startswith("/opt/goddev/staging/"):
            rel = task.target_file.lstrip("/")
            task.staging_file = f"/opt/goddev/staging/src/{rel}"

    # Log
    trace.start_run(
        run_id=blueprint.improvement_id,
        project_name=f"self-improve:{blueprint.improvement_id}",
        complexity="complex",
        n_milestones=len(blueprint.tasks),
        is_replan=bool(critic_feedback),
        critic_iter=critic_iter,
    )

    # Build announcement
    lines = [
        f"## 🔬 Meta CTO — Self-Improvement Plan: `{blueprint.improvement_id}`",
        f"",
        f"**Bottleneck**: {blueprint.highest_leverage_bottleneck}",
        f"**Analysis**: {blueprint.analysis_summary}",
        f"**Staging**: `{staging_dir}`",
        f"",
        f"### Improvement Tasks ({len(blueprint.tasks)})",
    ]
    for i, t in enumerate(blueprint.tasks, 1):
        lines.append(
            f"{i}. `[{t.improvement_type.upper()}]` `{t.target_file}` → `{t.agent}`"
        )
        lines.append(f"   _{t.specific_changes[:120]}_")
    lines += [
        f"",
        f"### Expected Outcomes",
        *[f"- {o}" for o in blueprint.expected_outcomes],
    ]

    plan_msg = AIMessage(content="\n".join(lines))

    sends = []
    for t in blueprint.tasks:
        # ── CRITICAL FIX: inject actual current source file content ──────────────
        # Workers have no file-reading tool. Without the actual code, they produce
        # generic improvements that don't match the real implementation.
        current_source = _read_source_file(t.target_file)
        source_preview = current_source[:4000] if len(current_source) > 4000 else current_source

        sends.append(Send(
            "worker",
            {
                **state,
                "project_blueprint": blueprint.model_dump(),
                "project_name": blueprint.improvement_id,
                "project_dir": staging_dir,
                "current_file_task": {
                    "task_id": t.task_id,
                    "file_path": t.staging_file,
                    "description": (
                        f"""SELF-IMPROVEMENT TASK [{t.improvement_type.upper()}]

## Target
Improve `{t.target_file}` in GodDev's source code.
Write the improved version to: `{t.staging_file}`

## Specific Changes Required
{t.specific_changes}

## Expected Impact
{t.expected_impact}

## Current Source Code (LIVE FILE — read this carefully)
```python
{source_preview}
```
{'(truncated — see above for full context)' if len(current_source) > 4000 else ''}

## Instructions
1. Read the current source carefully above
2. Apply EXACTLY the specified changes
3. Write the COMPLETE improved file — every line, not just the changed parts
4. The output MUST be a complete drop-in replacement (no stubs, no TODOs)
5. Do NOT change public interfaces or function signatures unless the task explicitly requires it
6. Python syntax must be valid — the file will be py_compile checked"""
                    ),
                    "agent": t.agent,
                    "squad_domain": "self-improvement",
                    "project_dir": "/opt/goddev/staging",
                    "tech_stack": {"languages": ["Python"], "frameworks": ["LangGraph", "FastAPI"], "databases": []},
                    "architecture_overview": "GodDev 3.0 hierarchical multi-agent system.",
                    "squad_context": f"Self-improvement run {blueprint.improvement_id}. All changes go to staging.",
                    "verification_cmd": t.verification_cmd or "python -m py_compile {file_path}",
                },
                "worker_outputs": [],
            },
        ))

    return Command(
        update={
            "user_request": f"Self-improvement run: {blueprint.improvement_id}",
            "project_blueprint": blueprint.model_dump(),
            "project_name": blueprint.improvement_id,
            "project_dir": staging_dir,
            "squad_plans": [],
            "worker_outputs": [],
            "critic_verdicts": [],
            "critic_approved": False,
            "messages": [plan_msg],
            "metadata": {
                **state.get("metadata", {}),
                "project_id": blueprint.improvement_id,
                "mode": "self_improvement",
                "staging_dir": staging_dir,
            },
        },
        goto=sends,
    )
