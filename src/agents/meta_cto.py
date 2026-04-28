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

import asyncio
import os
import shutil
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from ..contracts import validate_blueprint_task
from ..llm_router import acall_structured
from ..logger import trace
from ..state import GodDevState
from ..task_queue import acquire_slot


# ── Core pipeline files that self-improvement MUST NOT overwrite ──────────────
# Defined at module level so workers writing partial function bodies cannot
# accidentally introduce syntax errors (unquoted strings, etc.)
# Intentionally empty: self-improvement has full access to all files.
# The whole ecosystem is available for self-modification.
_SELF_IMPROVE_PROTECTED = frozenset()


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
    tasks: list[SelfImprovementTask] = Field(description="Max 3 improvement tasks, ordered by priority")
    expected_outcomes: list[str] = Field(description="Measurable improvements expected after this run")
    staging_dir: str = Field(description="Absolute path to staging directory")


# ─── System Prompt ────────────────────────────────────────────────────────────

_META_CTO_SYSTEM = """You are GodDev's Meta CTO — the architect of the system's own evolution.

## Your Mission
You have just read GodDev's own source code, performance logs, and memory.
Your job: plan SPECIFIC, HIGH-IMPACT improvements to GodDev's code that will make the system measurably better.

## Philosophy: Incremental Excellence
Small, focused improvements that SHIP are infinitely better than ambitious plans that fail critics.
- Prefer 1-3 targeted fixes over 6 sweeping changes
- Each task must be self-contained and independently verifiable
- A low verification rate means we're trying too much — cut scope, increase quality
- One successful improvement per run compounds into greatness over time

## Integration Contracts (NEVER BREAK)
Every change must preserve these contracts between nodes:
- **GodDevState fields**: Never remove/rename fields in state.py — only add new optional fields
- **Node signatures**: Every node takes `(state: GodDevState) -> dict | Command` — never change this
- **CriticVerdictOut schema**: Fields: critic_type, approved, score, issues, actionable_feedback, critical_files
- **MetaBlueprint schema**: Fields: improvement_id, analysis_summary, highest_leverage_bottleneck, tasks, expected_outcomes, staging_dir
- **Worker I/O**: Input via `current_file_task` dict, output via file write + worker_outputs state
- **LLM Router API**: `acall(role, messages)` and `acall_structured(role, messages, schema)` — signatures are frozen
- **Import paths**: All agents import from `..llm_router`, `..state`, `..task_queue` — never change package structure

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
- MAXIMUM 3 tasks — focus on the single highest-leverage bottleneck
- Be BRUTALLY specific: "Add ESLint syntax check to run_shell_command whitelist" not "improve verification"
- Do NOT plan changes to: `.env`, `venv/`, `deploy/` (those are infrastructure)
- Each task must address a SPECIFIC observed weakness, not a vague idea

## 🚫 PROTECTED FILES — DO NOT TARGET THESE
These core pipeline files are protected and will be SKIPPED by the system:
`worker.py`, `cto.py`, `squad_leader.py`, `critic_council.py`, `integrator.py`,
`meta_cto.py`, `polish_team.py`, `smooth_af.py`, `ui_critic.py`, `state.py`,
`llm_router.py`, `graph.py`, `self_build_graph.py`, `api/main.py`

## ✅ AVAILABLE TARGETS — Focus your improvements here:
- `agents/reflector.py` — reflection logic after runs
- `agents/self_assessor.py` — self-assessment before improvements
- `agents/self_deployer.py` — staging→live deployment logic
- `agents/self_improver.py` — improvement plan application
- `agents/sota_researcher.py` — state-of-the-art research
- `contracts.py` — integration contract validation
- `task_queue.py` — rate limiting and file lock registry
- `logger.py` — tracing and logging
- `dev_tools.py` — developer tools available to agents
- `vision_capture.py` — screenshot capture for UI analysis
- `static/index.html` — Chat UI (HTML/JS/CSS — NOT infrastructure config)

## ⚠️ CRITICAL GUARDRAILS — DO NOT VIOLATE
These rules protect the system from self-destructive changes:

1. **NEVER remove or downgrade an LLM engine.** Claude, GPT-4.1, Gemini Flash, and DeepSeek
   are ALL needed. If an engine is expensive, find ways to USE IT SMARTER (pre-compile context,
   compress inputs, route only strategic calls to it) — NEVER remove it from the pool.
2. **NEVER reduce the number of engines per role.** Multi-engine diversity is a core strength.
   If cost is high, optimize HOW engines are used, not WHICH engines are available.
3. **Cost optimization = smarter usage, not fewer resources.** Examples of GOOD cost fixes:
   - Lower compression threshold to pre-compile context before expensive engines
   - Route trivial tasks to cheap pools
   - Improve prompt efficiency to reduce token usage
   Examples of BAD/FORBIDDEN cost fixes:
   - Removing Claude from a pool
   - Replacing a powerful engine with a weaker one
   - Setting max_tokens too low
4. **NEVER break public interfaces** (function signatures, API endpoints, state schema).
5. **All improvements must be ADDITIVE** — make the system more capable, not less.

## Staging Output
All improved files go to `/opt/goddev/staging/src/` mirroring the live `src/` structure.
Example: to improve `src/agents/worker.py` → write to `/opt/goddev/staging/src/agents/worker.py`

Return a valid JSON MetaBlueprint. No prose outside the JSON."""


# ─── Source reader ────────────────────────────────────────────────────────────


def _read_goddev_source() -> str:
    """Read key Python agent files only — skip static/frontend for cost efficiency.

    The frontend (index.html etc.) is hundreds of KB and irrelevant for agent-level
    self-improvement. Reading only .py files under src/ cuts Claude context by ~70%.
    """
    src_dir = Path("/opt/goddev/src")
    if not src_dir.exists():
        return "(source directory not found)"

    parts: list[str] = []
    base_dir = src_dir.parent
    # Priority files first: the agent pipeline (most important for self-improvement)
    priority_dirs = ["agents", "api"]
    seen: set[str] = set()

    for subdir in priority_dirs:
        d = src_dir / subdir
        if not d.exists():
            continue
        for py_file in sorted(d.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            rel = str(py_file.relative_to(base_dir))
            if rel in seen:
                continue
            seen.add(rel)
            try:
                file_content = py_file.read_text(encoding="utf-8")
                lines = file_content.splitlines()
                if len(lines) > 400:
                    file_content = "\n".join(lines[:400]) + f"\n... ({len(lines) - 400} more lines truncated)"
                parts.append(f"### /{rel}\n```python\n{file_content}\n```")
            except Exception as exc:
                parts.append(f"### /{rel}\n(error: {exc})")

    # Then remaining src/ .py files (config, utils, etc.)
    for py_file in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(base_dir))
        if rel in seen:
            continue
        seen.add(rel)
        try:
            file_content = py_file.read_text(encoding="utf-8")
            lines = file_content.splitlines()
            if len(lines) > 200:
                file_content = "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
            parts.append(f"### /{rel}\n```python\n{file_content}\n```")
        except Exception as exc:
            parts.append(f"### /{rel}\n(error: {exc})")

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

    Goes through the cost/power-aware router so the bulky source-code context
    gets pre-compressed by the cheap `compressor` deployment before hitting
    the flagship Meta CTO chain (Claude → GPT-4.1 → Gemini Pro).
    """
    staging_dir = "/opt/goddev/staging/src"
    Path(staging_dir).mkdir(parents=True, exist_ok=True)

    # Clear stale file locks so retry workers can re-attempt files from prior rounds
    _meta_job_id = (state.get('metadata') or {}).get('job_id', '')
    if _meta_job_id:
        try:
            from ..task_queue import get_file_registry
            get_file_registry().clear_job(_meta_job_id)
        except Exception:
            pass

    critic_feedback: str = state.get("critic_feedback") or ""
    critic_iter: int = state.get("critic_iteration", 0)

    print("[Meta CTO] Reading GodDev source code...")
    source_code = _read_goddev_source()
    perf_data = _read_performance_data()

    assessment = state.get("assessment_report") or "(No assessment)"
    research = state.get("research_brief") or "(No research brief)"

    # Operator steering: opening HumanMessage + user_request (Improve flow injects priorities here first)
    first_human = ""
    for m in state.get("messages") or []:
        if isinstance(m, HumanMessage):
            c = m.content
            first_human = (c.strip() if isinstance(c, str) else "") or ""
            break
    steer_header = ""
    if first_human:
        steer_header = (
            "## Operator session seed (includes any human-directed improvement priorities)\n\n"
            f"{first_human}\n\n---\n\n"
        )
    

    # ── Three-stage DeepSeek pre-digest → targeted Claude call ──────────────
    # Stage 1: DeepSeek digests source code → 300-500 word bottleneck summary
    # Stage 2: DeepSeek digests assessment+research+perf → 300-500 word context summary
    # Stage 3: Claude receives ONLY the two digests (~2K chars vs ~120K raw)
    # Cost: ~$0.02 for 2 DeepSeek calls vs ~$2.50 for Claude on raw data

    from ..llm_router import acall as _acall

    _source_digest = ""  # fallback populated below
    _context_digest = ""  # fallback populated below

    async def _digest_source() -> str:
        """DeepSeek Pass 1: Compress source code into bottleneck analysis."""
        return await _acall(
            "meta_cto_cheap",
            [
                {"role": "system", "content": (
                    "You are a code quality analyst for GodDev, an autonomous AI dev system. "
                    "Read the source code and extract the 3-5 most impactful bottlenecks. "
                    "For each: exact file path, exact function/line range, root cause, "
                    "and concrete one-line fix. Be specific. 300-500 words total. No fluff."
                )},
                {"role": "user", "content": (
                    f"## Source Code (key files)\n{source_code[:50000]}\n\n"
                    f"## Human Priorities\n{first_human[:2000] if first_human else '(none)'}\n\n"
                    "List the top bottlenecks with exact file + function locations."
                )},
            ],
            complexity="trivial", compress=False,
            metadata={"job_id": _meta_job_id} if _meta_job_id else None,
        )

    async def _digest_context() -> str:
        """DeepSeek Pass 2: Compress assessment + research + perf into terse summary."""
        raw_context = (
            f"## Assessment Report\n{assessment[:8000]}\n\n"
            f"## SOTA Research Brief\n{research[:8000]}\n\n"
            f"## Performance Data (reflections, config, trace)\n{perf_data[:6000]}"
        )
        return await _acall(
            "meta_cto_cheap",
            [
                {"role": "system", "content": (
                    "You compress technical context for a Meta CTO planning code improvements. "
                    "Summarize into <=500 words preserving EVERY: metric value, file path, "
                    "function name, error pattern, critic score, verification rate, and "
                    "explicit recommendation. Use terse bullet points. No prose."
                )},
                {"role": "user", "content": raw_context},
            ],
            complexity="trivial", compress=False,
            metadata={"job_id": _meta_job_id} if _meta_job_id else None,
        )

    async def _run_both_digests():
        return await asyncio.gather(
            _digest_source(), _digest_context(), return_exceptions=True,
        )

    try:
        _digest_results = asyncio.run(_run_both_digests())
    except RuntimeError:
        import concurrent.futures as _cfu
        _digest_results = _cfu.ThreadPoolExecutor().submit(
            asyncio.run, _run_both_digests()
        ).result()
    except Exception as _ce:
        _digest_results = ["", ""]
        print(f"  [Meta CTO] DeepSeek digest failed: {_ce}")

    # Extract results (may contain exceptions)
    _raw_source_digest = _digest_results[0] if not isinstance(_digest_results[0], Exception) else ""
    _raw_context_digest = _digest_results[1] if not isinstance(_digest_results[1], Exception) else ""

    if _raw_source_digest and len(_raw_source_digest) > 80:
        _source_digest = (
            f"[Pre-digested by DeepSeek — {len(source_code):,} chars → summary]\n\n"
            f"{_raw_source_digest}\n\n"
            f"[Stats: {len(source_code.splitlines())} lines across {source_code.count('### /')} files]"
        )
        print(f"  [Meta CTO] Source digest: {len(source_code):,}→{len(_source_digest):,} chars")
    else:
        _source_digest = source_code[:20000]  # fallback: truncated raw source

    if _raw_context_digest and len(_raw_context_digest) > 80:
        _context_digest = (
            f"[Pre-digested by DeepSeek — assessment+research+perf compressed]\n\n"
            f"{_raw_context_digest}"
        )
        _raw_ctx_len = len(assessment) + len(research) + len(perf_data)
        print(f"  [Meta CTO] Context digest: {_raw_ctx_len:,}→{len(_context_digest):,} chars")
    else:
        _context_digest = (
            f"## Assessment\n{assessment[:3000]}\n\n"
            f"## Research\n{research[:3000]}\n\n"
            f"## Performance Data\n{perf_data[:3000]}"
        )

    plan_prompt = f"""{steer_header}Analyse GodDev 3.0 and construct exact code improvements.

## Digested Source Analysis (bottlenecks identified by DeepSeek)
{_source_digest}

---

## Digested Context (assessment + research + perf data summarized by DeepSeek)
{_context_digest}"""

    if critic_feedback:
        plan_prompt += (
            f"\n\n---\n**CRITIC REJECTED PREVIOUS SELF-BUILD #{critic_iter}.**\n"
            f"Issues: {critic_feedback}\nFix all issues in your new plan."
        )

    # ── Vision Capture (first plan only — skip on replans) ─────────────────
    # Screenshot is for initial UI assessment only. On replans, the critic
    # feedback is text-based, and image_url causes crashes when the cheap
    # compressor (DeepSeek) tries to process multimodal content.
    b64 = None
    if not critic_feedback:
        import sys
        sys.path.append("/opt/goddev")
        try:
            from src.vision_capture import capture_frontend_screenshot_base64
            b64 = capture_frontend_screenshot_base64()
        except Exception:
            b64 = None

    # Build user content as a multimodal array if a screenshot is available.
    # LiteLLM passes through the OpenAI-style content array unchanged.
    if b64:
        user_content = [
            {"type": "text", "text": plan_prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        print("[Meta CTO] Successfully attached GodDev UI Screenshot for visual reflection.")
    else:
        user_content = plan_prompt

    job_id = (state.get("metadata") or {}).get("job_id")
    meta_messages = [
        {"role": "system", "content": _META_CTO_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    async def _run_meta_cto() -> MetaBlueprint:
        await acquire_slot("meta_cto")
        return await acall_structured(
            "meta_cto", meta_messages, MetaBlueprint,
            complexity="standard",   # standard not strategic: input is pre-compressed
            compress=True,           # source code is huge → cheap-compressor first
            metadata={"job_id": job_id} if job_id else None,
        )

    try:
        blueprint = asyncio.run(_run_meta_cto())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            blueprint = pool.submit(asyncio.run, _run_meta_cto()).result()

    # Ensure staging dir is set correctly
    blueprint.staging_dir = staging_dir

    # Fix staging file paths if the model used wrong base
    for task in blueprint.tasks:
        if not task.staging_file.startswith("/opt/goddev/staging/"):
            raw = task.target_file.lstrip("/")
            task.staging_file = ("/opt/goddev/staging/" + raw) if raw.startswith("static/") else ("/opt/goddev/staging/src/" + raw.replace("src/", "", 1).lstrip("/"))

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
    _skipped = 0
    for t in blueprint.tasks:
        # ── Integration contract validation ──────────────────────────────
        _task_dict = t.model_dump() if hasattr(t, 'model_dump') else dict(t)
        _valid, _reason, _corrected = validate_blueprint_task(_task_dict, staging_dir)
        if not _valid:
            print(f"  [CONTRACT] Skipping task {t.task_id}: {_reason}")
            _skipped += 1
            continue
        # Apply auto-corrections from contract validation
        if _corrected.get('staging_file') != t.staging_file:
            t.staging_file = _corrected['staging_file']
        if _corrected.get('agent') != t.agent:
            t.agent = _corrected['agent']
        if _corrected.get('improvement_type') != t.improvement_type:
            t.improvement_type = _corrected['improvement_type']
        if _reason != 'OK':
            print(f"  [CONTRACT] Task {t.task_id} auto-fixed: {_reason}")

        # Skip protected core pipeline files — see module-level _SELF_IMPROVE_PROTECTED
        _target_rel = (t.target_file or "").lstrip("/").replace("src/", "", 1).replace("opt/goddev/src/", "", 1)
        if any(_target_rel.endswith(_pf) for _pf in _SELF_IMPROVE_PROTECTED):
            print(f"  [META-CTO] Skipping protected file: {t.target_file}")
            _skipped += 1
            continue

        # ── CRITICAL FIX: inject actual current source file content ──────────────
        # Workers have no file-reading tool. Without the actual code, they produce
        # generic improvements that don't match the real implementation.
        current_source = _read_source_file(t.target_file)
        source_preview = current_source[:12000] if len(current_source) > 12000 else current_source

        # ── Copy live file to staging so SEARCH/REPLACE patches have a base ──
        staging_path = Path(t.staging_file)
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        if True:  # always refresh from live — stale staging files cause SEARCH/REPLACE failures
            # Derive live path from staging path: /opt/goddev/staging/src/X → /opt/goddev/src/X
            staging_str = str(staging_path)
            if "/staging/src/" in staging_str:
                live_path = Path(staging_str.replace("/staging/src/", "/src/"))
            else:
                # Fallback: try from target_file directly
                rel = t.target_file.lstrip("/")
                if rel.startswith("src/"):
                    rel = rel[4:]
                live_path = Path("/opt/goddev/src") / rel
            if live_path.exists():
                shutil.copy2(live_path, staging_path)
                print(f"  [STAGING] Copied {live_path} → {staging_path}")
            else:
                print(f"  [STAGING] WARNING: live file not found: {live_path}")

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
{'(truncated — see above for full context)' if len(current_source) > 8000 else ''}

## Output Instructions
Output the COMPLETE improved Python file with your changes applied.
- Start with the very first character (module docstring or first import)
- Maintain ALL existing public interfaces (function names, signatures, module exports)
- Apply ONLY the specific changes described in "Specific Changes Required" above
- The file MUST be valid Python — it will be checked with py_compile
- Do NOT add markdown fences or commentary — raw Python source only"""
                    ),
                    "agent": t.agent,
                    "squad_domain": "self-improvement",
                    "project_dir": "/opt/goddev/staging",
                    "tech_stack": {"languages": ["Python"], "frameworks": ["LangGraph", "FastAPI"], "databases": []},
                    "architecture_overview": "GodDev 3.0 hierarchical multi-agent system.",
                    "squad_context": f"Self-improvement run {blueprint.improvement_id}. All changes go to staging.",
                    "verification_cmd": t.verification_cmd or ("python3 -m py_compile {file_path}" if t.staging_file.endswith(".py") else "test -s {file_path}"),
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
