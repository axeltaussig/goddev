"""
GodDev 3.0 — Squad Leader Agent
=================================

Squad Leaders are domain experts who translate the CTO's blueprint into
concrete, file-level work orders for their workers.

One Squad Leader runs per domain (architecture / backend / frontend / devops / qa).
Multiple squad leaders run IN PARALLEL via Send() from the CTO node.

Each Squad Leader produces a SquadPlan: an ordered list of FileTasks that
their workers will execute. Sequential tasks respect depend_on ordering;
parallel tasks run concurrently.
"""
from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.types import Command, Send

from ..state import GodDevState, SquadPlan

# ─── System Prompt ────────────────────────────────────────────────────────────

_SQUAD_LEAD_SYSTEM = """You are a Senior {domain_title} Lead at GodDev — an elite AI development team.

## Your Mission
Translate the CTO's ProjectBlueprint into a concrete, file-by-file execution plan for your squad.
You are the domain expert for **{domain}**. You do NOT write code — you plan what gets written.

## Your Responsibilities
1. Read the blueprint carefully — especially directory_structure and milestones for your domain
2. Decompose your domain's work into individual FileTasks (one per file)
3. Choose the right agent for each file (see agent guide below)
4. Set verification_cmd for every code file
5. Order tasks correctly — flag dependencies so workers execute in the right sequence

## Agent Selection Guide — Match Model to Task

| Agent | Best For | Your Domain Default |
|-------|----------|---------------------|
| **claude** | Architecture interfaces, data contracts, security-critical files | architecture domain |
| **openai** | APIs, business logic, React/Vue/Node code, complex integrations | backend domain |
| **gemini** | HTML/CSS/JS UI, animations, documentation, configuration files | frontend domain |
| **deepseek** | Test suites, algorithms, Dockerfiles, CI/CD, math-heavy code | devops + qa domains |

**Default for your domain**: `{worker_hint}` — prefer this unless file type suggests another agent.

## Verification Command Guide

| File Type | verification_cmd |
|-----------|-----------------|
| Python file | `python -m py_compile {{file_path}}` |
| JavaScript/TS | `node --check {{file_path}}` (or `tsc --noEmit`) |
| JSON | `python -c "import json; json.load(open('{{file_path}}'))"` |
| YAML | `python -c "import yaml; yaml.safe_load(open('{{file_path}}'))"` |
| Markdown/text | null (no verification needed) |

## Strategy Rules
- **parallel**: Use when files are INDEPENDENT (e.g. separate API routes, separate components)
- **sequential**: Use when files DEPEND on each other (e.g. models before routes before tests)
- MAX 8 file tasks per squad plan

## Output Format
Return a valid JSON SquadPlan — no prose outside the JSON."""

# ─── Domain Metadata ──────────────────────────────────────────────────────────

_DOMAIN_META = {
    "architecture": {
        "title": "Architecture & Design",
        "model": "openai",  # Squad leader uses GPT-4o for reliable structured output
        "worker_hint": "claude",  # Workers: Claude excels at interface design
        "description": "data models, interface specs, API contracts, type definitions",
    },
    "backend": {
        "title": "Backend Engineering",
        "model": "openai",
        "worker_hint": "openai",  # Workers: GPT-4o best for APIs and business logic
        "description": "server code, APIs, business logic, database access",
    },
    "frontend": {
        "title": "Frontend Engineering",
        "model": "openai",
        "worker_hint": "gemini",  # Workers: Gemini fast + great at HTML/CSS/JS
        "description": "UI components, HTML, CSS, JavaScript, animations",
    },
    "devops": {
        "title": "DevOps & Infrastructure",
        "model": "openai",
        "worker_hint": "deepseek",  # Workers: DeepSeek great at formulaic configs
        "description": "Dockerfile, CI/CD, nginx configs, deployment scripts",
    },
    "qa": {
        "title": "Quality Assurance",
        "model": "openai",
        "worker_hint": "deepseek",  # Workers: DeepSeek strong at test patterns
        "description": "test suites, fixtures, mocks, coverage config",
    },
}


# ─── LLM Factory ─────────────────────────────────────────────────────────────


def _get_lead_llm(model: str):
    if model == "claude":
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
            max_tokens=8192,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    if model == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    if model == "deepseek":
        from langchain_openai import ChatOpenAI as _OAI
        return _OAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=0,
            max_tokens=8192,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    # default: openai — use gpt-4.1 (configured via OPENAI_MODEL env var)
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        temperature=0, max_tokens=8192,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ─── Node ─────────────────────────────────────────────────────────────────────


def squad_leader_node(state: GodDevState) -> Command:
    """Squad Leader decomposes the blueprint for their domain and spawns workers."""
    squad_task: dict = state.get("current_squad_task") or {}
    domain: str = squad_task.get("domain", "backend")
    milestones: list = squad_task.get("milestones", [])
    blueprint: dict = squad_task.get("blueprint") or state.get("project_blueprint") or {}

    meta = _DOMAIN_META.get(domain, _DOMAIN_META["backend"])
    llm = _get_lead_llm(meta["model"]).with_structured_output(SquadPlan)

    project_dir = state.get("project_dir", "/opt/goddev/projects/unknown")
    project_name = state.get("project_name", "unknown")

    milestone_text = "\n".join(
        f"- **{m['title']}**: {m['description']}\n  Deliverables: {', '.join(m['deliverables'])}"
        for m in milestones
    )

    directory_structure = blueprint.get("directory_structure", [])
    # Filter to files likely relevant to this domain
    dir_text = "\n".join(f"  {f}" for f in directory_structure)

    tech = blueprint.get("tech_stack", {})
    tech_text = (
        f"Languages: {', '.join(tech.get('languages', []))}\n"
        f"Frameworks: {', '.join(tech.get('frameworks', []))}\n"
        f"Databases: {', '.join(tech.get('databases', []))}"
    )

    # ── Apply learned preferred_agents from runtime_config ──────────
    runtime_cfg = (state.get("metadata") or {}).get("runtime_config", {})
    preferred_agents = runtime_cfg.get("preferred_agents", {})
    # worker_hint default from domain meta, overridden by learned routing
    worker_hint = meta.get("worker_hint", "openai")
    for file_type, preferred in preferred_agents.items():
        if file_type in domain or domain in file_type:
            worker_hint = preferred
            break

    plan: SquadPlan = llm.invoke([
        SystemMessage(
            content=_SQUAD_LEAD_SYSTEM.format(
                domain=domain,
                domain_title=meta["title"],
                worker_hint=worker_hint,
            )
        ),
        HumanMessage(content=f"""**Project**: {project_name}
**Project Dir**: {project_dir}
**Your Domain**: {domain} \u2014 {meta["description"]}

**Tech Stack**:
{tech_text}

**Architecture Overview**:
{blueprint.get("architecture_overview", "Not specified")}

**Your Milestones**:
{milestone_text}

**Full Directory Structure (ALL files in the project)**:
{dir_text}

Create a SquadPlan covering ALL files that belong to the {domain} domain.
Every file_path in your plan MUST be an absolute path starting with {project_dir}.
IMPORTANT: If the file list above includes {domain}-relevant files like package.json, index.html,
or README.md, ensure they appear in your plan \u2014 do NOT skip any files from your domain."""
        ),
    ])

    # ── Auto-append missed deliverables ───────────────────────────────────────
    # Gather exactly what was explicitly assigned to this squad by the CTO
    assigned_files = set()
    for m in milestones:
        for d in m.get("deliverables", []):
            if d.endswith("/") or not "." in d.split("/")[-1]:
                continue # Probably a directory, skip
            
            # Map deliverable to an absolute path for matching
            abs_d = d if d.startswith(project_dir) else os.path.join(project_dir, d.lstrip("/"))
            assigned_files.add(abs_d)

    # Gather what the LLM actually planned
    planned_files = {ft.file_path for ft in plan.file_tasks}

    # Identify drops and auto-append
    for required in assigned_files:
        if required not in planned_files and not any(required in pf or pf in required for pf in planned_files):
            new_task = FileTask(
                file_path=required,
                description=f"Auto-appended task for missing deliverable: {required.replace(project_dir, '')}",
                agent="openai",  # safe fallback
            )
            plan.file_tasks.append(new_task)
            print(f"[{domain.upper()} Squad Lead] Auto-appended dropped file: {required}")

    # Announce the squad plan
    lines = [
        f"### [{domain.upper()} Squad Lead] Plan — `{plan.squad_id}`",
        f"**Strategy**: `{plan.strategy}` | **Files**: {len(plan.file_tasks)}",
        f"_{plan.context[:200]}_",
        "",
    ]
    for i, ft in enumerate(plan.file_tasks, 1):
        lines.append(
            f"  {i}. `{ft.agent.upper()}` → `{ft.file_path}` — {ft.description[:80]}"
        )

    plan_msg = AIMessage(content="\n".join(lines))

    # Propagate project_id + job_id into metadata so workers log correctly
    # and the file-lock registry can key by job_id to prevent cross-squad overwrites
    project_id = blueprint.get("project_id", "")
    existing_meta = state.get("metadata") or {}
    job_id = existing_meta.get("job_id", project_id)
    worker_metadata = {
        **existing_meta,
        "project_id": project_id,
        "job_id": job_id,
    }

    # Spawn one worker per file task
    sends = [
        Send(
            "worker",
            {
                **state,
                "metadata": worker_metadata,
                "current_file_task": {
                    **ft.model_dump(),
                    "squad_domain": domain,
                    "project_dir": project_dir,
                    "tech_stack": tech,
                    "architecture_overview": blueprint.get("architecture_overview", ""),
                    "squad_context": plan.context,
                },
                "worker_outputs": [],
            },
        )
        for ft in plan.file_tasks
    ]

    return Command(
        update={
            "squad_plans": [plan.model_dump()],
            "messages": [plan_msg],
        },
        goto=sends,
    )
