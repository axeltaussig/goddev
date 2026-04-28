"""
GodDev 3.5 — Squad Leader Agent (Multi-Tier Workforce)
========================================================

Squad Leaders are domain experts who translate the CTO's blueprint into
concrete, file-level work orders for their workers.

Each domain maps to a multi-tier role in the router:
  - architect:  Claude Opus (Lead) → Gemini 1.5 Pro (Worker) → GPT-4.1 (Fallback)
  - backend:    GPT-4.1 (Lead) → DeepSeek Chat (Worker) → Claude Opus (Fallback)
  - frontend:   Claude Opus (Lead) → Gemini 2.5 Flash (Worker) → DeepSeek Chat (Fallback)
  - devops:     DeepSeek Chat (Lead) → GPT-4.1 (Worker)
  - qa:         GPT-4.1 (Lead) → DeepSeek Chat (Worker) → Claude Opus (Fallback)

Squad Leaders use GPT-4.1 (via the router) for planning, then route
workers through the optimal multi-tier chain for execution.
"""
from __future__ import annotations

import json
import os

from langchain_core.messages import AIMessage
from langgraph.types import Command, Send

from ..state import GodDevState, SquadPlan, FileTask
from ..llm_router import acall as router_acall

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

## Affinity-Driven Orchestration (required)

The runtime hands you a **preferred_agents** map: task-class → provider (`claude`, `openai`, `gemini`, `deepseek`).
For **each** `file_tasks[]` entry, set `"agent"` to the provider that best matches the file's **affinity class**:

| Affinity class (infer from path) | Typical paths |
|----------------------------------|---------------|
| python_code | `*.py` |
| javascript_code / typescript_code | `*.js`, `*.jsx`, `*.mjs`, `*.ts`, `*.tsx` |
| tests | `*test*`, `tests/`, `__tests__/` |
| documentation | `*.md`, `*.rst` |
| config_files | `*.json`, `*.yaml`, `*.toml`, `.env`, nginx configs |
| docker | `Dockerfile*`, container-related |
| ci_cd | `.github/workflows/*`, CI configs |

Use **preferred_agents** when the file clearly matches a class; otherwise fall back to your domain default (`{worker_hint}`). The worker layer maps these providers onto the correct multi-tier router role, so precise affinity here materially improves quality and cost.

## Multi-Tier Agent Selection Guide — Match Model to Task

| Agent Role | Best For | Your Domain Default |
|------------|----------|---------------------|
| **architect** | Claude Opus (Lead) -> Gemini Pro (Worker) -> GPT-4.1 (Fallback) | architecture domain |
| **backend** | GPT-4.1 (Lead) -> DeepSeek (Worker) -> Claude Opus (Fallback) | backend domain |
| **frontend** | Claude Opus (Lead) -> Gemini Flash (Worker) -> DeepSeek (Fallback) | frontend domain |
| **devops** | DeepSeek (Lead) -> GPT-4.1 (Worker) | devops domain |
| **qa** | GPT-4.1 (Lead) -> DeepSeek (Worker) -> Claude Opus (Fallback) | qa domain |

**Domain baseline**: `{worker_hint}` — override per file using the affinity table above when `preferred_agents` supplies a better match.

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
Return ONLY valid JSON — no prose outside the JSON. The JSON must match this schema:
{{
  "squad_id": "string (e.g. arch-001)",
  "domain": "{domain}",
  "strategy": "parallel" or "sequential",
  "context": "string (brief explanation of plan)",
  "file_tasks": [
    {{
      "file_path": "string (absolute path starting with project_dir)",
      "description": "string",
      "agent": "string (one of: architect, backend, frontend, devops, qa)",
      "verification_cmd": "string or null",
      "depend_on": ["string (file_path)"] or []
    }}
  ],
  "estimated_files": number (total count of file_tasks)
}}"""


# ─── Domain Metadata ──────────────────────────────────────────────────────────

_DOMAIN_META = {
    "architecture": {
        "title": "Architecture & Design",
        "worker_hint": "architect",
        "description": "data models, interface specs, API contracts, type definitions",
    },
    "backend": {
        "title": "Backend Engineering",
        "worker_hint": "backend",
        "description": "server code, APIs, business logic, database access",
    },
    "frontend": {
        "title": "Frontend Engineering",
        "worker_hint": "frontend",
        "description": "UI components, HTML, CSS, JavaScript, animations",
    },
    "devops": {
        "title": "DevOps & Infrastructure",
        "worker_hint": "devops",
        "description": "Dockerfile, CI/CD, nginx configs, deployment scripts",
    },
    "qa": {
        "title": "Quality Assurance",
        "worker_hint": "qa",
        "description": "test suites, fixtures, mocks, coverage config",
    },
}


# ─── Node ─────────────────────────────────────────────────────────────────────


async def squad_leader_node(state: GodDevState) -> Command:
    """Squad Leader decomposes the blueprint for their domain and spawns workers.
    Async node — LangGraph supports async nodes natively when running with astream."""
    squad_task: dict = state.get("current_squad_task") or {}
    domain: str = squad_task.get("domain", "backend")
    milestones: list = squad_task.get("milestones", [])
    blueprint: dict = squad_task.get("blueprint") or state.get("project_blueprint") or {}

    meta = _DOMAIN_META.get(domain, _DOMAIN_META["backend"])
    job_id = (state.get("metadata") or {}).get("job_id", "unknown")

    project_dir = state.get("project_dir", "/opt/goddev/projects/unknown")
    project_name = state.get("project_name", "unknown")

    milestone_text = "\n".join(
        f"- **{m['title']}**: {m['description']}\n  Deliverables: {', '.join(m['deliverables'])}"
        for m in milestones
    )

    directory_structure = blueprint.get("directory_structure", [])
    dir_text = "\n".join(f"  {f}" for f in directory_structure)

    tech = blueprint.get("tech_stack", {})
    tech_text = (
        f"Languages: {', '.join(tech.get('languages', []))}\n"
        f"Frameworks: {', '.join(tech.get('frameworks', []))}\n"
        f"Databases: {', '.join(tech.get('databases', []))}"
    )

    # Apply learned preferred_agents from runtime_config (domain-aware hint)
    runtime_cfg = (state.get("metadata") or {}).get("runtime_config", {})
    preferred_agents = runtime_cfg.get("preferred_agents", {}) or {}
    if not isinstance(preferred_agents, dict):
        preferred_agents = {}
    worker_hint = meta.get("worker_hint", "openai")
    domain_key = {
        "architecture": "architecture",
        "backend": "python_code",
        "frontend": "javascript_code",
        "devops": "docker",
        "qa": "tests",
    }.get(domain, domain)
    if domain_key in preferred_agents:
        worker_hint = preferred_agents[domain_key]
    else:
        for file_type, preferred in preferred_agents.items():
            if file_type in domain or domain in str(file_type):
                worker_hint = preferred
                break

    squad_additions = (runtime_cfg.get("squad_leader_prompt_additions") or "").strip()
    preferred_blob = json.dumps(preferred_agents, indent=2) if preferred_agents else "{}"

    # Use multi-tier router with 'squad_lead' role (GPT-4.1 Lead -> DeepSeek Worker)
    system = _SQUAD_LEAD_SYSTEM.format(
        domain=domain,
        domain_title=meta["title"],
        worker_hint=worker_hint,
    )
    if squad_additions:
        system += f"\n\n## Additional squad guidance (from self-improvement memory)\n{squad_additions[:2000]}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"""**Project**: {project_name}
**Project Dir**: {project_dir}
**Your Domain**: {domain} - {meta["description"]}

**Tech Stack**:
{tech_text}

**Architecture Overview**:
{blueprint.get("architecture_overview", "Not specified")}

**Your Milestones**:
{milestone_text}

**Full Directory Structure (ALL files in the project)**:
{dir_text}

**preferred_agents (runtime affinity map — follow Affinity-Driven Orchestration rules)**:
```json
{preferred_blob}
```

Create a SquadPlan covering ALL files that belong to the {domain} domain.
Every file_path in your plan MUST be an absolute path starting with {project_dir}.
IMPORTANT: If the file list above includes {domain}-relevant files like package.json, index.html,
or README.md, ensure they appear in your plan - do NOT skip any files from your domain.

Return ONLY valid JSON matching the schema provided."""},
    ]

    # Call async router directly - LangGraph handles async nodes natively
    result_text = await router_acall("squad_lead", messages, metadata={"job_id": job_id})

    # Parse the JSON response into a SquadPlan
    try:
        # Find JSON block in response (handle code fences)
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
        else:
            json_str = result_text
        
        plan_dict = json.loads(json_str)
        # Convert file_tasks to proper FileTask objects
        file_tasks = []
        for ft in plan_dict.get("file_tasks", []):
            file_tasks.append(FileTask(
                file_path=ft["file_path"],
                description=ft.get("description", ""),
                agent=ft.get("agent", worker_hint),
                verification_cmd=ft.get("verification_cmd"),
                depend_on=ft.get("depend_on", []),
            ))
        
        plan = SquadPlan(
            squad_id=plan_dict.get("squad_id", f"{domain}-001"),
            domain=domain,
            strategy=plan_dict.get("strategy", "parallel"),
            context=plan_dict.get("context", f"Plan for {domain} domain"),
            file_tasks=file_tasks,
            estimated_files=plan_dict.get("estimated_files", len(file_tasks)),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"[{domain.upper()} Squad Lead] Failed to parse router response: {exc}")
        print(f"[{domain.upper()} Squad Lead] Raw response: {result_text[:500]}")
        # Fallback: create minimal plan
        plan = SquadPlan(
            squad_id=f"{domain}-fallback",
            domain=domain,
            strategy="parallel",
            context=f"Fallback plan for {domain} domain",
            file_tasks=[],
            estimated_files=0,
        )

    # Auto-append missed deliverables
    assigned_files = set()
    for m in milestones:
        for d in m.get("deliverables", []):
            if d.endswith("/") or not "." in d.split("/")[-1]:
                continue
            abs_d = d if d.startswith(project_dir) else os.path.join(project_dir, d.lstrip("/"))
            assigned_files.add(abs_d)

    planned_files = {ft.file_path for ft in plan.file_tasks}
    for required in assigned_files:
        if required not in planned_files and not any(required in pf or pf in required for pf in planned_files):
            plan.file_tasks.append(FileTask(
                file_path=required,
                description=f"Auto-appended task for missing deliverable: {required.replace(project_dir, '')}",
                agent=worker_hint,
            ))
            print(f"[{domain.upper()} Squad Lead] Auto-appended dropped file: {required}")

    # Announce the squad plan
    lines = [
        f"### [{domain.upper()} Squad Lead] Plan - `{plan.squad_id}`",
        f"**Strategy**: `{plan.strategy}` | **Files**: {len(plan.file_tasks)}",
        f"_{plan.context[:200]}_",
        "",
    ]
    for i, ft in enumerate(plan.file_tasks, 1):
        lines.append(
            f"  {i}. `{ft.agent.upper()}` -> `{ft.file_path}` - {ft.description[:80]}"
        )

    plan_msg = AIMessage(content="\n".join(lines))

    # Propagate metadata for workers
    project_id = blueprint.get("project_id", "")
    existing_meta = state.get("metadata") or {}
    worker_metadata = {
        **existing_meta,
        "project_id": project_id,
        "job_id": job_id,
    }

    # Spawn one worker per file task through the multi-tier workforce
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
