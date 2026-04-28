"""
GodDev 3.0 — CTO Agent (Claude Opus)
======================================

The CTO is the strategic brain. It receives the user's idea and produces a
complete ProjectBlueprint that every downstream agent follows.

On first entry: generates the ProjectBlueprint from the user request.
On re-entry (after Critic rejection): incorporates critic feedback and
  re-plans with corrective intent — can change tech stack, restructure
  milestones, or reassign squads.
"""
from __future__ import annotations

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, Send

from ..logger import trace
from ..state import GodDevState, ProjectBlueprint
from ..llm_router import acall_structured
from ..task_queue import acquire_slot

# ─── System Prompt ────────────────────────────────────────────────────────────

_CTO_SYSTEM = """You are the Chief Technology Officer of GodDev — the world's most capable autonomous AI development team.

## Your Role
You are the strategic brain. You receive an idea and produce a complete ProjectBlueprint that every downstream squad follows as their bible. You do NOT write code — you design systems.

## Your Squad Roster

| Squad | Domain | Responsibility |
|-------|--------|----------------|
| **architecture** | System Design | Data models, API contracts, interface definitions, ERDs |
| **backend** | Server/API | Business logic, server code, database queries, auth |
| **frontend** | UI/UX | Components, state management, routing, styling |
| **devops** | Infrastructure | Dockerfile, CI/CD, nginx, env configs, deployment |
| **qa** | Quality Assurance | Test suites, fixtures, mocks, coverage config |

## Blueprint Rules

**Project Directory:**
- ALWAYS use the PROJECTS_DIR env var base: `{projects_dir}`
- project_dir = `{projects_dir}/{{project_name}}`
- project_name MUST be kebab-case (e.g. `todo-api`, `snake-game`, `ecommerce-platform`)

**Directory Structure:**
- List EVERY file you plan to create, relative to project_dir
- Be exhaustive — missing files cause squad leaders to improvise badly
- Group by subdirectory (e.g. `src/routes/users.js`, `src/models/user.js`)

**Milestone Rules:**
- 1 milestone per squad domain (don't repeat squads)
- For simple apps: architecture + backend + frontend (3 squads)
- For complex apps: all 5 squads
- Each milestone's deliverables must map 1:1 to files in directory_structure

**Tech Stack Rules:**
- Pick ONE framework per concern (not React + Vue)
- Prefer battle-tested: Express/FastAPI/Rails for backend, React/Vue/Svelte for frontend
- Always include a testing framework in your stack

**CRITICAL FILE RULES (check before returning):**
- If the project uses Node.js/npm → `package.json` MUST be in directory_structure
- If the project is a web app → `index.html` MUST be in directory_structure  
- If the project uses Python → `requirements.txt` MUST be in directory_structure
- README.md MUST always be in directory_structure
- Every file listed in any milestone deliverable MUST appear in directory_structure
- Every file in directory_structure MUST be assigned to exactly ONE squad milestone

**Critic Feedback Integration:**
- If critic_feedback is present, directly address EVERY numbered issue
- You MAY change tech stack or restructure milestones if feedback demands it
- Be explicit in your reasoning about what changed

**Hard Limits:**
- Max 5 milestones (one per squad)
- Max 30 files in directory_structure for simple, 60 for complex
- estimated_complexity: simple (≤10 files), moderate (10-30 files), complex (30+ files)

Respond with a valid JSON ProjectBlueprint — no prose outside the JSON."""

# ─── Critical file guarantor ──────────────────────────────────────────────────


def _ensure_critical_files(blueprint) -> None:
    """
    Post-generation validation: auto-inject critical files that the LLM forgot.
    This prevents squad leaders from skipping files like package.json that the
    blueprint doesn't list but the project obviously needs.
    Modifies blueprint in-place. Works on any ProjectBlueprint object.
    """
    existing = {f.lower() for f in blueprint.directory_structure}
    langs = [lang.lower() for lang in blueprint.tech_stack.languages]
    frameworks = [fw.lower() for fw in blueprint.tech_stack.frameworks]

    needed: list[tuple[str, str]] = []  # (file_path, squad)

    # README always required
    if "readme.md" not in existing:
        needed.append(("README.md", "architecture"))

    # Node/npm projects need package.json
    if any(x in langs + frameworks for x in ["javascript", "typescript", "node", "react", "vue", "svelte", "next", "express"]):
        if "package.json" not in existing:
            needed.append(("package.json", "devops"))

    # Python projects need requirements.txt
    if "python" in langs:
        if "requirements.txt" not in existing and "pyproject.toml" not in existing:
            needed.append(("requirements.txt", "devops"))

    # Web apps need index.html
    if any(x in langs + frameworks for x in ["html", "javascript", "react"]):
        if "index.html" not in existing and not any("index.html" in f for f in existing):
            needed.append(("index.html", "frontend"))

    for file_path, preferred_squad in needed:
        blueprint.directory_structure.append(file_path)
        # Try to add to an existing milestone for this squad
        added = False
        for m in blueprint.milestones:
            if m.squad == preferred_squad:
                m.deliverables.append(file_path)
                added = True
                break
        if not added and blueprint.milestones:
            # Fall back to first milestone
            blueprint.milestones[0].deliverables.append(file_path)
        print(f"[CTO] Auto-injected missing critical file: {file_path} → {preferred_squad}")


# ─── Node ─────────────────────────────────────────────────────────────────────


def cto_node(state: GodDevState) -> Command:
    """CTO analyses/re-plans and spawns Squad Leaders via Send().

    Uses the cost/power-aware router (acall_structured) so the CTO benefits
    from context compression on bulky inputs and budget-paced routing.
    """
    # Extract user request
    user_request: str = state.get("user_request", "") or ""
    if not user_request:
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                user_request = str(msg.content)
                break

    projects_dir = os.getenv("PROJECTS_DIR", "/opt/goddev/projects")
    critic_feedback: str = state.get("critic_feedback") or ""
    critic_iter: int = state.get("critic_iteration", 0)
    is_replan = bool(critic_feedback)

    # Build planning prompt
    plan_prompt = f"Design and plan this project:\n\n{user_request}"
    if critic_feedback:
        plan_prompt += (
            f"\n\n---\n**CRITIC REJECTION #{critic_iter} — Address ALL issues below:**\n"
            f"{critic_feedback}\n\n"
            f"Previous blueprint was for project: {state.get('project_name', 'unknown')}\n"
            f"You MUST fix all issues. You MAY restructure the plan if needed."
        )

    # ── Runtime config integration: learned prompts + affinity map for orchestration ─
    runtime_cfg = (state.get("metadata") or {}).get("runtime_config", {})
    cto_additions = runtime_cfg.get("cto_prompt_additions", "")
    if cto_additions:
        plan_prompt += f"\n\n## Learned Configuration (from self-improvement runs):\n{cto_additions}"
    preferred = runtime_cfg.get("preferred_agents") or {}
    if isinstance(preferred, dict) and preferred:
        import json as _json
        plan_prompt += (
            "\n\n## Affinity-Driven Orchestration (runtime)\n"
            "Downstream squad leaders route file tasks using this task-class → provider map. "
            "Order milestones and file granularity so high-affinity paths own the riskiest surfaces.\n"
            f"```json\n{_json.dumps(preferred, indent=2)}\n```"
        )

    # ── Cost/power-aware blueprint generation via the router ─────────────────
    # The router will: compress oversized prompts via the cheap `compressor`
    # deployment, dispatch to the flagship CTO chain (Claude → Gemini Pro →
    # GPT-4.1), track cost per token, and respect any registered job budget.
    job_id = (state.get("metadata") or {}).get("job_id")
    cto_messages = [
        {"role": "system", "content": _CTO_SYSTEM.format(projects_dir=projects_dir)},
        {"role": "user",   "content": plan_prompt},
    ]

    async def _run_cto() -> ProjectBlueprint:
        await acquire_slot("cto")
        return await acall_structured(
            "cto", cto_messages, ProjectBlueprint,
            complexity="strategic",  # CTO is always strategic
            compress=True,           # large source/critic context → summarize
            metadata={"job_id": job_id} if job_id else None,
        )

    try:
        blueprint = asyncio.run(_run_cto())
    except RuntimeError:
        # Already inside an event loop — use a thread-safe runner
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            blueprint = pool.submit(asyncio.run, _run_cto()).result()

    _ensure_critical_files(blueprint)

    # ── Meta-logging ──────────────────────────────────────────────────────────
    trace.start_run(
        run_id=blueprint.project_id,
        project_name=blueprint.project_name,
        complexity=blueprint.estimated_complexity,
        n_milestones=len(blueprint.milestones),
        is_replan=is_replan,
        critic_iter=critic_iter,
    )

    # Announce the blueprint
    squads = [m.squad for m in blueprint.milestones]
    lines = [
        f"## 🏗️ CTO Blueprint: **{blueprint.project_name}**",
        f"",
        f"**Vision**: {blueprint.vision}",
        f"**Complexity**: `{blueprint.estimated_complexity}` | **Squads**: {len(squads)}",
        f"**Tech Stack**: {', '.join(blueprint.tech_stack.languages + blueprint.tech_stack.frameworks)}",
        f"**Project Dir**: `{blueprint.project_dir}`",
        f"",
        f"### Architecture Overview",
        blueprint.architecture_overview,
        f"",
        f"### Milestones ({len(blueprint.milestones)})",
    ]
    for i, m in enumerate(blueprint.milestones, 1):
        lines.append(
            f"{i}. **[{m.squad.upper()}]** {m.title} — {m.description}"
        )
    lines += [
        f"",
        f"### Files to Create ({len(blueprint.directory_structure)})",
        "```",
        *[f"  {f}" for f in blueprint.directory_structure[:20]],
        *(["  ..."] if len(blueprint.directory_structure) > 20 else []),
        "```",
        f"",
        f"### Success Criteria",
        *[f"- {c}" for c in blueprint.success_criteria],
    ]

    blueprint_msg = AIMessage(content="\n".join(lines))

    # Deduplicate squads and spawn one Squad Leader per unique domain
    seen_squads: set[str] = set()
    sends = []
    for milestone in blueprint.milestones:
        if milestone.squad in seen_squads:
            continue
        seen_squads.add(milestone.squad)

        # Gather all milestones for this squad
        squad_milestones = [
            m.model_dump() for m in blueprint.milestones if m.squad == milestone.squad
        ]

        updated_metadata = {
                **state.get("metadata", {}),
                "project_id": blueprint.project_id,
                "complexity": blueprint.estimated_complexity,
                "n_squads": len(seen_squads),
                "n_files": len(blueprint.directory_structure),
            }
        sends.append(
            Send(
                "squad_leader",
                {
                    **state,
                    "user_request": user_request,
                    "project_blueprint": blueprint.model_dump(),
                    "project_name": blueprint.project_name,
                    "project_dir": blueprint.project_dir,
                    "metadata": updated_metadata,
                    "current_squad_task": {
                        "domain": milestone.squad,
                        "milestones": squad_milestones,
                        "blueprint": blueprint.model_dump(),
                    },
                    # Reset accumulators for this planning round
                    "squad_plans": [],
                    "worker_outputs": [],
                    "critic_verdicts": [],
                    "critic_approved": False,
                },
            )
        )

    return Command(
        update={
            "user_request": user_request,
            "project_blueprint": blueprint.model_dump(),
            "project_name": blueprint.project_name,
            "project_dir": blueprint.project_dir,
            "iteration": state.get("iteration", 0),
            "messages": [blueprint_msg],
            "metadata": {
                **state.get("metadata", {}),
                "project_id": blueprint.project_id,
                "complexity": blueprint.estimated_complexity,
                "n_squads": len(seen_squads),
                "n_files": len(blueprint.directory_structure),
            },
        },
        goto=sends,
    )
