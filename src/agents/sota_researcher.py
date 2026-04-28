import asyncio
import os
from pydantic import BaseModel, Field

from ..state import GodDevState
from ..llm_router import acall_structured
from ..task_queue import acquire_slot

_RESEARCHER_SYSTEM = """You are the GodDev SOTA Architectural Researcher.
You receive a Hindsight Assessment Report quantifying GodDev's recent bottlenecks and failure rates.
Your goal is to bridge these empirical failures with State-Of-The-Art (SOTA) 2026 Multi-Agent Orchestration Theory.

You must:
1. Recommend specific architectural/prompting paradigms (e.g. LangGraph recursive validation, File Export Contracts, Sandboxing).
2. Propose concrete changes to the Agent Prompts and Python logic to fix the assessed bottlenecks.
3. Output a `research_brief` markdown string detailing the exact strategies the Meta-CTO must implement.

Do NOT write full source code replacements—you provide the logic and paradigm blueprints. The Meta-CTO writes the actual python modifications.
"""

class ResearchBrief(BaseModel):
    research_markdown: str = Field(description="Concrete architectural directives based on SOTA research")


def sota_researcher_node(state: GodDevState) -> dict:
    """Analyze assessment report and produce SOTA architectural directives.

    Uses cost/power-aware routing: standard complexity → flagship (GPT-4.1
    primary in backend chain), with auto-downgrade to cheap pool if budget
    is tight. Auto-escalates on schema validation failure.
    """
    print("🎓 [SOTA Researcher] Aligning failures with 2026 State-of-the-Art...")

    assessment = state.get("assessment_report", "No assessment provided.")
    steer = (state.get("user_request") or "").strip()
    steer_block = ""
    if steer and steer != "GodDev self-improvement":
        steer_block = (
            f"## Human-directed improvement priorities (align research to these)\n{steer}\n\n---\n\n"
        )
    prompt = f"{steer_block}## Empirical Assessment Report\n{assessment}\n\nProvide the SOTA Architectural Fixes."

    job_id = (state.get("metadata") or {}).get("job_id")
    msgs = [
        {"role": "system", "content": _RESEARCHER_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    async def _run_research() -> ResearchBrief:
        await acquire_slot("backend")
        return await acall_structured(
            "backend", msgs, ResearchBrief,
            complexity="standard",
            compress=True,
            metadata={"job_id": job_id} if job_id else None,
        )

    try:
        result = asyncio.run(_run_research())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(asyncio.run, _run_research()).result()

    print("🎓 [SOTA Researcher] Architecture constraints established.")
    return {"research_brief": result.research_markdown}
