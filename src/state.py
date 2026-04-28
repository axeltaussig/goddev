"""
GodDev 3.0 — Hierarchical State & Pydantic Schemas
====================================================

3-tier hierarchy:
  Level 1: CTO produces ProjectBlueprint (vision, tech stack, milestones)
  Level 2: Squad Leaders produce SquadPlan (file tasks per domain)
  Level 3: Workers produce FileOutput (actual code written to disk + verified)

Parallel safety throughout:
  - squad_plans    uses operator.add  (N squad leaders write simultaneously)
  - worker_outputs uses operator.add  (M workers write files simultaneously)
  - critic_verdicts uses operator.add (3 critics judge simultaneously)
"""
from __future__ import annotations

import operator
import uuid
from typing import Annotated, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ─── Domain Literals ─────────────────────────────────────────────────────────

AgentName = Literal[
    # Provider names (legacy / direct provider selection)
    "gemini", "openai", "deepseek", "claude",
    # Role names (Affinity-Driven Orchestration — let the router pick the
    # best deployment for the task type, with smart cost-effective routing)
    "architect", "backend", "frontend", "devops", "qa",
    # Meta / orchestration roles — used by self-improvement & multi-agent flows
    "meta_cto", "cto", "squad_lead", "integrator", "critic",
]
SquadDomain = Literal["architecture", "backend", "frontend", "devops", "qa"]
CriticType = Literal["code", "security", "performance"]

# ─── Level 1: CTO Blueprint ───────────────────────────────────────────────────


class TechStack(BaseModel):
    languages: list[str] = Field(description="Programming languages")
    frameworks: list[str] = Field(description="Frameworks and libraries")
    databases: list[str] = Field(default_factory=list)
    infrastructure: list[str] = Field(default_factory=list)


class ProjectMilestone(BaseModel):
    milestone_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    description: str
    deliverables: list[str] = Field(description="Concrete file or feature outputs")
    squad: SquadDomain
    priority: int = Field(default=1, ge=1, le=5, description="1=highest")


class ProjectBlueprint(BaseModel):
    """CTO's complete strategic plan — single source of truth for the entire run."""

    project_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    project_name: str = Field(description="kebab-case project name, e.g. todo-app")
    project_dir: str = Field(description="Absolute path where all files will be written")
    vision: str = Field(description="One paragraph: what this app is and who it serves")
    tech_stack: TechStack
    architecture_overview: str = Field(description="High-level architecture description")
    directory_structure: list[str] = Field(
        description="Complete list of files/dirs to create, relative to project_dir"
    )
    milestones: list[ProjectMilestone] = Field(
        description="2-5 milestones, one per squad domain"
    )
    success_criteria: list[str] = Field(description="Measurable definition of done")
    estimated_complexity: Literal["simple", "moderate", "complex"]


# ─── Level 2: Squad Plans ─────────────────────────────────────────────────────


class FileTask(BaseModel):
    """A single file to be written by one worker agent."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_path: str = Field(description="Absolute path of the file to create")
    description: str = Field(description="Detailed spec of what this file must contain")
    agent: AgentName = Field(description="Which model writes this file")
    depends_on: list[str] = Field(
        default_factory=list,
        description="task_ids this task depends on (for sequential ordering)",
    )
    verification_cmd: Optional[str] = Field(
        default=None,
        description="Shell command to verify the file after writing (e.g. 'python -m py_compile')",
    )


class SquadPlan(BaseModel):
    """Squad Leader's decomposed plan for their domain."""

    squad_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    domain: SquadDomain
    strategy: Literal["sequential", "parallel"]
    file_tasks: list[FileTask]
    context: str = Field(description="Shared context/constraints for all tasks in this squad")
    estimated_files: int


# ─── Level 3: Worker Output ───────────────────────────────────────────────────


class FileOutput(BaseModel):
    """Result of one worker writing and verifying one file."""

    task_id: str
    file_path: str
    agent: AgentName
    bytes_written: int
    verification_passed: bool
    verification_output: str = ""
    error: Optional[dict] = None
    squad_domain: str = ""


# ─── Level 4: Critic Verdicts ─────────────────────────────────────────────────


class CriticVerdict(BaseModel):
    """One critic's structured verdict on the entire worker output."""

    critic_type: CriticType
    approved: bool
    score: float = Field(ge=0.0, le=10.0, description="Quality score 0-10")
    issues: list[str] = Field(default_factory=list)
    actionable_feedback: str = Field(
        description="Numbered, specific instructions for the CTO re-plan"
    )
    critical_files: list[str] = Field(
        default_factory=list,
        description="Specific file paths with problems",
    )


# ─── Level 5: Integration Result ─────────────────────────────────────────────


class IntegrationResult(BaseModel):
    passed: bool
    files_verified: list[str] = Field(default_factory=list)
    integration_test_output: str = ""
    conflicts_resolved: list[str] = Field(default_factory=list)
    final_summary: str


# ─── Master State ─────────────────────────────────────────────────────────────


class GodDevState(TypedDict):
    """
    Hierarchical state for GodDev 3.0.

    Parallel safety:
    - squad_plans    → operator.add  (squad leaders append without overwriting)
    - worker_outputs → operator.add  (workers append without overwriting)
    - critic_verdicts → operator.add (critics append without overwriting)

    Per-agent injection (via Send()):
    - current_squad_task  → injected fresh per squad leader
    - current_file_task   → injected fresh per worker
    - current_critic_type → injected fresh per critic
    """

    # Conversation history
    messages: Annotated[list[BaseMessage], add_messages]

    # Level 1: Meta / CTO prep (Awakening)
    assessment_report: Optional[str]
    research_brief: Optional[str]

    # Level 1.5: CTO
    user_request: str
    project_blueprint: Optional[dict]
    project_name: str
    project_dir: str

    # Level 2: Squad Leaders (parallel per domain)
    squad_plans: Annotated[list[dict], operator.add]
    current_squad_task: Optional[dict]  # Injected per-squad via Send()

    # Level 3: Workers (parallel per file)
    worker_outputs: Annotated[list[dict], operator.add]
    current_file_task: Optional[dict]  # Injected per-worker via Send()

    # Level 4: Critic Council (3 critics in parallel)
    critic_verdicts: Annotated[list[dict], operator.add]
    current_critic_type: Optional[str]  # Injected per-critic via Send()
    critic_approved: bool
    critic_feedback: Optional[str]
    critic_iteration: int
    max_critic_iterations: int

    # Level 5: Integration
    integration_result: Optional[dict]
    integration_passed: bool

    # Level 6: Reflection
    reflection_notes: Optional[str]

    # Routing & metadata
    iteration: int
    max_iterations: int
    metadata: dict
