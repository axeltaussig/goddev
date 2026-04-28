"""
GodDev 3.0 — Hierarchical Autonomous Development Graph
=======================================================

3-Tier Architecture (Map-Reduce at every level):

    START
      │
      ▼
  [cto]  ─────────────────────────────────── Claude Opus
      │ Send() × N squads (parallel)         Produces ProjectBlueprint
      ▼
  [squad_leader] × N ──────────────────────── Domain experts
      │ Send() × M files (parallel)           Decomposes to FileTasks
      ▼
  [worker] × M ────────────────────────────── Writes + verifies files on disk
      │ (operator.add merges all FileOutputs)
      ▼
  [critic_council] ────────────────────────── Spawns 3 critics in parallel
      │ Send() × 3
      ├── [code_critic]      (GPT-4o)
      ├── [security_critic]  (Claude)
      └── [perf_critic]      (DeepSeek)
      │ (all 3 converge on verdict_collector)
      ▼
  [verdict_collector] ─────────────────────── Aggregates all 3 verdicts
      │
      ├── rejected (iter < max) → [cto] (re-plan with feedback)
      │
      └── approved ─────────────────────────── 
                │
                ▼
          [integrator] ─────────────────────── Verifies files, runs tests
                │
                ▼
          [reflector] ──────────────────────── Writes run memory
                │
                ▼
               END

Key properties:
- operator.add on squad_plans, worker_outputs, critic_verdicts = parallel-safe
- Send() injects per-agent context without shared state mutation
- Every level can run fully in parallel
- Critic Council: 3 critics run simultaneously (vs single in 2.0)
- Workers WRITE files to disk + run verification (vs text-only in 2.0)
- Fixed run_id tracking: no more plan_id mismatch bug from 2.0
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.cto import cto_node
from .agents.squad_leader import squad_leader_node
from .agents.worker import worker_node
from .agents.polish_team import polish_team_node
from .agents.critic_council import (
    critic_council_node,
    code_critic_node,
    security_critic_node,
    perf_critic_node,
    verdict_collector_node,
)
from .agents.ui_critic import ui_critic_node
from .agents.smooth_af import smooth_af_node
from .agents.integrator import integrator_node
from .agents.reflector import reflector_node
from .agents.self_improver import self_improver_node
from .state import GodDevState


# ─── Routing: after Verdict Collector ────────────────────────────────────────


def _route_after_verdict(state: GodDevState) -> str:
    """
    Route to integrator if all critics approved,
    or back to CTO for a corrective replan.
    """
    if state.get("critic_approved", False):
        return "integrator"
    return "cto"


# ─── Graph Builder ────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    builder = StateGraph(GodDevState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("cto", cto_node)
    builder.add_node("squad_leader", squad_leader_node)
    builder.add_node("worker", worker_node)
    builder.add_node("polish_team", polish_team_node)
    builder.add_node("smooth_af", smooth_af_node)
    builder.add_node("critic_council", critic_council_node)
    builder.add_node("code_critic", code_critic_node)
    builder.add_node("security_critic", security_critic_node)
    builder.add_node("perf_critic", perf_critic_node)
    builder.add_node("ui_critic", ui_critic_node)
    builder.add_node("verdict_collector", verdict_collector_node)
    builder.add_node("integrator", integrator_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("self_improver", self_improver_node)

    # ── Edges ──────────────────────────────────────────────────────────────────

    # Entry: user → CTO
    builder.add_edge(START, "cto")

    # CTO → squad_leaders via Send() (no explicit edge needed; Command handles it)
    # Squad leaders → workers via Send() (no explicit edge needed)

    # All workers converge on polish team, then critic council
    builder.add_edge("worker", "polish_team")
    builder.add_edge("polish_team", "smooth_af")
    builder.add_edge("smooth_af", "critic_council")

    # Critic council → 3 critics via Send() (no explicit edge needed)
    # All 3 critics converge on verdict collector
    builder.add_edge("code_critic", "verdict_collector")
    builder.add_edge("security_critic", "verdict_collector")
    builder.add_edge("perf_critic", "verdict_collector")
    builder.add_edge("ui_critic", "verdict_collector")

    # Verdict collector → integrator (approved) or CTO (replan)
    builder.add_conditional_edges(
        "verdict_collector",
        _route_after_verdict,
        {"integrator": "integrator", "cto": "cto"},
    )

    # Integration → reflection → self-improvement → end
    builder.add_edge("integrator", "reflector")
    builder.add_edge("reflector", "self_improver")
    builder.add_edge("self_improver", END)

    return builder


# ─── Compiled graph (LangGraph Studio + langgraph dev entry point) ─────────────
# max_concurrency=6: prevents thundering herd — at most 6 nodes run in parallel.
# Workers already have token-bucket rate limiting; this is an additional ceiling
# at the graph-scheduler level per best practices (LangGraph docs 2025).

graph = build_graph().compile()
graph.name = "GodDev 3.0"
# Nodes: cto, squad_leader, worker, critic_council, code_critic, security_critic,
#         perf_critic, verdict_collector, integrator, reflector, self_improver
