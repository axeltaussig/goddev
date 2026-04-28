"""
GodDev 3.0 — Self-Build Graph
================================

Same topology as the main graph but with two swaps:
  cto      → meta_cto      (reads own source + perf data, targets staging/)
  integrator → self_deployer (validates, backs up, deploys, smoke tests)

The worker, critic council, reflector, and self_improver nodes are
reused exactly from the main graph — the same critics that evaluate
user apps also evaluate GodDev's own improvements.

Graph:
  START → [meta_cto] → Send()×N tasks → [worker]×N
        → [polish_team] (DeepSeek fixers + Gemini coordinator)
        → [smooth_af] (3-tier pyramid: DeepSeek→Gemini+OpenAI→Claude)
        → [critic_council] → [code|sec|perf|ui critic] → [verdict_collector]
        → approved: [self_deployer] → [reflector] → [self_improver] → END
        → rejected: [meta_cto] (replan with critic feedback)
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.meta_cto import meta_cto_node
from .agents.worker import worker_node
from .agents.critic_council import (
    critic_council_node,
    code_critic_node,
    security_critic_node,
    perf_critic_node,
    verdict_collector_node,
)
from .agents.polish_team import polish_team_node
from .agents.ui_critic import ui_critic_node
from .agents.smooth_af import smooth_af_node
from .agents.self_deployer import self_deployer_node
from .agents.reflector import reflector_node
from .agents.self_improver import self_improver_node
from .state import GodDevState


from .agents.self_assessor import self_assessor_node
from .agents.sota_researcher import sota_researcher_node

def _route_after_verdict(state: GodDevState) -> str:
    if state.get("critic_approved", False):
        return "self_deployer"
    return "meta_cto"

def build_self_build_graph() -> StateGraph:
    builder = StateGraph(GodDevState)

    builder.add_node("self_assessor", self_assessor_node)
    builder.add_node("sota_researcher", sota_researcher_node)
    builder.add_node("meta_cto", meta_cto_node)
    builder.add_node("worker", worker_node)
    builder.add_node("polish_team", polish_team_node)
    builder.add_node("smooth_af", smooth_af_node)
    builder.add_node("critic_council", critic_council_node)
    builder.add_node("code_critic", code_critic_node)
    builder.add_node("security_critic", security_critic_node)
    builder.add_node("perf_critic", perf_critic_node)
    builder.add_node("ui_critic", ui_critic_node)
    builder.add_node("verdict_collector", verdict_collector_node)
    builder.add_node("self_deployer", self_deployer_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("self_improver", self_improver_node)

    builder.add_edge(START, "self_assessor")
    builder.add_edge("self_assessor", "sota_researcher")
    builder.add_edge("sota_researcher", "meta_cto")
    # meta_cto → worker: NO static edge — meta_cto_node uses Command(goto=Send("worker", ...))
    builder.add_edge("worker", "polish_team")
    builder.add_edge("polish_team", "smooth_af")
    builder.add_edge("smooth_af", "critic_council")
    builder.add_edge("code_critic", "verdict_collector")
    builder.add_edge("security_critic", "verdict_collector")
    builder.add_edge("perf_critic", "verdict_collector")
    builder.add_edge("ui_critic", "verdict_collector")

    builder.add_conditional_edges(
        "verdict_collector",
        _route_after_verdict,
        {"self_deployer": "self_deployer", "meta_cto": "meta_cto"},
    )

    builder.add_edge("self_deployer", "reflector")
    builder.add_edge("reflector", "self_improver")
    builder.add_edge("self_improver", END)

    return builder


self_build_graph = build_self_build_graph().compile()
self_build_graph.name = "GodDev 3.0 Self-Build"
