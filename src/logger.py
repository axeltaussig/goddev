"""
GodDev 3.0 — Meta-Optimization Logger
========================================

Records STRUCTURAL metadata only — zero payload, zero content.

Key fix from 2.0: run_id is now passed explicitly to every log call
(in 2.0, the plan_id mismatch caused worker records to be silently dropped).

Schema contract (PAYLOAD-AGNOSTIC):
LOGGED  : timestamps, durations, counts, booleans, run_id, project_name,
          agent names, squad domains, task_ids, scores, flags, byte counts.
NOT LOGGED: task content, briefs, file contents, user data, critic feedback text.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class _WorkerRecord:
    agent: str
    task_id: str
    file_path: str
    squad: str
    start_offset_s: float
    elapsed_s: float
    status: str
    bytes_written: int
    verification_passed: bool


@dataclass
class _CriticCouncilRecord:
    iteration: int
    max_iterations: int
    approved: bool
    force_approved: bool
    avg_score: float
    min_score: float
    n_total_issues: int
    verdicts: list[dict]


@dataclass
class _RunTrace:
    run_id: str
    project_name: str
    complexity: str
    n_milestones: int
    is_replan: bool
    critic_iter: int
    wall_start: float
    utc_start: str
    workers: list[_WorkerRecord] = field(default_factory=list)
    critic_council: Optional[_CriticCouncilRecord] = None
    closed: bool = False
    final_status: str = "RUNNING"
    n_nodes: int = 0


class GodDevLogger:
    """Thread-safe singleton that tracks per-run traces and flushes to execution_trace.md."""

    def __init__(self) -> None:
        self._traces: dict[str, _RunTrace] = {}
        self._lock = threading.Lock()
        self._log_path: Optional[Path] = None

    def _get_log_path(self) -> Path:
        if self._log_path is None:
            log_dir = os.getenv("LOG_DIR", str(Path(__file__).parent.parent / "logs"))
            self._log_path = Path(log_dir) / "execution_trace.md"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._log_path.exists() or self._log_path.stat().st_size == 0:
                self._log_path.write_text(
                    "# GodDev 3.0 — Execution Trace\n\n"
                    "> Payload-agnostic structural log.\n"
                    "> Schema: timestamps · durations · counts · flags · scores.\n\n---\n\n",
                    encoding="utf-8",
                )
        return self._log_path

    def start_run(
        self,
        run_id: str,
        project_name: str,
        complexity: str,
        n_milestones: int,
        is_replan: bool,
        critic_iter: int = 0,
    ) -> None:
        utc_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        trace = _RunTrace(
            run_id=run_id,
            project_name=project_name,
            complexity=complexity,
            n_milestones=n_milestones,
            is_replan=is_replan,
            critic_iter=critic_iter,
            wall_start=time.monotonic(),
            utc_start=utc_now,
        )
        with self._lock:
            self._traces[run_id] = trace

    def log_worker(
        self,
        run_id: str,
        agent: str,
        task_id: str,
        file_path: str,
        squad: str,
        elapsed_s: float,
        status: str = "ok",
        bytes_written: int = 0,
        verification_passed: bool = False,
    ) -> None:
        with self._lock:
            t = self._traces.get(run_id)
            if t is None:
                return
            offset = max(0.0, time.monotonic() - t.wall_start - elapsed_s)
            t.workers.append(_WorkerRecord(
                agent=agent, task_id=task_id, file_path=file_path,
                squad=squad, start_offset_s=offset, elapsed_s=elapsed_s,
                status=status, bytes_written=bytes_written,
                verification_passed=verification_passed,
            ))

    def log_critic_council(
        self,
        run_id: str,
        iteration: int,
        max_iterations: int,
        approved: bool,
        force_approved: bool,
        avg_score: float,
        min_score: float,
        n_total_issues: int,
        verdicts: list[dict],
    ) -> None:
        with self._lock:
            t = self._traces.get(run_id)
            if t is None:
                return
            t.critic_council = _CriticCouncilRecord(
                iteration=iteration, max_iterations=max_iterations,
                approved=approved, force_approved=force_approved,
                avg_score=avg_score, min_score=min_score,
                n_total_issues=n_total_issues, verdicts=verdicts,
            )

    def close_run(self, run_id: str, status: str, n_nodes: int) -> None:
        with self._lock:
            t = self._traces.get(run_id)
            if t is None or t.closed:
                return
            t.closed = True
            t.final_status = status
            t.n_nodes = n_nodes
            wall_time = time.monotonic() - t.wall_start

        md = self._format(t, wall_time)
        log_path = self._get_log_path()
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(md)

        # Print structured JSONL version
        import json
        jsonl_path = log_path.with_name("execution_trace.jsonl")
        import dataclasses
        try:
            trace_dict = dataclasses.asdict(t)
            trace_dict["wall_time_s"] = wall_time
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                json.dump(trace_dict, fh)
                fh.write("\n")
        except Exception:
            pass

    @staticmethod
    def _format(t: _RunTrace, wall_time_s: float) -> str:
        icon = {"SUCCESS": "OK", "FORCE_APPROVED": "WARN", "ERROR": "FAIL"}.get(
            t.final_status, t.final_status
        )
        lines: list[str] = [
            f"## [{icon}] {t.utc_start}  |  run_id=`{t.run_id}`  |  project=`{t.project_name}`",
            "",
            "### CTO Decision",
            "| field | value |",
            "|-------|-------|",
            f"| complexity | `{t.complexity}` |",
            f"| n_milestones | {t.n_milestones} |",
            f"| is_replan | {t.is_replan} |",
            f"| critic_iter_at_start | {t.critic_iter} |",
            "",
            "### Worker Execution",
        ]

        if t.workers:
            lines += [
                "| agent | squad | task_id | offset_s | elapsed_s | bytes | verif | status |",
                "|-------|-------|---------|----------|-----------|-------|-------|--------|",
            ]
            for w in t.workers:
                lines.append(
                    f"| {w.agent} | {w.squad} | `{w.task_id}` "
                    f"| {w.start_offset_s:.1f} | {w.elapsed_s:.1f} "
                    f"| {w.bytes_written:,} | {'PASS' if w.verification_passed else 'FAIL'} "
                    f"| {w.status} |"
                )
        else:
            lines.append("_(no worker records)_")

        lines += ["", "### Critic Council"]
        if t.critic_council:
            c = t.critic_council
            lines += [
                "| field | value |",
                "|-------|-------|",
                f"| round | {c.iteration}/{c.max_iterations} |",
                f"| approved | {c.approved} |",
                f"| force_approved | {c.force_approved} |",
                f"| avg_score | {c.avg_score:.1f}/10 |",
                f"| min_score | {c.min_score:.1f}/10 |",
                f"| n_total_issues | {c.n_total_issues} |",
            ]
            for v in c.verdicts:
                lines.append(
                    f"| {v.get('type', '?')}_score | {v.get('score', 0):.1f}/10 |"
                )
        else:
            lines.append("_(no critic record)_")

        # Optimization signals
        lines += ["", "### Optimization Signals",
                  "| signal | value | note |",
                  "|--------|-------|------|"]

        # Worker parallel efficiency
        if len(t.workers) > 1:
            elapsed_vals = [w.elapsed_s for w in t.workers]
            s = sum(elapsed_vals)
            par_eff = max(elapsed_vals) / s if s > 0 else 0.0
            lines.append(
                f"| parallel_efficiency | {par_eff:.2f} | 1.0=fully parallel |"
            )
        else:
            lines.append("| parallel_efficiency | N/A | single worker |")

        total_bytes = sum(w.bytes_written for w in t.workers)
        verif_rate = (
            sum(1 for w in t.workers if w.verification_passed) / len(t.workers)
            if t.workers else 0.0
        )
        lines += [
            f"| total_bytes_written | {total_bytes:,} | all workers combined |",
            f"| verification_pass_rate | {verif_rate:.1%} | files passing syntax check |",
            f"| wall_time_s | {wall_time_s:.1f} | total wall time |",
            f"| n_node_invocations | {t.n_nodes} | |",
        ]

        if t.critic_council:
            lines.append(
                f"| critic_first_pass | {t.critic_council.iteration == 1 and t.critic_council.approved}"
                f" | approved round 1 |"
            )

        lines += ["", "---", ""]
        return "\n".join(lines)


# ─── Singleton ────────────────────────────────────────────────────────────────

trace = GodDevLogger()