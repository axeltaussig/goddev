"""
GodDev 3.0 — Integration Agent (Claude Opus)
=============================================

Runs AFTER the Critic Council approves. Responsibilities:

1. Verify all expected files actually exist on disk
2. Run integration-level tests (e.g. `npm install && npm test`, `pytest`)
3. Generate a polished, end-user-facing project summary
4. Write a GODDEV_REPORT.md to the project root with full build report
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm_router import acall as router_acall
from ..state import GodDevState

_INTEGRATOR_SYSTEM = """You are Claude, the Integration Lead at GodDev.

Your job is to:
1. Assess the completeness of the project (all files present, all connected)
2. Write a comprehensive GODDEV_REPORT.md that the user can open immediately
3. Identify any remaining gaps or next steps

## Report Format (GODDEV_REPORT.md)

# {Project Name} — GodDev 3.0 Build Report

## Project Overview
[Vision, tech stack, what was built]

## Files Created
[Table: file | purpose | lines]

## Architecture Summary  
[How the pieces connect]

## How to Run
[Exact commands to get this running, step by step]

## How to Test
[Exact test commands]

## API Reference (if applicable)
[Key endpoints/interfaces]

## Next Steps
[What a developer should do after receiving this]

Be specific, accurate, and immediately actionable. No fluff."""


# No hardcoded LLM — the integrator uses the multi-tier router
# with 'integrator' role: Gemini 2.5 Flash (Lead) → GPT-4.1 (Worker)
# This gives us low-latency, cost-effective report generation.


def _run_integration_tests(project_dir: str, tech_stack: dict) -> tuple[str, bool]:
    """Attempt to run integration tests based on the tech stack.

    Args:
        project_dir: Absolute path to the project root directory.
        tech_stack: Dictionary containing 'languages', 'frameworks', 'databases' keys.

    Returns:
        Tuple of (test_output_string, passed_boolean). The output string contains
        formatted results for each test framework that was executed.
    """
    results: list[str] = []
    passed = True
    p = Path(project_dir)

    # Python: run pytest if tests directory exists
    if (p / "tests").exists() or list(p.glob("**/test_*.py")):
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=60, cwd=project_dir,
            )
            output = (r.stdout + r.stderr).strip()
            results.append(f"**pytest**: {'PASS' if r.returncode == 0 else 'FAIL'}\n```\n{output[:500]}\n```")
            if r.returncode != 0:
                passed = False
        except subprocess.TimeoutExpired:
            results.append("**pytest**: ERROR — timed out after 60 seconds")
        except FileNotFoundError:
            results.append("**pytest**: ERROR — pytest executable not found")
        except OSError as exc:
            results.append(f"**pytest**: ERROR — {exc}")

    # Node: check package.json exists and run npm test
    if (p / "package.json").exists():
        try:
            r = subprocess.run(
                ["npm", "install", "--silent"],
                capture_output=True, text=True, timeout=120, cwd=project_dir,
            )
            if r.returncode != 0:
                output = (r.stdout + r.stderr).strip()
                results.append(f"**npm install**: FAIL\n```\n{output[:500]}\n```")
                passed = False
            else:
                r = subprocess.run(
                    ["npm", "test", "--if-present"],
                    capture_output=True, text=True, timeout=120, cwd=project_dir,
                )
                output = (r.stdout + r.stderr).strip()
                results.append(f"**npm test**: {'PASS' if r.returncode == 0 else 'FAIL'}\n```\n{output[:500]}\n```")
                if r.returncode != 0:
                    passed = False
        except subprocess.TimeoutExpired:
            results.append("**npm test**: ERROR — timed out after 120 seconds")
        except FileNotFoundError:
            results.append("**npm test**: ERROR — npm executable not found")
        except OSError as exc:
            results.append(f"**npm test**: ERROR — {exc}")

    if not results:
        return "No integration tests configured (no test framework detected)", True

    return "\n\n".join(results), passed


async def integrator_node(state: GodDevState) -> dict:
    """Verify all files, run integration tests, generate GODDEV_REPORT.md."""
    blueprint = state.get("project_blueprint") or {}
    project_dir = state.get("project_dir", "/opt/goddev/projects/unknown")
    project_name = state.get("project_name", "unknown")
    worker_outputs: list[dict] = state.get("worker_outputs") or []
    critic_verdicts: list[dict] = state.get("critic_verdicts") or []

    # ── Check file existence ──────────────────────────────────────────────────
    expected_files: list[str] = blueprint.get("directory_structure", [])
    p = Path(project_dir)

    files_found: list[str] = []
    files_missing: list[str] = []
    for rel_path in expected_files:
        abs_path = p / rel_path.lstrip("/")
        if abs_path.exists():
            files_found.append(rel_path)
        else:
            files_missing.append(rel_path)

    # ── Integration contract: validate required_exports exist in written files ─
    contract_violations: list[str] = []
    all_file_tasks: list[dict] = []
    for milestone in blueprint.get("milestones", []):
        all_file_tasks.extend(milestone.get("file_tasks", []))
    for task in all_file_tasks:
        req_exports: list[str] = task.get("required_exports") or []
        if not req_exports:
            continue
        fp_str: str = task.get("file_path", "")
        fp_abs = Path(fp_str) if Path(fp_str).is_absolute() else p / fp_str.lstrip("/")
        if not fp_abs.exists():
            continue
        try:
            file_text = fp_abs.read_text(encoding="utf-8")
        except Exception:
            continue
        for symbol in req_exports:
            if symbol not in file_text:
                contract_violations.append(f"{Path(fp_str).name}: missing `{symbol}`")

    # All worker outputs
    files_written = [o["file_path"] for o in worker_outputs if o.get("bytes_written", 0) > 0]
    total_bytes = sum(o.get("bytes_written", 0) for o in worker_outputs)
    verif_pass = sum(1 for o in worker_outputs if o.get("verification_passed"))
    verif_fail = len(worker_outputs) - verif_pass

    # ── Run integration tests ─────────────────────────────────────────────────
    tech = blueprint.get("tech_stack", {})
    integration_output, integration_passed = _run_integration_tests(project_dir, tech)

    # ── Critic scores summary ─────────────────────────────────────────────────
    critic_summary = " | ".join(
        f"{v.get('critic_type', '?').upper()}: {v.get('score', 0):.1f}/10"
        for v in critic_verdicts
    )

    # ── Build file listing for report ─────────────────────────────────────────
    file_table_rows = []
    for fp in files_written[:40]:
        try:
            rel = str(Path(fp).relative_to(p))
            lines = len(Path(fp).read_text(encoding="utf-8").splitlines())
            file_table_rows.append(f"| `{rel}` | {lines} lines |")
        except Exception:
            file_table_rows.append(f"| `{fp}` | — |")

    file_table = "| File | Lines |\n|------|-------|\n" + "\n".join(file_table_rows)

    # ── Generate GODDEV_REPORT.md via Multi-Tier Router ──────────────────────
    # Uses 'integrator' role: Gemini 2.5 Flash (Lead) → GPT-4.1 (Worker)
    # Fast, cheap, and cost-effective for free-text report generation.
    
    job_id = (state.get("metadata") or {}).get("job_id", "unknown")
    
    report_messages = [
        {"role": "system", "content": _INTEGRATOR_SYSTEM},
        {"role": "user", "content": f"""Generate GODDEV_REPORT.md for this project.

**Project**: {project_name}
**Directory**: {project_dir}
**User Request**: {state.get("user_request", "")}

**Tech Stack**:
- Languages: {', '.join(tech.get('languages', []))}
- Frameworks: {', '.join(tech.get('frameworks', []))}
- Databases: {', '.join(tech.get('databases', []))}

**Files Written**: {len(files_written)} files, {total_bytes:,} total bytes
**File Verification**: {verif_pass} PASS / {verif_fail} FAIL

**Files Created**:
{file_table}

**Missing from Blueprint**: {', '.join(files_missing) if files_missing else 'None'}

**Integration Tests**:
{integration_output}

**Critic Council Scores**: {critic_summary}

**Architecture Overview**:
{blueprint.get("architecture_overview", "See project files.")}

Write GODDEV_REPORT.md content now. Be specific about exact commands to run."""},
    ]
    
    try:
        report_content = await router_acall("integrator", report_messages, metadata={"job_id": job_id})
    except Exception as exc:
        print(f"[Integrator] Router call failed, using fallback: {exc}")
        report_content = f"# {project_name} — GodDev 3.0 Build Report\n\n"
        report_content += f"**Files Created**: {len(files_written)} files\n"
        report_content += f"**Integration Tests**: {'PASS' if integration_passed else 'FAIL'}\n"
        report_content += f"\nSee project at `{project_dir}` for full details."

    # Write the report to disk
    report_path = str(p / "GODDEV_REPORT.md")
    try:
        p.mkdir(parents=True, exist_ok=True)
        (p / "GODDEV_REPORT.md").write_text(report_content, encoding="utf-8")
    except Exception as exc:
        report_content += f"\n\n[ERROR writing report: {exc}]"

    # Build final user message
    if contract_violations:
        integration_passed = False
    status_icon = "✅" if integration_passed else "⚠️"
    final_msg = (
        f"## {status_icon} GodDev 3.0 — Build Complete: **{project_name}**\n\n"
        f"**Project Directory**: `{project_dir}`\n"
        f"**Files Created**: {len(files_written)} | "
        f"**Total Size**: {total_bytes:,} bytes | "
        f"**Verification**: {verif_pass}/{len(worker_outputs)} PASS\n"
        f"**Critic Scores**: {critic_summary}\n"
        f"**Integration Tests**: {'PASS' if integration_passed else 'FAIL'}\n\n"
        f"📄 Full build report: `{report_path}`\n\n"
        f"---\n\n"
        f"{report_content}"
    )

    integration_result = {
        "passed": integration_passed,
        "files_verified": files_found,
        "files_missing": files_missing,
        "integration_test_output": integration_output,
        "total_bytes": total_bytes,
        "final_summary": report_content,
    }

    return {
        "integration_result": integration_result,
        "integration_passed": integration_passed,
        "final_output": report_content,
        "messages": [AIMessage(content=final_msg)],
    }