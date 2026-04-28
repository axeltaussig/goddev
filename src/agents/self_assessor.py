import asyncio
import json
import os
import collections # Added for deque
from pathlib import Path
from pydantic import BaseModel, Field

from ..state import GodDevState
from ..llm_router import acall_structured
from ..task_queue import acquire_slot

_ASSESSOR_SYSTEM = """You are the GodDev Self-Assessor (System 2 Reflection).
Your job is to read raw telemetry logs and statistical performance history for the GodDev multi-agent system,
and output an incredibly dense, brutally objective Hindsight Assessment Report.

You must identify:
1. Exact statistical failure rates (e.g. Code Critic scores, Verification drop-offs).
2. Root causes of recurring failures (e.g. Integrator failing due to missing backend contracts, path traversal errors).
3. Efficacy of the current LLM routing topology (which model is failing at which task).

Output a structured markdown report quantifying exact telemetry metrics. Do NOT propose code fixes — only assess the damage and bottleneck realities.
"""

def _read_telemetry() -> str:
    """
    Reads the last 100 entries from the execution trace log file.

    This function efficiently reads only the required lines without loading
    the entire file into memory, addressing potential performance and memory issues.

    Returns:
        A string containing the last 100 lines of the telemetry log,
        or an error message if the file is not found or cannot be read.
    """
    log_file = Path("/opt/goddev/logs/execution_trace.jsonl")
    if not log_file.exists():
        return "No execution_trace.jsonl found."
    
    last_n_lines = collections.deque(maxlen=100)
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                last_n_lines.append(line.strip())
    except IOError as e:
        return f"Error reading telemetry log: {e}"
        
    return "\n".join(last_n_lines)

def _read_projects_reports() -> str:
    """
    Reads the most recent 5 project reports, taking only the first 1000 characters
    of each report to avoid excessive memory usage.

    Returns:
        A markdown-formatted string combining snippets of the latest project reports,
        or a message indicating no reports were found or could be read.
    """
    parts = []
    proj_dir = Path("/opt/goddev/projects")
    if not proj_dir.exists():
        return "No project reports."

    project_metadata = []
    for p in proj_dir.iterdir():
        if p.is_dir():
            try:
                mtime = p.stat().st_mtime
                project_metadata.append((mtime, p))
            except OSError as e:
                print(f"Warning: Could not get modification time for {p}: {e}")
    
    project_metadata.sort(key=lambda x: x[0], reverse=True)
    
    for _, p in project_metadata[:5]:
        rep = p / "GODDEV_REPORT.md"
        if rep.exists():
            try:
                with open(rep, 'r', encoding='utf-8') as f:
                    content = f.read(1000) # Read only first 1000 characters
                parts.append(f"### Project {p.name}\n" + content.strip() + "\n...")
            except IOError as e:
                parts.append(f"### Project {p.name} (Error reading report: {e})")
                print(f"Warning: Could not read report for {p.name}: {e}")
    
    return "\n\n".join(parts) if parts else "No project reports."

def _format_score(score: float) -> str:
    """
    Formats a numeric score into a descriptive, colored label.

    Args:
        score: The numeric score to format (expected range 0.0 to 10.0).

    Returns:
        A string label (e.g., "EXCELLENT", "GOOD", "FAIR", "POOR")
        based on the provided score.
    """
    if score > 8.0:
        return "EXCELLENT"
    elif score >= 6.0:
        return "GOOD"
    elif score >= 4.0:
        return "FAIR"
    else:
        return "POOR"

class AssessmentReport(BaseModel):
    report_markdown: str = Field(description="Dense statistical markdown assessment")

def self_assessor_node(state: GodDevState) -> dict:
    """
    The self-assessor node reads logs and quantifies GodDev failures
    before the research phase, generating a hindsight assessment report.

    It compiles telemetry and recent project reports, then uses an LLM
    to create a structured markdown assessment. A synthetic overall score
    is derived and formatted into the final report.

    Args:
        state: The current GodDevState containing user requests and metadata.

    Returns:
        A dictionary with the key "assessment_report" containing the
        formatted markdown assessment report.
    """
    print("📋 [Self-Assessor] Compiling telemetry and hindsight metrics...")
    
    telemetry = _read_telemetry()
    reports = _read_projects_reports()
    steer = (state.get("user_request") or "").strip()
    steer_block = ""
    if steer and steer != "GodDev self-improvement":
        steer_block = (
            f"## Human-directed improvement priorities (weight these in the assessment)\n{steer}\n\n---\n\n"
        )
    prompt = f"""{steer_block}
## Telemetry Trace (Last 100 actions)
```jsonl
{telemetry}
```

## Recent GODDEV_REPORTs (Integration Pass/Fail records)
{reports}
    """
    
    job_id = (state.get("metadata") or {}).get("job_id")
    msgs = [
        {"role": "system", "content": _ASSESSOR_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    async def _run_assessor() -> AssessmentReport:
        await acquire_slot("backend_cheap")
        return await acall_structured(
            "backend", msgs, AssessmentReport,
            complexity="trivial",
            compress=True,
            metadata={"job_id": job_id} if job_id else None,
        )

    try:
        try:
            result = asyncio.run(_run_assessor())
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, _run_assessor()).result()
        report_md = result.report_markdown
    except Exception as exc:
        print(f"📋 [Self-Assessor] LLM call failed ({exc}) — using minimal assessment.")
        report_md = f"## Self-Assessment (auto-fallback)\n\nTelemetry lines: {len(telemetry.splitlines())}. LLM assessment unavailable: {exc}"
    
    print("📋 [Self-Assessor] Hindsight quantification complete.")

    # Derive a synthetic score for demonstration purposes.
    # In a more advanced system, this score would be derived from structured metrics
    # parsed from telemetry or the LLM's structured output.
    synthetic_score = 7.5 # Default starting point
    
    # Simple heuristic to adjust score based on keywords in reports and LLM output
    if "FAIL" in reports.upper() or "FAILURE" in reports.upper():
        synthetic_score -= 2.0
    
    if "FAILURE RATE" in report_md.upper() or "CRITICAL BOTTLENECK" in report_md.upper():
        synthetic_score -= 1.5
    
    # Ensure score remains within a logical range (0 to 10)
    synthetic_score = max(0.0, min(10.0, synthetic_score))
    
    formatted_score_label = _format_score(synthetic_score)
    
    # Prepend the formatted score and its label to the assessment report
    final_report_md = f"## Overall Assessment: {formatted_score_label} (Score: {synthetic_score:.1f}/10)\n\n" + report_md
    
    return {"assessment_report": final_report_md}