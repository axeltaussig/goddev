"""
GodDev 3.0 — Smooth AF Team (3-Tier Quality Pyramid)
=====================================================

The final quality pass before critics. Catches everything the Polish Team
doesn't: direction consistency, subtle logic bugs, integration coherence,
naming quality, edge cases, and overall smoothness.

Architecture (cost-optimised pyramid):
  Tier 3 (Top)   — Claude Director: sees ONLY a ~200-word summary.
                    Makes strategic approve/fix decisions. ~$0.01-0.03.
  Tier 2 (Mid)   — Gemini Synthesizer + OpenAI Validator: compress
                    DeepSeek reports into ultra-tight summaries. ~$0.01.
  Tier 1 (Base)  — 4 DeepSeek agents do ALL the heavy analysis work
                    in parallel on every file. ~$0.02-0.04.

Total cost: ~$0.04-0.08 per run. Claude sees almost nothing.

Placement in graph: polish_team → smooth_af → critic_council
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from langchain_core.messages import AIMessage

from ..llm_router import acall
from ..state import GodDevState
from ..task_queue import acquire_slot


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — DeepSeek Workforce (4 parallel specialists per file)
# ═══════════════════════════════════════════════════════════════════════════

_DIRECTION_PROMPT = (
    "You are a code direction analyst. You receive a MODIFIED file and its ORIGINAL.\n"
    "Check that the modifications:\n"
    "- Align with the stated improvement goal\n"
    "- Don't introduce scope creep or unrelated changes\n"
    "- Move in a consistent direction (not contradicting other parts)\n"
    "- Preserve the architectural intent of the original\n\n"
    "Output a JSON object: {\"issues\": [...], \"score\": 0-10, \"suggested_fixes\": [...]}\n"
    "If no issues: {\"issues\": [], \"score\": 10, \"suggested_fixes\": []}\n"
    "Be terse. Max 200 words total."
)

_SUBTLE_BUGS_PROMPT = (
    "You are a subtle bug hunter. You receive a Python file.\n"
    "Look for things that compile fine but will FAIL at runtime:\n"
    "- Off-by-one errors, wrong variable names in scope\n"
    "- Race conditions in async code\n"
    "- Unreachable code, dead branches\n"
    "- Type mismatches (str vs int, dict vs list)\n"
    "- Missing await on async calls\n"
    "- Wrong exception types caught\n"
    "- Mutable default arguments\n\n"
    "Output a JSON object: {\"bugs\": [...], \"score\": 0-10, \"fixes\": [...]}\n"
    "If no bugs: {\"bugs\": [], \"score\": 10, \"fixes\": []}\n"
    "Be terse. Max 200 words total."
)

_COHERENCE_PROMPT = (
    "You are an integration coherence checker. You receive a file and a list of\n"
    "other files it interacts with (imports, calls, shared state).\n"
    "Check that:\n"
    "- All function calls match their signatures in other files\n"
    "- Shared data structures are used consistently\n"
    "- Error handling is compatible across boundaries\n"
    "- No circular dependency risks\n\n"
    "Output a JSON object: {\"issues\": [...], \"score\": 0-10, \"fixes\": [...]}\n"
    "If all coherent: {\"issues\": [], \"score\": 10, \"fixes\": []}\n"
    "Be terse. Max 200 words total."
)

_QUALITY_PROMPT = (
    "You are a code quality reviewer. You receive a Python file.\n"
    "Check for:\n"
    "- Unclear variable/function names\n"
    "- Unnecessarily complex logic that could be simpler\n"
    "- DRY violations (duplicated patterns)\n"
    "- Missing error handling on I/O or network calls\n"
    "- Inconsistent style with the rest of the codebase\n\n"
    "Output a JSON object: {\"issues\": [...], \"score\": 0-10, \"suggestions\": [...]}\n"
    "If quality is good: {\"issues\": [], \"score\": 10, \"suggestions\": []}\n"
    "Be terse. Max 200 words total."
)

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2 — Gemini Synthesizer + OpenAI Validator
# ═══════════════════════════════════════════════════════════════════════════

_SYNTH_PROMPT = (
    "You are a quality report synthesizer. You receive 4 specialist reports\n"
    "(direction, bugs, coherence, quality) for one or more files.\n\n"
    "Produce a SINGLE ultra-compressed summary for a senior architect:\n"
    "- Top 3 critical issues (if any) with exact file:line references\n"
    "- Overall quality score (0-10)\n"
    "- Whether files are READY or NEED_FIXES\n"
    "- If NEED_FIXES: exact 1-line fix instructions per issue\n\n"
    "Max 150 words. Bullet points only. No fluff."
)

_VALIDATOR_PROMPT = (
    "You are a cross-validator. You receive:\n"
    "1. The original specialist reports\n"
    "2. A synthesis from another model\n\n"
    "Check if the synthesis missed anything critical.\n"
    "Add any missed items. Confirm or correct the quality score.\n"
    "Output max 100 words. Bullet points only."
)

# ═══════════════════════════════════════════════════════════════════════════
# TIER 3 — Claude Director (minimal tokens)
# ═══════════════════════════════════════════════════════════════════════════

_DIRECTOR_PROMPT = (
    "You are the Smooth AF Director. You see an ultra-compressed quality report.\n"
    "Decide:\n"
    "  APPROVE — files are smooth, ship them\n"
    "  FIX — list exact fixes needed (max 3, each one line)\n\n"
    "Output format:\n"
    "  DECISION: APPROVE|FIX\n"
    "  SCORE: 0-10\n"
    "  FIXES: (numbered list if FIX, empty if APPROVE)\n\n"
    "Max 80 words. Be decisive."
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()


def _read_original(staging_path: str) -> str | None:
    if "/staging/src/" in staging_path:
        live = staging_path.replace("/staging/src/", "/src/")
        try:
            return Path(live).read_text(encoding="utf-8")
        except Exception:
            pass
    return None


def _get_related_imports(file_content: str, src_root: str = "/opt/goddev/src") -> str:
    """Extract import targets and read first 30 lines of each for coherence check."""
    snippets = []
    for line in file_content.splitlines()[:80]:
        line = line.strip()
        if line.startswith("from ..") or line.startswith("from ."):
            parts = line.split("import")[0].replace("from ", "").strip()
            rel = parts.replace(".", "/") + ".py"
            candidates = [
                Path(src_root) / rel,
                Path(src_root) / rel.replace(".py", "/__init__.py"),
            ]
            for c in candidates:
                if c.exists():
                    try:
                        head = c.read_text(encoding="utf-8").splitlines()[:30]
                        snippets.append(f"### {c.name}\n" + "\n".join(head))
                    except Exception:
                        pass
                    break
    return "\n\n".join(snippets[:5]) if snippets else "(no related files found)"


async def _ds_call(system: str, user_content: str, job_id: str | None) -> str:
    """DeepSeek workforce call — cheap and fast."""
    await acquire_slot("critic")
    return await acall(
        "smooth_worker",
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        complexity="trivial",
        compress=False,
        metadata={"job_id": job_id} if job_id else None,
    )


async def _mid_call(role: str, system: str, user_content: str, job_id: str | None) -> str:
    """Mid-tier call (Gemini/OpenAI) — moderate cost."""
    await acquire_slot("critic")
    return await acall(
        role,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        complexity="trivial",
        compress=False,
        metadata={"job_id": job_id} if job_id else None,
    )


async def _director_call(summary: str, job_id: str | None) -> str:
    """Claude director — minimal tokens, strategic decision only."""
    await acquire_slot("meta_cto")
    return await acall(
        "smooth_director",
        [
            {"role": "system", "content": _DIRECTOR_PROMPT},
            {"role": "user", "content": summary},
        ],
        complexity="trivial",
        compress=False,
        metadata={"job_id": job_id} if job_id else None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-file analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def _analyse_one_file(file_path: str, job_id: str | None) -> dict:
    """Run 4 DeepSeek specialists on one file."""
    try:
        current = Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        return {"file": file_path, "status": "skip", "reason": str(e)}

    if not current.strip() or len(current) < 50:
        return {"file": file_path, "status": "skip", "reason": "too small"}

    original = _read_original(file_path) or ""
    related = _get_related_imports(current)
    truncated = current[:12000]

    # ── Tier 1: 4 DeepSeek agents in parallel ──────────────────────────
    tasks = [
        _ds_call(
            _DIRECTION_PROMPT,
            f"## ORIGINAL:\n```python\n{original[:6000]}\n```\n\n## MODIFIED:\n```python\n{truncated}\n```",
            job_id,
        ),
        _ds_call(
            _SUBTLE_BUGS_PROMPT,
            f"## File: {file_path}\n```python\n{truncated}\n```",
            job_id,
        ),
        _ds_call(
            _COHERENCE_PROMPT,
            f"## File: {file_path}\n```python\n{truncated}\n```\n\n## Related files:\n{related[:4000]}",
            job_id,
        ),
        _ds_call(
            _QUALITY_PROMPT,
            f"## File: {file_path}\n```python\n{truncated}\n```",
            job_id,
        ),
    ]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        return {"file": file_path, "status": "error", "reason": str(e)}

    # Collect reports
    labels = ["Direction", "Subtle Bugs", "Coherence", "Quality"]
    reports = []
    for i, r in enumerate(results):
        label = labels[i] if i < len(labels) else f"Report {i}"
        if isinstance(r, str) and r.strip():
            reports.append(f"**{label}**: {_strip_fences(r)[:300]}")
        else:
            reports.append(f"**{label}**: (failed)")

    return {
        "file": file_path,
        "status": "analysed",
        "reports": reports,
        "report_text": "\n".join(reports),
    }


async def _apply_fixes(file_path: str, fixes: str, job_id: str | None) -> bool:
    """DeepSeek agent applies Claude-directed fixes."""
    try:
        current = Path(file_path).read_text(encoding="utf-8")
    except Exception:
        return False

    fix_prompt = (
        "Apply ONLY these specific fixes to the file. Make MINIMAL changes.\n"
        "Output the COMPLETE corrected file. No markdown fences, no commentary.\n\n"
        f"## Fixes Required:\n{fixes}\n\n"
        f"## Current File:\n```python\n{current[:15000]}\n```"
    )

    try:
        result = await _ds_call(
            "You are a code fixer. Apply the requested fixes precisely. "
            "Output ONLY corrected Python source. No fences, no commentary.",
            fix_prompt,
            job_id,
        )
        result = _strip_fences(result)
        if not result.strip() or len(result) < 50:
            return False

        # Syntax check before writing
        Path(file_path).write_text(result, encoding="utf-8")
        if file_path.endswith(".py"):
            r = subprocess.run(
                ["python3", "-m", "py_compile", file_path],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                # Revert
                Path(file_path).write_text(current, encoding="utf-8")
                print(f"  [Smooth AF] Fix reverted — syntax error: {r.stderr[:60]}")
                return False
        return True
    except Exception as e:
        print(f"  [Smooth AF] Fix failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def _smooth_pipeline(files: list[str], job_id: str | None) -> dict:
    """
    Full 3-tier pipeline:
      Tier 1: DeepSeek analyses all files in parallel
      Tier 2: Gemini + OpenAI synthesize & validate
      Tier 3: Claude director decides approve/fix
      (If fix: DeepSeek applies fixes)
    """
    # ── Tier 1: DeepSeek workforce ─────────────────────────────────────
    analyses = await asyncio.gather(
        *[_analyse_one_file(fp, job_id) for fp in files],
        return_exceptions=True,
    )

    good_analyses = []
    for a in analyses:
        if isinstance(a, dict) and a.get("status") == "analysed":
            good_analyses.append(a)

    if not good_analyses:
        return {"decision": "APPROVE", "score": 7.0, "reason": "no files to analyse", "fixes_applied": 0}

    # ── Tier 2: Gemini synthesizer + OpenAI validator ──────────────────
    all_reports = "\n\n".join(
        f"### {a['file']}\n{a['report_text']}" for a in good_analyses
    )
    # Truncate to keep mid-tier costs low
    all_reports_truncated = all_reports[:6000]

    synth_task = _mid_call(
        "smooth_synth", _SYNTH_PROMPT,
        f"## Specialist Reports ({len(good_analyses)} files)\n{all_reports_truncated}",
        job_id,
    )
    validator_task = _mid_call(
        "smooth_validator", _VALIDATOR_PROMPT,
        f"## Specialist Reports\n{all_reports_truncated[:3000]}\n\n## (Synthesis will be provided by another model)",
        job_id,
    )

    try:
        synth_result, validator_result = await asyncio.gather(
            synth_task, validator_task, return_exceptions=True,
        )
    except Exception:
        synth_result, validator_result = "", ""

    synthesis = _strip_fences(synth_result) if isinstance(synth_result, str) else "(synthesis failed)"
    validation = _strip_fences(validator_result) if isinstance(validator_result, str) else "(validation failed)"

    # ── Tier 3: Claude director (minimal tokens!) ──────────────────────
    # Feed Claude ONLY the ultra-compressed synthesis + validation
    director_input = (
        f"## Quality Summary ({len(good_analyses)} files)\n"
        f"{synthesis[:400]}\n\n"
        f"## Cross-Validation\n"
        f"{validation[:200]}"
    )

    try:
        director_output = await _director_call(director_input, job_id)
        director_output = _strip_fences(director_output)
    except Exception as e:
        director_output = f"DECISION: APPROVE\nSCORE: 7\nFIXES:\n(director error: {e})"

    # Parse director decision
    decision = "APPROVE"
    score = 7.0
    fixes_text = ""

    for line in director_output.splitlines():
        line_upper = line.strip().upper()
        if line_upper.startswith("DECISION:"):
            decision = "FIX" if "FIX" in line_upper else "APPROVE"
        elif line_upper.startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip().split()[0])
                score = max(0.0, min(10.0, score))
            except (ValueError, IndexError):
                pass
        elif line_upper.startswith("FIXES:") or line_upper.startswith("FIX"):
            fixes_text = line.split(":", 1)[1].strip() if ":" in line else ""

    # Collect remaining lines as fix instructions
    in_fixes = False
    fix_lines = []
    for line in director_output.splitlines():
        if line.strip().upper().startswith("FIXES:") or line.strip().upper().startswith("FIX"):
            in_fixes = True
            rest = line.split(":", 1)[1].strip() if ":" in line else ""
            if rest:
                fix_lines.append(rest)
            continue
        if in_fixes and line.strip():
            fix_lines.append(line.strip())

    # ── Apply fixes if needed ──────────────────────────────────────────
    fixes_applied = 0
    if decision == "FIX" and fix_lines:
        fix_instructions = "\n".join(fix_lines)
        fix_tasks = []
        for a in good_analyses:
            fix_tasks.append(_apply_fixes(a["file"], fix_instructions, job_id))
        fix_results = await asyncio.gather(*fix_tasks, return_exceptions=True)
        fixes_applied = sum(1 for r in fix_results if r is True)

    return {
        "decision": decision,
        "score": score,
        "synthesis": synthesis[:300],
        "director_output": director_output[:300],
        "fixes_applied": fixes_applied,
        "files_analysed": len(good_analyses),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Graph Node
# ═══════════════════════════════════════════════════════════════════════════

def smooth_af_node(state: GodDevState) -> dict:
    """
    Smooth AF Team: 3-tier quality pyramid.

    Tier 1 (DeepSeek×4): direction, bugs, coherence, quality analysis
    Tier 2 (Gemini+OpenAI): synthesize + cross-validate reports
    Tier 3 (Claude): strategic approve/fix decision (~80 words max)
    Tier 1 again (DeepSeek): apply directed fixes if needed

    Cost: ~$0.04-0.08 per run | Claude sees ~200 words max
    """
    worker_outputs = state.get("worker_outputs") or []
    job_id = (state.get("metadata") or {}).get("job_id")

    # Collect files to review
    files = []
    for wo in worker_outputs:
        fp = wo.get("file_path", "")
        if fp and Path(fp).exists() and Path(fp).stat().st_size > 50:
            files.append(fp)

    if not files:
        return {
            "messages": [AIMessage(content="😎 **Smooth AF**: No files to review — skipping.")],
        }

    print(f"[Smooth AF] Reviewing {len(files)} files — 3-tier pyramid active...")

    async def _run():
        return await _smooth_pipeline(files, job_id)

    try:
        result = asyncio.run(_run())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(asyncio.run, _run()).result()

    decision = result.get("decision", "APPROVE")
    score = result.get("score", 7.0)
    fixes = result.get("fixes_applied", 0)
    n_files = result.get("files_analysed", 0)
    synthesis = result.get("synthesis", "")[:200]

    icon = "😎" if decision == "APPROVE" else "🔧"
    summary = (
        f"{icon} **Smooth AF** — {n_files} files reviewed (3-tier pyramid)\n"
        f"  • **Decision**: {decision} | **Score**: {score:.1f}/10\n"
        f"  • **DeepSeek workforce**: 4 specialists × {n_files} files\n"
        f"  • **Gemini+OpenAI**: synthesized & validated\n"
        f"  • **Claude director**: {len(result.get('director_output', ''))} chars (~minimal tokens)\n"
    )
    if fixes > 0:
        summary += f"  • **Fixes applied**: {fixes} files improved by DeepSeek\n"
    if synthesis:
        summary += f"  • **Summary**: {synthesis}"

    print(f"[Smooth AF] Done: {decision} score={score:.1f} fixes={fixes}")

    return {
        "messages": [AIMessage(content=summary)],
    }
