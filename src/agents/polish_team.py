"""
GodDev 3.0 — Polish Team (DeepSeek micro-fixers + Gemini coordinator)
======================================================================

Sits between workers and critics in the self-build pipeline.
Multiple DeepSeek agents each review worker outputs for a specific concern,
then a Gemini coordinator merges their micro-fixes into the final file.

This dramatically increases critic pass rates by catching:
- Syntax errors (py_compile failures)
- Missing/wrong imports
- Broken public interfaces (integration contracts)

Cost: ~$0.01-0.03 per file (all DeepSeek + Gemini Flash)
Value: +20-30% verification pass rate → fewer critic rejections → fewer reruns
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from langchain_core.messages import AIMessage

from ..llm_router import acall
from ..state import GodDevState
from ..task_queue import acquire_slot


# ── DeepSeek Specialist Prompts ──────────────────────────────────────────────

_SYNTAX_PROMPT = (
    "You are a Python syntax fixer. You receive a Python file that may have syntax errors.\n"
    "Your ONLY job: fix syntax errors so py_compile passes. Make MINIMAL changes.\n"
    "If the file is already valid Python, return it UNCHANGED.\n"
    "Output ONLY the corrected Python source code. No markdown fences, no commentary."
)

_IMPORT_PROMPT = (
    "You are a Python import fixer. You receive a Python file and its package location.\n"
    "Your ONLY job: ensure all imports are correct and resolvable.\n"
    "- Fix relative imports based on the file's location in the package\n"
    "- Remove imports of modules that don't exist\n"
    "- Add missing imports that are referenced but not imported\n"
    "Make MINIMAL changes. If imports are already correct, return the file UNCHANGED.\n"
    "Output ONLY the corrected Python source code. No markdown fences, no commentary."
)

_CONTRACT_PROMPT = (
    "You are an integration contract checker. You receive:\n"
    "1. The ORIGINAL source file (before modification)\n"
    "2. The MODIFIED source file (worker's output)\n\n"
    "Your ONLY job: ensure the modified file preserves ALL public interfaces from the original:\n"
    "- Same function names and signatures\n"
    "- Same class names and methods\n"
    "- Same module-level exports\n"
    "- No removed public APIs\n\n"
    "If any public interface was broken, restore it while keeping the worker's improvements.\n"
    "If contracts are preserved, return the modified file UNCHANGED.\n"
    "Output ONLY the corrected Python source code. No markdown fences, no commentary."
)

_COORDINATOR_PROMPT = (
    "You are the Polish Team coordinator. You receive multiple versions of a Python file,\n"
    "each fixed by a different specialist (syntax, imports, contracts).\n\n"
    "Your job: merge ALL fixes into ONE final version that:\n"
    "1. Has valid syntax\n"
    "2. Has correct imports\n"
    "3. Preserves all public interfaces from the original\n"
    "4. Keeps the intended improvements from the worker\n\n"
    "If all versions are identical to the current file, return it unchanged.\n"
    "Output ONLY the final merged Python source code. No markdown fences, no commentary."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()


def _read_original(staging_path: str) -> str | None:
    """Read the live (pre-modification) version of a staging file."""
    if "/staging/src/" in staging_path:
        live = staging_path.replace("/staging/src/", "/src/")
        try:
            return Path(live).read_text(encoding="utf-8")
        except Exception:
            pass
    return None


async def _polish_call(role: str, system: str, user_content: str, job_id: str | None) -> str:
    """Run a single DeepSeek/Gemini polish call with rate limiting."""
    await acquire_slot("critic")
    return await acall(
        role,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        complexity="trivial",
        compress=False,
        affinity="polish",
        metadata={"job_id": job_id} if job_id else None,
    )


async def _polish_one_file(file_path: str, job_id: str | None) -> dict:
    """Run 3 DeepSeek specialists + Gemini coordinator on one file."""
    try:
        current = Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        return {"file": file_path, "status": "skip", "reason": str(e)}

    if not current.strip():
        return {"file": file_path, "status": "skip", "reason": "empty"}

    original = _read_original(file_path) or current

    # ── Fan out to 3 DeepSeek specialists in parallel ────────────────────
    specialist_tasks = [
        _polish_call(
            "polisher", _SYNTAX_PROMPT,
            f"## File: {file_path}\n```python\n{current[:15000]}\n```",
            job_id,
        ),
        _polish_call(
            "polisher", _IMPORT_PROMPT,
            f"## File: {file_path}\n## Package root: /opt/goddev/src/\n```python\n{current[:15000]}\n```",
            job_id,
        ),
        _polish_call(
            "polisher", _CONTRACT_PROMPT,
            (
                f"## ORIGINAL:\n```python\n{original[:10000]}\n```\n\n"
                f"## MODIFIED:\n```python\n{current[:15000]}\n```"
            ),
            job_id,
        ),
    ]

    try:
        results = await asyncio.gather(*specialist_tasks, return_exceptions=True)
    except Exception as e:
        return {"file": file_path, "status": "error", "reason": str(e)}

    # Collect successful specialist outputs
    versions = []
    for r in results:
        if isinstance(r, str) and r.strip():
            cleaned = _strip_fences(r)
            if cleaned.strip():
                versions.append(cleaned)

    if not versions:
        return {"file": file_path, "status": "skip", "reason": "all specialists failed"}

    # If all versions are identical to current, no changes needed
    if all(v.strip() == current.strip() for v in versions):
        return {"file": file_path, "status": "unchanged"}

    # ── Gemini coordinator merges fixes ──────────────────────────────────
    labels = ["Syntax-fixed", "Import-fixed", "Contract-preserved"]
    merge_parts = [f"## Current file:\n```python\n{current[:12000]}\n```\n"]
    for i, v in enumerate(versions):
        label = labels[i] if i < len(labels) else f"Version {i+1}"
        merge_parts.append(f"## {label}:\n```python\n{v[:12000]}\n```\n")

    try:
        final = await _polish_call(
            "polish_lead", _COORDINATOR_PROMPT,
            "\n".join(merge_parts),
            job_id,
        )
        final = _strip_fences(final)
    except Exception:
        final = versions[0]

    if not final.strip():
        return {"file": file_path, "status": "skip", "reason": "coordinator empty"}

    # Write polished file back
    try:
        Path(file_path).write_text(final, encoding="utf-8")
    except Exception as e:
        return {"file": file_path, "status": "error", "reason": f"write: {e}"}

    # Re-verify with py_compile
    verified = False
    if file_path.endswith(".py"):
        try:
            r = subprocess.run(
                ["python3", "-m", "py_compile", file_path],
                capture_output=True, text=True, timeout=10,
            )
            verified = r.returncode == 0
        except Exception:
            pass

    return {"file": file_path, "status": "polished", "verified": verified}


# ── Graph Node ───────────────────────────────────────────────────────────────

def polish_team_node(state: GodDevState) -> dict:
    """
    Polish Team: DeepSeek micro-fixers + Gemini coordinator.

    For each worker output file:
      1. 3 DeepSeek agents check syntax, imports, and contracts in parallel
      2. Gemini Flash merges their fixes into one clean version
      3. File is re-verified with py_compile

    Cost: ~$0.01-0.03 per file | Value: +20-30% critic pass rate
    """
    worker_outputs = state.get("worker_outputs") or []
    job_id = (state.get("metadata") or {}).get("job_id")

    # Collect .py files that workers actually wrote
    files = []
    for wo in worker_outputs:
        fp = wo.get("file_path", "")
        if fp and fp.endswith(".py") and Path(fp).exists():
            files.append(fp)

    if not files:
        return {
            "messages": [AIMessage(content="🔧 **Polish Team**: No Python files to review — skipping.")],
        }

    print(f"[Polish Team] Reviewing {len(files)} files with DeepSeek specialists...")

    async def _polish_all():
        return await asyncio.gather(
            *[_polish_one_file(fp, job_id) for fp in files],
            return_exceptions=True,
        )

    try:
        results = asyncio.run(_polish_all())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results = pool.submit(asyncio.run, _polish_all()).result()

    polished = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "polished")
    unchanged = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "unchanged")
    errors = sum(1 for r in results if isinstance(r, dict) and r.get("status") in ("error", "skip"))
    verified = sum(1 for r in results if isinstance(r, dict) and r.get("verified"))

    summary = (
        f"🔧 **Polish Team** — {len(files)} files reviewed by DeepSeek specialists + Gemini coordinator\n"
        f"  • **Polished**: {polished} files improved\n"
        f"  • **Unchanged**: {unchanged} (already clean)\n"
        f"  • **Skipped/Errors**: {errors}\n"
        f"  • **Re-verified**: {verified}/{polished} pass py_compile"
    )
    print(f"[Polish Team] Done: {polished} polished, {unchanged} unchanged, {errors} errors")

    return {
        "messages": [AIMessage(content=summary)],
    }
