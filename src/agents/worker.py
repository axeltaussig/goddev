"""
GodDev 3.0 — Worker Agent (v3 — LiteLLM Router + File Lock)
=============================================================

State-of-the-art improvements over v2:

  v1: create_react_agent → 5-10 LLM calls per file (tool loop chaos)
  v2: direct llm.invoke() + manual semaphores → still thundering herd
  v3: LiteLLM Router → auto-fallback, circuit breaker, token-bucket
      rate limiting, file-lock registry (no more overwrite conflicts)

Key architecture decisions:
  - LiteLLM Router: unified async interface, fallback chains per role
    (e.g. architect: Claude → GPT-4o → DeepSeek if rate-limited)
  - Token bucket: each role has a sustained RPS budget; workers queue
    naturally without explicit sleep() hacks
  - Circuit breaker: failed providers auto-cool-down, router skips them
  - File lock: only one worker writes each file; conflicts logged+skipped
  - Code fence stripping: robust against LLM format deviations
  - Dependency pre-loading: reads already-written deps for context
  - Auto-fix: one retry cycle with full error context if verify fails
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from langchain_core.messages import AIMessage

from ..logger import trace
from ..state import GodDevState, FileOutput
from ..llm_router import acall
from ..task_queue import acquire_slot, get_file_registry

# ─── Role mapping: squad domain → LiteLLM router role ────────────────────────
# This is the ONE place that decides which provider handles each domain.
# The router handles fallbacks automatically if primary is rate-limited.

_DOMAIN_TO_ROLE: dict[str, str] = {
    "architecture": "architect",   # Claude → GPT-4o → DeepSeek
    "backend":      "backend",     # GPT-4o → Claude → DeepSeek
    "frontend":     "frontend",    # Gemini → GPT-4o
    "devops":       "devops",      # DeepSeek → GPT-4o
    "qa":           "qa",          # DeepSeek → GPT-4o → Claude
}

_AGENT_TO_ROLE: dict[str, str] = {
    "claude":    "architect",
    "openai":    "backend",
    "gemini":    "frontend",
    "deepseek":  "devops",
}

# Affinity routing applies to normal product squads only — self-improvement tasks
# carry an explicit Meta CTO agent; do not override with file-extension affinity.
_AFFINITY_SQUAD_DOMAINS = frozenset({"architecture", "backend", "frontend", "devops", "qa"})

# Router role chains are keyed by task *role* (architect, backend, …).
# Affinity maps runtime_config preferred_agents (provider → task class) onto these roles.


def _file_affinity_key(file_path: str) -> str:
    """Classify path for preferred_agents lookup (matches config keys)."""
    if not file_path:
        return "default"
    p = file_path.replace("\\", "/").lower()
    name = Path(file_path).name.lower()
    parts = Path(file_path).parts
    if name == "dockerfile" or "dockerfile" in name:
        return "docker"
    if ".github" in parts or "/.github/" in p:
        return "ci_cd"
    if "test" in p or name.startswith("test_") or name.endswith("_test.py") or ".test." in name:
        return "tests"
    if p.endswith(".py"):
        return "python_code"
    if p.endswith((".ts", ".tsx")):
        return "typescript_code"
    if p.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "javascript_code"
    if p.endswith((".html", ".htm", ".css", ".scss", ".sass", ".less", ".svg", ".vue", ".svelte")):
        return "markup_code"
    if p.endswith((".md", ".rst")):
        return "documentation"
    if p.endswith((".json", ".toml", ".ini", ".yaml", ".yml", ".cfg", ".env")):
        return "config_files"
    return "default"


def _provider_to_router_role(provider: str) -> str | None:
    p = (provider or "").strip().lower()
    return _AGENT_TO_ROLE.get(p)


def _affinity_router_role(
    file_path: str,
    squad_domain: str,
    preferred_agents: dict,
) -> str | None:
    """
    Affinity-Driven Orchestration: pick LiteLLM role from runtime_config + file path.

    preferred_agents maps task classes → provider (claude|openai|gemini|deepseek).
    We translate provider → primary router role, then nudge toward the active squad
    so work stays domain-coherent when the affinity still fits that domain.
    """
    if not preferred_agents:
        return None
    key = _file_affinity_key(file_path)
    prov = preferred_agents.get(key) or preferred_agents.get("default")
    if not prov:
        return None
    base = _provider_to_router_role(str(prov))
    if not base:
        return None
    domain = (squad_domain or "").lower()
    # Prefer router chains that match both affinity provider and squad responsibility
    if domain == "qa" and key == "tests":
        return "qa"
    if domain == "devops" and key in ("docker", "ci_cd", "config_files"):
        return "devops"
    if domain == "frontend" and key in ("javascript_code", "typescript_code", "documentation"):
        return "frontend"
    if domain == "architecture":
        return "architect"
    if domain == "backend":
        return "backend"
    if domain == "qa":
        return "qa"
    return base

_LABELS: dict[str, str] = {
    "architect":  "Claude/Arch",
    "backend":    "GPT4o/Back",
    "frontend":   "Gemini/Front",
    "devops":     "DeepSeek/Ops",
    "qa":         "DeepSeek/QA",
    "critic":     "GPT4o/Critic",
}

# ─── Complexity classification per file ──────────────────────────────────────
# Drives `acall(..., complexity=...)`: trivial → cheap pool (DeepSeek/Gemini
# Flash), strategic → flagship + context compression, standard → flagship.
# Universal trivial affinities (cheap pool always handles these well)
_TRIVIAL_AFFINITIES = {"documentation", "config_files", "tests", "ci_cd", "docker"}

# Affinities that the cheap pool handles well WHEN the squad is frontend or
# devops — boilerplate UI / shell scripts that don't need flagship reasoning.
_TRIVIAL_FOR_FRONTEND = {"markup_code", "javascript_code", "typescript_code"}
_TRIVIAL_FOR_DEVOPS   = {"javascript_code", "typescript_code"}

# Strategic affinities — always send to flagship pool (with compression).
_STRATEGIC_AFFINITIES: set[str] = set()  # nothing forced strategic by extension


def _complexity_for(file_path: str, squad_domain: str, fix_attempts: int) -> str:
    """
    Decide how strategic a single worker call is.

    Cost-effective bias: prefer cheap pool by default; escalate only when there
    is a *real* reason. The squad domain is no longer a blanket override —
    the actual file type drives complexity (an architect writing HTML doesn't
    need Claude Opus, but architecting a Python core does).

    Rules (precedence):
      1. Self-heal iteration ≥2  → strategic  (escalate when stuck).
      2. Trivial-by-affinity     → trivial    (docs/config/tests/CI/Docker).
      3. Frontend markup/JS/TS   → trivial    (cheap pool first).
      4. DevOps JS/TS scripts    → trivial    (cheap pool).
      5. Architecture + Python   → strategic  (real system design).
      6. Architecture + markup   → trivial    (writing HTML, not designing).
      7. Anything else           → standard.
    """
    domain = (squad_domain or "").lower()

    # Self-improvement workers: stay cheap. Patches are small edits, not full apps.
    # Only escalate to "standard" on retries — never "strategic" (avoids Claude $15/$75).
    if domain == "self-improvement":
        if fix_attempts >= 2:
            return "standard"
        return "trivial"

    if fix_attempts >= 2:
        return "strategic"

    aff = _file_affinity_key(file_path)

    if aff in _TRIVIAL_AFFINITIES:
        return "trivial"
    if domain == "frontend" and aff in _TRIVIAL_FOR_FRONTEND:
        return "trivial"
    if domain == "devops" and aff in _TRIVIAL_FOR_DEVOPS:
        return "trivial"
    if domain == "architecture":
        # Real system design = Python core code; everything else is just writing
        # files that happen to belong to the architecture squad.
        if aff == "python_code":
            return "strategic"
        if aff in _TRIVIAL_FOR_FRONTEND:
            return "trivial"
        return "standard"
    if aff in _STRATEGIC_AFFINITIES:
        return "standard"
    return "standard"

# ─── System Prompts ───────────────────────────────────────────────────────────

_RAW_RULE = (
    "\n\n## ABSOLUTE OUTPUT RULE\n"
    "Output ONLY the raw file content — exact bytes to write to disk.\n"
    "- NO markdown fences (no ```)\n"
    "- NO preamble like 'Here is the file:'\n"
    "- NO explanations after the code\n"
    "- Begin with the VERY FIRST CHARACTER of the file\n"
    "- NEVER truncate — output the COMPLETE file, no ellipses"
)

_QUALITY_RULE = (
    "\n\n## CODE QUALITY REQUIREMENTS (Code Critic scores this 1-10, target 8+)\n"
    "- Module-level docstring describing this file's role in the overall architecture\n"
    "- Type hints on EVERY function signature (Python) or JSDoc with @param/@returns (JS)\n"
    "- Docstring/JSDoc on every public function: purpose, params, returns, side-effects\n"
    "- Functions ≤40 lines; extract helpers for complex logic\n"
    "- Explicit error handling on all I/O, API calls, and parse operations\n"
    "- No TODO stubs, no commented-out code, no placeholder text\n"
    "- Descriptive variable/function names (no single letters except loop i/j/k)\n"
    "- Separate concerns: data access, business logic, and presentation in different functions\n"
    "- For JS: use const/let (never var), async/await (never callbacks), strict mode\n"
    "- For Python: use pathlib not os.path, f-strings not %, dataclasses for data"
)

_SYSTEMS: dict[str, str] = {
    "architect": (
        "You are Claude Opus, Chief Architect at GodDev — world's best AI dev team." + _RAW_RULE + _QUALITY_RULE +
        "\nRole: architecture docs, interface contracts, security config, system design. "
        "Standards: full type annotations, zero ambiguity, anticipate integration. "
        "Production quality only — no TODOs, no stubs."
    ),
    "backend": (
        "You are GPT-4.1, Senior Backend Engineer at GodDev." + _RAW_RULE + _QUALITY_RULE +
        "\nRole: APIs, business logic, server code, database access, auth. "
        "Standards: SOLID principles, proper error handling, input validation, SQL injection prevention. "
        "Production quality only — no TODOs, no stubs."
    ),
    "frontend": (
        "You are Gemini 2.5 Flash, Senior Frontend Engineer at GodDev." + _RAW_RULE + _QUALITY_RULE +
        "\nRole: HTML/CSS/JS UI, animations, canvas games, documentation. "
        "Standards: modern ESM, responsive design, smooth UX, WCAG accessibility, semantic HTML. "
        "Production quality only — no TODOs, no stubs."
    ),
    "devops": (
        "You are DeepSeek V3, DevOps & Systems Engineer at GodDev." + _RAW_RULE + _QUALITY_RULE +
        "\nRole: Dockerfiles, CI/CD, nginx configs, shell scripts, package.json, deployment. "
        "Standards: reproducible builds, security hardening, minimal images, pinned versions. "
        "Production quality only — no TODOs, no stubs."
    ),
    "qa": (
        "You are DeepSeek V3, QA Engineer at GodDev." + _RAW_RULE + _QUALITY_RULE +
        "\nRole: test suites, algorithms, fixtures, mocks, coverage configs. "
        "Standards: high branch coverage, edge cases, error paths tested, hermetic tests. "
        "Production quality only — no TODOs, no stubs."
    ),
    "self-improvement": (
        "You are GodDev's Self-Improvement Agent. You improve GodDev's own Python source files.\n\n"
        "## OUTPUT FORMAT\n"
        "Output the COMPLETE improved Python file — raw content, no markdown fences.\n"
        "Start with the very first character of the file (the module docstring or import).\n\n"
        "## RULES\n"
        "- Output the ENTIRE file with your improvements applied\n"
        "- Maintain ALL existing public interfaces (function signatures, class names, module exports)\n"
        "- The file MUST pass `python3 -m py_compile` — syntax errors cause automatic revert\n"
        "- Make targeted, specific improvements described in the task\n"
        "- Do NOT remove existing functionality — only add or improve\n"
        "- NO markdown fences, NO preamble, NO explanations — just the complete file\n\n"
        "## INCREMENTAL EXCELLENCE\n"
        "A small, correct change is infinitely better than an ambitious rewrite that breaks things.\n"
        "- Make the MINIMUM changes needed to address the specific task\n"
        "- If unsure about a change, DON'T make it — leave the original code intact\n"
        "- Every line you change is a chance to introduce a bug — minimize your diff\n"
        "- Preserve ALL imports, ALL class definitions, ALL function signatures\n\n"
        "## INTEGRATION CONTRACTS (NEVER BREAK)\n"
        "- GodDevState fields: never remove/rename — only add optional fields\n"
        "- Node signatures: (state: GodDevState) -> dict | Command — frozen\n"
        "- LLM Router API: acall(role, messages), acall_structured(role, messages, schema) — frozen\n"
        "- Import paths: ..llm_router, ..state, ..task_queue — never change package structure\n"
        "- Pydantic schemas: CriticVerdictOut, MetaBlueprint, ImprovementPlan — field names frozen\n\n"
        + _QUALITY_RULE +
        "\nRole: improve GodDev's own Python source files with complete file rewrites."
    ),
}

# ─── SEARCH/REPLACE patch support (self-improvement mode) ────────────────────

_SR_PATTERN = re.compile(
    r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
    re.DOTALL,
)

def _is_search_replace(text: str) -> bool:
    """Detect if text contains SEARCH/REPLACE blocks."""
    return bool(_SR_PATTERN.search(text))


def _apply_search_replace(patch_text: str, original: str) -> tuple[bool, str, str]:
    """
    Apply SEARCH/REPLACE blocks to original file content.

    Returns (success, result_or_error, details).
    """
    blocks = _SR_PATTERN.findall(patch_text)
    if not blocks:
        return False, original, "No SEARCH/REPLACE blocks found"

    result = original
    applied = 0
    for i, (search, replace) in enumerate(blocks):
        search_norm = search.replace('\r\n', '\n')
        replace_norm = replace.replace('\r\n', '\n')

        if search_norm in result:
            result = result.replace(search_norm, replace_norm, 1)
            applied += 1
            continue

        # Whitespace-drift fallback: match by stripped lines, replace by line range
        search_stripped = '\n'.join(l.rstrip() for l in search_norm.splitlines())
        result_orig_lines = result.splitlines(keepends=True)
        result_stripped = '\n'.join(l.rstrip() for l in result.splitlines())

        if search_stripped in result_stripped:
            idx = result_stripped.index(search_stripped)
            start_line = result_stripped[:idx].count('\n')
            end_line = start_line + search_stripped.count('\n')
            rep_lines = replace_norm.splitlines(keepends=True)
            if rep_lines and not rep_lines[-1].endswith('\n'):
                rep_lines[-1] += '\n'
            result = ''.join(result_orig_lines[:start_line] + rep_lines + result_orig_lines[end_line + 1:])
            applied += 1
            continue

        return False, original, f"SEARCH block not found (block {i+1}): {search_norm[:120]}..."

    return True, result, f"Applied {applied} SEARCH/REPLACE block(s)"


# ─── Code fence stripper ──────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Strip markdown code fences that LLMs sometimes add despite instructions."""
    s = text.strip()
    # Full ```lang\n...\n``` block
    m = re.match(r'^```[^\n]*\n(.*?)(?:\n```\s*)?$', s, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    # Partial fence at top only
    if s.startswith("```"):
        lines = s.split('\n')
        body = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            body.append(line)
        if body:
            return '\n'.join(body)
    return s

# ─── Brief builder ────────────────────────────────────────────────────────────

def _build_brief(file_task: dict, critic_fb: str = "") -> str:
    fp       = file_task.get("file_path", "")
    desc     = file_task.get("description", "")
    veri_cmd = file_task.get("verification_cmd") or "none"
    tech     = file_task.get("tech_stack") or {}
    arch     = file_task.get("architecture_overview", "")
    ctx      = file_task.get("squad_context", "")
    deps     = file_task.get("depend_on") or file_task.get("imports_from") or []

    tech_str = (
        f"Languages: {', '.join(tech.get('languages', []))}\n"
        f"Frameworks: {', '.join(tech.get('frameworks', []))}\n"
        f"Databases: {', '.join(tech.get('databases', []))}"
    )

    dep_contents: list[str] = []
    for dep in deps[:5]:
        try:
            txt = Path(dep).read_text(encoding="utf-8")
            dep_contents.append(f"### Already written: `{dep}`\n```\n{txt[:2500]}\n```")
        except Exception:
            pass

    brief = f"""## Task: Write `{fp}`

**Absolute path**: `{fp}`
**Verification**: `{veri_cmd}`

## Tech Stack
{tech_str}

## Architecture
{arch[:1200] if arch else "See description."}

## Squad Context
{ctx[:600] if ctx else "N/A"}

## File Specification
{desc}
"""
    if dep_contents:
        brief += "\n## Dependency Files (for integration context)\n" + "\n\n".join(dep_contents)
    if critic_fb:
        brief += f"\n\n## ⚠️ Critic Issues to Fix\n{critic_fb}"

    brief += f"\n\n---\nWrite the COMPLETE content of `{fp}` now. Raw content only — no fences, no commentary."
    return brief

# ─── Verification ─────────────────────────────────────────────────────────────

def _verify(cmd: str, file_path: str, project_dir: str) -> tuple[bool, str]:
    actual = cmd.replace("{file_path}", file_path).replace("{{file_path}}", file_path)
    try:
        r = subprocess.run(actual, shell=True, capture_output=True,
                           text=True, timeout=30, cwd=project_dir)
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out or ("OK" if r.returncode == 0 else "non-zero exit")
    except subprocess.TimeoutExpired:
        return False, "timed out (30s)"
    except Exception as exc:
        return False, str(exc)

# ─── Worker Node ──────────────────────────────────────────────────────────────

async def worker_node(state: GodDevState) -> dict:
    """
    LiteLLM-router worker: token-bucket throttling + auto-fallback + file lock.
    Now fully async — no asyncio.run() inside thread-pool.

    Flow:
      1. Resolve role from agent/domain
      2. Check file lock registry — skip if another worker already claimed this file
      3. Acquire token-bucket slot (awaits briefly if rate-limited, avoids 429s)
      4. Call LiteLLM router.acompletion — auto-fallbacks to next model if needed
      5. Strip code fences, write file to disk
      6. Run verification → auto-fix if fails
    """
    file_task:    dict = state.get("current_file_task") or {}
    agent:        str  = file_task.get("agent", "openai")
    file_path:    str  = file_task.get("file_path", "")
    task_id:      str  = file_task.get("task_id", "?")
    squad_domain: str  = file_task.get("squad_domain", "unknown")
    veri_cmd:     str  = file_task.get("verification_cmd") or ""
    project_dir:  str  = file_task.get("project_dir", "/opt/goddev/projects")
    run_id:       str  = (state.get("metadata") or {}).get("project_id", "")
    job_id:       str  = (state.get("metadata") or {}).get("job_id", run_id)
    critic_fb:    str  = state.get("critic_feedback") or ""

    # ── Hard guard: skip if file_path is empty or a directory ──────────────
    if not file_path or not file_path.strip() or file_path.strip() in (".", "/"):
        print(f"  [WORKER] Skipping task {task_id}: empty/invalid file_path '{file_path}'")
        output = FileOutput(
            task_id=task_id, file_path=file_path or "(empty)",
            agent=agent, bytes_written=0, verification_passed=False,
            verification_output="skipped: empty file_path",
            error="empty file_path", squad_domain=squad_domain,
        )
        return {"worker_outputs": [output.model_dump()],
                "messages": [AIMessage(content=f"⏭️ **[Worker]** Skipped task `{task_id}`: empty file_path")]}

    # ── Resolve LiteLLM router role (affinity overrides domain default on product squads)
    runtime_cfg = (state.get("metadata") or {}).get("runtime_config") or {}
    preferred = (runtime_cfg.get("preferred_agents") or {})
    if squad_domain in _AFFINITY_SQUAD_DOMAINS and isinstance(preferred, dict):
        aff_role = _affinity_router_role(file_path, squad_domain, preferred)
    else:
        aff_role = None
    if aff_role:
        role = aff_role
    else:
        role = _DOMAIN_TO_ROLE.get(squad_domain) or _AGENT_TO_ROLE.get(str(agent).lower(), "backend")
    label = _LABELS.get(role, role.upper())

    # ── File lock registry ────────────────────────────────────────────────────
    registry = get_file_registry()
    worker_id = f"{task_id}:{squad_domain}"

    if file_path and not registry.claim(job_id, file_path, worker_id):
        owner = registry.get_owner(job_id, file_path)
        msg = (f"⏭️ **[{label} Worker]** `{file_path}`\n"
               f"  SKIPPED — already claimed by `{owner}`")
        output = FileOutput(
            task_id=task_id, file_path=file_path, agent=agent,
            bytes_written=0, verification_passed=True,
            verification_output="skipped: file claimed by another worker",
            error=None, squad_domain=squad_domain,
        )
        return {"worker_outputs": [output.model_dump()], "messages": [AIMessage(content=msg)]}

    t0 = time.monotonic()
    status    = "ok"
    error_msg: dict | None = None
    veri_out  = ""
    veri_pass = False
    written   = 0

    # ── Worker prompt additions from self-improvement runtime_config ──────────
    worker_additions = runtime_cfg.get("worker_prompt_additions", "")

    try:
        Path(project_dir).mkdir(parents=True, exist_ok=True)
        # Self-improvement workers always get the SEARCH/REPLACE prompt
        _is_self_improve = (squad_domain == "self-improvement")
        if _is_self_improve and "self-improvement" in _SYSTEMS:
            system = _SYSTEMS["self-improvement"]
        else:
            system = _SYSTEMS.get(role, _SYSTEMS["backend"])
        # Append learned worker prompt additions from self-improvement
        if worker_additions:
            system = system + f"\n\n## ADDITIONAL QUALITY REQUIREMENTS (from system self-improvement):\n{worker_additions[:1500]}"
        brief  = _build_brief(file_task, critic_fb)

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": brief},
        ]

        # ── Generation attempt 1 ──────────────────────────────────────────────
        # Worker is now async def — safe to await directly on the LangGraph event loop.
        await acquire_slot(role)
        job_id = state.get("metadata", {}).get("job_id")
        complexity = _complexity_for(file_path, squad_domain, fix_attempts=0)
        affinity = _file_affinity_key(file_path)
        content = await acall(role, messages, complexity=complexity,
                              affinity=affinity,
                              metadata={"job_id": job_id})
        content = _strip_fences(content)

        if not content or len(content) < 10:
            raise ValueError(f"Router returned empty content for role={role}")

        fp = Path(file_path)
        fp.parent.mkdir(parents=True, exist_ok=True)

        # ── SEARCH/REPLACE patch mode for self-improvement ───────────────
        _patch_applied = False
        if _is_self_improve and _is_search_replace(content):
            original = ""
            if fp.exists():
                original = fp.read_text(encoding="utf-8")
            if original:
                ok, result, detail = _apply_search_replace(content, original)
                if ok:
                    fp.write_text(result, encoding="utf-8")
                    # Immediate syntax check — revert if patch broke Python syntax
                    if file_path.endswith(".py"):
                        _syn = subprocess.run(
                            ["python3", "-m", "py_compile", file_path],
                            capture_output=True, text=True,
                        )
                        if _syn.returncode != 0:
                            fp.write_text(original, encoding="utf-8")
                            print(f"  [PATCH-REVERTED] Syntax error — original restored: {_syn.stderr[:80]}")
                            written = len(original.encode())
                        else:
                            written = len(result.encode())
                            _patch_applied = True
                            print(f"  [PATCH] {detail} -> {fp.name}")
                    else:
                        written = len(result.encode())
                        _patch_applied = True
                        print(f"  [PATCH] {detail} -> {fp.name}")
                else:
                    # Patch failed — keep original, do NOT write raw patch text
                    print(f"  [PATCH-FAIL] {detail} — keeping original file intact")
                    written = len(original.encode())
                    # written set so byte gate passes; veri_pass stays False → self-heal
            else:
                # No original file — write as-is (new file)
                fp.write_text(content, encoding="utf-8")
                written = len(content.encode())
        else:
            if _is_self_improve:
                # Full-file improvement mode: write and validate syntax
                fp.write_text(content, encoding="utf-8")
                written = len(content.encode())
                if file_path.endswith(".py") and written > 100:
                    _syn = subprocess.run(
                        ["python3", "-m", "py_compile", file_path],
                        capture_output=True, text=True,
                    )
                    if _syn.returncode != 0:
                        # Syntax error — revert to original
                        _orig = ""
                        try:
                            _live = Path(file_path.replace("/staging/src/", "/src/"))
                            if _live.exists():
                                _orig = _live.read_text(encoding="utf-8")
                        except Exception:
                            pass
                        if _orig:
                            fp.write_text(_orig, encoding="utf-8")
                            written = len(_orig.encode())
                            print(f"  [SELF-IMPROVE] Syntax error in full-file — reverted to live: {_syn.stderr[:60]}")
                        else:
                            written = 0
            else:
                fp.write_text(content, encoding="utf-8")
                written = len(content.encode())


        # ── Self-improvement size guard ──────────────────────────────────────────
        # Note: Protected file check removed — the whole ecosystem is available for self-modification.
        if _is_self_improve and file_path.endswith('.py') and written > 0:
            _live_ref = Path(file_path.replace('/staging/src/', '/src/'))
            _fname = Path(file_path).name
            # Size guard: revert if written file is <40% of the live original
            if _live_ref.exists():
                _live_size = _live_ref.stat().st_size
                if _live_size > 2000 and written < _live_size * 0.4:
                    _live_content = _live_ref.read_text(encoding='utf-8')
                    fp.write_text(_live_content, encoding='utf-8')
                    written = len(_live_content.encode())
                    print(f'  [SIZE-GUARD] {fp.name}: reverted (too small)')
        # Size guard for static UI files (HTML/JS/CSS) — also protect them from tiny stubs
        elif _is_self_improve and file_path.endswith(('.html', '.js', '.css')) and written > 0:
            _live_ref = Path(file_path.replace('/staging/static/', '/static/').replace('/staging/src/../static/', '/static/'))
            if _live_ref.exists():
                _live_size = _live_ref.stat().st_size
                if _live_size > 5000 and written < _live_size * 0.3:
                    _live_content = _live_ref.read_text(encoding='utf-8')
                    fp.write_text(_live_content, encoding='utf-8')
                    written = len(_live_content.encode())
                    print(f'  [SIZE-GUARD] {fp.name}: HTML/JS/CSS too small ({written}b << {_live_size}b live) — reverted')

        # ── Verification & Self-Healing Loop ──────────────────────────────────
        MAX_AUTO_FIX = 4
        fix_attempts = 0
        fix_msgs = messages.copy()

        while True:
            # 1. Byte Gate Check
            if written < 50:
                veri_pass = False
                veri_out = f"Blocked by Byte Gate: Content only {written} bytes. Must be >= 50 bytes."
                veri_cmd = veri_cmd or "echo 'Byte Gate Check'"
            elif veri_cmd:
                veri_pass, veri_out = _verify(veri_cmd, file_path, project_dir)
            else:
                veri_pass = True
                veri_out = "OK (No verification config or skipped)"

            # 2. Exit condition Check
            if veri_pass or fix_attempts >= MAX_AUTO_FIX:
                break

            # 3. Prepare Self-Healing Iteration
            fix_attempts += 1
            fix_msgs.append({"role": "assistant", "content": content})
            if _is_self_improve:
                # For self-improvement: ask for SEARCH/REPLACE fix blocks
                # Read current file state (may have been patched)
                current_src = ""
                if fp.exists():
                    current_src = fp.read_text(encoding="utf-8")[:6000]
                fix_msgs.append({"role": "user", "content": (
                    f"## ❌ Verification FAILED (Attempt {fix_attempts}/{MAX_AUTO_FIX})\n"
                    f"Command: `{veri_cmd}`\n"
                    f"Error:\n```\n{veri_out[:1000]}\n```\n\n"
                    f"Current file state (after previous patch):\n```python\n{current_src}\n```\n\n"
                    f"Fix EVERY error using SEARCH/REPLACE blocks. Match the current file state EXACTLY."
                )})
            else:
                fix_msgs.append({"role": "user", "content": (
                    f"## ❌ Verification FAILED (Attempt {fix_attempts}/{MAX_AUTO_FIX})\n"
                    f"Command: `{veri_cmd}`\n"
                    f"Error:\n```\n{veri_out[:1000]}\n```\n\n"
                    f"Fix EVERY error based on the compiler/linter feedback. Output the COMPLETE corrected file — no fences, no commentary."
                )})

            # 4. Agentic Execution — escalate complexity as fix attempts grow
            await acquire_slot(role)
            fix_complexity = _complexity_for(file_path, squad_domain, fix_attempts=fix_attempts)
            content = _strip_fences(await acall(
                role, fix_msgs,
                complexity=fix_complexity,
                affinity=affinity,
                metadata={"job_id": job_id},
            ))
            
            # Write new file content to disk
            if content and len(content) >= 10:
                if _is_self_improve and _is_search_replace(content):
                    original = fp.read_text(encoding="utf-8") if fp.exists() else ""
                    if original:
                        ok, result, detail = _apply_search_replace(content, original)
                        if ok:
                            fp.write_text(result, encoding="utf-8")
                            written = len(result.encode())
                            print(f"  [PATCH-FIX] {detail} → {fp.name}")
                            continue
                        else:
                            print(f"  [PATCH-FIX-FAIL] {detail} — keeping current state")
                            written = len(fp.read_bytes()) if fp.exists() else 0
                            continue  # retry with current file state shown to model
                if not _is_self_improve:
                    fp.write_text(content, encoding="utf-8")
                    written = len(content.encode())
                else:
                    # Malformed output in self-heal — keep current staging file
                    written = len(fp.read_bytes()) if fp.exists() else 0
            else:
                written = 0

    except Exception as exc:
        status    = "error"
        import traceback
        error_msg = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()
        }
        # Release file lock so another worker can retry
        if file_path:
            registry.release(job_id, file_path)
    finally:
        elapsed = time.monotonic() - t0

    if run_id:
        try:
            trace.log_worker(run_id=run_id, agent=role, task_id=task_id,
                             file_path=file_path, squad=squad_domain,
                             elapsed_s=elapsed, status=status,
                             bytes_written=written, verification_passed=veri_pass)
        except Exception:
            pass

    icon = "✅" if veri_pass else ("⚠️" if status == "ok" else "❌")
    msg  = (f"{icon} **[{label} Worker]** `{file_path}`\n"
            f"  bytes={written:,} | verification={'PASS' if veri_pass else 'FAIL'} "
            f"| elapsed={elapsed:.1f}s")
    if error_msg:
        msg += f"\n  ⚠️ {error_msg['type']}: {error_msg['message'][:80]}"

    output = FileOutput(
        task_id=task_id, file_path=file_path, agent=agent,
        bytes_written=written, verification_passed=veri_pass,
        verification_output=veri_out, error=error_msg, squad_domain=squad_domain,
    )
    return {"worker_outputs": [output.model_dump()], "messages": [AIMessage(content=msg)]}
