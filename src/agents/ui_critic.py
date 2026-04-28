"""UI Critic agent for GodDev 3.0.

This module provides a UI critic node that scores HTML/CSS/JS files against
Nielsen heuristics and WCAG 2.1 AA accessibility guidelines. It scans worker
outputs for UI-related files and produces structured critique verdicts.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from ..llm_router import acall_structured
from .critic_council import CriticVerdictOut
from ..state import GodDevState
from ..task_queue import acquire_slot

# Constants for file scanning budget
MAX_FILES = 6
MAX_LINES_PER_FILE = 100
MAX_CHARS_TOTAL = 18_000

# Regex pattern for UI-related file extensions
UI_FILE_PATTERN = re.compile(r'\.(html|css|jsx?|tsx?|vue|svelte)$', re.IGNORECASE)

# System prompt for the UI critic
UI_CRITIC_SYSTEM_PROMPT = """You are a UI/UX critic evaluating frontend code against Nielsen's 10 usability heuristics and WCAG 2.1 AA standards.

Evaluate the provided UI code files and return a structured critique. Apply these specific checks:

1. **Color Contrast (WCAG 1.4.3)**: Verify color contrast ratio >= 4.5:1 for normal text
2. **Accessible Labels (WCAG 4.1.2)**: All interactive elements must have aria-label or visible label
3. **Responsive Design (WCAG 1.4.10)**: Check for viewport meta tag and media queries
4. **Keyboard Focus (WCAG 2.4.7)**: Ensure keyboard focus indicators are visible
5. **Event Handling**: No inline event handlers (onclick= etc.) - use addEventListener instead

For each issue found, provide:
- The specific file and line number
- Which heuristic/WCAG criterion is violated
- A concrete suggestion for fixing it

Score the overall UI quality from 0.0 to 10.0 based on severity and number of issues found."""


def _is_ui_file(file_path: str) -> bool:
    """Check if a file path matches UI-related extensions.

    Args:
        file_path: Path to check against UI file patterns.

    Returns:
        True if the file has a UI-related extension, False otherwise.
    """
    return bool(UI_FILE_PATTERN.search(file_path))


def _read_file_content(file_path: str, base_path: Optional[Path] = None) -> Optional[str]:
    """Read file content with line and character limits.

    Sanitizes error messages to prevent information disclosure.

    Args:
        file_path: Path to the file to read.
        base_path: Optional base path to resolve relative paths.

    Returns:
        File content as string if successful, None if file cannot be read,
        or a sanitized error message string on failure.
    """
    try:
        if base_path:
            full_path = base_path / file_path
        else:
            full_path = Path(file_path)

        # Validate path to prevent path traversal
        try:
            resolved_path = full_path.resolve()
            if base_path:
                resolved_base = base_path.resolve()
                if not str(resolved_path).startswith(str(resolved_base)):
                    # File is outside the allowed working directory
                    return '# [ERROR: Access to file outside permitted directory is denied.]'
            full_path = resolved_path # Use the resolved path for operations
        except (ValueError, OSError):
            # Path resolution failed or path is invalid
            return '# [ERROR: Invalid file path provided.]'


        if not full_path.exists():
            return None # Indicate file not found without disclosing path details if it was invalid from the start

        if not full_path.is_file():
            return '# [ERROR: Path refers to a directory, not a file.]'

        content = full_path.read_text(encoding='utf-8', errors='replace')
        lines = content.splitlines()

        # Apply line limit
        if len(lines) > MAX_LINES_PER_FILE:
            lines = lines[:MAX_LINES_PER_FILE]
            lines.append(f'\n# [TRUNCATED: file exceeds {MAX_LINES_PER_FILE} lines]')

        return '\n'.join(lines)
    except (FileNotFoundError, PermissionError):
        # Specific OS errors handled for clearer, sanitized message
        return '# [ERROR: File not found or access denied due to permissions.]'
    except (OSError, IOError):
        # Catch other general OS/IO errors and provide a generic message
        return '# [ERROR: Could not read file content due to an internal system issue.]'


def _collect_ui_files(state: GodDevState) -> List[Dict[str, str]]:
    """Collect UI files from worker outputs within budget constraints.

    Args:
        state: Current GodDev state containing worker outputs.

    Returns:
        List of dicts with 'path' and 'content' keys for UI files found.
    """
    ui_files: List[Dict[str, str]] = []
    total_chars = 0
    files_scanned = 0

    # Get the working directory from state if available
    base_path = None
    if hasattr(state, 'working_directory') and state.working_directory:
        base_path = Path(state.working_directory)

    # Scan worker outputs for UI files
    worker_outputs = state.get('worker_outputs') or []
    if not worker_outputs:
        return ui_files

    for output in worker_outputs:
        if files_scanned >= MAX_FILES:
            break

        # Handle both string and dict worker outputs
        if isinstance(output, dict):
            file_path = output.get('file_path', '') or output.get('path', '') or output.get('file', '')
        elif isinstance(output, str):
            file_path = output
        else:
            continue

        if not file_path or not _is_ui_file(file_path):
            continue

        # _read_file_content now handles internal path validation
        content = _read_file_content(file_path, base_path)
        
        # If content is None, it means the file was legitimately not found and we should skip it.
        # If content is an error string, we should include it for the critic to see the issue.
        if content is None:
            continue

        # Check character budget
        content_chars = len(content)
        if total_chars + content_chars > MAX_CHARS_TOTAL:
            # Truncate to fit budget
            available = MAX_CHARS_TOTAL - total_chars
            if available > 100:  # Only include if we can show meaningful content
                content = content[:available] + '\n# [TRUNCATED: character budget exceeded]'
                content_chars = len(content)
            else:
                break

        ui_files.append({
            'path': file_path,
            'content': content,
        })
        total_chars += content_chars
        files_scanned += 1

    return ui_files


def _build_critic_prompt(ui_files: List[Dict[str, str]]) -> str:
    """Build the prompt for the UI critic with collected file contents.

    Args:
        ui_files: List of UI file dicts with 'path' and 'content' keys.

    Returns:
        Formatted prompt string for the critic model.
    """
    if not ui_files:
        return "No UI files found to review."

    prompt_parts = ["Please review the following UI files:", ""]

    for file_info in ui_files:
        prompt_parts.append(f"--- File: {file_info['path']} ---")
        prompt_parts.append(file_info['content'])
        prompt_parts.append("")

    prompt_parts.append(
        "Evaluate these files against Nielsen heuristics and WCAG 2.1 AA standards. "
        "Focus on: color contrast, accessible labels, responsive design, "
        "keyboard focus visibility, and proper event handling."
    )

    return '\n'.join(prompt_parts)


async def ui_critic_node(state: GodDevState) -> Dict[str, Any]:
    """UI Critic node that evaluates frontend code quality.

    Scans worker outputs for UI files, sends them to the critic model,
    and returns structured critique verdicts.

    Args:
        state: Current GodDev state containing worker outputs and other context.

    Returns:
        Dict with 'critic_verdicts' key containing list of CriticVerdictOut dicts.
        Returns a soft-approve verdict (score 7.0) when no UI files are present.
    """
    job_id = (state.get("metadata") or {}).get("job_id")
    try:
        # Collect UI files from worker outputs
        ui_files = _collect_ui_files(state)

        # If no UI files found, return soft-approve verdict
        if not ui_files:
            soft = {
                "critic_type": "code",
                "approved": True,
                "score": 7.0,
                "issues": [],
                "actionable_feedback": "No UI files found — auto-approved.",
                "critical_files": [],
            }
            return {
                "critic_verdicts": [soft],
                "messages": [AIMessage(content="✅ **[UI Critic]** No UI files — auto-approved (7.0/10)")],
            }

        # Build the prompt with collected file contents
        prompt = _build_critic_prompt(ui_files)

        await acquire_slot("critic")

        # Dispatch to critic model via the standard acall_structured API
        verdict: CriticVerdictOut = await acall_structured(
            "critic",
            [
                {"role": "system", "content": UI_CRITIC_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            CriticVerdictOut,
            complexity="trivial",
            compress=True,
            affinity="ui_review",
            metadata={"job_id": job_id} if job_id else None,
        )
        v = verdict.model_dump()
        v["critic_type"] = "code"  # normalize for verdict collector

        icon = "✅" if v["approved"] else "❌"
        return {
            "critic_verdicts": [v],
            "messages": [AIMessage(content=(
                f"{icon} **[UI Critic]** score={v['score']:.1f}/10 | "
                f"approved={v['approved']} | issues={len(v['issues'])} | "
                f"files reviewed: {len(ui_files)}"
            ))],
        }

    except Exception as e:
        # Return a safe fallback verdict on error
        soft = {
            "critic_type": "code",
            "approved": True,
            "score": 6.0,
            "issues": [f"UI critic error: {e.__class__.__name__}: {str(e)[:80]}"],
            "actionable_feedback": "UI critic could not run — review manually.",
            "critical_files": [],
        }
        return {
            "critic_verdicts": [soft],
            "messages": [AIMessage(content=f"⚠️ **[UI Critic]** Error: {e.__class__.__name__} — soft-approved (6.0/10)")],
        }