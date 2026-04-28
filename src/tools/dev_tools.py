"""
GodDev 3.0 — Developer Tool Suite
===================================

Tools available to every worker agent:
  - write_file       : Write content to disk, create parent dirs
  - read_file        : Read file contents
  - list_directory   : List project directory structure
  - run_shell_command: Execute whitelisted shell commands (broader than 2.0)
  - run_python_code  : Execute Python in isolated subprocess
  - search_web       : Tavily web search (optional)

Key upgrade over 2.0: run_shell_command whitelist expanded to cover
node/npm/npx, cargo, go, pytest, jest — workers can actually BUILD and TEST.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool, tool


# ─── File I/O ─────────────────────────────────────────────────────────────────


_GODDEV_ROOT = "/opt/goddev"
_PROTECTED = [
    "/opt/goddev/src/",
    "/opt/goddev/venv/",
    "/opt/goddev/deploy/",
    "/opt/goddev/.env",
    "/opt/goddev/memory/runtime_config.json",
]
# Staging is EXPLICITLY allowed — self-improvement workers write here
_ALWAYS_ALLOWED_PREFIXES = [
    "/opt/goddev/staging/",
    "/opt/goddev/projects/",
    "/opt/goddev/logs/",
    "/opt/goddev/memory/",
    "/opt/goddev/backups/",
    "/tmp/",
]


def _is_protected(file_path: str) -> bool:
    """Return True if the path is inside GodDev's own source tree."""
    normalized = str(Path(file_path).resolve())
    # Check explicit allowlist first (staging overrides protection)
    for allowed in _ALWAYS_ALLOWED_PREFIXES:
        if normalized.startswith(allowed):
            return False
    for protected in _PROTECTED:
        if normalized.startswith(protected) or normalized == protected.rstrip("/"):
            return True
    return False


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Args:
        file_path: Absolute path for the output file.
        content: Text content to write.

    Returns:
        Success message with byte count, or error.
    """
    if _is_protected(file_path):
        return (
            f"ERROR: WRITE BLOCKED — '{file_path}' is inside GodDev's protected "
            f"system directories. All project files must go inside /opt/goddev/projects/."
        )
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            return "ERROR: Cannot write to a directory path. Specify a file name."
        path.write_text(content, encoding="utf-8")
        return f"OK wrote {len(content.encode()):,} bytes → {file_path}"
    except Exception as exc:
        return f"ERROR write_file: {exc}"


@tool
def read_file(file_path: str) -> str:
    """Read and return the full contents of a file.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        File contents or error message.
    """
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: File not found — {file_path}"
    except Exception as exc:
        return f"ERROR read_file: {exc}"


@tool
def list_directory(directory_path: str) -> str:
    """Recursively list all files in a directory (up to 200 entries).

    Args:
        directory_path: Absolute path to the directory.

    Returns:
        Newline-separated list of relative file paths, or error.
    """
    try:
        base = Path(directory_path)
        if not base.exists():
            return f"ERROR: Directory not found — {directory_path}"
        files = sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())
        if not files:
            return "(empty directory)"
        truncated = files[:200]
        suffix = f"\n... ({len(files) - 200} more)" if len(files) > 200 else ""
        return "\n".join(truncated) + suffix
    except Exception as exc:
        return f"ERROR list_directory: {exc}"


@tool
def create_directory(directory_path: str) -> str:
    """Create a directory and all parent directories.

    Args:
        directory_path: Absolute path to create.

    Returns:
        Success or error message.
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return f"OK created directory: {directory_path}"
    except Exception as exc:
        return f"ERROR create_directory: {exc}"


# ─── Code Execution ───────────────────────────────────────────────────────────


@tool
def run_python_code(code: str) -> str:
    """Execute Python code in an isolated subprocess and return stdout/stderr.

    Args:
        code: Valid Python source code.

    Returns:
        Combined output, or error with traceback.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        tmp = fh.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=30,
        )
        out = result.stdout
        if result.returncode != 0 and result.stderr:
            out += f"\n[STDERR]:\n{result.stderr}"
        return out.strip() or "OK executed (no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: timeout after 30s"
    except Exception as exc:
        return f"ERROR run_python_code: {exc}"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ─── Shell Commands ───────────────────────────────────────────────────────────

_ALLOWED = frozenset({
    # Python ecosystem
    "python", "python3", "pip", "pip3", "pytest", "ruff", "black", "mypy",
    "py_compile",
    # Node ecosystem
    "node", "npm", "npx", "yarn", "pnpm",
    # Go
    "go",
    # Rust
    "cargo",
    # General
    "git", "echo", "ls", "cat", "mkdir", "touch", "cp", "mv",
    # Linters/formatters
    "eslint", "prettier", "tsc",
    # Build tools
    "make", "cmake",
})


@tool
def run_shell_command(command: str, working_dir: str = "") -> str:
    """Run a whitelisted shell command and return its output.

    Args:
        command: Shell command string to execute.
        working_dir: Optional working directory (defaults to current dir).

    Returns:
        stdout + stderr combined, or error.
    """
    parts = command.strip().split()
    if not parts:
        return "ERROR: empty command"
    base = os.path.basename(parts[0]).lower().rstrip(".exe")
    if base not in _ALLOWED:
        return (
            f"ERROR: '{base}' not in allowlist. Allowed: "
            + ", ".join(sorted(_ALLOWED))
        )
    cwd = working_dir if working_dir and os.path.isdir(working_dir) else None
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=cwd,
        )
        out = result.stdout
        if result.stderr:
            out += f"\n[STDERR]:\n{result.stderr}"
        return out.strip() or "OK (no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: timeout after 120s"
    except Exception as exc:
        return f"ERROR run_shell_command: {exc}"


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def get_dev_tools() -> List[BaseTool]:
    """Return the full GodDev 3.0 tool suite."""
    tools: List[BaseTool] = [
        write_file,
        read_file,
        list_directory,
        create_directory,
        run_python_code,
        run_shell_command,
    ]

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key and not tavily_key.startswith("tvly-..."):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tools.append(TavilySearchResults(max_results=6))
        except ImportError:
            pass

    return tools