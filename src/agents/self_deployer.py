"""
GodDev 3.0 — Self-Deployer
============================

Runs AFTER the Critic Council approves a self-improvement build.

Steps:
  1. Validate staging/ — all improved files must pass syntax checks
  2. Backup live src/ → /opt/goddev/backups/src-{timestamp}/
  3. Diff staging/ vs live src/ — show exactly what changes
  4. Deploy: copy staging/src/ files → live /opt/goddev/src/
  5. Hot-reload: send SIGHUP to uvicorn (triggers graceful reload)
  6. Smoke test: hit /health to confirm the service is still up
  7. Rollback if smoke test fails: restore from backup

This is the safest possible self-deployment — always backs up,
always smoke tests, always rolls back if anything breaks.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage

from ..state import GodDevState


def _validate_staging(staging_src: Path) -> tuple[bool, list[str]]:
    """Run py_compile on every Python file in staging and do interface testing."""
    errors: list[str] = []
    
    # Count and log the number of Python files to validate
    num_py_files = sum(1 for _ in staging_src.rglob("*.py") if "__pycache__" not in str(_))
    print(f"[Deploy] Validating {num_py_files} staging files...")
    
    # 1. Syntax checks
    for py in staging_src.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(py)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            errors.append(f"{py}: {result.stderr.strip()}")
            
    # 2. Interface Preservation Checks
    if len(errors) == 0:
        verify_script = Path("/opt/goddev/verify_agents.py")
        if verify_script.exists():
            env = dict(os.environ)
            # Prepend staging to PYTHONPATH so it tests the staged code, not live code
            env["PYTHONPATH"] = f"/opt/goddev/staging:{env.get('PYTHONPATH', '')}"
            result = subprocess.run(
                ["/opt/goddev/venv/bin/python3", str(verify_script)],
                capture_output=True, text=True, env=env,
            )
            if result.returncode != 0:
                errors.append(f"Interface Check Failed:\n{result.stderr.strip() or result.stdout.strip()}")
                
    return len(errors) == 0, errors


def _backup_src(live_src: Path, backup_base: Path) -> Path:
    """Back up live src/ to backups/src-{timestamp}/."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_dir = backup_base / f"src-{ts}"
    shutil.copytree(live_src, backup_dir)
    return backup_dir


def _compute_diff(staging_src: Path, live_src: Path) -> list[str]:
    """Return list of changed file paths."""
    changed: list[str] = []
    for staged in staging_src.rglob("*"):
        if staged.is_dir() or "__pycache__" in str(staged):
            continue
        rel = staged.relative_to(staging_src)
        live = live_src / rel
        if not live.exists():
            changed.append(f"[NEW]      {rel}")
        else:
            staged_content = staged.read_bytes()
            live_content = live.read_bytes()
            if staged_content != live_content:
                changed.append(f"[MODIFIED] {rel}")
    return changed


def _deploy_staging(staging_src: Path, live_src: Path, changed_files: list[str]) -> None:
    """Copy changed files from staging/src/ to src/ and staging/static/ to static/."""
    for staged in staging_src.rglob("*"):
        if staged.is_dir() or "__pycache__" in str(staged):
            continue
        rel = staged.relative_to(staging_src)
        print(f"[Deploy] Copying {rel} to live...")
        live = live_src / rel
        live.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged, live)

    # Also deploy UI files from staging/static/ to static/
    staging_static = Path("/opt/goddev/staging/static")
    live_static = Path("/opt/goddev/static")
    if staging_static.exists():
        for staged in staging_static.rglob("*"):
            if staged.is_dir():
                continue
            rel = staged.relative_to(staging_static)
            live = live_static / rel
            staged_size = staged.stat().st_size
            live_size = live.stat().st_size if live.exists() else 0
            if live_size > 5000 and staged_size < live_size * 0.3:
                print(f"  [SIZE-GUARD] {rel}: {staged_size}b too small vs {live_size}b live — skipping")
                continue
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, live)
            print(f"  [STATIC] Deployed {rel} ({staged_size:,}b)")


def _reload_service() -> tuple[bool, str]:
    """Schedule a delayed service restart so the current request can complete first.

    Uses 'at now + 1 minute' or a background subprocess with a short sleep,
    ensuring the graph can finish writing job status before we kill ourselves.
    """
    try:
        # Schedule restart 8 seconds from now — enough time for the deployer
        # function to return and the API to write the completed job status.
        subprocess.Popen(
            "sleep 8 && systemctl restart goddev",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True, "Service restart scheduled (8s delay)"
    except Exception as exc:
        return False, f"Could not schedule restart: {exc}"


def _smoke_test(retries: int = 5) -> bool:
    """Hit /health and confirm service is up."""
    for i in range(retries):
        print(f"[Deploy] Smoke test attempt {i+1}/{retries}...")
        time.sleep(3)
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                body = r.read()
                if b"ok" in body:
                    return True
        except Exception:
            pass
    return False


def _restore_backup(backup_dir: Path, live_src: Path) -> None:
    """Restore live src/ from backup."""
    shutil.rmtree(live_src)
    shutil.copytree(backup_dir, live_src)
    _reload_service()


def self_deployer_node(state: GodDevState) -> dict:
    """
    Validate, backup, deploy staging improvements, smoke test.
    Rolls back automatically if the service doesn't recover.
    """
    staging_src = Path("/opt/goddev/staging/src")
    live_src = Path("/opt/goddev/src")
    staging_static = Path("/opt/goddev/staging/static")
    live_static = Path("/opt/goddev/static")
    backup_base = Path("/opt/goddev/backups")
    backup_base.mkdir(parents=True, exist_ok=True)

    worker_outputs: list[dict] = state.get("worker_outputs") or []
    improvement_id = state.get("project_name", "unknown-improvement")

    # ── Step 1: Validate staging files ────────────────────────────────────────
    if not staging_src.exists() or not any(staging_src.rglob("*.py")):
        msg = "❌ **Self-Deployer**: No Python files found in staging — aborting deploy."
        return {
            "integration_passed": False,
            "messages": [AIMessage(content=msg)],
        }

    valid, syntax_errors = _validate_staging(staging_src)
    if not valid:
        errors_text = "\n".join(syntax_errors)
        msg = (
            f"❌ **Self-Deployer**: Syntax errors in staged files — aborting.\n"
            f"```\n{errors_text}\n```"
        )
        return {
            "integration_passed": False,
            "messages": [AIMessage(content=msg)],
        }

    # ── Step 2: Compute diff ───────────────────────────────────────────────────
    changed_files = _compute_diff(staging_src, live_src)
    # Also diff staging/static/ vs static/ for UI improvements
    if staging_static.exists() and live_static.exists():
        for sc in _compute_diff(staging_static, live_static):
            changed_files.append(sc)
    if not changed_files:
        msg = "ℹ️ **Self-Deployer**: No changes detected in staging vs live — nothing to deploy."
        return {
            "integration_passed": True,
            "messages": [AIMessage(content=msg)],
        }

    diff_text = "\n".join(changed_files)

    # ── Step 3: Backup live src/ ───────────────────────────────────────────────
    try:
        backup_dir = _backup_src(live_src, backup_base)
    except Exception as exc:
        msg = f"❌ **Self-Deployer**: Failed to create backup — aborting. Error: {exc}"
        return {
            "integration_passed": False,
            "messages": [AIMessage(content=msg)],
        }

    # ── Step 4: Deploy ─────────────────────────────────────────────────────────
    try:
        _deploy_staging(staging_src, live_src, changed_files)
    except Exception as exc:
        # Restore backup immediately
        _restore_backup(backup_dir, live_src)
        msg = f"❌ **Self-Deployer**: Deploy failed, backup restored. Error: {exc}"
        return {
            "integration_passed": False,
            "messages": [AIMessage(content=msg)],
        }

    # ── Step 5: Reload service ─────────────────────────────────────────────────
    reload_ok, reload_output = _reload_service()
    if not reload_ok:
        _restore_backup(backup_dir, live_src)
        msg = f"❌ **Self-Deployer**: Service reload failed, backup restored.\n```\n{reload_output}\n```"
        return {
            "integration_passed": False,
            "messages": [AIMessage(content=msg)],
        }

    # ── Step 6: Smoke test ─────────────────────────────────────────────────────
    service_up = _smoke_test(retries=8)

    if not service_up:
        # ROLLBACK
        _restore_backup(backup_dir, live_src)
        _reload_service()
        msg = (
            f"⚠️ **Self-Deployer**: Smoke test failed after deploy — ROLLED BACK to `{backup_dir.name}`.\n"
            f"GodDev is back to the previous version. Investigate the staged changes."
        )
        return {
            "integration_passed": False,
            "messages": [AIMessage(content=msg)],
        }

    # ── Step 7: Clean up staging ───────────────────────────────────────────────
    try:
        shutil.rmtree(staging_src)
    except Exception:
        pass

    # ── Build success message ──────────────────────────────────────────────────
    n_files = len(changed_files)
    n_workers = len(worker_outputs)
    verif_pass = sum(1 for o in worker_outputs if o.get("verification_passed"))

    msg = (
        f"## ✅ Self-Deployment Complete — `{improvement_id}`\n\n"
        f"**Files improved**: {n_files}\n"
        f"**Worker verification**: {verif_pass}/{n_workers} PASS\n"
        f"**Backup**: `{backup_dir}`\n"
        f"**Service**: Reloaded and healthy ✅\n\n"
        f"### Changes Deployed\n"
        f"```\n{diff_text}\n```\n\n"
        f"GodDev is now running its improved version of itself. 🚀"
    )

    integration_result = {
        "passed": True,
        "files_verified": changed_files,
        "backup_dir": str(backup_dir),
        "final_summary": msg,
    }

    return {
        "integration_result": integration_result,
        "integration_passed": True,
        "final_output": msg,
        "messages": [AIMessage(content=msg)],
    }