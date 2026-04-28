"""
GodDev 3.0 — Persistent Session Store
=======================================

Persists job state, events, project chats, and WebSocket connections to disk.
Survives server restarts and allows cross-device session access.

Architecture:
- Each job/session is stored as a JSON file under SESSION_DIR
- In-memory dicts serve as a fast cache, synced to disk on every mutation
- WebSocket connections remain in-memory only (transient by nature)
- On reconnect, the full event history is replayed from disk
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ─── Configuration ────────────────────────────────────────────────────────────

SESSION_DIR = Path(os.getenv("SESSION_DIR", "/opt/goddev/sessions"))
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ─── In-memory cache ──────────────────────────────────────────────────────────
# These sync to disk on every write operation

_jobs: dict[str, dict[str, Any]] = {}
_job_events: dict[str, list[str]] = {}
_project_chats: dict[str, list[dict]] = {}
_ws_connections: dict[str, list] = {}  # WebSocket objects stay in-memory only

_lock = threading.Lock()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _job_path(job_id: str) -> Path:
    return SESSION_DIR / f"job_{job_id}.json"


def _events_path(job_id: str) -> Path:
    return SESSION_DIR / f"events_{job_id}.jsonl"


def _chat_path(project_name: str) -> Path:
    # Sanitize project name for filename
    safe = project_name.replace("/", "_").replace("\\", "_")
    return SESSION_DIR / f"chat_{safe}.json"


def _sync_job_to_disk(job_id: str) -> None:
    """Write job state to disk synchronously."""
    with _lock:
        job = _jobs.get(job_id)
        if job:
            _job_path(job_id).write_text(
                json.dumps(job, indent=2, default=str),
                encoding="utf-8",
            )


def _sync_events_to_disk(job_id: str) -> None:
    """Append all pending events to the events log file."""
    with _lock:
        events = _job_events.get(job_id, [])
        if events:
            path = _events_path(job_id)
            existing_count = 0
            if path.exists():
                existing_count = sum(1 for _ in path.read_text().splitlines() if _)
            new_events = events[existing_count:]
            if new_events:
                with path.open("a", encoding="utf-8") as f:
                    for event in new_events:
                        f.write(json.dumps({"event": event, "timestamp": time.time()}) + "\n")


def _sync_chat_to_disk(project_name: str) -> None:
    """Write project chat history to disk."""
    with _lock:
        chat = _project_chats.get(project_name)
        if chat:
            _chat_path(project_name).write_text(
                json.dumps(chat, indent=2, default=str),
                encoding="utf-8",
            )


# ─── Public API ───────────────────────────────────────────────────────────────


def get_job(job_id: str) -> Optional[dict]:
    """Get a job by ID. Tries memory first, then disk."""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return job
    # Try loading from disk
    path = _job_path(job_id)
    if path.exists():
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            with _lock:
                _jobs[job_id] = job
            return job
        except Exception:
            pass
    return None


def set_job(job_id: str, job: dict) -> None:
    """Set a job and persist to disk."""
    with _lock:
        _jobs[job_id] = job
    _sync_job_to_disk(job_id)


def update_job(job_id: str, updates: dict) -> None:
    """Update specific fields of a job and persist."""
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.update(updates)
        else:
            _jobs[job_id] = updates
    _sync_job_to_disk(job_id)


def delete_job(job_id: str) -> None:
    """Remove a job from memory and disk."""
    with _lock:
        _jobs.pop(job_id, None)
        _job_events.pop(job_id, None)
        _ws_connections.pop(job_id, None)
    try:
        _job_path(job_id).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        _events_path(job_id).unlink(missing_ok=True)
    except Exception:
        pass


def clear_session(job_id: str) -> None:
    """Fully clear a session from memory and disk."""
    delete_job(job_id)


def get_events(job_id: str) -> list[str]:
    """Get all events for a job. Loads from disk if needed."""
    with _lock:
        events = _job_events.get(job_id)
        if events is not None:
            return list(events)
    path = _events_path(job_id)
    if path.exists():
        try:
            loaded = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        loaded.append(data["event"])
                    except Exception:
                        loaded.append(line)
            with _lock:
                _job_events[job_id] = list(loaded)
            return loaded
        except Exception:
            pass
    return []


def add_event(job_id: str, event: str) -> None:
    """Add an event to a job's event list and persist."""
    with _lock:
        _job_events.setdefault(job_id, []).append(event)
    _sync_events_to_disk(job_id)


def get_chat(project_name: str) -> list[dict]:
    """Get chat history for a project."""
    with _lock:
        chat = _project_chats.get(project_name)
        if chat is not None:
            return list(chat)
    path = _chat_path(project_name)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            with _lock:
                _project_chats[project_name] = list(loaded)
            return loaded
        except Exception:
            pass
    return []


def add_chat_event(project_name: str, event: dict) -> None:
    """Add a chat event for a project and persist."""
    with _lock:
        _project_chats.setdefault(project_name, []).append(event)
    _sync_chat_to_disk(project_name)


def register_ws(job_id: str, ws) -> None:
    """Register a WebSocket connection (in-memory only)."""
    with _lock:
        _ws_connections.setdefault(job_id, []).append(ws)


def unregister_ws(job_id: str, ws) -> None:
    """Remove a WebSocket connection."""
    with _lock:
        conns = _ws_connections.get(job_id, [])
        if ws in conns:
            conns.remove(ws)


def get_ws_connections(job_id: str) -> list:
    """Get all WebSocket connections for a job."""
    with _lock:
        return list(_ws_connections.get(job_id, []))


def list_active_jobs() -> list[dict]:
    """List all active jobs (recently updated)."""
    jobs = []
    with _lock:
        for job_id, job in _jobs.items():
            jobs.append({
                "job_id": job_id,
                "status": job.get("status", "unknown"),
                "project_name": job.get("project_name"),
                "started_at": job.get("started_at"),
                "mode": job.get("mode", "build"),
            })
    if not jobs:
        for f in sorted(SESSION_DIR.glob("job_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
                job_id = f.stem.replace("job_", "")
                jobs.append({
                    "job_id": job_id,
                    "status": job.get("status", "unknown"),
                    "project_name": job.get("project_name"),
                    "started_at": job.get("started_at"),
                    "mode": job.get("mode", "build"),
                })
            except Exception:
                pass
    return jobs


# ─── Initialize: load existing sessions on startup ───────────────────────────


def _load_existing_sessions() -> None:
    """On startup, load any existing sessions from disk into memory."""
    for f in SESSION_DIR.glob("job_*.json"):
        try:
            job_id = f.stem.replace("job_", "")
            job = json.loads(f.read_text(encoding="utf-8"))
            with _lock:
                if job_id not in _jobs:
                    _jobs[job_id] = job
            get_events(job_id)
        except Exception:
            pass


# Call on module import
_load_existing_sessions()
