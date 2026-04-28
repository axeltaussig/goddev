"""
GodDev 3.0 — FastAPI Production Server
========================================

Endpoints:
  GET  /               — Chat UI (static/index.html)
  POST /build          — Submit an app idea, get a job_id back
  GET  /build/{job_id} — Poll job status + progress events + cost summary
  GET  /projects       — List all built projects
  GET  /projects/{name}/files   — List project files
  GET  /projects/{name}/report  — GODDEV_REPORT.md
  GET  /memory/stats   — Self-improvement version + run count
  GET  /health         — Health check
  WS   /ws/{job_id}   — WebSocket: real-time agent stream + cost updates
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GodDev 3.0",
    description="Hierarchical Autonomous AI Development Team",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
_STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ─── In-memory job store ──────────────────────────────────────────────────────

_jobs: dict[str, dict[str, Any]] = {}
_job_events: dict[str, list[str]] = {}  # job_id → list of event strings
_ws_connections: dict[str, list[WebSocket]] = {}  # job_id → list of websockets
_project_chats: dict[str, list[dict]] = {}  # project_name → list of chat messages


# ─── Models ───────────────────────────────────────────────────────────────────


class BuildRequest(BaseModel):
    request: str
    max_critic_iterations: int = 2
    max_iterations: int = 3


class BuildResponse(BaseModel):
    job_id: str
    status: str
    message: str
    started_at: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | completed | failed
    project_name: Optional[str] = None
    project_dir: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
    progress_events: list[str] = []
    error: Optional[str] = None
    result_summary: Optional[str] = None
    cost_summary: Optional[dict] = None


# ─── Background build runner ──────────────────────────────────────────────────


def _classify_event(event: str) -> str:
    """Classify event string into a type for the UI."""
    if any(x in event for x in ["✅", "❌", "⚠️", "Worker]"]):
        return "worker"
    if "Squad Lead" in event or "Squad Plan" in event:
        return "squad"
    if "Blueprint" in event or "🏗️" in event or "CTO" in event:
        return "blueprint"
    if "Critic" in event or "⚖️" in event:
        return "critic"
    if "Integrat" in event or "🔗" in event:
        return "integrator"
    if "✅ Build complete" in event or "Self-improvement complete" in event:
        return "complete"
    if "❌ Build failed" in event or "failed" in event.lower():
        return "error"
    return "info"


async def _broadcast_event(job_id: str, event: str, **extra) -> None:
    """Send a typed event to all WebSocket connections for this job."""
    _job_events.setdefault(job_id, []).append(event)
    payload = {"type": _classify_event(event), "content": event, **extra}
    raw = json.dumps(payload)
    dead: list = []
    for ws in list(_ws_connections.get(job_id, [])):
        try:
            await ws.send_text(raw)
        except Exception:
            # Connection already closed — mark for removal, never crash the build
            dead.append(ws)
    for ws in dead:
        conns = _ws_connections.get(job_id, [])
        if ws in conns:
            conns.remove(ws)


async def _broadcast_cost(job_id: str, cost_summary: dict) -> None:
    """Send a cost update to all WebSocket connections for this job."""
    payload = json.dumps({"type": "cost", "data": cost_summary})
    dead: list = []
    for ws in list(_ws_connections.get(job_id, [])):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        conns = _ws_connections.get(job_id, [])
        if ws in conns:
            conns.remove(ws)


def _task_error_callback(task) -> None:
    """Log asyncio task exceptions that would otherwise be swallowed silently."""
    if not task.cancelled() and task.exception():
        import traceback
        print(f"[GodDev] BACKGROUND TASK FAILED: {task.exception()}")
        traceback.print_exception(type(task.exception()), task.exception(),
                                  task.exception().__traceback__)


async def _run_build(job_id: str, request: str, max_critic_iterations: int, max_iterations: int) -> None:
    """Execute the GodDev graph and stream events with cost tracking."""
    print(f"[GodDev] _run_build starting for job {job_id}")
    _jobs[job_id]["status"] = "running"

    try:
        print(f"[GodDev] importing graph...")
        from ..graph import graph
        from ..cost_tracker import get_tracker, remove_tracker
        from langchain_core.messages import HumanMessage
        print(f"[GodDev] imports OK, streaming graph...")

        tracker = get_tracker(job_id)

        initial_state = {
            "messages": [HumanMessage(content=request)],
            "user_request": request,
            "squad_plans": [],
            "worker_outputs": [],
            "critic_verdicts": [],
            "critic_approved": False,
            "critic_feedback": None,
            "critic_iteration": 0,
            "max_critic_iterations": max_critic_iterations,
            "integration_passed": False,
            "iteration": 0,
            "max_iterations": max_iterations,
            "metadata": {"job_id": job_id},
            "project_name": "",
            "project_dir": "",
            "project_blueprint": None,
            "current_squad_task": None,
            "current_file_task": None,
            "current_critic_type": None,
            "integration_result": None,
            "reflection_notes": None,
        }

        # ── Load self-improvement learnings from runtime_config.json ──────────────
        # The self-improver writes config updates; we inject them into metadata
        # so the CTO and workers can read them at runtime.
        try:
            import json as _json
            from pathlib import Path as _Path
            rc_path = _Path(os.getenv("MEMORY_DIR", "/opt/goddev/memory")) / "runtime_config.json"
            if rc_path.exists():
                runtime_cfg = _json.loads(rc_path.read_text())
                initial_state["metadata"]["runtime_config"] = runtime_cfg.get("agent_config", {})
                print(f"[GodDev] Loaded runtime_config v{runtime_cfg.get('version', '?')}")
        except Exception as rc_exc:
            print(f"[GodDev] runtime_config not loaded: {rc_exc}")

        # ── Graph execution config ───────────────────────────────────────────────
        # max_concurrency: LangGraph 2025 best practice — limits parallel branches
        # to prevent thundering herd at the scheduler level (token buckets handle
        # the per-provider rate limiting, this handles scheduler fan-out width).
        graph_config = {"max_concurrency": 6}

        await _broadcast_event(job_id, "🚀 GodDev 3.0 starting build...")

        # Run the graph with streaming
        final_state = None
        prev_msg_count = 0
        async for chunk in graph.astream(initial_state, config=graph_config, stream_mode="values"):
            final_state = chunk

            msgs = chunk.get("messages", [])
            if len(msgs) > prev_msg_count:
                for msg in msgs[prev_msg_count:]:
                    content = getattr(msg, "content", "")
                    if content:
                        # Send full content — UI handles display length
                        await _broadcast_event(job_id, str(content))
                prev_msg_count = len(msgs)

            if chunk.get("project_name"):
                _jobs[job_id]["project_name"] = chunk["project_name"]
            if chunk.get("project_dir"):
                _jobs[job_id]["project_dir"] = chunk["project_dir"]

            # Push cost update every chunk
            cost_summary = tracker.summary()
            await _broadcast_cost(job_id, cost_summary)

        # Final cost summary
        final_cost = tracker.summary()
        cost_md = tracker.to_markdown_table()
        remove_tracker(job_id)

        completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _jobs[job_id].update({
            "status": "completed",
            "completed_at": completed_at,
            "result_summary": (final_state or {}).get("final_output", "Build complete."),
            "cost_summary": final_cost,
        })

        project_name = _jobs[job_id].get("project_name", "unknown")
        project_dir  = _jobs[job_id].get("project_dir", "")
        play_url     = f"/play/{project_name}/"
        complete_msg = (
            f"\u2705 Build complete! Project: **{project_name}**\n"
            f"Play at: [{play_url}]({play_url})\n"
            f"Total cost: ${final_cost['total_cost_usd']:.4f} | "
            f"Tokens: {final_cost['total_tokens']:,}"
        )
        await _broadcast_event(
            job_id,
            complete_msg,
            play_url=play_url,
            project_name=project_name,
        )
        # Store the build event in the per-project chat history
        _project_chats.setdefault(project_name, []).append({
            "type": "complete",
            "content": complete_msg,
            "play_url": play_url,
            "cost": final_cost,
        })

    except Exception as exc:
        _jobs[job_id].update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        await _broadcast_event(job_id, f"\u274c Build failed: {exc}")


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "GodDev 3.0", "version": "3.0.0"}


@app.get("/")
async def serve_ui():
    """Serve the GodDev chat UI."""
    ui_path = Path(__file__).parent.parent.parent / "static" / "index.html"
    if ui_path.exists():
        return FileResponse(str(ui_path), media_type="text/html")
    return JSONResponse({"service": "GodDev 3.0", "docs": "/docs", "build": "POST /build"})


@app.post("/self-improve")
async def trigger_self_improvement():
    """
    Trigger GodDev to analyse its own source code and improve itself.
    Uses the full agent team (Meta CTO → Workers → Critics → Self-Deployer).
    """
    job_id = "self-" + str(uuid.uuid4())[:8]
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "request": "GodDev self-improvement run",
        "project_name": None,
        "project_dir": "/opt/goddev/staging",
        "started_at": started_at,
        "completed_at": None,
        "error": None,
        "result_summary": None,
        "mode": "self_improvement",
    }
    _job_events[job_id] = []

    task = asyncio.create_task(_run_self_build(job_id))
    task.add_done_callback(_task_error_callback)

    return BuildResponse(
        job_id=job_id,
        status="queued",
        message=f"Self-improvement started. GodDev is analysing its own code. Watch at /ws/{job_id}",
        started_at=started_at,
    )


async def _run_self_build(job_id: str) -> None:
    """Run the self-build graph: Meta CTO analyses + improves GodDev's source."""
    _jobs[job_id]["status"] = "running"
    try:
        from ..self_build_graph import self_build_graph
        from ..cost_tracker import get_tracker, remove_tracker
        from langchain_core.messages import HumanMessage

        tracker = get_tracker(job_id)

        initial_state = {
            "messages": [HumanMessage(content="Analyse GodDev 3.0 and improve its own source code based on performance data.")],
            "user_request": "GodDev self-improvement",
            "squad_plans": [],
            "worker_outputs": [],
            "critic_verdicts": [],
            "critic_approved": False,
            "critic_feedback": None,
            "critic_iteration": 0,
            "max_critic_iterations": 2,
            "integration_passed": False,
            "iteration": 0,
            "max_iterations": 2,
            "metadata": {"job_id": job_id, "mode": "self_improvement"},
        }

        await _broadcast_event(job_id, "🔬 GodDev Self-Improvement starting... reading own source code")

        graph_config = {"max_concurrency": 6}

        final_state = None
        prev_msg_count = 0
        async for chunk in self_build_graph.astream(initial_state, config=graph_config, stream_mode="values"):
            final_state = chunk
            msgs = chunk.get("messages", [])
            if len(msgs) > prev_msg_count:
                for msg in msgs[prev_msg_count:]:
                    content = getattr(msg, "content", "")
                    if content:
                        await _broadcast_event(job_id, str(content))
                prev_msg_count = len(msgs)
            cost_summary = tracker.summary()
            await _broadcast_cost(job_id, cost_summary)

        final_cost = tracker.summary()
        remove_tracker(job_id)
        completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _jobs[job_id].update({
            "status": "completed",
            "completed_at": completed_at,
            "project_name": (final_state or {}).get("project_name", "self-improve"),
            "result_summary": (final_state or {}).get("final_output", "Self-improvement complete."),
            "cost_summary": final_cost,
        })
        await _broadcast_event(job_id, f"✅ Self-improvement complete! Cost: ${final_cost['total_cost_usd']:.4f}")

    except Exception as exc:
        _jobs[job_id].update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        await _broadcast_event(job_id, f"❌ Self-improvement failed: {exc}")


@app.get("/memory/stats")
async def memory_stats():
    """Return self-improvement stats from runtime_config.json."""
    try:
        from ..config import load_runtime_config
        config = load_runtime_config()
        return {
            "config_version": config.get("version", 1),
            "last_updated": config.get("last_updated", "never"),
            "total_improvements": config.get("total_improvements", 0),
            "total_runs": config["performance_history"].get("total_runs", 0),
            "critic_score_threshold": config["agent_config"]["critic_score_threshold"],
            "preferred_agents": config["agent_config"].get("preferred_agents", {}),
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/models")
async def model_status():
    """Show LiteLLM router: all configured deployments, roles, and fallback chains."""
    try:
        from ..llm_router import get_model_status
        deployments = get_model_status()
        roles: dict = {}
        for d in deployments:
            r = d["role"]
            roles.setdefault(r, []).append({
                "model":    d["model"],
                "order":    d["order"],
                "parallel": d["parallel"],
                "provider": d.get("provider", "unknown"),
                "note":     d["description"],
            })
        return {"status": "ok", "roles": roles, "total_deployments": len(deployments)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.get("/health/models")
async def health_models():
    """
    Live health of all 4 providers (Claude, OpenAI, Gemini, DeepSeek).
    Status: ok | configured | error | rate_limited | unknown
    Updated in real-time as builds run — no extra API calls, zero cost.
    """
    try:
        from ..llm_router import get_provider_health
        health = get_provider_health()
        # Also report current model versions from env
        import os
        versions = {
            "claude":   os.getenv("ANTHROPIC_MODEL",  "claude-opus-4-7"),
            "openai":   os.getenv("OPENAI_MODEL",     "gpt-4.1"),
            "gemini":   os.getenv("GEMINI_MODEL",     "gemini-2.5-flash"),
            "deepseek": os.getenv("DEEPSEEK_MODEL",   "deepseek-chat"),
        }
        for p, h in health.items():
            h["model_version"] = versions.get(p, "unknown")
        return {"status": "ok", "providers": health}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.post("/build", response_model=BuildResponse)
async def submit_build(body: BuildRequest):
    """Submit an app build request. Returns a job_id for polling/streaming."""
    job_id = str(uuid.uuid4())[:12]
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "request": body.request,
        "project_name": None,
        "project_dir": None,
        "started_at": started_at,
        "completed_at": None,
        "error": None,
        "result_summary": None,
    }
    _job_events[job_id] = []

    # Fire and forget — run in background (error callback surfaces any failures)
    task = asyncio.create_task(
        _run_build(job_id, body.request, body.max_critic_iterations, body.max_iterations)
    )
    task.add_done_callback(_task_error_callback)

    return BuildResponse(
        job_id=job_id,
        status="queued",
        message=f"Build queued. Connect to /ws/{job_id} for live output.",
        started_at=started_at,
    )


@app.get("/build/{job_id}", response_model=JobStatus)
async def get_build_status(job_id: str):
    """Poll the status of a build job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        project_name=job.get("project_name"),
        project_dir=job.get("project_dir"),
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
        progress_events=_job_events.get(job_id, [])[-50:],
        error=job.get("error"),
        result_summary=job.get("result_summary"),
        cost_summary=job.get("cost_summary"),
    )


@app.get("/projects")
async def list_projects():
    """List all projects built by GodDev."""
    projects_dir = Path(os.getenv("PROJECTS_DIR", "/opt/goddev/projects"))
    if not projects_dir.exists():
        return {"projects": []}

    projects = []
    for p in sorted(projects_dir.iterdir()):
        if p.is_dir():
            report = p / "GODDEV_REPORT.md"
            file_count = len(list(p.rglob("*")))
            projects.append({
                "name": p.name,
                "path": str(p),
                "has_report": report.exists(),
                "file_count": file_count,
                "created_at": datetime.fromtimestamp(
                    p.stat().st_ctime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            })

    return {"projects": projects}


@app.get("/projects/{project_name}/files")
async def get_project_files(project_name: str):
    """List all files in a project."""
    projects_dir = Path(os.getenv("PROJECTS_DIR", "/opt/goddev/projects"))
    project_dir = projects_dir / project_name

    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project {project_name} not found")

    files = []
    for f in sorted(project_dir.rglob("*")):
        if f.is_file():
            files.append({
                "path": str(f.relative_to(project_dir)),
                "size_bytes": f.stat().st_size,
            })

    return {"project": project_name, "files": files}


@app.get("/projects/{project_name}/report")
async def get_project_report(project_name: str):
    """Get the GODDEV_REPORT.md for a project."""
    projects_dir = Path(os.getenv("PROJECTS_DIR", "/opt/goddev/projects"))
    report_path = projects_dir / project_name / "GODDEV_REPORT.md"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No report found for this project")

    return {"project": project_name, "report": report_path.read_text(encoding="utf-8")}


@app.delete("/projects/{project_name}")
async def delete_project(project_name: str):
    """
    Delete a project: removes files from disk and clears in-memory session state.
    The corresponding chat session in the UI is also invalidated.
    """
    import shutil
    projects_dir = Path(os.getenv("PROJECTS_DIR", "/opt/goddev/projects"))
    project_dir  = projects_dir / project_name

    deleted_files = False
    if project_dir.exists():
        shutil.rmtree(str(project_dir))
        deleted_files = True

    # Clear in-memory job entries for this project
    dead_jobs = [
        jid for jid, j in _jobs.items()
        if j.get("project_name") == project_name
    ]
    for jid in dead_jobs:
        _jobs.pop(jid, None)
        _job_events.pop(jid, None)
        _ws_connections.pop(jid, None)

    # Clear per-project chat history
    _project_chats.pop(project_name, None)

    return {
        "deleted": project_name,
        "files_removed": deleted_files,
        "jobs_cleared": len(dead_jobs),
    }


@app.get("/projects/{project_name}/chat")
async def get_project_chat(project_name: str):
    """
    Return stored chat events for a project session.
    Used by the UI to restore session state when switching tabs.
    """
    events = _project_chats.get(project_name, [])
    return {"project": project_name, "events": events}


@app.get("/play/{project_name}")
@app.get("/play/{project_name}/")
async def play_project_index(project_name: str):
    """Serve the index.html for a built project (browser-playable games)."""
    projects_dir = Path(os.getenv("PROJECTS_DIR", "/opt/goddev/projects"))
    index = projects_dir / project_name / "index.html"
    if index.exists():
        content = index.read_text(encoding="utf-8")
        # Rewrite absolute paths to go through /play/{project_name}/
        content = content.replace('src="/', f'src="/play/{project_name}/')
        content = content.replace('href="/', f'href="/play/{project_name}/')
        return HTMLResponse(content)
    raise HTTPException(404, detail=f"No index.html found for project '{project_name}'")


@app.get("/play/{project_name}/{file_path:path}")
async def play_project_file(project_name: str, file_path: str):
    """Serve static files for a built project (JS, CSS, images, etc.)."""
    projects_dir = Path(os.getenv("PROJECTS_DIR", "/opt/goddev/projects"))
    target = (projects_dir / project_name / file_path).resolve()
    # Security: ensure we stay within the project directory
    if not str(target).startswith(str((projects_dir / project_name).resolve())):
        raise HTTPException(403, detail="Path traversal not allowed")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, detail=f"File not found: {file_path}")
    mime, _ = mimetypes.guess_type(str(target))
    return FileResponse(str(target), media_type=mime or "application/octet-stream")


# ─── WebSocket endpoint ───────────────────────────────────────────────────────


@app.websocket("/ws/{job_id}")
async def websocket_stream(websocket: WebSocket, job_id: str):
    """Stream real-time agent output for a build job. Handles reconnection and closed-WS gracefully."""
    await websocket.accept()

    if job_id not in _jobs:
        try:
            await websocket.send_text(json.dumps({"error": f"Job {job_id} not found"}))
            await websocket.close()
        except Exception:
            pass
        return

    # Register this connection
    _ws_connections.setdefault(job_id, []).append(websocket)

    async def _safe_send(payload: str) -> bool:
        """Send to WS; return False and deregister if connection is gone."""
        try:
            await websocket.send_text(payload)
            return True
        except Exception:
            return False

    # Catch-up: replay all events so far (supports page refresh mid-build)
    past_events = _job_events.get(job_id, [])
    for event in past_events:
        ok = await _safe_send(json.dumps({"type": _classify_event(event), "content": event}))
        if not ok:
            _ws_connections.get(job_id, []).remove(websocket) if websocket in _ws_connections.get(job_id, []) else None
            return

    # Poll for completion, keeping connection alive
    try:
        while True:
            status = _jobs.get(job_id, {}).get("status", "unknown")
            if status in ("completed", "failed"):
                pname = _jobs[job_id].get("project_name")
                final = json.dumps({
                    "type": "status",
                    "status": status,
                    "project_name": pname,
                    "project_dir": _jobs[job_id].get("project_dir"),
                    "play_url": f"/play/{pname}/" if pname else None,
                    "error": _jobs[job_id].get("error"),
                    "cost_update": _jobs[job_id].get("cost_summary"),
                })
                await _safe_send(final)
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        conns = _ws_connections.get(job_id, [])
        if websocket in conns:
            conns.remove(websocket)


# ─── Entry point ─────────────────────────────────────────────────────────────


def start():
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    start()
