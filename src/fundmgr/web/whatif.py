"""
What-If Lab routes — mounted at /whatif.

Generation runs in a daemon thread (an N-sample consensus call at high
reasoning effort takes minutes) and the page polls the job endpoint. One job
at a time: these runs are LLM-cost-bearing and the Pi doesn't need a queue.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from fundmgr.engine.whatif import (
    MAX_RUNS, MODEL_OPTIONS, generate_whatif, list_profiles, list_results,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)

router = APIRouter(prefix="/whatif")

# Single-slot job registry. Survives page reloads (module-level), not restarts —
# completed results are on disk either way via engine.whatif.
_job_lock = threading.Lock()
_job: dict | None = None  # {id, status: running|done|error, params, result, error, started}


MAX_CAPITAL_SEK = 1_000_000_000  # sanity ceiling on the amount being placed


class GenerateRequest(BaseModel):
    profile: str
    provider: str | None = None
    model_id: str | None = None
    n_runs: int = Field(default=1, ge=1, le=MAX_RUNS)
    include_macro: bool = True
    # None = use the profile's own capital
    capital_sek: float | None = Field(default=None, gt=0, le=MAX_CAPITAL_SEK)
    deploy_full: bool = False


def _run_job(job_id: str, req: GenerateRequest) -> None:
    global _job
    try:
        result = generate_whatif(
            config_name=req.profile,
            provider=req.provider,
            model_id=req.model_id,
            n_runs=req.n_runs,
            include_macro=req.include_macro,
            capital_sek=req.capital_sek,
            deploy_full=req.deploy_full,
        )
        with _job_lock:
            if _job and _job["id"] == job_id:
                _job.update(status="done", result=result)
    except Exception as e:
        with _job_lock:
            if _job and _job["id"] == job_id:
                _job.update(status="error", error=str(e))


@router.get("/", response_class=HTMLResponse)
def whatif_page(request: Request):
    with _job_lock:
        active_job_id = _job["id"] if _job and _job["status"] == "running" else None
    tmpl = _jinja_env.get_template("whatif.html")
    return HTMLResponse(tmpl.render(
        request=request,
        profiles=list_profiles(),
        model_options=MODEL_OPTIONS,
        max_runs=MAX_RUNS,
        results=list_results(limit=20),
        active_job_id=active_job_id,
        active_page="whatif",
    ))


@router.post("/api/generate")
def api_generate(req: GenerateRequest):
    global _job
    valid_profiles = {p["config"] for p in list_profiles()}
    if req.profile not in valid_profiles:
        raise HTTPException(status_code=400, detail=f"Unknown profile: {req.profile}")
    # Model override must be one of the offered pairs (or absent = profile default)
    if req.provider or req.model_id:
        valid_models = {(m["provider"], m["model_id"]) for m in MODEL_OPTIONS}
        if (req.provider, req.model_id) not in valid_models:
            raise HTTPException(status_code=400, detail="Unknown provider/model combination")

    with _job_lock:
        if _job and _job["status"] == "running":
            raise HTTPException(status_code=409, detail="A generation is already running")
        job_id = uuid.uuid4().hex[:12]
        _job = {
            "id": job_id,
            "status": "running",
            "params": req.model_dump(),
            "result": None,
            "error": None,
            "started": time.time(),
        }

    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@router.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    with _job_lock:
        if not _job or _job["id"] != job_id:
            raise HTTPException(status_code=404, detail="Unknown job")
        return {
            "id": _job["id"],
            "status": _job["status"],
            "params": _job["params"],
            "elapsed_s": round(time.time() - _job["started"], 1),
            "result": _job["result"],
            "error": _job["error"],
        }
