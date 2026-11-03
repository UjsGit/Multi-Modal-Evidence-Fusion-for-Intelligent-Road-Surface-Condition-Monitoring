"""
FastAPI REST API Server for Tiered Road Quality Map POC.
Provides segment GeoJSON, inspector history, tier ingestion triggers,
and demo time-decay simulation. Serves the Leaflet frontend.
"""

import os
import sys
from pathlib import Path
import shutil
import re
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

sys.path.insert(0, str(ROOT_DIR))

from backend.db import (
    init_db,
    get_all_segments_geojson,
    get_segment_history,
    advance_simulated_time,
    reset_simulated_time,
    get_simulated_now,
    recompute_all_edges
)
from backend.ingest_tier0 import run_tier0_ingest
from backend.ingest_tier1 import run_tier1_ingest, ingest_uploaded_video
from backend.ingest_tier2 import run_tier2_ingest

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Tiered Crowdsourced Road Quality Map API",
    description="POC API for tiered ingestion (Street View -> Dashcam -> Multi-Sensor Vibration) and OSM road quality scoring.",
    version="1.0.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
def on_startup():
    init_db()

# Request schemas
class IngestTier0Request(BaseModel):
    api_key: Optional[str] = None
    sample_step_m: Optional[float] = 40.0

class IngestTier1Request(BaseModel):
    video_source: Optional[str] = None
    target_edge_ids: Optional[List[str]] = None
    num_frames: Optional[int] = None
    sample_rate: Optional[float] = 10.0
    frame_step: Optional[int] = None
    conf_threshold: Optional[float] = 0.35

class IngestTier2Request(BaseModel):
    video_source: Optional[str] = None
    imu_source: Optional[str] = None
    target_edge_ids: Optional[List[str]] = None
    vibration_mode: Optional[str] = "real_imu"
    num_frames: Optional[int] = 15

class GSVKeyConfigRequest(BaseModel):
    api_key: str

class SimulateDecayRequest(BaseModel):
    days: float = 14.0

# ----------------- Core API Endpoints ----------------- #

@app.get("/segments")
def get_segments(tier_filter: Optional[str] = Query(None, description="Optional tier filter")):
    """Returns GeoJSON FeatureCollection of all road segments with scores, severity, confidence, and tier breakdowns."""
    return get_all_segments_geojson(tier_filter=tier_filter)

@app.get("/segments/{edge_id}")
def get_segment_details(edge_id: str):
    """Returns full observation history and metadata for a single edge (used by the inspector panel)."""
    details = get_segment_history(edge_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Segment edge_id '{edge_id}' not found.")
    return details

@app.post("/ingest/tier0")
def trigger_ingest_tier0(req: Optional[IngestTier0Request] = None):
    """Triggers/reruns Tier 0 ingestion (Google Street View baseline)."""
    api_key = req.api_key if req else None
    step = req.sample_step_m if req and req.sample_step_m else 40.0
    res = run_tier0_ingest(api_key=api_key, sample_step_m=step)
    return res

@app.post("/ingest/tier1")
def trigger_ingest_tier1(req: Optional[IngestTier1Request] = None):
    """Triggers/reruns Tier 1 ingestion (YouTube dashcam / video without IMU)."""
    vid = req.video_source if req else None
    edges = req.target_edge_ids if req else None
    frames = req.num_frames if (req and req.num_frames) else None
    sample_rate = req.sample_rate if (req and req.sample_rate is not None) else 10.0
    frame_step = req.frame_step if (req and req.frame_step is not None) else None
    conf_threshold = req.conf_threshold if (req and req.conf_threshold is not None) else 0.35
    res = run_tier1_ingest(
        video_source=vid,
        target_edge_ids=edges,
        num_frames=frames,
        sample_rate=sample_rate,
        frame_step=frame_step,
        conf_threshold=conf_threshold
    )
    return res

@app.post("/ingest/tier1/upload")
async def upload_and_ingest_tier1(
    video_file: UploadFile = File(...),
    corridor: Optional[str] = Form("auto"),
    sample_rate: Optional[float] = Form(3.0),
    max_frames: Optional[int] = Form(100),
    conf_threshold: Optional[float] = Form(0.35)
):
    """New Feature: Upload a road video file (.mp4, .mov, .avi) for automated
    YOLOv8 defect detection, route geotag interpolation, and corridor quality re-scoring."""
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    upload_dir = DATA_DIR / "raw" / "tier1" / "uploaded_videos"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    clean_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", video_file.filename)
    dest_path = upload_dir / clean_filename

    # Save uploaded file
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    finally:
        video_file.file.close()

    print(f"[Upload Tier 1] Saved uploaded video to: {dest_path} ({dest_path.stat().st_size} bytes)")

    # Run ingestion on uploaded video
    try:
        result = ingest_uploaded_video(
            video_path=str(dest_path),
            target_corridor=corridor,
            sample_rate=sample_rate,
            max_frames=max_frames,
            conf_threshold=conf_threshold
        )
        return result
    except Exception as e:
        print(f"[Upload Tier 1] Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion analysis failed: {str(e)}")

@app.post("/ingest/tier2")
def trigger_ingest_tier2(req: Optional[IngestTier2Request] = None):
    """Triggers/reruns Tier 2 ingestion (Dashcam + synchronized vibration)."""
    vid = req.video_source if req else None
    imu = req.imu_source if req else None
    edges = req.target_edge_ids if req else None
    vmode = req.vibration_mode if req and req.vibration_mode else "real_imu"
    frames = req.num_frames if req and req.num_frames else 15
    res = run_tier2_ingest(
        video_source=vid,
        imu_source=imu,
        target_edge_ids=edges,
        vibration_mode=vmode,
        num_frames=frames
    )
    return res

@app.post("/ingest/tier2/upload_imu")
async def upload_tier2_imu(imu_file: UploadFile = File(...)):
    """Uploads a real accelerometer CSV (timestamp, acc_z) for Tier 2 vibration cross-verification."""
    if not imu_file.filename:
        raise HTTPException(status_code=400, detail="No IMU CSV file provided.")

    imu_dir = DATA_DIR / "raw" / "tier2" / "sample_imu"
    imu_dir.mkdir(parents=True, exist_ok=True)

    clean_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", imu_file.filename)
    dest_path = imu_dir / clean_filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(imu_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save IMU file: {str(e)}")
    finally:
        imu_file.file.close()

    from backend.ingest_common import VibrationAnalyzer
    records = VibrationAnalyzer.load_imu_csv(str(dest_path))

    return {
        "status": "success",
        "saved_path": str(dest_path),
        "filename": clean_filename,
        "sample_count": len(records),
        "message": f"Successfully uploaded IMU stream '{clean_filename}' with {len(records)} accelerometer records."
    }

@app.post("/config/gsv_key")
def configure_gsv_key(req: GSVKeyConfigRequest):
    """Configures Google Street View API Key in the environment for live API ingestion."""
    os.environ["GOOGLE_STREET_VIEW_API_KEY"] = req.api_key.strip()
    return {
        "status": "success",
        "has_key": bool(req.api_key.strip()),
        "message": "Google Street View API key updated successfully."
    }

# ----------------- Demo Controls Endpoints ----------------- #

@app.post("/demo/simulate_decay")
def simulate_decay(req: SimulateDecayRequest):
    """Advances the simulated clock by N days to visibly demonstrate confidence decay and staleness live."""
    new_time = advance_simulated_time(req.days)
    return {
        "status": "success",
        "simulated_now": new_time.isoformat(),
        "days_advanced": req.days,
        "message": f"Advanced simulation clock by {req.days} days. Edge weights and scores updated."
    }

@app.post("/demo/reset")
def reset_demo():
    """Resets database and reinitializes with clean road network."""
    init_db(force_reinit=True)
    reset_simulated_time()
    return {
        "status": "success",
        "message": "Reset database and simulated time to initial baseline."
    }

@app.get("/demo/status")
def get_demo_status():
    """Returns current simulated clock time and observation counts."""
    now = get_simulated_now()
    from backend.db import get_db_connection
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM edges")
    edge_cnt = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) as cnt FROM observations")
    obs_cnt = cur.fetchone()["cnt"]
    cur.execute("SELECT source_tier, COUNT(*) as cnt FROM observations GROUP BY source_tier")
    tier_counts = {f"tier{r['source_tier']}": r["cnt"] for r in cur.fetchall()}
    conn.close()

    return {
        "simulated_now": now.isoformat(),
        "total_edges": edge_cnt,
        "total_observations": obs_cnt,
        "observations_by_tier": tier_counts
    }

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Frontend Static Files & Thumbnails ----------------- #

app.mount("/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_index():
        return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    print("Starting Road Quality API server on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=True)
