# How to Run & Test the Road Quality Map System

A complete guide for launching, operating, and testing the **Tiered Crowdsourced Road-Quality Map (Proof-of-Concept)**.

---

## 1. Quick Start

### Python Environment Requirements
> [!NOTE]
> This project requires **Python 3.13** (or Python 3.11/3.12) because computer vision and machine learning dependencies (PyTorch `torch`, `torchvision`, and `ultralytics` YOLOv8) provide precompiled wheels up to Python 3.13. If your default `py` launcher resolves to Python 3.15, make sure to specify Python 3.13 when setting up or activating the virtual environment.

From the project root (`d:\Projects\road_condition`), the virtual environment is already configured at `.venv` with all required dependencies.

If you ever need to recreate it from scratch:
```powershell
py -3.13 -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Start the Backend Server & Interactive Map
You can launch the project with any of the following quick options:

**Option A (One-click batch / PowerShell script):**
```powershell
.\run.bat
# Or in PowerShell:
.\run.ps1
```

**Option B (From activated virtual environment):**
```powershell
.\.venv\Scripts\python.exe backend/api.py
```

*Or using uvicorn directly with the venv:*
```powershell
.\.venv\Scripts\uvicorn.exe backend.api:app --host 127.0.0.1 --port 8000 --reload
```

Once started, open your web browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 2. Testing the Automated Pipeline

To run the automated verification test suite verifying all acceptance criteria plus the new video upload feature:
```powershell
python scripts/test_pipeline.py
```
This runs 7 automated end-to-end checks:
1. **Tier 0 Standalone Cold Start**: Populates the full ORR corridor (205 segments) with baseline Good-condition scores.
2. **Tier 1 Isolated Segment Updates**: Modifies only the road edges traversed by dashcam footage.
3. **Tier 2 Multi-Sensor Correlation & False-Positive Suppression**: Confirms high trust on matching vibration spikes and suppresses false positives (shadows, manholes).
4. **Segment Inspector Telemetry API**: Verifies granular tier-by-tier evidence breakdown and observation history.
5. **Simulated Time Decay**: Validates exponential confidence decay over simulated time skips.
6. **Zero Hardware Dependency**: Validates that all pipelines execute locally using public data and video/IMU proxies.
7. **New Feature (Video Upload)**: Verifies automated frame scanning, YOLOv8 defect detection, and segment re-scoring for user-uploaded videos.

---

## 3. Using the New Feature: Tier 1 Video Upload

### From the Web UI:
1. In the left GIS panel under **TIER 1 (Dashcam Video)**, click **UPLOAD ROAD VIDEO**.
2. A technical modal dialog will open:
   - Drag and drop any road dashcam video (`.mp4`, `.mov`, `.avi`, `.webm`) or click **BROWSE FILES**.
   - Choose target ORR corridor stretch (Auto-Route, West Stretch, or East Stretch).
   - Select sampling density (Standard 3 FPS, Dense 5 FPS, Ultra 10 FPS).
   - Select frame limit (30 frames for a rapid demo, 60 frames, 120 frames, or All).
3. Click **ANALYZE & INGEST VIDEO**.
4. A progress bar will track video upload, YOLOv8 frame analysis, GPS interpolation, and database updates.
5. Once complete, a summary card displays total potholes detected, observations logged, and touched segments.
6. Click **VIEW ON MAP & INSPECT** to smoothly fly to the updated road segments and inspect the YOLO detection snapshots in the right-hand Telemetry Inspector drawer.

### From the REST API (curl / Python):
```bash
curl -X POST "http://127.0.0.1:8000/ingest/tier1/upload" \
  -F "video_file=@data/raw/tier1/sample_videos/clip1.mp4" \
  -F "corridor=west" \
  -F "sample_rate=3.0" \
  -F "max_frames=30" \
  -F "conf_threshold=0.35"
```

---

## 4. UI Features & Exploration Walkthrough

1. **Discrete Condition Overlays**:
   - Segments are rendered as distinct flat-color lines: **Good** (Green: 80–100), **Fair** (Yellow: 60–79), **Poor** (Orange: 40–59), **Bad** (Red: 20–39), **Severe** (Purple: 0–19).
2. **Segment Inspector Drawer**:
   - Click any road segment on the map to open the drawer on the right.
   - Inspect overall score, confidence percentage, dominant defect, road length, and last ingested timestamp.
   - Examine the **Multi-Tier Evidence Matrix** (breakdown across Tier 0, Tier 1, and Tier 2).
   - View the **Observation Telemetry Log** with YOLO detection crops and bounding boxes.
   - Click **PIN LOC 📍** on any observation card to drop a marker pin on its exact GPS coordinates on the map.
   - Click any detection image thumbnail to view an enlarged snapshot in the Lightbox viewer.
3. **Independent Ingestion Triggers**:
   - **RUN TIER 0 INGEST**: Generates baseline Street View observations across all 205 segments.
   - **RUN TIER 1 INGEST**: Ingests dashcam footage and updates only the traversed segments.
   - **RUN TIER 2 INGEST**: Correlates visual detections with real 100Hz IMU vibration data (or optical flow jitter proxy) to boost confidence or suppress false positives.
4. **View Filter Radios**:
   - Toggle between **FUSED STATE**, **TIER 0 ONLY (BASELINE)**, **TIER 0 + TIER 1**, and **TIER 0 + TIER 1 + TIER 2** to inspect each tier's independent contribution.
5. **Simulated Clock & Staleness Decay**:
   - Click **SIMULATE +14 DAYS DECAY** or **+30 DAYS** to observe confidence decaying exponentially on stale segments.
   - Click **RESET BASELINE** to restore a clean road graph.

---

## 5. Re-Seeding Demo Telemetry

To reset the database and re-run all baseline tiers (Tier 0 baseline, Tier 1 dashcam clips, and Tier 2 vibration correlation):
```powershell
python scripts/seed_demo_data.py
```