# Multi-Modal Evidence Fusion for Intelligent Road Surface Condition Monitoring

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688.svg)](https://fastapi.tiangolo.com/)
[![Leaflet.js](https://img.shields.io/badge/Leaflet-1.9.4-199900.svg)](https://leafletjs.com/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-blueviolet.svg)](https://github.com/ultralytics/ultralytics)
[![Acceptance Checks](https://img.shields.io/badge/Acceptance%20Tests-7%2F7%20Passed%20(100%25)-brightgreen.svg)](#9-automated-verification--test-suite)

An end-to-end automated platform that dynamically evaluates, scores, and visualizes road surface quality across urban road corridors using a **tiered evidence fusion model**.

Traditional road quality assessments rely on infrequent, capital-intensive survey vehicles equipped with specialized laser profilometers and LiDAR rigs. This project demonstrates a defensible, multi-tier alternative: combining ubiquitous low-cost public data (Google Street View static imagery, dashcam video clips, smartphone IMU accelerometer recordings, and optical flow video vibration proxies) into a single, high-confidence road condition index (0–100) mapped onto real OpenStreetMap (OSM) roadway segments.

---

## Table of Contents

1. [Corridor Geographic Scope & Road Network](#1-corridor-geographic-scope--road-network)
2. [Multi-Tier Ingestion Architecture](#2-multi-tier-ingestion-architecture)
3. [Road-Quality Index & Mathematical Fusion Model](#3-road-quality-index--mathematical-fusion-model)
4. [Hardware & Dataset Assets](#4-hardware--dataset-assets)
5. [System Architecture & Data Flow](#5-system-architecture--data-flow)
6. [Interactive GIS Frontend Walkthrough](#6-interactive-gis-frontend-walkthrough)
7. [REST API Documentation](#7-rest-api-documentation)
8. [Installation & Execution Guide](#8-installation--execution-guide)
9. [Automated Verification & Test Suite](#9-automated-verification--test-suite)
10. [Repository Structure](#10-repository-structure)

---

## 1. Corridor Geographic Scope & Road Network

The demo focuses on one of India's most critical and traffic-heavy urban corridors: the **Outer Ring Road (ORR), Bengaluru**, spanning the stretch between **Marathahalli**, **Kadubeesanahalli**, and **Bellandur / Silk Board**.

```
                           BENGALURU OUTER RING ROAD (ORR)
                           
  [Marathahalli] ============================================= [Silk Board]
      (North)       Kadubeesanahalli     Bellandur Junction       (South)
      
                     CORRIDOR BOUNDING BOX (WGS 84):
                     • Latitude : 12.9250° N  to  12.9450° N
                     • Longitude: 77.6600° E  to  77.7000° E
```

- **Road Graph Source:** 205 OpenStreetMap (OSM) ways extracted via the Overpass API (`data/demo_area_roads.geojson`).
- **Attributes Preserved:** Unique OSM edge IDs, road names, highway classifications (`trunk`, `primary`, `secondary`, `tertiary`, `service`), lane counts, surface types, and sub-meter polyline geometries.
- **Corridor Distance:** ~1.5 km arterial highway corridor with service roads, flyovers, and underpasses.

---

## 2. Multi-Tier Ingestion Architecture

The pipeline processes observations across three operational tiers of increasing trust:

```
+---------------------------------------------------------------------------------------------------+
|                                  OPENSTREETMAP ROAD NETWORK GRAPH                                 |
|                                (205 Road Edges // Bengaluru ORR)                                  |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
       +------------------------------------------+------------------------------------------+
       |                                          |                                          |
       v                                          v                                          v
+-----------------------------+    +-----------------------------+    +-----------------------------+
|           TIER 0            |    |           TIER 1            |    |           TIER 2            |
|  Google Street View Static  |    |     Dashcam Road Video      |    |  Dashcam + 100Hz Real IMU   |
|   (Cold Start / Baseline)   |    |    (Isolated Edge Scan)     |    | (Cross-Sensor Verification) |
|-----------------------------|    |-----------------------------|    |-----------------------------|
| • Road-facing 360° heading  |    | • YouTube/Uploaded Dashcam  |    | • Smartphone 100Hz Accel    |
| • 40m waypoint sampling     |    | • 3 to 15 FPS frame scan    |    | • Vertical acc_z correlation|
| • YOLO defect inspection    |    | • Waypoint GPS interpolation|    | • Optical-flow jitter proxy |
| • Full network coverage     |    | • Updates traversed edges   |    | • False-positive suppression|
| • Trust Weight: LOW (1.0)   |    | • Trust Weight: MEDIUM (2.0)|    | • Trust Weight: HIGH (3.0)  |
+-----------------------------+    +-----------------------------+    +-----------------------------+
       |                                          |                                          |
       +------------------------------------------+------------------------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |       EVIDENCE FUSION ENGINE        |
                               | • Trust-weighted aggregation        |
                               | • Exponential temporal decay        |
                               | • Confidence & severity scoring     |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |        SQLITE STORE & REST API      |
                               |      LEAFLET GIS INTERACTIVE MAP    |
                               +-------------------------------------+
```

### Tier 0: Google Street View (Baseline / Cold Start)
- **Objective:** Solves the cold-start problem by guaranteeing every road edge in the network has baseline condition telemetry before crowdsourced videos arrive.
- **Mechanism:** Samples representative points along all 205 OSM road edges at 40-meter steps. Calculates the exact road heading (compass bearing $0^\circ \text{--} 360^\circ$) pointing down the lane of travel. Concurrent multi-threaded queries (10 worker threads) fetch images from the Google Street View Static API with local disk caching.
- **Trust Level:** `LOW` (Weight = 1.0). Street View imagery is static and often several months old.

### Tier 1: Dashcam Video Footage (Isolated Traversal Updates)
- **Objective:** Ingests recent, public dashcam video footage (`.mp4`, `.mov`, `.avi`, `.webm`) traversing specific road edges.
- **Mechanism:** Extracts frames at configurable scan densities (Standard 3 FPS, Dense 10 FPS, Ultra 15 FPS, or Complete 100%). Interpolates GPS coordinates along the sequence of traversed road edges. Evaluates each frame with Ultralytics YOLOv8, annotates defect bounding boxes, and generates thumbnail crops.
- **Isolation Guarantee:** Modifies **only** the road edges directly traversed by the vehicle, leaving untouched edges in their existing baseline state.
- **Trust Level:** `MEDIUM` (Weight = 2.0). Video represents recent ground-truth conditions.

### Tier 2: Multi-Sensor Correlation & False-Positive Suppression
- **Objective:** Cross-references visual computer vision detections with physical vehicle vibration dynamics.
- **Mechanism:** Ingests 100Hz accelerometer recordings (`timestamp, acc_x, acc_y, acc_z`) from mounted smartphones, or computes a dense Farnebäck optical-flow vertical jitter proxy (`cv2.calcOpticalFlowFarneback`) when physical IMU is absent.
- **Dual Outcomes:**
  1. **Confirmed Vibration Match (True Positive):** A visual pothole coincides with a vertical acceleration spike ($|acc\_z - 9.81| > 1.8\,\text{m/s}^2$). The observation is elevated to Tier 2 with `HIGH` trust (Weight = 3.0).
  2. **False-Positive Suppression:** A visual defect detection (e.g. shadow, manhole cover, painted road marking, dark tar patch) without a corresponding vibration spike is flagged as a suppressed false positive (`vibration_match = 0`), preventing the segment score from being penalized.
- **Trust Level:** `HIGH` (Weight = 3.0).

### Tier 3: LiDAR Survey Vehicle (Architectural Stub)
- Preserved in the database schema (`source_tier = 3`) and UI drawer as a designated slot for dedicated highway authority survey vehicles.

---

## 3. Road-Quality Index & Mathematical Fusion Model

### Temporal Exponential Decay
Road condition evidence naturally decays in confidence as time elapses:

$$\text{decay\_weight}(obs) = \exp(-\lambda \cdot \Delta t_{\text{days}})$$

Where $\lambda = 0.05$ per day (half-life $\approx 14$ days), and $\Delta t_{\text{days}}$ is the elapsed simulated time between the observation's capture date and the current simulated clock.

### Observation Weighting
Each observation's final contribution is a function of its tier trust weight, temporal decay, and raw model confidence:

$$W(obs) = \text{TIER\_TRUST}[obs.\text{source\_tier}] \times \exp(-\lambda \cdot \Delta t_{\text{days}}) \times obs.\text{confidence}$$

$$\text{TIER\_TRUST} = \{ \text{Tier 0}: 1.0, \quad \text{Tier 1}: 2.0, \quad \text{Tier 2}: 3.0 \}$$

### Segment Quality Score (0–100 Scale)
Every segment aggregates good evidence (clear road detections) against bad evidence (weighted defect severity):

$$\text{Severity Multipliers}: \quad \text{Low} = 0.5, \quad \text{Medium} = 1.0, \quad \text{High} = 1.6, \quad \text{Severe} = 2.2$$

$$\text{Score} = 100 \times \frac{\sum W_{\text{good}}}{\sum W_{\text{good}} + \sum (W_{\text{defect}} \times \text{Severity})}$$

### Discrete Condition Bands

| Condition Band | Score Range | Hex Code | Visual Style |
| :--- | :---: | :---: | :--- |
| **Good** | 80 – 100 | `#2ECC71` | Vibrant Green |
| **Fair** | 60 – 79 | `#F1C40F` | Amber Yellow |
| **Poor** | 40 – 59 | `#E67E22` | Safety Orange |
| **Bad** | 20 – 39 | `#E74C3C` | Crimson Red |
| **Severe** | 0 – 19 | `#8E44AD` | Deep Purple |

The map line stroke weight dynamically reflects measurement confidence: higher observation counts render thicker lines on the map canvas.

---

## 4. Hardware & Dataset Assets

All models, sample videos, and real IMU sensor traces are integrated in the repository:

| Asset Name | Location | Details & Description |
| :--- | :--- | :--- |
| **YOLOv8 Weights** | `models/best.pt` | PyTorch checkpoint trained for pothole and road defect detection |
| **Sample Dashcam 1** | `data/raw/tier1/sample_videos/clip1.mp4` | 84 MB dashcam clip along ORR Bengaluru West corridor |
| **Sample Dashcam 2** | `data/raw/tier1/sample_videos/clip2.mp4` | 79 MB dashcam clip along ORR Bengaluru East corridor |
| **Standardized IMU** | `data/raw/tier2/sample_imu/corridor_vibration.csv` | 19,154 rows of 100Hz synchronized acceleration (`timestamp,acc_x,acc_y,acc_z`) |
| **Raw Real IMU** | `data/raw/tier2/raw_real/potholes_imu_formatted.csv` | Real-world smartphone accelerometer dataset covering 97 isolated potholes |
| **Street View Cache** | `data/raw/tier0/*.jpg` | 274 live Google Street View 360° road heading images |
| **Annotated Thumbnails**| `data/thumbnails/*.jpg` | 450+ cropped YOLO detection snapshots with bounding boxes and defect labels |
| **Road Graph GeoJSON** | `data/demo_area_roads.geojson` | 205 OSM segments with WGS84 geometries for Bengaluru ORR |
| **SQLite Database** | `data/road_quality.db` | High-performance relational store for road edges and observation history |

---

## 5. System Architecture & Data Flow

```
+------------------------------------------------------------------------------------+
|                                CLIENT APPLICATION                                  |
|   • Leaflet.js Web Map Canvas            • Slide-in Segment Telemetry Inspector    |
|   • CARTO Voyager Basemaps               • Drag-and-Drop Video Upload Dialog       |
|   • Multi-Tier Filter Controls           • Time-Decay Simulation Cockpit           |
+------------------------------------------+-----------------------------------------+
                                           | HTTP Requests (REST JSON / Multipart)
                                           v
+------------------------------------------------------------------------------------+
|                                FASTAPI BACKEND (api.py)                            |
|   GET  /segments                 GET  /segments/{edge_id}      GET  /demo/status   |
|   POST /ingest/tier0             POST /ingest/tier1            POST /ingest/tier2  |
|   POST /ingest/tier1/upload      POST /demo/simulate_decay     POST /demo/reset    |
+------------------------------------------+-----------------------------------------+
                                           | Internal Calls
                                           v
+------------------------------------------------------------------------------------+
|                         MODULAR INGESTION & CV ENGINES                             |
|  • ingest_tier0.py: Street View heading calculation, concurrent HTTP, disk cache   |
|  • ingest_tier1.py: Dashcam frame extraction, route interpolation, upload pipeline |
|  • ingest_tier2.py: 100Hz IMU correlation, Farnebäck optical flow jitter proxy     |
|  • ingest_common.py: RoadDefectDetector (YOLOv8 wrapper) & RoadGraphSnapper (GIS)  |
+------------------------------------------+-----------------------------------------+
                                           | SQL Queries / Commits
                                           v
+------------------------------------------------------------------------------------+
|                           SQLITE DATA STORE (db.py)                                |
|  • edges: edge_id, name, length_m, score, severity_bucket, confidence, geometry    |
|  • observations: edge_id, lat, lon, defect_type, severity, source_tier, metadata   |
|  • system_state: simulated_now clock timestamp                                     |
+------------------------------------------------------------------------------------+
```

---

## 6. Interactive GIS Frontend Walkthrough

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to access the dashboard.

### 1. Left GIS Control Sidebar
- **Corridor Monitor Badge:** Displays active bounding box, target stretch, and current simulation timestamp.
- **Independent Ingestion Triggers:**
  - **RUN TIER 0 INGEST:** Recomputes baseline from Street View imagery.
  - **RUN TIER 1 INGEST:** Recomputes dashcam clips along traversed corridor stretches.
  - **RUN TIER 2 INGEST:** Correlates detections with 100Hz accelerometer traces.
  - **UPLOAD ROAD VIDEO:** Opens the modal to ingest custom dashcam footage.
- **View Filter Radios:** Toggle between **Fused State**, **Tier 0 Only**, **Tier 0 + Tier 1**, and **All Tiers**.
- **Simulation Time Controls:** Fast-forward simulated time by `+14 Days` or `+30 Days` to observe real-time exponential staleness decay on older segments.

### 2. Segment Inspector Drawer (Right Slide-in)
Click any colored road segment on the map to inspect its telemetry:
- **Header:** Segment Name, Highway Type, Length (meters), Score (0–100), Severity Bucket, and Measurement Confidence percentage.
- **Multi-Tier Evidence Matrix:** Visual breakdown of observations by tier (Tier 0 vs Tier 1 vs Tier 2).
- **Observation History Log:** Chronological list of all detections recorded on that edge.
- **Observation Filtering:** Filter between `All Observations`, `Defects Only`, `Verified (Tier 2)`, and `Suppressed False Positives`.
- **PIN LOC 📍:** Drops a locator pin at the exact GPS coordinates of any observation.
- **Image Lightbox:** Click any defect thumbnail to inspect the full-resolution YOLO detection image with bounding box overlays.

---

## 7. REST API Documentation

FastAPI provides interactive OpenAPI documentation at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

### Core Endpoints

#### `GET /segments`
Returns a GeoJSON `FeatureCollection` of all 205 road segments.
- **Query Parameter:** `tier_filter` (`optional`): `all`, `tier0`, `tier1`, `tier2`.
- **Response Structure:**
  ```json
  {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": { "type": "LineString", "coordinates": [[77.6812, 12.9341], [77.6825, 12.9348]] },
        "properties": {
          "edge_id": "25186966",
          "name": "Outer Ring Road",
          "highway": "primary",
          "length_m": 142.6,
          "score": 68.4,
          "severity_bucket": "Fair",
          "confidence": 0.88,
          "dominant_defect": "pothole",
          "tier_breakdown": { "tier0": 1, "tier1": 3, "tier2": 10 }
        }
      }
    ]
  }
  ```

#### `GET /segments/{edge_id}`
Returns complete observation telemetry and defect history for a single road edge.

#### `POST /ingest/tier1/upload`
Uploads a road video file for automated YOLOv8 scanning, waypoint interpolation, and edge re-scoring.
- **Form Parameters:**
  - `video_file`: Video binary (`.mp4`, `.mov`, `.avi`, `.webm`).
  - `corridor`: Target stretch (`auto`, `west`, `east`).
  - `sample_rate`: Sampling rate in frames per second (e.g. `3.0`, `5.0`, `10.0`).
  - `max_frames`: Frame scan ceiling (`10`, `30`, `60`, `120`).
  - `conf_threshold`: YOLO confidence cutoff (`0.35`).
- **Example cURL:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/ingest/tier1/upload" \
    -F "video_file=@data/raw/tier1/sample_videos/clip1.mp4" \
    -F "corridor=west" \
    -F "sample_rate=3.0" \
    -F "max_frames=30" \
    -F "conf_threshold=0.35"
  ```

#### `POST /demo/simulate_decay`
Advances the simulated clock by `days` (default `14.0`) and recalculates confidence and scores across all edges.

#### `GET /demo/status`
Returns current system time, total road edges, total observations, and tier counts.

---

## 8. Installation & Execution Guide

### Prerequisites
- **Python Version:** **Python 3.13** (recommended) or **Python 3.11 / 3.12**.
  > [!IMPORTANT]
  > PyTorch (`torch`), `torchvision`, and `ultralytics` YOLOv8 distribute precompiled binary wheels for **Python &le; 3.13**. If your default system `py` command defaults to Python 3.15, make sure to use `py -3.13`.

### 1. Setup Virtual Environment
```powershell
# From the project root:
py -3.13 -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Seed Initial Demo Telemetry
Populates all 205 segments with baseline Street View observations, scans sample dashcam clips, and correlates 19,000+ real IMU vibration points:
```powershell
.\.venv\Scripts\python.exe scripts/seed_demo_data.py
```

### 3. Launch the Server
Use any of the following convenient launcher options:

**Option A: 1-Click Launchers**
- Windows Command Prompt / Double-click:
  ```cmd
  run.bat
  ```
- PowerShell:
  ```powershell
  .\run.ps1
  ```

**Option B: Direct Python Launch**
```powershell
.\.venv\Scripts\python.exe backend/api.py
```

**Option C: Direct Uvicorn Launch**
```powershell
.\.venv\Scripts\uvicorn.exe backend.api:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 9. Automated Verification & Test Suite

The test suite validates all functional criteria defined in the project specification:
```powershell
.\.venv\Scripts\python.exe scripts/test_pipeline.py
```

### Acceptance Checklist Results

| # | Acceptance Criterion | Status | Verification Details |
| :---: | :--- | :---: | :--- |
| **1** | **Tier 0 Baseline Cold Start** | ✅ PASSED | Populates all 205 segments in Bengaluru ORR with baseline scores (85–92) and LOW trust. |
| **2** | **Tier 1 Isolated Recompute** | ✅ PASSED | Scans dashcam video, detects potholes via YOLOv8, and updates **only** traversed edges. |
| **3** | **Tier 2 Multi-Sensor Correlation** | ✅ PASSED | Confirms high trust on matching accelerometer spikes and suppresses false positives (shadows, manholes). |
| **4** | **Segment Telemetry Inspector** | ✅ PASSED | Validates granular tier-by-tier evidence breakdown and observation history API. |
| **5** | **Simulated Staleness Decay** | ✅ PASSED | Validates exponential confidence decay when advancing clock (+14d / +30d). |
| **6** | **Zero Hardware Requirement** | ✅ PASSED | All pipelines execute locally using public OSM data and video/IMU proxies. |
| **7** | **New Feature: Video Upload** | ✅ PASSED | Verifies user video upload, YOLO inference, GPS interpolation, and segment re-scoring. |

---

## 10. Repository Structure

```text
road_condition/
├── .env                              # Environment configuration (Google Street View API key)
├── .env.example                      # Example environment template
├── .venv/                            # Python 3.13 virtual environment
├── README.md                         # Project documentation and architecture guide
├── docs_how_to_run.md                # Quick run and testing instructions
├── requirements.txt                  # Python dependencies (FastAPI, Ultralytics, OpenCV, PyTorch)
├── run.bat                           # 1-click Windows CMD launcher
├── run.ps1                           # 1-click Windows PowerShell launcher
├── backend/
│   ├── api.py                        # FastAPI REST API, endpoints, file upload handler & static mounts
│   ├── db.py                         # SQLite database operations, decay math, GeoJSON serializer
│   ├── ingest_common.py              # YOLOv8 wrapper, OSM graph snapper, optical-flow jitter analyzer
│   ├── ingest_tier0.py               # Google Street View Static API collector (multi-threaded, cached)
│   ├── ingest_tier1.py               # Dashcam video frame scanner & user video upload ingest pipeline
│   └── ingest_tier2.py               # 100Hz IMU correlation & false-positive suppression engine
├── frontend/
│   ├── index.html                    # Dark-asphalt civic GIS interface & control cockpit
│   ├── map.js                        # Leaflet map logic, condition color styling & inspector drawer
│   └── style.css                     # Custom design tokens, glassmorphism UI & status badges
├── data/
│   ├── demo_area_roads.geojson       # 205 OpenStreetMap road segments for Bengaluru ORR
│   ├── road_quality.db               # SQLite database (edges, observations, system_state)
│   ├── thumbnails/                   # 450+ cropped YOLO detection snapshots with bounding boxes
│   └── raw/
│       ├── tier0/                    # Cached Google Street View images (274 road heading views)
│       ├── tier1/
│       │   ├── sample_videos/        # Sample dashcam clips (clip1.mp4, clip2.mp4)
│       │   └── uploaded_videos/      # Storage for user-uploaded videos
│       └── tier2/
│           ├── sample_imu/           # corridor_vibration.csv (19,154 100Hz IMU samples)
│           └── raw_real/             # potholes_imu_formatted.csv (97 isolated potholes dataset)
├── models/
│   └── best.pt                       # Pretrained Ultralytics YOLOv8 road defect detection weights
└── scripts/
    ├── convert_real_imu.py           # Formats raw smartphone IMU recordings into standardized CSVs
    ├── fetch_osm_roads.py            # Extracts OSM road networks via Overpass API
    ├── seed_demo_data.py             # Re-seeds database with baseline, video, and IMU observations
    └── test_pipeline.py              # Automated 7-point acceptance test suite
```
