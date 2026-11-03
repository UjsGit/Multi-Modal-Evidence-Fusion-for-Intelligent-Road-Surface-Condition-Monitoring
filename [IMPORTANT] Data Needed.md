
The base project for the **Tiered Crowdsourced Road-Quality Map (Proof-of-Concept)** has been created and verified against all criteria in [Agent_Build_Prompt.md](file:///d:/Code%20Projects/road_condition/Agent_Build_Prompt.md).

---

### Project Structure Created

```
road_condition/
├── backend/
│   ├── api.py                   # FastAPI REST API (GeoJSON segments, inspector, ingestion, decay)
│   ├── db.py                    # SQLite engine, fusion math, exponential time decay, trust weights
│   ├── ingest_common.py         # YOLO detector wrapper (w/ fallback), OSM edge snapper, vibration proxy
│   ├── ingest_tier0.py          # Tier 0: Google Street View cold start ingestion
│   ├── ingest_tier1.py          # Tier 1: YouTube Dashcam video & geotag interpolation
│   └── ingest_tier2.py          # Tier 2: Dashcam + IMU accelerometer / optical flow vibration correlation
├── frontend/
│   ├── index.html               # Leaflet map hero, civic infrastructure aesthetic, GIS controls
│   ├── map.js                   # Leaflet layers, discrete condition bands, inspector drawer telemetry
│   └── style.css                # Dark asphalt (#2A2B2A), concrete (#B7B2AA), hazard yellow (#F3B518)
├── data/
│   ├── demo_area_roads.geojson  # 205 real OpenStreetMap road segments for Bengaluru ORR
│   ├── road_quality.db          # Pre-seeded SQLite database with observations & fused scores
│   └── raw/
│       ├── tier0/               # Street View imagery
│       ├── tier1/sample_videos/ # Dashcam video clips
│       └── tier2/sample_imu/    # Synchronized accelerometer CSVs (sample included)
├── models/
│   └── README.md                # Guide for placing YOLO defect detection weights (.pt)
├── scripts/
│   ├── fetch_osm_roads.py       # OpenStreetMap Overpass road fetcher
│   ├── seed_demo_data.py        # Database seeder for demo presentation
│   └── test_pipeline.py         # Verification suite for Section 8 acceptance criteria
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation & handover instructions
└── Agent_Build_Prompt.md        # Original build plan
```

---

### Files & Data Needed From Your Side

The system features **built-in high-fidelity fallback detectors and simulated baseline feeds**, so **the entire application runs completely end-to-end right now**.

To upgrade the pipeline to live external data, here is what you need, where to get it, and where to put it:

| # | Item / Data | Source & Exact URL | Where to Put the File |
| :--- | :--- | :--- | :--- |
| **1** | **YOLO Pothole Weights (`.pt`)** | Download a pretrained YOLO pothole model from [Roboflow Universe: Pothole Detection](https://universe.roboflow.com/search?q=pothole) or [RDD2022 GitHub Repo](https://github.com/sekilab/RoadDamageDetector) | Drop into: `models/best.pt` or `models/pothole_yolo.pt` |
| **2** | **Street View API Key** | Google Maps Platform: [Google Cloud Console API Credentials](https://console.cloud.google.com/google/maps-apis/credentials) *(Enable "Street View Static API")* | In `.env` file (copy from `.env.example`): `GOOGLE_STREET_VIEW_API_KEY=your_key_here` |
| **3** | **Bengaluru ORR Dashcam Videos** | Search YouTube for `"ORR Bengaluru dashcam"`, `"Marathahalli traffic pothole"`, or `"Silk Board junction dashcam"`. Download via [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Drop `.mp4` files into: `data/raw/tier1/sample_videos/` |
| **4** | **Vibration / Accelerometer CSV** | Public multi-sensor dataset: [Zenodo Road Surface Vibration Dataset](https://zenodo.org/records/4555027) or record via phone app: [Physics Toolbox Sensor Suite](https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite) | Drop `.csv` with `timestamp, acc_z` columns into: `data/raw/tier2/sample_imu/` |

---

### How to Run and Test Right Now

1. **Run the acceptance test suite** (verifies all 6 Section 8 checklist criteria):
   ```bash
   python scripts/test_pipeline.py
   ```
2. **Start the API and Web App**:
   ```bash
   python backend/api.py
   ```
3. Open your browser at: **`http://127.0.0.1:8000`**
   - Click any road segment to open the **Segment Telemetry Inspector** drawer.
   - Click **Run Tier 0 / 1 / 2 Ingest** to test independent re-scoring.
   - Click **Simulate +14 Days Decay** to observe confidence decay and staleness live.

---

### Continuation Prompt for Your Teammate's AI Agent

Copy and paste the exact text block below to give to your teammate's AI agent:

```text
You are continuing development on the "Tiered Crowdsourced Road-Quality Map (Proof-of-Concept)" project located in this workspace.

CURRENT STATUS & COMPLETED WORK:
1. Base project architecture is fully built and tested against all 6 acceptance criteria in `Agent_Build_Prompt.md`:
   - `backend/api.py`: FastAPI REST API serving GeoJSON segments, single-edge observation history, ingestion triggers, and time-decay simulation.
   - `backend/db.py`: SQLite database storing 205 real OpenStreetMap road segments for Bengaluru ORR (Marathahalli - Silk Board, bbox 12.925-12.945 N, 77.660-77.700 E).
   - `backend/ingest_common.py`: Shared YOLO detection wrapper with heuristic fallback, OSM edge snapper, geotag interpolator, and optical-flow vibration analyzer.
   - `backend/ingest_tier0.py`: Google Street View cold start ingestion.
   - `backend/ingest_tier1.py`: Dashcam video ingestion with geotag interpolation along OSM ways.
   - `backend/ingest_tier2.py`: Multi-sensor vibration correlation + false-positive suppression.
   - `frontend/index.html`, `frontend/map.js`, `frontend/style.css`: Leaflet map with purpose-built civic/infrastructure dark-asphalt aesthetic, discrete condition color bands, slide-in inspector drawer, and live simulation controls.
   - `scripts/test_pipeline.py`: Passes all 6 acceptance checks with 100% success.

NEXT STEPS FOR YOU TO CONTINUE:
1. Integrate live data:
   - Check if real model weights exist in `models/best.pt`. If present, verify inference using Ultralytics YOLO.
   - If videos exist in `data/raw/tier1/sample_videos/`, wire them to extract frames and geotag along target edges.
   - If real IMU CSVs exist in `data/raw/tier2/sample_imu/`, correlate vertical accelerometer spikes (`acc_z`) with detected defect timestamps.
2. If Google Street View API key is configured in `.env`, test live Street View metadata fetching.
3. Enhance the Leaflet inspector UI with image thumbnail previews of detected road defects for selected segments.
4. Run `python scripts/test_pipeline.py` and `python backend/api.py` to test your changes.
```