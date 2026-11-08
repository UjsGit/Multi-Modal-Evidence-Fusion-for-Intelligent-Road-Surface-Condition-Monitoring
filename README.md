A system that scores road segments from 0 to 100 by combining evidence of different reliability and age: Google Street View imagery, dashcam video, and vibration data. It is demonstrated on 205 OpenStreetMap ways around the Outer Ring Road in Bengaluru.

Python 3.13 · FastAPI · SQLite · Leaflet 1.9.4 · Ultralytics YOLO

## How it works

Evidence comes in three tiers. Each detection is stored as an append-only observation tied to an OSM edge, and scores are recomputed from those observations.

| Tier | Source | Trust weight |
| :-- | :-- | :-: |
| 0 | Street View images (baseline) | 1.0 |
| 1 | Dashcam video, YOLO pothole detection | 2.0 |
| 2 | Tier 1 detections checked against an IMU trace or an optical-flow vibration proxy | 3.0 |

Tier 3 (LiDAR) is reserved in the schema (`source_tier = 3`) and has no ingestion code.

Each observation weight is `W = T_tier * exp(-lambda * age_days) * confidence`. Defect evidence is scaled by a severity multiplier (Low 0.5, Medium 1.0, High 1.6, Severe 2.2). The segment score is the clear-evidence share of the total, with small prior terms (see `backend/db.py`). Confidence is computed separately.

| Band | Score |
| :-- | :-: |
| Good | 80-100 |
| Fair | 60-79 |
| Poor | 40-59 |
| Bad | 20-39 |
| Severe | 0-19 |

Decay rate: `VERIFY` (code default lambda = 0.035 per day; older docs said 0.05). Vibration spike threshold: `VERIFY` (code 2.0 m/s^2; older docs said 1.8).


## Run it

```powershell
py -3.13 -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/seed_demo_data.py      # seed baseline, video and IMU observations
python backend/api.py                 # serve at http://127.0.0.1:8000
python scripts/test_pipeline.py       # 7 functional checks
```

Python 3.11 or 3.12 also work. A Street View API key goes in `.env` (see `.env.example`); without one, Tier 0 falls back to a synthetic baseline. The seed script and test script show results for the data in this repo only.

## API

Interactive docs are at `/docs`.

| Endpoint | Purpose |
| :-- | :-- |
| `GET /segments` | GeoJSON of all edges, optional tier filter |
| `GET /segments/{edge_id}` | Edge record and observation history |
| `POST /ingest/tier0`, `/tier1`, `/tier2` | Run each tier |
| `POST /ingest/tier1/upload` | Upload a video (corridor, rate, frame limit, threshold) |
| `POST /ingest/tier2/upload_imu` | Upload an accelerometer CSV |
| `POST /demo/simulate_decay` | Advance the simulated clock and recompute |
| `POST /demo/reset` | Reinitialise database and clock |
| `GET /demo/status` | Clock, edge and observation counts |

## Layout

```text
backend/    api.py, db.py, ingest_common.py, ingest_tier0.py, ingest_tier1.py, ingest_tier2.py
frontend/   index.html, map.js, style.css
data/       demo_area_roads.geojson, road_quality.db, raw/ (tier0, tier1, tier2), thumbnails/
models/     best.pt
scripts/    fetch_osm_roads.py, convert_real_imu.py, seed_demo_data.py, test_pipeline.py
```

## Tests

`scripts/test_pipeline.py` runs seven functional checks (cold start, Tier 1 isolation, Tier 2 correlation, segment inspector, staleness decay, no special hardware, video upload). They show the pipeline behaves as specified. They do not measure detection accuracy.
