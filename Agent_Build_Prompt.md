# Build Prompt: Tiered Crowdsourced Road-Quality Map (Proof-of-Concept)

Copy everything below into your coding agent as its task brief.

---

## What you're building

A small, fully working proof-of-concept web app that shows a **road-quality map** for one real, small area. Road segments are colored by condition, and that condition is computed by a pipeline that ingests data in three independent tiers of increasing trust. Each tier can be re-run at any time on newly found data, and each tier's contribution updates the map on its own. No physical driving or hardware is involved — every input is either a public API, a public video, or a public dataset.

This is a **proof of concept for a methodology**, not a production system. Prioritize working end-to-end over completeness. Use the simplest, most standard open-source library available for every non-novel piece (web framework, DB, map rendering, HTTP, video I/O) — none of that is where the value of this project is. The value is in the tiered ingestion → fusion → map-update pipeline, so put your care there.

---

## 1. Demo area (use this, don't ask the user to pick)

**Outer Ring Road, Bengaluru — the Marathahalli to Silk Board stretch, clipped to a ~1.5 km sub-segment.**

Why this location: it has dense, current Google Street View coverage (it's inside Bengaluru's tech corridor), it is one of the most-filmed stretches of road in India for dashcam/traffic YouTube content (searches like "ORR Bengaluru dashcam", "Marathahalli traffic pothole", "Silk Board junction dashcam" return real footage), and it is widely reported as having poor road-surface quality — meaning a small clip of it will contain real, visible potholes and patch repairs, so the demo's output will look obviously correct rather than empty.

Bounding box (approximate, refine slightly with OpenStreetMap/Nominatim if needed):
```
min_lat: 12.925   max_lat: 12.945
min_lon: 77.660   max_lon: 77.700
```
Pull the actual road graph for this box with OSMnx at build time — don't hardcode segments by hand.

---

## 2. The three tiers — exact data flow

All three tiers write into the **same segment table**, keyed by OSM edge ID, and each write independently triggers a map refresh for that segment. This independence is the thing to get right: Tier 0 running alone must already produce a usable colored map; Tier 1 running later must visibly improve/change specific segments without touching others; Tier 2 must visibly boost confidence only where it has data.

### Tier 0 — Google Street View (baseline / cold start)
1. Sample points every ~20–30 m along every OSM edge inside the bounding box.
2. Pull a Street View Static API image at each point (heading pointed along the direction of travel), plus its metadata (capture date).
3. Run the detector (see §3) on each image.
4. Write one "observation" row per detection: `{edge_id, lat, lon, defect_type, severity, source_tier=0, source_trust=LOW, capture_date, ingested_at}`.
5. Recompute the road-quality index (§4) for every touched edge and push the update to the frontend.
Stay inside the free-tier Street View API quota: this bbox needs on the order of a few hundred calls, which fits comfortably.

### Tier 1 — YouTube dashcam / open dashcam datasets (video, no IMU)
1. Source footage two ways, either is fine, use both if time allows:
   - Download 2–4 public YouTube dashcam videos that plausibly cover this ORR stretch (search terms above), using `yt-dlp` (open-source).
   - Or pull a subset of an open dashcam dataset (e.g. RDD2022 India subset, or BharatPotHole) and treat its frames as if they were dashcam frames from this stretch.
2. Extract frames at a fixed rate (e.g. 1 fps).
3. Geotag frames: if the video has no real GPS track, approximate by interpolating along the chosen OSM edges in upload order (a reasonable proxy for a POC — state this assumption plainly in the UI/README, don't hide it).
4. Run the same detector (§3) on each frame.
5. Write observation rows with `source_tier=1, source_trust=MEDIUM`.
6. Recompute and push updates for touched edges only.

### Tier 2 — Dashcam + synchronized vibration data (video + accelerometer)
1. Prefer an **open dataset that already has real synchronized accelerometer + video** (e.g. the multi-sensor pothole datasets referenced in the project's own literature review — accelerometer+video crowdsourcing datasets). Use its accelerometer stream as-is where available.
2. Where only plain YouTube video exists (no real IMU channel), **do not use random noise** — derive a vibration-proxy signal from the video itself: run optical-flow / frame-jitter analysis (e.g. `cv2.calcOpticalFlowFarneback` or simple frame-to-frame pixel-shift magnitude) as a stand-in for vertical accelerometer spikes. This is a real, defensible approximation (camera shake correlates with road roughness) rather than fabricated data, and should be clearly labeled in the data as `vibration_source=proxy` vs `vibration_source=real_imu`.
3. Correlate: for each visual defect detection, check whether the vibration signal has a matching spike within a small time window at that timestamp.
   - Match → confidence boost, `source_trust=HIGH`.
   - No match (e.g. shadow, manhole cover, faded paint mistaken for a crack) → confidence stays low or the detection is suppressed.
4. Write observation rows with `source_tier=2`, including `vibration_source` and the match result.
5. Recompute and push updates for touched edges only.

### Tier 3 — real LiDAR reference vehicles
Out of scope for this POC. Leave a clearly labeled stub in the data model (`source_tier=3` reserved) and a disabled toggle in the UI so the extensibility story is visible without building it.

### Re-run behavior
Build one ingestion command per tier (`ingest_tier0.py`, `ingest_tier1.py`, `ingest_tier2.py`), each idempotent and appendable — running it again with a new video/image adds new observations and re-triggers fusion for the affected edges only, it doesn't require reprocessing the whole area. This is what "when a new video is found, update the map" means in practice.

---

## 3. Detector (shared by all tiers)

Use **Ultralytics YOLO** (open-source) for defect detection — pothole, crack, faded/missing lane marking, damaged signage, manhole cover as an explicit negative/ignore class. Fine-tune a small pretrained checkpoint on a public pothole dataset (RDD2022 or a Roboflow Universe pothole checkpoint) rather than training from scratch — a few hundred images and under an hour on a free-tier GPU is enough for a POC-quality model. Use the same model's monocular depth capability (or bounding-box-area as a simpler proxy if time is short) to bucket severity into Low/Medium/High/Severe.

---

## 4. Road-quality index (keep this simple and inspectable)

Per segment, maintain a running score computed from all observations on it:

```
weight(obs) = TIER_TRUST[obs.source_tier] * exp(-decay_rate * days_since(obs.capture_or_ingest_date))
score = 100 * (weighted_good_evidence) / (weighted_good_evidence + weighted_bad_evidence)
```
`TIER_TRUST = {0: 1, 1: 2, 2: 3}` (Tier 2 confirmed observations count for more). Pick `decay_rate` so that a Tier-0 observation with no fresher data noticeably fades over the demo's simulated timeline (a few tunable weeks), so you can show staleness live.

Store per segment: `score (0–100), dominant_defect_type, severity_bucket, last_observed_at, confidence (based on total weight, not just score), tier_breakdown`.

---

## 5. Tech stack — simplest available, nothing exotic

- **Backend:** Python, FastAPI (or Flask if the agent prefers less boilerplate). One process, no message queue, no microservices.
- **Detection/CV:** Ultralytics YOLO, OpenCV, `yt-dlp` for video download.
- **Geo:** OSMnx for the road graph, GeoPandas for edge/segment handling, plain Shapely for nearest-edge snapping.
- **Storage:** SQLite with the SpatiaLite extension, or even a single GeoJSON/JSON file if that's faster to stand up — this is a demo for ~20–40 segments, don't reach for PostGIS/Docker unless it's genuinely no extra effort.
- **API:** A handful of REST endpoints (below) serving plain JSON/GeoJSON — no vector tiles needed at this scale.
- **Frontend:** Plain HTML/CSS/JS (or a minimal React app if the agent strongly prefers) + **Leaflet.js** with OpenStreetMap base tiles. No frontend build pipeline required.

### Minimum API surface
```
GET  /segments                 -> GeoJSON FeatureCollection, one feature per OSM edge, with score/severity/tier_breakdown/confidence
GET  /segments/{edge_id}       -> full observation history for one edge (for the inspector panel)
POST /ingest/tier0             -> trigger/rerun Tier 0 ingestion for the bbox
POST /ingest/tier1             -> trigger/rerun Tier 1 ingestion (accepts a video path/URL or dataset subset id)
POST /ingest/tier2             -> trigger/rerun Tier 2 ingestion (accepts a video path/URL + optional accelerometer file)
```

---

## 6. Frontend requirements

### Functional
- Full-map hero: the Leaflet map *is* the app, not a card inside a dashboard.
- Segments drawn as colored line overlays on top of the OSM base layer, using **discrete condition bands**, not a smooth/blended gradient fill:
  - e.g. 5 bands — Good / Fair / Poor / Bad / Severe — each a distinct flat color, not interpolated between them.
- A layer toggle per tier (Tier 0 only / +Tier 1 / +Tier 2 / all) so you can visibly demonstrate each tier's independent contribution.
- Click a segment → an inspector panel (not a bouncy tooltip) showing: current score, defect type(s), severity, last-observed date, confidence, and a small breakdown of which tiers contributed.
- A "re-run ingestion" control (even a simple button that calls the POST endpoints) so the live demo can visibly show the map updating when new data is fed in.
- A legend that explains the color bands and the tier icons/badges.

### Visual direction — avoid generic AI-app look
Do **not** use: gradients anywhere in the UI chrome (buttons, panels, backgrounds), the SaaS rounded-card-with-soft-shadow kit applied uniformly to everything, a warm cream background with a terracotta accent, or a near-black background with a single neon accent. These are the current default "AI-generated app" tells — this app should look like a purpose-built civic/infrastructure tool, not a generic dashboard template.

Ground the visual language in the subject matter — road infrastructure:
- **Palette:** dark asphalt charcoal (`#2A2B2A`) and concrete grey (`#B7B2AA`) as the base UI colors; road-sign/hazard yellow (`#F3B518`) used sparingly as the one accent (active toggle, primary button); the condition scale (green → red) is reserved strictly for data on the map, never used as a UI brand color.
- **Type:** pair a clean grotesk for UI text (e.g. Public Sans / IBM Plex Sans) with a monospace face for all data readouts — scores, coordinates, timestamps, edge IDs (e.g. IBM Plex Mono) — like a technical instrument readout, not marketing copy.
- **Layout:** a left-hand panel styled like a GIS tool's layer list (hairline dividers, flat rows, no rounded card chrome) rather than floating dashboard cards. The inspector panel slides in from the right or bottom as a flat technical readout, not a popup bubble.
- Keep motion minimal: one smooth transition when the inspector panel opens/closes and when segment colors update after a re-run; no hover animations on every element.

---

## 7. Suggested project structure

```
/backend
  ingest_common.py     # detector wrapper, OSM edge snapping, fusion/index math
  ingest_tier0.py
  ingest_tier1.py
  ingest_tier2.py
  api.py               # FastAPI app, the 4 endpoints above
  db.py                # SQLite/SpatiaLite or GeoJSON store
/frontend
  index.html
  map.js               # Leaflet setup, layer toggles, inspector panel
  style.css
/data
  demo_area_roads.geojson
  raw/                 # downloaded GSV images, YouTube frames, dataset subsets
README.md              # states the bbox, the data-source assumptions (geotag interpolation,
                        # optical-flow vibration proxy vs real IMU), and how to re-run each tier
```

---

## 8. Acceptance checklist (what "done" looks like)

1. Running Tier 0 alone populates the whole bbox with a colored map from Street View data only.
2. Running Tier 1 on a fed-in video changes only the segments that video actually passed over, visibly.
3. Running Tier 2 on a fed-in video+vibration source further changes confidence/score only where a real detection+vibration match occurs, and visibly suppresses at least one false-positive-looking detection that has no vibration match.
4. Clicking any segment shows a clear tier-by-tier breakdown of what produced its current score.
5. The map, when left alone, visibly shows a segment's confidence decaying over a simulated time skip (a demo/testing control is fine for this — a "skip N days" button).
6. Nothing in the pipeline required physically visiting the ORR stretch or any hardware beyond a laptop.

Be explicit in the README about every simplifying assumption made (frame geotag interpolation, optical-flow vibration proxy, which dataset stood in for which tier) — reviewers will ask about these, and stating them plainly is part of the credibility of the demo, not a weakness to hide.
