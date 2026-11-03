"""
Database and Data Access Layer for Road Quality Map POC.
Uses SQLite for lightweight, zero-dependency local persistence.
Stores OSM edge geometries, road-quality indices, and observation history.
"""

import sqlite3
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "road_quality.db"
GEOJSON_FILE = DATA_DIR / "demo_area_roads.geojson"

# Default Trust Weights as specified in Build Prompt:
# Tier 0 (Street View): 1.0
# Tier 1 (YouTube/Dashcam video): 2.0
# Tier 2 (Dashcam + synchronized vibration): 3.0
TIER_TRUST = {
    0: 1.0,
    1: 2.0,
    2: 3.0
}

# Default daily staleness decay rate (tunable)
DEFAULT_DECAY_RATE = 0.035  # ~50% weight after 20 days

SEVERITY_WEIGHTS = {
    "Low": 0.5,
    "Medium": 1.0,
    "High": 1.6,
    "Severe": 2.2
}

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(force_reinit: bool = False):
    """Initializes SQLite schema and imports road network from GeoJSON if empty."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    cur = conn.cursor()

    if force_reinit:
        cur.execute("DROP TABLE IF EXISTS observations")
        cur.execute("DROP TABLE IF EXISTS edges")
        cur.execute("DROP TABLE IF EXISTS system_state")

    # Table for OSM road segments
    cur.execute("""
    CREATE TABLE IF NOT EXISTS edges (
        edge_id TEXT PRIMARY KEY,
        name TEXT,
        highway TEXT,
        surface TEXT,
        lanes TEXT,
        length_m REAL,
        score REAL DEFAULT 85.0,
        severity_bucket TEXT DEFAULT 'Good',
        dominant_defect TEXT DEFAULT 'none',
        confidence REAL DEFAULT 0.0,
        last_observed_at TEXT,
        tier_breakdown_json TEXT,
        geometry_json TEXT
    )
    """)

    # Table for all observations ingested across all tiers
    cur.execute("""
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        edge_id TEXT,
        lat REAL,
        lon REAL,
        defect_type TEXT,
        severity TEXT,
        source_tier INTEGER,
        source_trust TEXT,
        capture_date TEXT,
        ingested_at TEXT,
        vibration_source TEXT,
        vibration_match INTEGER DEFAULT 1,
        confidence_raw REAL DEFAULT 1.0,
        metadata_json TEXT,
        FOREIGN KEY (edge_id) REFERENCES edges(edge_id)
    )
    """)

    # Table for system simulation state (e.g. simulated clock time)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS system_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    conn.commit()

    # Populate edges from demo_area_roads.geojson if empty
    cur.execute("SELECT COUNT(*) as count FROM edges")
    edge_count = cur.fetchone()["count"]

    if edge_count == 0 and GEOJSON_FILE.exists():
        with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        features = geojson_data.get("features", [])
        print(f"Populating DB with {len(features)} road segments from GeoJSON...")
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            edge_id = str(props.get("edge_id") or feat.get("id"))
            name = props.get("name", "Outer Ring Road")
            highway = props.get("highway", "primary")
            surface = props.get("surface", "asphalt")
            lanes = str(props.get("lanes", "3"))
            length_m = float(props.get("length_m", 150.0))
            score = float(props.get("score", 85.0))
            tier_json = json.dumps({"tier0": 0, "tier1": 0, "tier2": 0})
            geom_json = json.dumps(geom)

            cur.execute("""
            INSERT OR IGNORE INTO edges 
            (edge_id, name, highway, surface, lanes, length_m, score, severity_bucket, dominant_defect, confidence, last_observed_at, tier_breakdown_json, geometry_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Good', 'none', 0.0, NULL, ?, ?)
            """, (edge_id, name, highway, surface, lanes, length_m, score, tier_json, geom_json))

        conn.commit()

    # Initialize simulated date if absent
    cur.execute("SELECT value FROM system_state WHERE key='simulated_now'")
    if not cur.fetchone():
        now_iso = datetime.now(timezone.utc).isoformat()
        cur.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('simulated_now', ?)", (now_iso,))
        cur.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('decay_rate', ?)", (str(DEFAULT_DECAY_RATE),))
        conn.commit()

    conn.close()

def get_simulated_now() -> datetime:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM system_state WHERE key='simulated_now'")
    row = cur.fetchone()
    conn.close()
    if row and row["value"]:
        return datetime.fromisoformat(row["value"].replace("Z", "+00:00"))
    return datetime.now(timezone.utc)

def advance_simulated_time(days: float) -> datetime:
    current = get_simulated_now()
    new_time = current + timedelta(days=days)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('simulated_now', ?)", (new_time.isoformat(),))
    conn.commit()
    conn.close()
    recompute_all_edges()
    return new_time

def reset_simulated_time() -> datetime:
    now = datetime.now(timezone.utc)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('simulated_now', ?)", (now.isoformat(),))
    conn.commit()
    conn.close()
    recompute_all_edges()
    return now

def get_decay_rate() -> float:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM system_state WHERE key='decay_rate'")
    row = cur.fetchone()
    conn.close()
    if row and row["value"]:
        return float(row["value"])
    return DEFAULT_DECAY_RATE

def determine_severity_bucket(score: float) -> str:
    """Discrete condition bands as specified in section 6:
    Good (80-100), Fair (60-79), Poor (40-59), Bad (20-39), Severe (0-19)"""
    if score >= 80:
        return "Good"
    elif score >= 60:
        return "Fair"
    elif score >= 40:
        return "Poor"
    elif score >= 20:
        return "Bad"
    else:
        return "Severe"

def record_observations(obs_list: List[Dict[str, Any]]) -> List[str]:
    """Records new observations and recomputes only touched edges."""
    if not obs_list:
        return []

    conn = get_db_connection()
    cur = conn.cursor()
    touched_edges = set()

    for obs in obs_list:
        edge_id = str(obs["edge_id"])
        touched_edges.add(edge_id)
        
        lat = float(obs.get("lat", 0.0))
        lon = float(obs.get("lon", 0.0))
        defect_type = obs.get("defect_type", "none")
        severity = obs.get("severity", "Low")
        source_tier = int(obs.get("source_tier", 0))
        source_trust = obs.get("source_trust", "LOW")
        capture_date = obs.get("capture_date") or datetime.now(timezone.utc).isoformat()
        ingested_at = obs.get("ingested_at") or datetime.now(timezone.utc).isoformat()
        vibration_source = obs.get("vibration_source", "none")
        vibration_match = 1 if obs.get("vibration_match", True) else 0
        confidence_raw = float(obs.get("confidence_raw", 1.0))
        meta_json = json.dumps(obs.get("metadata", {}))

        cur.execute("""
        INSERT INTO observations (
            edge_id, lat, lon, defect_type, severity, source_tier, 
            source_trust, capture_date, ingested_at, vibration_source, 
            vibration_match, confidence_raw, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge_id, lat, lon, defect_type, severity, source_tier,
            source_trust, capture_date, ingested_at, vibration_source,
            vibration_match, confidence_raw, meta_json
        ))

    conn.commit()
    conn.close()

    for eid in touched_edges:
        recompute_edge(eid)

    return list(touched_edges)

def recompute_edge(edge_id: str):
    """Recomputes score, confidence, and metrics for a single edge based on §4 formula."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM observations WHERE edge_id = ?", (edge_id,))
    rows = cur.fetchall()

    if not rows:
        # Edge exists with no observations
        cur.execute("""
        UPDATE edges SET 
            score = 85.0,
            severity_bucket = 'Good',
            dominant_defect = 'none',
            confidence = 0.0,
            tier_breakdown_json = ?
        WHERE edge_id = ?
        """, (json.dumps({"tier0": 0, "tier1": 0, "tier2": 0}), edge_id))
        conn.commit()
        conn.close()
        return

    simulated_now = get_simulated_now()
    decay_rate = get_decay_rate()

    weighted_good = 0.0
    weighted_bad = 0.0
    defect_counts: Dict[str, float] = {}
    tier_counts = {"tier0": 0, "tier1": 0, "tier2": 0}
    latest_capture = None

    for r in rows:
        tier = r["source_tier"]
        tier_key = f"tier{tier}"
        if tier_key in tier_counts:
            tier_counts[tier_key] += 1

        try:
            cdate_str = r["capture_date"].replace("Z", "+00:00")
            cdate = datetime.fromisoformat(cdate_str)
        except Exception:
            cdate = simulated_now

        if latest_capture is None or cdate > latest_capture:
            latest_capture = cdate

        days_since = max(0.0, (simulated_now - cdate).total_seconds() / 86400.0)
        time_decay = math.exp(-decay_rate * days_since)
        tier_weight = TIER_TRUST.get(tier, 1.0)
        raw_weight = tier_weight * time_decay * r["confidence_raw"]

        defect = r["defect_type"].lower()
        sev = r["severity"]
        sev_mult = SEVERITY_WEIGHTS.get(sev, 1.0)

        # Tier 2 correlation handling:
        # If Tier 2 observation and vibration_match is False (e.g. shadow/manhole cover falsely flagged as crack)
        # Suppress bad evidence!
        if tier == 2 and r["vibration_match"] == 0:
            # Suppressed false positive: don't penalize the road score
            weighted_good += raw_weight * 0.5
            continue

        if defect in ["none", "clear", "smooth"]:
            weighted_good += raw_weight
        else:
            impact = raw_weight * sev_mult
            weighted_bad += impact
            defect_counts[defect] = defect_counts.get(defect, 0.0) + impact

    # Baseline prior to prevent wild swings on a single detection:
    prior_good = 1.0
    prior_bad = 0.15

    total_good = weighted_good + prior_good
    total_bad = weighted_bad + prior_bad

    score = 100.0 * (total_good / (total_good + total_bad))
    score = max(0.0, min(100.0, round(score, 1)))

    severity_bucket = determine_severity_bucket(score)
    dominant_defect = max(defect_counts.items(), key=lambda x: x[1])[0] if defect_counts else "none"

    # Confidence derived from cumulative effective weight (asymptotes toward 1.0)
    total_effective_weight = weighted_good + weighted_bad
    confidence = round(1.0 - math.exp(-0.4 * total_effective_weight), 2)

    last_observed_str = latest_capture.isoformat() if latest_capture else simulated_now.isoformat()

    cur.execute("""
    UPDATE edges SET 
        score = ?,
        severity_bucket = ?,
        dominant_defect = ?,
        confidence = ?,
        last_observed_at = ?,
        tier_breakdown_json = ?
    WHERE edge_id = ?
    """, (
        score,
        severity_bucket,
        dominant_defect,
        confidence,
        last_observed_str,
        json.dumps(tier_counts),
        edge_id
    ))

    conn.commit()
    conn.close()

def recompute_all_edges():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT edge_id FROM edges")
    edges = [r["edge_id"] for r in cur.fetchall()]
    conn.close()

    for eid in edges:
        recompute_edge(eid)

def get_all_segments_geojson(tier_filter: Optional[str] = None) -> Dict[str, Any]:
    """Generates GeoJSON FeatureCollection for the Leaflet frontend.
    Allows filtering to show condition when only certain tiers are considered."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM edges")
    rows = cur.fetchall()

    tier_map_filter = None
    if tier_filter in ["tier0", "tier0_only"]:
        tier_map_filter = 0
    elif tier_filter in ["tier1", "tier0_tier1"]:
        tier_map_filter = 1
    elif tier_filter in ["tier2"]:
        tier_map_filter = 2

    # If tier_map_filter is specified, pre-query observations up to that tier
    filtered_scores = {}
    if tier_map_filter is not None:
        cur.execute("SELECT * FROM observations WHERE source_tier <= ?", (tier_map_filter,))
        obs_rows = cur.fetchall()
        simulated_now = get_simulated_now()
        decay_rate = get_decay_rate()

        from collections import defaultdict
        obs_by_edge = defaultdict(list)
        for ob in obs_rows:
            obs_by_edge[ob["edge_id"]].append(ob)

        for edge_id, e_rows in obs_by_edge.items():
            weighted_good = 0.0
            weighted_bad = 0.0
            defect_counts = {}
            for r in e_rows:
                tier = r["source_tier"]
                try:
                    cdate = datetime.fromisoformat(r["capture_date"].replace("Z", "+00:00"))
                except Exception:
                    cdate = simulated_now
                days_since = max(0.0, (simulated_now - cdate).total_seconds() / 86400.0)
                time_decay = math.exp(-decay_rate * days_since)
                tier_weight = TIER_TRUST.get(tier, 1.0)
                raw_weight = tier_weight * time_decay * r["confidence_raw"]

                defect = r["defect_type"].lower()
                sev = r["severity"]
                sev_mult = SEVERITY_WEIGHTS.get(sev, 1.0)

                if tier == 2 and r["vibration_match"] == 0:
                    weighted_good += raw_weight * 0.5
                    continue

                if defect in ["none", "clear", "smooth"]:
                    weighted_good += raw_weight
                else:
                    impact = raw_weight * sev_mult
                    weighted_bad += impact
                    defect_counts[defect] = defect_counts.get(defect, 0.0) + impact

            prior_good = 1.0
            prior_bad = 0.15
            total_good = weighted_good + prior_good
            total_bad = weighted_bad + prior_bad
            sc = round(100.0 * (total_good / (total_good + total_bad)), 1)
            bucket = determine_severity_bucket(sc)
            conf = round(1.0 - math.exp(-0.4 * (weighted_good + weighted_bad)), 2)
            dominant = max(defect_counts.items(), key=lambda x: x[1])[0] if defect_counts else "none"
            filtered_scores[edge_id] = (sc, bucket, conf, dominant)

    conn.close()

    features = []
    for r in rows:
        geom = json.loads(r["geometry_json"])
        tier_breakdown = json.loads(r["tier_breakdown_json"]) if r["tier_breakdown_json"] else {}
        edge_id = r["edge_id"]

        if tier_map_filter is not None and edge_id in filtered_scores:
            display_score, display_bucket, display_confidence, dom_defect = filtered_scores[edge_id]
        elif tier_map_filter is not None:
            display_score = 85.0
            display_bucket = "Good"
            display_confidence = 0.0
            dom_defect = "none"
        else:
            display_score = r["score"]
            display_bucket = r["severity_bucket"]
            display_confidence = r["confidence"]
            dom_defect = r["dominant_defect"]

        features.append({
            "type": "Feature",
            "id": r["edge_id"],
            "properties": {
                "edge_id": r["edge_id"],
                "name": r["name"],
                "highway": r["highway"],
                "surface": r["surface"],
                "lanes": r["lanes"],
                "length_m": r["length_m"],
                "score": display_score,
                "severity_bucket": display_bucket,
                "dominant_defect": dom_defect,
                "confidence": display_confidence,
                "last_observed_at": r["last_observed_at"],
                "tier_breakdown": tier_breakdown
            },
            "geometry": geom
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }

def get_segment_history(edge_id: str) -> Dict[str, Any]:
    """Returns detailed inspector data for a clicked edge."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM edges WHERE edge_id = ?", (edge_id,))
    edge = cur.fetchone()
    if not edge:
        conn.close()
        return {}

    cur.execute("""
    SELECT id, lat, lon, defect_type, severity, source_tier, 
           source_trust, capture_date, ingested_at, vibration_source, 
           vibration_match, confidence_raw, metadata_json
    FROM observations 
    WHERE edge_id = ? 
    ORDER BY capture_date DESC
    """, (edge_id,))
    observations = [dict(r) for r in cur.fetchall()]
    conn.close()

    for obs in observations:
        if obs.get("metadata_json"):
            try:
                obs["metadata"] = json.loads(obs["metadata_json"])
            except Exception:
                obs["metadata"] = {}
        del obs["metadata_json"]

    edge_dict = dict(edge)
    edge_dict["tier_breakdown"] = json.loads(edge_dict["tier_breakdown_json"]) if edge_dict.get("tier_breakdown_json") else {}
    del edge_dict["tier_breakdown_json"]
    del edge_dict["geometry_json"]

    return {
        "edge": edge_dict,
        "observations_count": len(observations),
        "observations": observations
    }
