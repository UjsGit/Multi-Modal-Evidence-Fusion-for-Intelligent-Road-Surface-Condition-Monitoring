"""
Seed script to initialize the database and run Tier 0 baseline, Tier 1 dashcam,
and Tier 2 multi-sensor ingestion so the demo is immediately rich and interactive.
"""

import sys
from pathlib import Path

# Set up import path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.db import init_db, recompute_all_edges
from backend.ingest_tier0 import run_tier0_ingest
from backend.ingest_tier1 import run_tier1_ingest
from backend.ingest_tier2 import run_tier2_ingest

def seed():
    print("=== SEEDING ROAD QUALITY DATABASE ===")
    
    print("\n1. Initializing database schema and road graph...")
    init_db(force_reinit=True)
    
    print("\n2. Executing Tier 0 baseline ingestion (Street View)...")
    res0 = run_tier0_ingest()
    print(f"Tier 0 Result: {res0['message']}")
    
    print("\n3. Executing Tier 1 ingestion (Dashcam video)...")
    res1 = run_tier1_ingest(sample_rate=5.0, conf_threshold=0.35)
    print(f"Tier 1 Result: {res1['message']}")
    
    print("\n4. Executing Tier 2 ingestion (Multi-sensor vibration correlation)...")
    res2 = run_tier2_ingest()
    print(f"Tier 2 Result: {res2['message']}")
    
    print("\n5. Computing global fused scores across all segments...")
    recompute_all_edges()
    
    print("\n=== SEED COMPLETE! Launch the API server to view the map: ===")
    print("python backend/api.py")
    print("Or open http://127.0.0.1:8000 in your browser.")

if __name__ == "__main__":
    seed()
