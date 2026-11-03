"""
Converter: Raw Real IMU Dataset to Road Quality Map Specification Format.
Converts multi-channel smartphone sensor recordings (such as 97_isolated_potholes.csv)
into the standard vibration CSV format:
    timestamp,acc_x,acc_y,acc_z
as specified in:
    - data/raw/tier2/sample_imu/sample_corridor_vibration.csv
    - [IMPORTANT] Data Needed.md
    - Agent_Build_Prompt.md
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_REAL_PATH = ROOT_DIR / "data" / "raw" / "tier2" / "raw_real" / "97_isolated_potholes.csv"
OUTPUT_IMU_DIR = ROOT_DIR / "data" / "raw" / "tier2" / "sample_imu"
OUTPUT_RAW_REAL_DIR = ROOT_DIR / "data" / "raw" / "tier2" / "raw_real"

def convert_real_imu(
    input_path: Path = RAW_REAL_PATH,
    output_dir: Path = OUTPUT_IMU_DIR
):
    if not input_path.exists():
        print(f"Error: Raw file not found at {input_path}")
        sys.exit(1)

    print(f"[Converter] Reading raw real IMU file: {input_path}...")
    df = pd.read_csv(input_path)
    print(f"[Converter] Loaded {len(df)} rows with {len(df.columns)} columns.")

    # 1. Normalize timestamp to start at 0.000s
    t0 = df["seconds_elapsed"].iloc[0]
    timestamp = (df["seconds_elapsed"] - t0).round(3)

    # 2. Gravity alignment for true vertical acceleration
    # The phone was mounted in portrait mode (gravity mostly on Y with slight backward tilt on Z)
    gx_mean = df["gravity_x"].mean()
    gy_mean = df["gravity_y"].mean()
    gz_mean = df["gravity_z"].mean()
    g_vec = np.array([gx_mean, gy_mean, gz_mean])
    g_norm = np.linalg.norm(g_vec)
    g_unit = g_vec / g_norm

    print(f"[Converter] Estimated Gravity Unit Vector: x={g_unit[0]:.4f}, y={g_unit[1]:.4f}, z={g_unit[2]:.4f}")

    # Vertical acceleration (projected along gravity vector onto vehicle vertical axis)
    acc_z = (
        df["totalAcceleration_x"] * g_unit[0] +
        df["totalAcceleration_y"] * g_unit[1] +
        df["totalAcceleration_z"] * g_unit[2]
    ).round(2)

    # Lateral acceleration (vehicle sway, lateral axis)
    acc_x = df["totalAcceleration_x"].round(2)

    # Longitudinal acceleration (forward/backward motion, gravity removed)
    acc_y = (df["totalAcceleration_z"] - df["gravity_z"]).round(2)

    # Core project specification format (matches sample_corridor_vibration.csv)
    formatted_df = pd.DataFrame({
        "timestamp": timestamp,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save primary real vibration file in sample_imu/
    primary_out = output_dir / "real_potholes_vibration.csv"
    formatted_df.to_csv(primary_out, index=False)
    print(f"[Converter] Saved primary formatted dataset to: {primary_out}")

    # Also save as corridor_vibration.csv (the default name cited in README.md)
    corridor_out = output_dir / "corridor_vibration.csv"
    formatted_df.to_csv(corridor_out, index=False)
    print(f"[Converter] Saved to standard active location: {corridor_out}")

    # Also save formatted copy inside raw_real/ folder for user convenience
    raw_real_out = OUTPUT_RAW_REAL_DIR / "potholes_imu_formatted.csv"
    formatted_df.to_csv(raw_real_out, index=False)
    print(f"[Converter] Saved copy in raw_real: {raw_real_out}")

    # Optional enriched version with GPS and original annotation metadata
    enriched_df = pd.DataFrame({
        "timestamp": timestamp,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "latitude": df["location_latitude"].round(6),
        "longitude": df["location_longitude"].round(6),
        "annotation": df["annotation_text"]
    })
    enriched_out = output_dir / "real_potholes_vibration_with_gps.csv"
    enriched_df.to_csv(enriched_out, index=False)
    print(f"[Converter] Saved enriched GPS dataset to: {enriched_out}")

    # Print summary statistics
    spikes = (np.abs(acc_z - 9.8) > 2.5).sum()
    print("\n==================================================")
    print("CONVERSION SUMMARY")
    print("==================================================")
    print(f"Total Records: {len(formatted_df)}")
    print(f"Duration: {timestamp.iloc[-1]:.2f} seconds ({timestamp.iloc[-1]/60:.1f} minutes)")
    print(f"Sampling Rate: ~100 Hz")
    print(f"Detected Pothole Vibration Spikes (|acc_z - 9.8| > 2.5 m/s²): {spikes} samples")
    print(f"acc_z (vertical): min={acc_z.min()} m/s², mean={acc_z.mean()} m/s², max={acc_z.max()} m/s²")
    print(f"acc_x (lateral):  min={acc_x.min()} m/s², mean={acc_x.mean()} m/s², max={acc_x.max()} m/s²")
    print(f"acc_y (longit.):  min={acc_y.min()} m/s², mean={acc_y.mean()} m/s², max={acc_y.max()} m/s²")
    print("==================================================")

if __name__ == "__main__":
    convert_real_imu()
