"""
Tier 1 Ingestion: YouTube Dashcam / Open Dashcam Video (No IMU).
Processes dashcam clips (clip1.mp4 and clip2.mp4) using Ultralytics YOLOv8 pothole detection.
Extracts frames at ~1 fps, geotags them via route interpolation along Bengaluru ORR OSM edges,
runs YOLO defect detector, saves annotated thumbnails of detections, and writes observations
with source_tier=1, source_trust=MEDIUM.
Crucially: recomputes and modifies ONLY the segments traversed by the video footage.
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.db import init_db, record_observations, get_simulated_now
from backend.ingest_common import RoadDefectDetector, RoadGraphSnapper

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_TIER1_DIR = DATA_DIR / "raw" / "tier1"
SAMPLE_VIDEOS_DIR = RAW_TIER1_DIR / "sample_videos"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"

# Contiguous Outer Ring Road (ORR) corridor stretches for sample clips
ORR_CORRIDOR_WEST = ['311412704', '367594185', '25186966', '674050818', '308856658', '25187663']
ORR_CORRIDOR_EAST = ['367594184', '377104672', '377104670', '1239003786', '380929043']


def download_youtube_clip(url: str, output_path: str) -> bool:
    """Uses yt-dlp to download a dashcam clip if installed."""
    try:
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",
            "--download-sections", "*0:00-0:20",
            "-o", output_path,
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"[Tier 1] yt-dlp download failed: {e}")
        return False


def process_single_video(
    video_path: str,
    target_edges: List[str],
    detector: RoadDefectDetector,
    snapper: RoadGraphSnapper,
    simulated_now: datetime,
    clip_id: int = 1,
    num_frames: Optional[int] = None,
    sample_rate: Optional[float] = 10.0,
    frame_step: Optional[int] = None,
    conf_threshold: float = 0.35
) -> List[Dict[str, Any]]:
    """Extracts frames from a video file at a dense sampling rate, runs YOLO detection, and returns observations."""
    import cv2
    
    video_name = os.path.basename(video_path)
    stem = Path(video_path).stem
    print(f"[Tier 1] Processing '{video_name}' along ORR corridor ({len(target_edges)} segments)...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Tier 1] Error: Cannot open video file {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sampling step:
    # If frame_step is given (e.g. 1 for full 100% scan, or 2 for every 2nd frame), use directly.
    # Otherwise derive from sample_rate (default 10.0 fps -> step=3 on 30fps clip, triple coverage vs old 3fps).
    if frame_step is not None and frame_step > 0:
        step = int(frame_step)
    elif sample_rate is not None and sample_rate > 0:
        step = max(1, int(round(fps / sample_rate)))
    else:
        step = 2  # default dense scanning (every 2nd frame, ~15 fps)

    actual_scan_fps = round(fps / step, 1)
    
    # Calculate how many frames will be sampled across the entire clip
    expected_samples = (total_frames + step - 1) // step if total_frames > 0 else 100
    if num_frames is not None and num_frames > 0:
        expected_samples = min(expected_samples, num_frames)
    expected_samples = max(expected_samples, len(target_edges))

    # Generate route waypoints along the target ORR corridor matching total sample count
    waypoints = snapper.interpolate_points_along_edges(target_edges, num_points=expected_samples)

    capture_date = (simulated_now - timedelta(days=4 + clip_id)).isoformat()
    observations = []
    pothole_count = 0

    frame_idx = 0
    sample_idx = 0  # counts how many samples we've scanned

    print(f"[Tier 1] '{video_name}': Scanning at step={step} (~{actual_scan_fps} fps, expected {expected_samples} frames, conf={conf_threshold})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % step == 0:
            if num_frames is not None and num_frames > 0 and sample_idx >= num_frames:
                break

            wp = waypoints[min(sample_idx, len(waypoints) - 1)]
            lat, lon = wp["lat"], wp["lon"]
            timestamp_sec = round(frame_idx / fps, 2)

            # Snap coordinates to target edge
            edge_id, dist = snapper.snap(lat, lon)
            if not edge_id or edge_id not in target_edges:
                edge_id = target_edges[sample_idx % len(target_edges)]

            # Run YOLO detector on frame with tuned confidence threshold
            detections = detector.detect_frame(frame, frame_id=sample_idx, conf_threshold=conf_threshold)

            pothole_dets = [d for d in detections if d.get("defect_type") == "pothole"]

            if pothole_dets:
                pothole_count += len(pothole_dets)
                # Generate unique annotated snapshot with frame index to prevent collision
                thumb_filename = f"{stem}_f{frame_idx:05d}_t{int(timestamp_sec*100):05d}.jpg"
                thumb_path = str(THUMBNAILS_DIR / thumb_filename)
                detector.annotate_and_save_frame(frame, pothole_dets, thumb_path)
                thumbnail_url = f"/thumbnails/{thumb_filename}"

                for det in pothole_dets:
                    observations.append({
                        "edge_id": edge_id,
                        "lat": lat,
                        "lon": lon,
                        "defect_type": "pothole",
                        "severity": det["severity"],
                        "source_tier": 1,
                        "source_trust": "MEDIUM",
                        "capture_date": capture_date,
                        "ingested_at": simulated_now.isoformat(),
                        "vibration_source": "none",
                        "vibration_match": True,
                        "confidence_raw": det.get("confidence", 0.75),
                        "metadata": {
                            "source": "Dashcam Video (Tier 1)",
                            "video_source": video_name,
                            "frame_index": frame_idx,
                            "frame_sec": timestamp_sec,
                            "timestamp_sec": timestamp_sec,
                            "bbox": det.get("bbox", []),
                            "area_ratio": det.get("area_ratio", 0.0),
                            "thumbnail_url": thumbnail_url
                        }
                    })
            sample_idx += 1

        frame_idx += 1
        
    cap.release()
    print(f"[Tier 1] '{video_name}': Scanned {sample_idx} frames (step={step}, ~{actual_scan_fps} fps across {frame_idx} video frames), {pothole_count} potholes detected.")
    return observations


def run_tier1_ingest(
    video_source: Optional[str] = None,
    target_edge_ids: Optional[List[str]] = None,
    num_frames: Optional[int] = None,
    sample_rate: Optional[float] = 10.0,
    frame_step: Optional[int] = None,
    conf_threshold: float = 0.35
) -> Dict[str, Any]:
    """Runs Tier 1 ingestion on dashcam videos (clip1.mp4 & clip2.mp4) along ORR edges."""
    init_db()
    RAW_TIER1_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

    snapper = RoadGraphSnapper()
    detector = RoadDefectDetector()

    if not snapper.segments:
        return {"status": "error", "message": "No road segments found in database."}

    # Determine videos to process
    videos_to_process = []
    
    if video_source and video_source.startswith("http"):
        download_target = str(SAMPLE_VIDEOS_DIR / "dashcam_temp.mp4")
        if download_youtube_clip(video_source, download_target):
            videos_to_process.append((download_target, target_edge_ids or ORR_CORRIDOR_WEST))
    elif video_source and os.path.exists(video_source):
        videos_to_process.append((video_source, target_edge_ids or ORR_CORRIDOR_WEST))
    else:
        # Process clip1.mp4 and clip2.mp4 from sample_videos
        clip1_path = SAMPLE_VIDEOS_DIR / "clip1.mp4"
        clip2_path = SAMPLE_VIDEOS_DIR / "clip2.mp4"
        
        if target_edge_ids:
            if clip1_path.exists():
                videos_to_process.append((str(clip1_path), target_edge_ids))
            elif clip2_path.exists():
                videos_to_process.append((str(clip2_path), target_edge_ids))
        else:
            if clip1_path.exists():
                videos_to_process.append((str(clip1_path), ORR_CORRIDOR_WEST))
            if clip2_path.exists():
                videos_to_process.append((str(clip2_path), ORR_CORRIDOR_EAST))
            
        if not videos_to_process:
            # Check any other mp4 in directory
            existing = list(SAMPLE_VIDEOS_DIR.glob("*.mp4"))
            for i, vid in enumerate(existing):
                corridor = target_edge_ids if target_edge_ids else (ORR_CORRIDOR_WEST if i % 2 == 0 else ORR_CORRIDOR_EAST)
                videos_to_process.append((str(vid), corridor))

    if not videos_to_process:
        return {"status": "error", "message": "No dashcam video footage found in data/raw/tier1/sample_videos."}

    simulated_now = get_simulated_now()
    all_observations = []
    
    for idx, (vpath, corridor) in enumerate(videos_to_process):
        obs = process_single_video(
            video_path=vpath,
            target_edges=corridor,
            detector=detector,
            snapper=snapper,
            simulated_now=simulated_now,
            clip_id=idx + 1,
            num_frames=num_frames,
            sample_rate=sample_rate,
            frame_step=frame_step,
            conf_threshold=conf_threshold
        )
        all_observations.extend(obs)

    # Persist observations and recompute ONLY touched segments
    touched = record_observations(all_observations)
    potholes_total = sum(1 for o in all_observations if o.get("defect_type") == "pothole")
    
    print(f"[Tier 1] Ingestion complete: {len(all_observations)} observations recorded ({potholes_total} potholes).")
    print(f"[Tier 1] Updated {len(touched)} segments: {touched}")

    return {
        "status": "success",
        "tier": 1,
        "mode": "real_yolo_dashcam",
        "videos_processed": [os.path.basename(v[0]) for v in videos_to_process],
        "observations_count": len(all_observations),
        "potholes_detected": potholes_total,
        "sample_rate": sample_rate,
        "frame_step": frame_step,
        "conf_threshold": conf_threshold,
        "touched_edges": touched,
        "touched_edges_count": len(touched),
        "message": f"Successfully processed {len(videos_to_process)} dashcam videos via YOLO with increased frame scan coverage. Detected {potholes_total} potholes across {len(touched)} ORR segments."
    }


def ingest_uploaded_video(
    video_path: str,
    target_corridor: Optional[str] = "auto",
    target_edge_ids: Optional[List[str]] = None,
    sample_rate: Optional[float] = 3.0,
    max_frames: Optional[int] = 100,
    frame_step: Optional[int] = None,
    conf_threshold: float = 0.35
) -> Dict[str, Any]:
    """Ingests and analyzes a user-uploaded road dashcam video.
    Runs YOLOv8 defect detection, interpolates GPS waypoints along target OSM edges,
    generates annotated bounding-box thumbnails, records Tier 1 observations,
    and recalculates fused quality scores for touched segments."""
    init_db()
    RAW_TIER1_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

    snapper = RoadGraphSnapper()
    detector = RoadDefectDetector()

    if not snapper.segments:
        return {"status": "error", "message": "No road segments found in database."}

    # Determine corridor segments
    if target_edge_ids and len(target_edge_ids) > 0:
        corridor = target_edge_ids
    elif target_corridor == "west":
        corridor = ORR_CORRIDOR_WEST
    elif target_corridor == "east":
        corridor = ORR_CORRIDOR_EAST
    else:
        # Default auto corridor combining primary stretch
        corridor = ORR_CORRIDOR_WEST + ORR_CORRIDOR_EAST

    simulated_now = get_simulated_now()
    observations = process_single_video(
        video_path=video_path,
        target_edges=corridor,
        detector=detector,
        snapper=snapper,
        simulated_now=simulated_now,
        clip_id=99,
        num_frames=max_frames,
        sample_rate=sample_rate,
        frame_step=frame_step,
        conf_threshold=conf_threshold
    )

    touched = record_observations(observations)
    potholes_total = sum(1 for o in observations if o.get("defect_type") == "pothole")
    video_name = os.path.basename(video_path)

    print(f"[Tier 1 Upload] Ingestion complete: {len(observations)} observations recorded ({potholes_total} potholes).")
    print(f"[Tier 1 Upload] Updated {len(touched)} segments: {touched}")

    return {
        "status": "success",
        "tier": 1,
        "mode": "user_uploaded_video",
        "video_filename": video_name,
        "observations_count": len(observations),
        "potholes_detected": potholes_total,
        "sample_rate": sample_rate,
        "max_frames": max_frames,
        "conf_threshold": conf_threshold,
        "touched_edges": touched,
        "touched_edges_count": len(touched),
        "message": f"Analyzed '{video_name}' via YOLOv8. Detected {potholes_total} potholes across {len(touched)} ORR segments."
    }


if __name__ == "__main__":
    result = run_tier1_ingest()
    print(json.dumps(result, indent=2))
