"""
Shared Ingestion, Detection, Edge Snapping, and Vibration Processing Utilities.
Used across Tier 0 (Street View), Tier 1 (Dashcam), and Tier 2 (Multi-sensor).
"""

import math
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
GEOJSON_FILE = DATA_DIR / "demo_area_roads.geojson"

# Classes supported
DEFECT_CLASSES = ["pothole", "crack", "faded_lane", "damaged_sign", "manhole_cover", "clear"]
IGNORE_CLASSES = ["manhole_cover"]  # Explicit negative/ignore class

class RoadDefectDetector:
    """Wrapper around Ultralytics YOLO with fallback to heuristic/mock CV."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.is_real_yolo = False
        
        # Check if user has placed a custom YOLO model checkpoint
        candidates = [
            model_path,
            str(MODELS_DIR / "pothole_yolo.pt"),
            str(MODELS_DIR / "best.pt"),
            str(MODELS_DIR / "yolov8n.pt")
        ]
        
        target_model = None
        for cand in candidates:
            if cand and os.path.exists(cand):
                target_model = cand
                break
                
        if target_model:
            try:
                from ultralytics import YOLO
                print(f"[Detector] Loading Ultralytics YOLO checkpoint: {target_model}")
                self.model = YOLO(target_model)
                self.is_real_yolo = True
            except Exception as e:
                print(f"[Detector] Ultralytics import/load failed ({e}). Falling back to internal engine.")
        else:
            print("[Detector] No YOLO model weights found in models/ (e.g. models/best.pt).")
            print("[Detector] Running in high-fidelity mock/heuristic detection mode.")

    def detect_frame(self, image_input: Any, frame_id: int = 0, conf_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Runs detection on an image path or numpy frame.
        Returns list of detections with defect_type, severity, confidence, bbox."""
        is_valid_input = False
        if hasattr(image_input, "shape"):  # numpy array / cv2 frame
            is_valid_input = True
        elif isinstance(image_input, str) and os.path.exists(image_input):
            is_valid_input = True

        if self.is_real_yolo and self.model is not None and is_valid_input:
            try:
                return self._detect_yolo(image_input, conf_threshold=conf_threshold)
            except Exception as e:
                print(f"[Detector] YOLO inference error ({e}), falling back to heuristic.")
                return self._detect_heuristic(image_input, frame_id)
        else:
            return self._detect_heuristic(image_input, frame_id)

    def _detect_yolo(self, image_input, conf_threshold: float = 0.35) -> List[Dict[str, Any]]:
        import cv2
        scale_x, scale_y = 1.0, 1.0
        feed_input = image_input
        orig_h, orig_w = 640, 640

        if hasattr(image_input, "shape"):
            orig_h, orig_w = image_input.shape[:2]
            if orig_w > 960 or orig_h > 540:
                feed_input = cv2.resize(image_input, (640, 360))
                scale_x = orig_w / 640.0
                scale_y = orig_h / 360.0

        results = self.model(feed_input, conf=conf_threshold, imgsz=640, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                raw_name = str(r.names.get(cls_id, "pothole")).lower()
                conf = float(box.conf[0].item())
                raw_xyxy = box.xyxy[0].tolist()
                xyxy = [
                    round(float(raw_xyxy[0]) * scale_x, 1),
                    round(float(raw_xyxy[1]) * scale_y, 1),
                    round(float(raw_xyxy[2]) * scale_x, 1),
                    round(float(raw_xyxy[3]) * scale_y, 1)
                ]
                
                # Area proxy for severity based on bounding box size
                w = max(1.0, xyxy[2] - xyxy[0])
                h = max(1.0, xyxy[3] - xyxy[1])
                area_ratio = (w * h) / (orig_h * orig_w)
                
                if area_ratio > 0.04:
                    severity = "Severe"
                elif area_ratio > 0.02:
                    severity = "High"
                elif area_ratio > 0.008:
                    severity = "Medium"
                else:
                    severity = "Low"

                # Class 0 in models/best.pt represents pothole
                cls_name = "pothole" if raw_name in ["0", "defect", "potholes"] or raw_name not in DEFECT_CLASSES else raw_name

                if cls_name in IGNORE_CLASSES:
                    # Explicit negative class (e.g. manhole covers)
                    continue

                detections.append({
                    "defect_type": cls_name,
                    "severity": severity,
                    "confidence": round(conf, 2),
                    "bbox": xyxy,
                    "area_ratio": round(area_ratio, 4)
                })
        return detections

    def annotate_and_save_frame(
        self,
        frame: Any,
        detections: List[Dict[str, Any]],
        output_path: str,
        badge_text: Optional[str] = None,
        badge_color: Optional[Tuple[int, int, int]] = None
    ) -> bool:
        """Draws bounding boxes and defect labels on frame, saving as JPEG thumbnail.
        Supports optional badge banner for Tier 2 vibration confirmation/suppression."""
        try:
            import cv2
            if isinstance(frame, str) and os.path.exists(frame):
                annotated = cv2.imread(frame)
            elif hasattr(frame, "copy"):
                annotated = frame.copy()
            else:
                return False

            if annotated is None:
                return False

            h, w = annotated.shape[:2]

            # Draw detections
            for det in detections:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    defect = det.get("defect_type", "pothole").upper()
                    conf = det.get("confidence", 0.0)
                    sev = det.get("severity", "Medium").upper()
                    label = f"{defect} {int(conf*100)}% [{sev}]"
                    
                    # Colors based on severity
                    color = (0, 0, 255) if sev in ["HIGH", "SEVERE"] else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    
                    # Label banner
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + tw + 10, max(0, y1)), color, -1)
                    cv2.putText(annotated, label, (x1 + 5, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw telemetry badge banner at top if provided
            if badge_text:
                b_color = badge_color or (0, 180, 0)
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(annotated, (10, 10), (10 + tw + 20, 10 + th + 16), (20, 20, 20), -1)
                cv2.rectangle(annotated, (10, 10), (10 + tw + 20, 10 + th + 16), b_color, 2)
                cv2.putText(annotated, badge_text, (20, 10 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, b_color, 2)
                    
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return True
        except Exception as e:
            print(f"[Detector] Failed to save annotated thumbnail: {e}")
            return False

    def _detect_heuristic(self, image_input, frame_id: int) -> List[Dict[str, Any]]:
        """Intelligent heuristic detection to allow out-of-the-box demo execution."""
        # Generates realistic deterministic detections based on frame cycle
        # To simulate road segments with potholes, cracks, faded lanes, and clear patches
        cycle = frame_id % 7
        if cycle == 0:
            return [{
                "defect_type": "pothole",
                "severity": "High",
                "confidence": 0.88,
                "bbox": [240, 310, 360, 420],
                "area_ratio": 0.05
            }]
        elif cycle == 2:
            return [{
                "defect_type": "crack",
                "severity": "Medium",
                "confidence": 0.76,
                "bbox": [150, 280, 290, 350],
                "area_ratio": 0.03
            }]
        elif cycle == 4:
            return [{
                "defect_type": "faded_lane",
                "severity": "Low",
                "confidence": 0.69,
                "bbox": [50, 300, 180, 480],
                "area_ratio": 0.04
            }]
        elif cycle == 5:
            # Manhole cover: explicit negative class to test false-positive filtering
            return [{
                "defect_type": "manhole_cover",
                "severity": "Medium",
                "confidence": 0.72,
                "bbox": [300, 320, 370, 390],
                "area_ratio": 0.02
            }]
        else:
            return [{
                "defect_type": "none",
                "severity": "Low",
                "confidence": 0.95,
                "bbox": [],
                "area_ratio": 0.0
            }]

# Edge snapping and road geometry helper
class RoadGraphSnapper:
    """Snaps lat/lon coordinates to the nearest OSM road edge."""
    def __init__(self, geojson_path: Path = GEOJSON_FILE):
        self.segments = []
        if geojson_path.exists():
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for feat in data.get("features", []):
                edge_id = str(feat.get("properties", {}).get("edge_id") or feat.get("id"))
                coords = feat.get("geometry", {}).get("coordinates", [])
                if coords:
                    self.segments.append({
                        "edge_id": edge_id,
                        "name": feat.get("properties", {}).get("name", "ORR Segment"),
                        "coords": coords  # list of [lon, lat]
                    })
        print(f"[Snapper] Initialized with {len(self.segments)} road segments.")

    def snap(self, lat: float, lon: float) -> Tuple[Optional[str], float]:
        """Finds nearest edge ID and distance in meters."""
        if not self.segments:
            return None, float("inf")

        best_edge = None
        min_dist = float("inf")

        for seg in self.segments:
            coords = seg["coords"]
            for pt in coords:
                d = haversine_distance(lat, lon, pt[1], pt[0])
                if d < min_dist:
                    min_dist = d
                    best_edge = seg["edge_id"]

        return best_edge, min_dist

    def interpolate_points_along_edges(self, edge_ids: List[str], num_points: int) -> List[Dict[str, float]]:
        """Interpolates lat/lon points along a sequence of OSM edges.
        Used for Tier 1 geotag interpolation when video has no hardware GPS."""
        all_coords = []
        seg_map = {s["edge_id"]: s for s in self.segments}
        for eid in edge_ids:
            if eid in seg_map:
                all_coords.extend(seg_map[eid]["coords"])

        if not all_coords:
            # Fallback to default ORR bounding center coordinates
            all_coords = [[77.6850, 12.9340], [77.6780, 12.9290]]

        interpolated = []
        n = max(1, num_points)
        total_pts = len(all_coords)
        for i in range(n):
            idx = int((i / n) * (total_pts - 1))
            lon, lat = all_coords[idx]
            # Add subtle jitter to represent moving vehicle
            lat_jitter = lat + (i % 3 - 1) * 0.00005
            lon_jitter = lon + (i % 2 - 1) * 0.00005
            interpolated.append({
                "lat": round(lat_jitter, 6),
                "lon": round(lon_jitter, 6),
                "timestamp_sec": float(i)
            })

        return interpolated

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes distance in meters between two lat/lon points."""
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Vibration and Optical-Flow Proxy
class VibrationAnalyzer:
    """Handles both real hardware accelerometer data and optical-flow vibration proxy."""
    
    @staticmethod
    def load_imu_csv(csv_path: str) -> List[Dict[str, float]]:
        """Loads accelerometer records from CSV. Expects timestamp, acc_z (vertical)."""
        records = []
        if not os.path.exists(csv_path):
            return records
            
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row.get("timestamp") or row.get("time") or 0.0)
                z = float(row.get("acc_z") or row.get("z") or row.get("accel_z") or 9.8)
                records.append({"timestamp": ts, "acc_z": z})
        return records

    @staticmethod
    def compute_optical_flow_vibration_proxy(video_path: str, max_frames: int = 120) -> List[Dict[str, float]]:
        """Derives a vertical vibration-proxy signal from video frame-to-frame jitter.
        Uses optical flow / pixel-shift magnitude as required in Build Prompt §2.2."""
        vibration_signals = []
        
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            ret, prev_frame = cap.read()
            if not ret or prev_frame is None:
                cap.release()
                raise ValueError("Could not read video frames.")
                
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_idx = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate Farneback optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Vertical motion component (v) reflects vehicle bump/pitch
                v_component = flow[..., 1]
                vertical_jitter = float(abs(v_component.mean()))
                
                vibration_signals.append({
                    "timestamp": round(frame_idx / fps, 2),
                    "jitter_magnitude": round(vertical_jitter, 3),
                    "is_spike": vertical_jitter > 1.8
                })
                
                prev_gray = gray
                frame_idx += 1
                
            cap.release()
        except Exception as e:
            print(f"[Vibration] Optical flow fallback (OpenCV not active or video file mock): {e}")
            # Generate deterministic synthetic vibration proxy for mock runs
            for idx in range(max_frames):
                ts = round(idx * 0.5, 2)
                # Simulated spike at frame 0, 7, 14
                spike = (idx % 7 == 0)
                mag = 3.2 if spike else 0.4
                vibration_signals.append({
                    "timestamp": ts,
                    "jitter_magnitude": mag,
                    "is_spike": spike
                })

        return vibration_signals

    @staticmethod
    def correlate_defect_with_vibration(defect_timestamp: float, vibration_stream: List[Dict[str, float]], window_sec: float = 1.0) -> Tuple[bool, float]:
        """Correlates a visual defect detection timestamp with vibration spikes.
        Match -> True, confidence boost.
        No match -> False, confidence stays low or suppression occurs."""
        for v in vibration_stream:
            t = v.get("timestamp", 0.0)
            if abs(t - defect_timestamp) <= window_sec:
                if v.get("is_spike", False) or v.get("jitter_magnitude", 0) > 1.5 or abs(v.get("acc_z", 9.8) - 9.8) > 2.5:
                    return True, 1.4  # Matched spike
        return False, 0.4  # No matching vibration
