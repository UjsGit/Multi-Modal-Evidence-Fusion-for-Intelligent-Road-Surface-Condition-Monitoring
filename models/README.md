# Models Directory

This directory stores the object detection model checkpoints for road defect detection (Potholes, Cracks, Faded Lanes, Damaged Signs, Manhole covers).

## Checkpoint Placement
Drop your trained or pretrained YOLO checkpoint here:
- **Recommended filename**: `models/pothole_yolo.pt` or `models/best.pt`

## Where to Obtain Model Weights
1. **Roboflow Universe (Pothole Detection Checkpoints)**:
   - URL: https://universe.roboflow.com/search?q=pothole
   - Dataset / Model: "Pothole Detection Computer Vision Project" (YOLOv8 / YOLOv11)
   - Download Format: PyTorch (`best.pt`) or export directly via Roboflow Python SDK.
2. **RDD2022 (Road Damage Detection) Benchmark**:
   - URL: https://github.com/sekilab/RoadDamageDetector
   - Checkpoints trained on Indian road subsets (RDD2022 India subset covers potholes, longitudinal & lateral cracks).
3. **Ultralytics YOLO Pretrained Baselines**:
   - URL: https://github.com/ultralytics/ultralytics
   - Standard baseline: `yolov8n.pt` / `yolov8s.pt`

## Zero-Dependency Fallback Mode
If no `.pt` checkpoint file is placed in this folder, the system automatically runs the internal **High-Fidelity Heuristic Engine** in `backend/ingest_common.py`. This ensures all three tiers, the API, and the Leaflet frontend work completely end-to-end out of the box!
