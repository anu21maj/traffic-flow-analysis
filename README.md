# Traffic Flow Analysis (3 Lanes, Tracking + Counting)

This repo contains a complete, minimal solution for the **Traffic Flow Analysis** task:
- Detect vehicles with a pre-trained COCO model (YOLOv8).
- Track vehicles across frames (ByteTrack via Ultralytics).
- Define **three lanes** (polygons) and **count vehicles per lane** (unique track IDs).
- Export a CSV with `VehicleID, Lane, Frame, Timestamp`.
- Save an annotated video with real-time overlays and final per-lane summary.

> Input video: https://www.youtube.com/watch?v=MNn9qKG2UFI

---

## 1) Setup (Python 3.10+ recommended)

```bash
# (Recommended) Create and activate a virtual environment
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

# 1. Install PyTorch (choose CPU or CUDA build)
# CPU-only (works everywhere):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# (Optional) CUDA 12.1 (if you have a compatible NVIDIA GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies
pip install -r requirements.txt
```

If `yt-dlp` fails due to network restrictions, download the MP4 manually and pass `--video-file path/to/file.mp4` when running.

---

## 2) Calibrate Lane Polygons (one-time)

Run the calibration tool to **draw 3 lane polygons** on the first frame:

```bash
python main.py --mode calibrate --video-url https://www.youtube.com/watch?v=MNn9qKG2UFI --lanes lanes.json
```
**Instructions (on-screen):**
- Click points around **Lane 1** in order (counter-clockwise or clockwise).
- Press **`n`** to finish Lane 1 and start Lane 2.
- Repeat for **Lane 2** and **Lane 3**.
- Use **`u`** to undo the last point, **`r`** to reset current lane, **`q`** to abort.
- When all 3 lanes are defined, the JSON file is saved (default: `lanes.json`).

> Tip: Try to tightly wrap each lane with a clean polygon that does not overlap others.

---

## 3) Run Analysis

```bash
# Using the original YouTube URL (downloaded automatically to ./inputs/input.mp4)
python main.py --mode run --video-url https://www.youtube.com/watch?v=MNn9qKG2UFI --lanes lanes.json --output-dir outputs

# OR if you already have the MP4
python main.py --mode run --video-file path/to/your.mp4 --lanes lanes.json --output-dir outputs
```

Key optional flags:
- `--model yolov8n.pt` (default) – use `yolov8s.pt` or above for higher accuracy.
- `--conf 0.25` – detection confidence threshold.
- `--device 0` – GPU index or `cpu`.

Outputs placed in `outputs/`:
- `counts.csv` – rows: `VehicleID,Lane,Frame,Timestamp`.
- `annotated.mp4` – overlay video with lanes, boxes, IDs, and live counts.
- `summary.txt` – total count per lane.

---

## 4) What Counts As a Vehicle?
We include COCO classes: **car(2), motorcycle(3), bus(5), truck(7)**. Edit `--classes` if you want to change.

---

## 5) Troubleshooting
- If PyTorch is missing CUDA, it will fall back to CPU.
- If you see no tracks/IDs, ensure `--conf` is not too high and the lanes cover the roadway.
- If FPS is very low, try a smaller model (`yolov8n.pt`) or reduce the input frame size (`--max-width 1280`).

---

## 6) Demo Video
## 6) Demo Video

👉 [Watch Demo Video](https://youtu.be/MmqZMJf6Tz8)

The demo shows the annotated output (`annotated.mp4`) with:
- Lane polygons overlaid in yellow
- Vehicle detections with bounding boxes and unique IDs
- Live per-lane counts in the top-left corner

For submission, the annotated video was trimmed to ~1–2 minutes for clarity using `ffmpeg`:
```bash
ffmpeg -y -ss 00:00:10 -t 00:01:30 -i outputs/annotated.mp4 -c copy outputs/demo.mp4


## 7) License
For educational use. Ensure your use of the input video respects YouTube's Terms of Service.
