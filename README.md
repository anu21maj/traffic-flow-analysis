# Traffic Flow Analysis: Detection, Tracking and Per-Lane Counting

A Python pipeline that detects vehicles in roadside video, tracks them across frames, and counts each vehicle once per lane. Lanes are defined by polygons you draw on the first frame.

**Demo:** [Watch the annotated output](https://youtu.be/MmqZMJf6Tz8)

## What this project is

The detector and tracker are pretrained models from [Ultralytics](https://github.com/ultralytics/ultralytics): YOLOv8 (COCO weights) and ByteTrack. The code in this repository is the part built around them:

- a lane calibration tool (click polygon points on the first frame, save to JSON)
- per-lane counting logic based on tracked IDs and point-in-polygon tests
- CSV, summary and annotated-video outputs

## Pipeline

1. **Detect** cars, motorcycles, buses and trucks (COCO classes 2, 3, 5, 7) with YOLOv8.
2. **Track** detections across frames with ByteTrack, giving each vehicle an ID.
3. **Assign lanes.** For each tracked box, take the centroid and test it against the three lane polygons with `cv2.pointPolygonTest`. A point on an edge counts as inside.
4. **Count.** A vehicle ID is added to a lane's set the first time its centroid falls inside that lane. The lane count is the size of the set.
5. **Export** a CSV log, a text summary and an annotated video.

## Setup (Python 3.10+)

```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

# PyTorch: CPU-only build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Or, for a compatible NVIDIA GPU (CUDA 12.1):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`requirements.txt` should contain: `ultralytics`, `opencv-python`, `numpy`, `pandas`, `yt-dlp`.

If `yt-dlp` fails because of network restrictions, download the MP4 manually and pass `--video-file path/to/video.mp4`.

## Usage

The script has two subcommands, `calibrate` and `run`.

### 1. Calibrate the lanes (one time per camera view)

```bash
python main.py calibrate --video-url https://www.youtube.com/watch?v=MNn9qKG2UFI --lanes lanes.json
```

Or with a local file:

```bash
python main.py calibrate --video-file path/to/video.mp4 --lanes lanes.json
```

Controls in the calibration window:

| Key / action | Effect |
|---|---|
| Left click | Add a point to the current lane polygon |
| `n` | Finish the current lane (needs at least 3 points) and start the next |
| `u` | Undo the last point |
| `r` | Reset the current lane |
| `q` | Abort without saving |

After the third lane is finished, the polygons are saved to `lanes.json`. Keep the polygons tight around each lane and avoid overlaps.

### 2. Run the analysis

```bash
python main.py run --video-url https://www.youtube.com/watch?v=MNn9qKG2UFI --lanes lanes.json --output-dir outputs
```

Or with a local file:

```bash
python main.py run --video-file path/to/video.mp4 --lanes lanes.json --output-dir outputs
```

Options for `run`:

| Option | Default | Meaning |
|---|---|---|
| `--model` | `yolov8n.pt` | Ultralytics model name or path (`yolov8s.pt` or larger is more accurate but slower) |
| `--conf` | `0.25` | Detection confidence threshold |
| `--device` | auto | `cpu` or a CUDA index such as `0` |
| `--classes` | `2 3 5 7` | COCO class IDs to include |
| `--max-width` | `1280` | Resize frames to this width for speed (`0` disables resizing) |

Press `q` in the preview window to stop early.

## Outputs (in `outputs/`)

- `counts.csv`: `VehicleID, Lane, Frame, Timestamp`, one row per vehicle per lane, logged when it is first counted
- `annotated.mp4`: lane polygons, bounding boxes, track IDs and live per-lane counts
- `summary.txt`: final count per lane

## Limitations

- **ID switches inflate counts.** A vehicle that is occluded and re-identified with a new track ID is counted again.
- **Lane changes.** A vehicle that moves between lanes is counted once in each lane it enters.
- **Cumulative counts, not flow rates.** Counts are totals over the clip, not vehicles per minute.
- **Static camera only.** Lane polygons are fixed in image coordinates.
- **Pretrained classes only.** The model is not fine-tuned for this scene, so small, distant or heavily overlapping vehicles may be missed.


## Troubleshooting

- **No tracks or IDs appear:** lower `--conf` and check that the lane polygons cover the road.
- **Low frame rate:** use `yolov8n.pt`, or reduce `--max-width`.
- **PyTorch has no CUDA support:** it falls back to CPU automatically.

## Project structure

```
traffic-flow-analysis/
├── main.py
├── requirements.txt
├── lanes.json        # created by the calibrate step
├── inputs/           # downloaded video
└── outputs/          # annotated.mp4, counts.csv, summary.txt
```
