#!/usr/bin/env python3
import argparse, os, sys, json, time, math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Optional: Ultralytics YOLO for detection + ByteTrack for tracking
from ultralytics import YOLO

# Optional: YouTube download
try:
    from yt_dlp import YoutubeDL
except Exception:
    YoutubeDL = None


VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: car=2, motorcycle=3, bus=5, truck=7


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def download_youtube(url: str, out_file: Path) -> Path:
    """
    Download a YouTube video using yt-dlp into the given out_file (MP4).
    """
    if out_file.exists():
        print(f"[info] Video already exists: {out_file}")
        return out_file

    if YoutubeDL is None:
        raise RuntimeError("yt-dlp is not installed. Install it or provide --video-file.")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_file),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
    }
    print(f"[info] Downloading {url} -> {out_file}")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    if not out_file.exists():
        raise RuntimeError("Download failed; file not found after yt-dlp.")
    return out_file


def get_first_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read first frame for calibration.")
    return frame


def point_in_polygon(pt, polygon_np):
    """
    Determine if point pt (x, y) lies inside polygon (Nx2 np.array).
    Uses cv2.pointPolygonTest which returns +1 (inside), -1 (outside), 0 (on edge).
    """
    x, y = pt
    res = cv2.pointPolygonTest(polygon_np.astype(np.float32), (float(x), float(y)), False)
    return res >= 0  # treat edge as inside


def draw_polygon(frame, poly_np, color=(0, 255, 255), thickness=2, fill_alpha=0.15):
    overlay = frame.copy()
    if len(poly_np) >= 3:
        cv2.fillPoly(overlay, [poly_np.astype(np.int32)], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    cv2.polylines(frame, [poly_np.astype(np.int32)], isClosed=True, color=color, thickness=thickness)


def draw_box_with_id(frame, xyxy, track_id, cls_name="veh"):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"ID {track_id}"
    cv2.putText(frame, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def centroid(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def calibrate(video_path: Path, lanes_json: Path, window_name="Calibration"):
    frame = get_first_frame(video_path)
    h, w = frame.shape[:2]
    clone = frame.copy()

    lanes = [[] for _ in range(3)]  # three lanes
    lane_idx = 0
    current = []

    instructions = [
        "Calibration Controls: click to add points for the current lane polygon.",
        "Press 'n' to finalize current lane and move to next.",
        "Press 'u' to undo last point. Press 'r' to reset current lane.",
        "Press 'q' to abort, or finish after 3 lanes to save."
    ]

    def on_mouse(event, x, y, flags, userdata):
        nonlocal current, frame
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            # Draw provisional polygon
            frame[:] = clone
            for li in range(lane_idx):
                if len(lanes[li]) >= 3:
                    draw_polygon(frame, np.array(lanes[li]))
            # Draw current lane points/edges
            if len(current) >= 2:
                cv2.polylines(frame, [np.array(current, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)
            for (px, py) in current:
                cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, w), min(720, h))
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame[:] = clone
        # Draw finished lanes
        for li in range(lane_idx):
            if len(lanes[li]) >= 3:
                draw_polygon(frame, np.array(lanes[li]))
        # Draw current lane as you click
        if len(current) >= 2:
            cv2.polylines(frame, [np.array(current, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)
        for (px, py) in current:
            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

        y0 = 24
        for line in instructions:
            cv2.putText(frame, line, (18, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (18, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            y0 += 26
        cv2.putText(frame, f"Defining Lane {lane_idx+1}/3", (18, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Defining Lane {lane_idx+1}/3", (18, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            print("[warn] Calibration aborted, not saving.")
            cv2.destroyWindow(window_name)
            return
        elif key == ord('u'):
            if current:
                current.pop()
        elif key == ord('r'):
            current = []
        elif key == ord('n'):
            if len(current) >= 3:
                lanes[lane_idx] = current.copy()
                lane_idx += 1
                current = []
                if lane_idx >= 3:
                    # Save and exit
                    data = {"lanes": lanes}
                    lanes_json.write_text(json.dumps(data, indent=2))
                    print(f"[ok] Saved lane polygons -> {lanes_json}")
                    cv2.destroyWindow(window_name)
                    return
            else:
                print("[info] Need at least 3 points to finalize a polygon.")


def run_analysis(video_path: Path, lanes_json: Path, output_dir: Path, model_name: str = "yolov8n.pt",
                 conf: float = 0.25, device: str = "", classes = VEHICLE_CLASS_IDS, max_width: int = 1280):
    ensure_dir(output_dir)
    out_video = output_dir / "annotated.mp4"
    out_csv = output_dir / "counts.csv"
    out_summary = output_dir / "summary.txt"

    # Load lanes
    data = json.loads(lanes_json.read_text())
    lanes = [np.array(poly, dtype=np.int32) for poly in data["lanes"]]
    if len(lanes) != 3:
        raise RuntimeError("lanes.json must contain exactly 3 lane polygons.")

    # Model
    print(f"[info] Loading model: {model_name}")
    model = YOLO(model_name)

    # Probe FPS + size
    cap0 = cv2.VideoCapture(str(video_path))
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None  # lazy init on first frame (after optional resize)

    # Counting state
    counted_ids_per_lane = [set(), set(), set()]
    csv_rows = []

    # Tracking inference stream
    print("[info] Starting tracking... (press 'q' to stop)")
    stream = model.track(source=str(video_path),
                         stream=True,
                         tracker="bytetrack.yaml",
                         conf=conf,
                         iou=0.5,
                         device=device if device else None,
                         classes=classes,
                         persist=True,  # keep track IDs between frames
                         verbose=False)

    frame_idx = -1
    for result in stream:
        frame_idx += 1
        frame = result.orig_img  # numpy array BGR
        if frame is None:
            continue

        # Optional resize for speed
        h, w = frame.shape[:2]
        scale = 1.0
        if max_width and w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # Prepare writer if needed
        if writer is None:
            out_w, out_h = frame.shape[1], frame.shape[0]
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (out_w, out_h))

        # Draw lane polygons
        for i, poly in enumerate(lanes):
            poly_scaled = (poly * scale).astype(np.int32)
            draw_polygon(frame, poly_scaled, color=(0, 255, 255))

        # Read tracked boxes and IDs
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()  # Nx4
            ids = boxes.id.cpu().numpy().astype(int)  # N
            clses = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(ids), dtype=int)

            # Iterate detections
            for det_i in range(len(ids)):
                cls_id = int(clses[det_i])
                if cls_id not in classes:
                    continue

                box = xyxy[det_i]
                if scale != 1.0:
                    box = box * scale
                cX, cY = centroid(box)

                # Determine which lane contains the centroid (if any)
                lane_hit = None
                for li, poly in enumerate(lanes):
                    poly_scaled = (poly * scale).astype(np.int32)
                    if point_in_polygon((cX, cY), poly_scaled):
                        lane_hit = li
                        break

                # Draw box & ID
                draw_box_with_id(frame, box, ids[det_i])

                # If inside a lane and not yet counted in that lane, record it
                if lane_hit is not None and ids[det_i] not in counted_ids_per_lane[lane_hit]:
                    counted_ids_per_lane[lane_hit].add(int(ids[det_i]))
                    timestamp = frame_idx / float(fps)
                    csv_rows.append([int(ids[det_i]), int(lane_hit+1), int(frame_idx), float(timestamp)])

        # Overlay live counts
        y0 = 28
        for i in range(3):
            text = f"Lane {i+1}: {len(counted_ids_per_lane[i])}"
            cv2.putText(frame, text, (18, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (18, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            y0 += 32

        # Write and show
        writer.write(frame)
        cv2.imshow("Traffic Flow Analysis (press 'q' to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("[info] Stopping early by user request.")
            break

    # Cleanup
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Save CSV
    df = pd.DataFrame(csv_rows, columns=["VehicleID", "Lane", "Frame", "Timestamp"])
    df.to_csv(out_csv, index=False)
    print(f"[ok] Wrote CSV -> {out_csv}")

    # Summary
    with open(out_summary, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(f"Lane {i+1}: {len(counted_ids_per_lane[i])}\n")
    print(f"[ok] Wrote summary -> {out_summary}")
    print(f"[ok] Annotated video -> {out_video}")


def main():
    parser = argparse.ArgumentParser(description="Traffic Flow Analysis: Detect, Track, Count per Lane")
    mode = parser.add_subparsers(dest="mode", required=True)

    # calibrate
    pcal = mode.add_parser("calibrate", help="Draw 3 lane polygons on first frame, save to JSON.")
    pcal.add_argument("--video-url", type=str, default="https://www.youtube.com/watch?v=MNn9qKG2UFI",
                      help="YouTube URL to download (ignored if --video-file provided).")
    pcal.add_argument("--video-file", type=str, default="",
                      help="Path to local MP4 (skips download).")
    pcal.add_argument("--lanes", type=str, default="lanes.json",
                      help="Output JSON file for lane polygons.")

    # run
    prun = mode.add_parser("run", help="Run detection + tracking + per-lane counting")
    prun.add_argument("--video-url", type=str, default="https://www.youtube.com/watch?v=MNn9qKG2UFI",
                      help="YouTube URL to download (ignored if --video-file provided).")
    prun.add_argument("--video-file", type=str, default="", help="Path to local MP4 (skips download).")
    prun.add_argument("--lanes", type=str, default="lanes.json", help="Lane polygons JSON file.")
    prun.add_argument("--output-dir", type=str, default="outputs", help="Directory for outputs.")
    prun.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model name or path.")
    prun.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    prun.add_argument("--device", type=str, default="", help="'cpu' or CUDA index like '0'.")
    prun.add_argument("--classes", type=int, nargs="+", default=VEHICLE_CLASS_IDS, help="COCO class IDs to include.")
    prun.add_argument("--max-width", type=int, default=1280, help="Resize width for speed (0 to disable).")

    args = parser.parse_args()

    inputs_dir = Path("inputs")
    ensure_dir(inputs_dir)

    # Determine video path
    video_path = None
    if args.mode == "calibrate":
        lanes_json = Path(args.lanes)
        if args.video_file:
            video_path = Path(args.video_file)
            if not video_path.exists():
                raise FileNotFoundError(video_path)
        else:
            video_path = inputs_dir / "input.mp4"
            download_youtube(args.video_url, video_path)

        calibrate(video_path, lanes_json)

    elif args.mode == "run":
        lanes_json = Path(args.lanes)
        if not lanes_json.exists():
            raise FileNotFoundError(f"Lane file not found: {lanes_json}. Run calibration first.")
        out_dir = Path(args.output_dir)
        ensure_dir(out_dir)

        if args.video_file:
            video_path = Path(args.video_file)
            if not video_path.exists():
                raise FileNotFoundError(video_path)
        else:
            video_path = inputs_dir / "input.mp4"
            download_youtube(args.video_url, video_path)

        run_analysis(video_path, lanes_json, out_dir, model_name=args.model, conf=args.conf,
                     device=args.device, classes=args.classes, max_width=args.max_width)


if __name__ == "__main__":
    main()
