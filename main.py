"""
Real-Time Object Detection & Tracking
======================================
Stack  : YOLOv8 (Ultralytics) + DeepSORT (deep-sort-realtime) + OpenCV
Author : Generated for VSCode / Python 3.9+

HOW IT WORKS (high level):
  1. OpenCV reads one frame at a time from the video/webcam.
  2. Every N frames, YOLOv8 detects objects in a downscaled copy of that frame.
  3. DeepSORT receives those detections and assigns each object a unique Track ID
     that persists across frames — even when the object is briefly not detected.
  4. The bounding boxes are scaled back up to the original resolution and drawn
     on the full-quality frame before displaying fullscreen.

Usage  :
    python main.py                              # webcam (default)
    python main.py --source video.mp4           # video file
    python main.py --source 0                   # explicit webcam index
    python main.py --conf 0.4                   # confidence threshold (default 0.5)
    python main.py --model yolov8s.pt           # larger model for better accuracy
    python main.py --proc-width 640             # smaller = faster inference (default 960)
    python main.py --skip 4                     # skip more frames for extra speed
"""

# ── Standard library imports ────────────────────────────────────────────────
import argparse          # for command-line argument parsing
import time              # for FPS timing
import sys               # for sys.exit() on errors
from pathlib import Path # for extracting filename from path

# ── Third-party imports ─────────────────────────────────────────────────────
import cv2               # OpenCV — reads frames, draws on them, displays the window
import numpy as np       # NumPy — array operations (colour palette, frame math)
from ultralytics import YOLO                              # YOLOv8 detection model
from deep_sort_realtime.deepsort_tracker import DeepSort  # multi-object tracker
import torch
torch.set_num_threads(8)   # limit PyTorch CPU threads to avoid system overload


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# These are the default values used unless overridden by command-line arguments.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "yolov8n.pt"  # 'n' = nano model (fastest, smallest, least accurate)
                                 # alternatives: yolov8s.pt / yolov8m.pt / yolov8l.pt
DEFAULT_CONF    = 0.5            # only accept detections with ≥50% confidence
DEFAULT_IOU     = 0.45           # Intersection-over-Union threshold for NMS
                                 # (removes duplicate boxes that overlap by >45%)
DEFAULT_PROC_W  = 960            # width to resize frames to before inference
                                 # smaller = faster but less detail for the model
MAX_AGE         = 30             # how many consecutive frames DeepSORT will keep a
                                 # track alive without seeing the object (then it dies)
EMBEDDER        = "mobilenet"    # appearance model DeepSORT uses to tell objects apart
                                 # mobilenet extracts a feature vector (embedding) from
                                 # each detected crop to re-identify objects over time
WINDOW_TITLE    = "Object Detection & Tracking  |  Q = quit"
FONT            = cv2.FONT_HERSHEY_SIMPLEX   # OpenCV font used for all text
DISPLAY_W       = 1280           # legacy constant (kept for reference; window is fullscreen)


# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# Pre-generate 300 distinct bright colours, one per class/track.
# Using a fixed seed (42) means colours are the same every run.
# ──────────────────────────────────────────────────────────────────────────────
_RNG    = np.random.default_rng(42)
PALETTE = _RNG.integers(80, 255, size=(300, 3)).tolist()
# Values range 80–255 to avoid very dark colours that are hard to see on video.


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# Lets the user control behaviour from the terminal without editing the code.
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 + DeepSORT real-time tracker")

    p.add_argument("--source", default="0",
                   help="Video source: '0' for webcam, or path to video file")
    # If the value is a digit string like '0', it's treated as a webcam index.
    # Otherwise it's treated as a file path.

    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"YOLO model weights file (default: {DEFAULT_MODEL})")
    # The .pt file is downloaded automatically on first run if not found locally.

    p.add_argument("--conf", type=float, default=DEFAULT_CONF,
                   help=f"Detection confidence threshold 0–1 (default: {DEFAULT_CONF})")
    # Lower = more detections but more false positives.
    # Higher = fewer but more reliable detections.

    p.add_argument("--iou", type=float, default=DEFAULT_IOU,
                   help=f"NMS IoU threshold 0–1 (default: {DEFAULT_IOU})")
    # Controls how aggressively overlapping duplicate boxes are suppressed.

    p.add_argument("--skip", type=int, default=9,
                   help="Skip N frames between YOLO detections (default: 9)")
    # With skip=9, YOLO runs every 10th frame.
    # DeepSORT still updates every frame using Kalman filter prediction,
    # so boxes remain smooth even without a new detection.

    p.add_argument("--proc-width", type=int, default=DEFAULT_PROC_W,
                   help=f"Resize width before inference for speed (default: {DEFAULT_PROC_W})")
    # The frame is downscaled to this width before passing to YOLO.
    # Bounding box coordinates are then scaled back up for display.
    # This is the biggest performance lever: 960→640 nearly doubles FPS on CPU.

    p.add_argument("--save", action="store_true",
                   help="Save annotated output to output.mp4")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────────────────────────────────────
def load_model(weights: str) -> YOLO:
    """
    Load a YOLOv8 model from a .pt weights file.
    If the file doesn't exist locally, Ultralytics downloads it automatically
    from GitHub (~6 MB for yolov8n.pt).
    """
    print(f"[INFO] Loading model: {weights}")
    model = YOLO(weights)
    print(f"[INFO] Model loaded  — {len(model.names)} classes")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# OBJECT DETECTION  (YOLOv8)
# ──────────────────────────────────────────────────────────────────────────────
def detect_objects(frame: np.ndarray,
                   model: YOLO,
                   conf: float,
                   iou: float) -> list[list]:
    """
    Run YOLOv8 inference on a single BGR frame.

    YOLOv8 divides the image into a grid and predicts bounding boxes +
    class probabilities at each grid cell simultaneously (single-pass = fast).

    Returns detections in the format DeepSORT expects:
        [ ([x1, y1, width, height], confidence, class_name), ... ]
    """
    # model.predict() returns a list of Results objects (one per image).
    # imgsz=640 tells YOLO the internal processing size — it handles its own
    # resize internally but we already passed a small frame for extra speed.
    # verbose=False silences per-frame console output.
    results = model.predict(frame, conf=conf, iou=iou,
                            imgsz=640, verbose=False, stream=False)[0]

    detections = []
    for box in results.boxes:
        # box.xyxy gives [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w, h            = x2 - x1, y2 - y1      # convert to width/height for DeepSORT
        conf_score      = float(box.conf[0])      # confidence score 0.0–1.0
        class_id        = int(box.cls[0])          # numeric class index (e.g. 0 = person)
        class_name      = model.names[class_id]   # human-readable name (e.g. "person")
        detections.append(([x1, y1, w, h], conf_score, class_name))

    return detections


# ──────────────────────────────────────────────────────────────────────────────
# DRAW TRACKS
# Renders each confirmed track's bounding box and label onto the frame.
# ──────────────────────────────────────────────────────────────────────────────
def draw_tracks(frame: np.ndarray,
                tracks: list,
                class_names: dict,
                sx: float = 1.0,
                sy: float = 1.0) -> np.ndarray:
    """
    Draw bounding boxes and labels for all confirmed tracks.

    sx, sy — scale factors to convert from inference resolution back to the
             full-resolution frame where we're drawing.
             e.g. if inference was at 960px wide and display is 3840px wide,
             sx = 3840/960 = 4.0 so every x coordinate is multiplied by 4.
    """
    for track in tracks:
        # DeepSORT tracks go through three states:
        #   Tentative → needs n_init (3) consecutive detections to be confirmed
        #   Confirmed → actively being tracked and displayed
        #   Deleted   → removed after MAX_AGE frames without a detection
        # We only draw confirmed tracks to avoid phantom boxes.
        if not track.is_confirmed():
            continue

        track_id = track.track_id   # unique integer ID for this object's lifetime
        ltrb     = track.to_ltrb()  # Kalman-predicted box: [Left, Top, Right, Bottom]

        # Scale coordinates from the small inference frame back to full resolution
        x1 = int(ltrb[0] * sx)
        y1 = int(ltrb[1] * sy)
        x2 = int(ltrb[2] * sx)
        y2 = int(ltrb[3] * sy)

        # Retrieve class name and confidence stored by DeepSORT when it matched
        # the detection. Guard against None — DeepSORT sets these to None for
        # tracks that were not matched in the current frame (Kalman-predicted only).
        det_class = (track.det_class
                     if hasattr(track, "det_class") and track.det_class is not None
                     else "object")
        conf      = (float(track.det_conf)
                     if hasattr(track, "det_conf") and track.det_conf is not None
                     else 0.0)

        # ── Scale-aware font / thickness ────────────────────────────────────
        # At 1280px wide: scale_f = 1.0 → font_scale 0.55, thickness 2
        # At 3840px wide: scale_f = 3.0 → font_scale 1.65, thickness 6
        # This ensures text stays readable regardless of frame resolution.
        scale_f    = max(1.0, frame.shape[1] / 1280)
        font_scale = round(0.55 * scale_f, 2)
        box_thick  = max(2, int(2 * scale_f))    # bounding box line thickness
        txt_thick  = max(1, int(1.5 * scale_f))  # text stroke thickness
        pad        = max(4, int(6 * scale_f))    # padding inside label background

        # ── Consistent colour per class ──────────────────────────────────────
        # hash() converts the class name string to a number, then we wrap it
        # into the palette range. Same class always gets the same colour.
        colour_idx = hash(det_class) % len(PALETTE)
        colour     = tuple(PALETTE[colour_idx])

        # ── Bounding box ─────────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, box_thick)

        # ── Label: measure text size first, then draw background behind it ───
        label = f"ID:{track_id}  {det_class}  {conf:.0%}"
        # getTextSize returns the pixel width/height of the rendered text
        (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, txt_thick)
        # Draw filled rectangle as label background (thickness -1 = filled)
        cv2.rectangle(frame,
                      (x1, y1 - th - baseline - pad),   # top-left of background
                      (x1 + tw + pad, y1),               # bottom-right of background
                      colour, -1)

        # ── Label text (black on coloured background) ─────────────────────────
        cv2.putText(frame, label,
                    (x1 + pad // 2, y1 - baseline - pad // 2),
                    FONT, font_scale, (0, 0, 0), txt_thick, cv2.LINE_AA)
        # cv2.LINE_AA = anti-aliased rendering for smooth text edges

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# HUD OVERLAY
# Draws a semi-transparent status bar at the top of the frame.
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray,
             fps: float,
             n_tracks: int,
             source_label: str) -> np.ndarray:
    """
    Draw a heads-up display (HUD) showing FPS, object count, and source name.
    Uses addWeighted() to blend a dark bar semi-transparently so the video
    behind it is still partially visible.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()  # copy needed for alpha blending trick below

    # ── Scale all HUD dimensions with frame resolution ───────────────────────
    scale_f    = max(1.0, w / 1280)
    font_scale = round(0.65 * scale_f, 2)
    thickness  = max(2, int(2 * scale_f))
    bar_h      = max(36, int(48 * scale_f))  # height of the dark top bar
    x_fps      = int(10  * scale_f)          # x position of FPS text
    x_obj      = int(220 * scale_f)          # x position of object count text
    x_src      = int(500 * scale_f)          # x position of source name text
    y_text     = int(bar_h * 0.68)           # vertical baseline for all text

    # ── Semi-transparent dark bar ─────────────────────────────────────────────
    # Draw solid dark rectangle on the overlay copy, then blend it 60/40
    # with the original frame → creates a translucent effect.
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    # addWeighted formula: dst = src1*alpha + src2*beta + gamma
    # overlay*0.6 + frame*0.4 = 60% dark bar, 40% original video showing through

    # ── Text elements ─────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:5.1f}",       (x_fps, y_text),
                FONT, font_scale, (0, 230, 0),    thickness, cv2.LINE_AA)  # green
    cv2.putText(frame, f"Objects: {n_tracks}",   (x_obj, y_text),
                FONT, font_scale, (0, 200, 255),  thickness, cv2.LINE_AA)  # cyan
    cv2.putText(frame, f"Source: {source_label}", (x_src, y_text),
                FONT, round(font_scale * 0.85, 2), (200, 200, 200),
                max(1, thickness - 1), cv2.LINE_AA)                         # grey
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# Ties everything together: open video → load model → process frames → display
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Resolve video source ──────────────────────────────────────────────────
    # If --source is a digit string (e.g. "0"), treat it as a webcam index.
    # Otherwise treat it as a file path string.
    source       = int(args.source) if args.source.isdigit() else args.source
    source_label = "Webcam" if isinstance(source, int) else Path(str(source)).name

    # ── Open video capture ────────────────────────────────────────────────────
    # cv2.VideoCapture works for webcam indices AND video file paths.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # native video width
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # native video height
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30          # native FPS (fallback 30)
    print(f"[INFO] Capture opened — {frame_w}×{frame_h} @ {src_fps:.1f} FPS")

    # ── Compute inference (processing) resolution ─────────────────────────────
    # We never exceed the native width (min guard), and keep aspect ratio intact.
    proc_w = min(args.proc_width, frame_w)
    proc_h = int(frame_h * proc_w / frame_w)
    print(f"[INFO] Processing at  — {proc_w}×{proc_h} (use --proc-width to change)")

    # ── Optional video writer ─────────────────────────────────────────────────
    # Only created if the user passes --save. Writes full-resolution annotated frames.
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # codec for .mp4 output
        writer = cv2.VideoWriter("output.mp4", fourcc, src_fps, (frame_w, frame_h))
        print("[INFO] Saving output to output.mp4")

    # ── Load YOLOv8 model ─────────────────────────────────────────────────────
    model = load_model(args.model)

    # ── Initialise DeepSORT tracker ───────────────────────────────────────────
    # DeepSORT combines two ideas:
    #   1. Kalman Filter  — predicts where each object WILL be next frame
    #                       (handles gaps when YOLO misses a detection)
    #   2. Appearance Embedding (MobileNet) — extracts a visual fingerprint
    #                       from each crop so the same person/car is re-identified
    #                       even after being briefly occluded.
    tracker = DeepSort(
        max_age  = MAX_AGE,  # delete track after this many frames without a match
        n_init   = 3,        # require 3 consecutive detections before confirming a track
                             # (filters out one-frame false positives)
        embedder = EMBEDDER, # MobileNet extracts appearance features from each crop
        half     = False,    # FP16 half-precision — set True if you have a CUDA GPU
        bgr      = True,     # our frames are BGR (OpenCV default, not RGB)
    )
    print("[INFO] DeepSORT tracker initialised")
    print("[INFO] Press  Q  to quit\n")

    # ── Create fullscreen window ──────────────────────────────────────────────
    # WINDOW_NORMAL allows the window to be resized/fullscreened.
    # WND_PROP_FULLSCREEN then expands it to cover the entire monitor.
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ── Frame counters and state ──────────────────────────────────────────────
    frame_idx         = 0      # total frames read so far
    fps_timer         = time.perf_counter()
    fps_smooth        = 0.0    # exponential moving average of FPS
    cached_detections = []     # last YOLO detections, reused on skipped frames

    # ── Main processing loop ──────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()   # read one frame from the video/webcam
        if not ret:
            # ret is False when the video ends or the webcam disconnects
            print("[INFO] End of stream.")
            break


        # ── Mirror Mode ───────────────────────────────────────────────────────
        # If using a webcam, flip the frame horizontally so it acts like a mirror.
        if isinstance(source, int):
            frame = cv2.flip(frame, 1)

      
        # ── FPS calculation (Exponential Moving Average) ──────────────────────
        # EMA smooths out frame-to-frame jitter in the displayed FPS number.
        # Formula: new_avg = 0.9 * old_avg + 0.1 * current_fps
        # Weight 0.9 means recent history is smoothed over ~10 frames.
        now        = time.perf_counter()
        elapsed    = now - fps_timer
        fps_timer  = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(elapsed, 1e-6))

        # ── Downscale frame for fast inference ────────────────────────────────
        # Running YOLO on 3840×2160 is ~16x slower than on 960×540.
        # INTER_LINEAR is a good balance between speed and quality for downscaling.
        small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

        # ── Detection: run YOLO every (skip+1) frames ─────────────────────────
        # On skipped frames we reuse the previous detection list.
        # DeepSORT's Kalman filter predicts box positions for the skipped frames,
        # so tracking remains smooth even without fresh detections.
        if args.skip == 0 or frame_idx % (args.skip + 1) == 0:
            cached_detections = detect_objects(small, model, args.conf, args.iou)

        # ── Tracking: update DeepSORT with latest (or cached) detections ──────
        # update_tracks() does three things internally:
        #   1. Runs MobileNet on each detection crop → appearance embedding
        #   2. Runs the Kalman filter to predict current positions of existing tracks
        #   3. Hungarian algorithm matches predictions to detections (minimises cost)
        tracks = tracker.update_tracks(cached_detections, frame=small)

        # ── Draw on the FULL-RESOLUTION frame ─────────────────────────────────
        # Compute scale factors: how much bigger is the original vs inference frame?
        # e.g. 3840/960 = 4.0 → every coordinate must be multiplied by 4
        sx    = frame_w / proc_w
        sy    = frame_h / proc_h
        frame = draw_tracks(frame, tracks, model.names, sx=sx, sy=sy)

        # Count only confirmed (active) tracks for the HUD display
        active_tracks = sum(1 for t in tracks if t.is_confirmed())
        frame = draw_hud(frame, fps_smooth, active_tracks, source_label)

        # ── Display and optionally save ───────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)  # render to the fullscreen window
        if writer:
            writer.write(frame)          # save full-res annotated frame to file

        # ── Keyboard quit ─────────────────────────────────────────────────────
        # waitKey(1) processes GUI events and waits 1 ms.
        # '& 0xFF' masks to the low byte (handles NumLock / modifier keys).
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
            break

        frame_idx += 1  # increment counter used by the skip logic

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()            # release the video file / webcam handle
    if writer:
        writer.release()     # flush and close the output video file
    cv2.destroyAllWindows()  # close the display window
    print("[INFO] Done.")


# Only run main() when this file is executed directly,
# not when it is imported as a module by another script.
if __name__ == "__main__":
    main()
