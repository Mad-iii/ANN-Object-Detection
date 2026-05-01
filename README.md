# 🎯 Real-Time Object Detection & Tracking
### YOLOv8 + DeepSORT + OpenCV

---

## 📁 Project Structure

```
object_tracker/
├── main.py            ← Main script
├── requirements.txt   ← Dependencies
└── README.md          ← This file
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with webcam
python main.py --source 0 --skip 2 --conf 0.5

# 3. Run with a video file
python main.py --source source.mp4 --model yolov8n.pt --conf 0.5 --skip 2

# 4. Save output to output.mp4
python main.py --source video.mp4 --save
```

---

## 🎛️ All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | `0` = webcam, or path to video file |
| `--model`  | `yolov8n.pt` | YOLO weights (`n/s/m/l/x`) |
| `--conf`   | `0.5` | Detection confidence threshold |
| `--iou`    | `0.45` | NMS IoU threshold |
| `--skip`   | `0` | Skip N frames between detections |
| `--save`   | off | Save output to `output.mp4` |

---

## 🏃 Speed vs Accuracy Trade-offs

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------|-------------|-------------|----------|
| `yolov8n.pt` | 6 MB | ~15 FPS | ~100+ FPS | Good |
| `yolov8s.pt` | 22 MB | ~8 FPS | ~80 FPS | Better |
| `yolov8m.pt` | 52 MB | ~4 FPS | ~60 FPS | Best |

---

## 🔑 Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |

---

## 📌 Notes

- YOLOv8 weights download automatically on first run (~6 MB for nano)
- DeepSORT's MobileNet embedder also downloads on first run
- Output saved as `output.mp4` in the project folder when `--save` is used
