# MOT16 Multi-Object Tracking Pipeline

**MOTA 26.1% → 68.8%** — ByteTrack · NSA Kalman · ResNet-18 Re-ID · FastAPI

Master's project — Deep Learning for Video Processing (UAM)  
Eric Ayestaran · Marcel Hofmann · Unax Murua

---

## Results on MOT16

| | MOTA ↑ | MOTP ↑ | IDF1 ↑ | Precision ↑ | Recall ↑ | FP ↓ | FN ↓ |
|---|---|---|---|---|---|---|---|
| Baseline (FRCNN only) | 26.1% | 0.111 | 47.2% | 66.9% | 52.4% | 29,113 | 53,405 |
| **Ours** | **68.8%** | **0.109** | **67.1%** | **94.6%** | **73.9%** | **4,707** | **29,275** |

Demo — sequence MOT16-09: [best approach](https://youtu.be/4KkDkTVgfUA) · [finetuned model](https://youtu.be/i-wtiM7GUUw)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Input Video / Frames                     │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Faster R-CNN + FPN   │  ResNet-50 backbone
                    │  (person detection)   │  NMS thresh 0.5
                    └───────────┬───────────┘
                                │  boxes + scores
              ┌─────────────────▼──────────────────┐
              │          ByteTrack splitter         │
              │   high conf ≥ 0.6 │ low 0.1–0.6    │
              └──────┬──────────────────────┬───────┘
                     │                      │
        ┌────────────▼──────┐   ┌───────────▼───────────┐
        │  NSA Kalman +     │   │   IoU-only association │
        │  ResNet-18 Re-ID  │   │   (unmatched tracks)  │
        │  cost = 0.4·IoU   │   └───────────────────────┘
        │       + 0.6·App   │
        └────────────┬──────┘
                     │  matched / new tracks
        ┌────────────▼──────────────────────┐
        │  Linear interpolation (≤10 frames) │
        └────────────┬──────────────────────┘
                     │
        ┌────────────▼──────────────────────┐
        │  Tracking results  {id: {frame:   │
        │  [x1,y1,x2,y2,score]}}            │
        └───────────────────────────────────┘
```

### Key components

| Component | Detail |
|---|---|
| **Detection** | Faster R-CNN, ResNet-50 + FPN backbone, pretrained on COCO |
| **ByteTrack** | Two-stage association — retains low-confidence detections to survive occlusions |
| **NSA Kalman** | Noise-scaled by detector confidence `R ∝ (1 − score)`: uncertain detections increase filter uncertainty |
| **Re-ID** | ResNet-18 gallery embeddings (size 30); cosine distance weighted 60% vs IoU 40% |
| **Interpolation** | Fills gaps ≤ 10 frames with linearly interpolated bounding boxes |

---

## Running with Docker

### Build

```bash
docker build -t mot16-tracker .
```

### Run the API server

```bash
docker run --rm -p 8000:8000 mot16-tracker
```

GPU support (requires NVIDIA Container Toolkit):

```bash
docker run --rm --gpus all -p 8000:8000 mot16-tracker
```

### Check the service is up

```bash
curl http://localhost:8000/health
# {"status":"ok","device":"cuda"}
```

Interactive API docs: http://localhost:8000/docs

---

## API Reference

### `POST /track`

Upload a video and receive per-frame tracking results.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | file | Video file (`.mp4`, `.avi`, `.mov`, `.mkv`) |

**Response** — `application/json`

```json
{
  "video_name": "clip.mp4",
  "num_frames": 450,
  "num_tracks": 12,
  "processing_time_s": 38.2,
  "detections": [
    {
      "track_id": 0,
      "frame": 0,
      "x1": 412.3,
      "y1": 201.7,
      "x2": 478.1,
      "y2": 389.5,
      "score": 0.94
    }
  ]
}
```

**Example with curl**

```bash
curl -X POST http://localhost:8000/track \
     -F "file=@your_video.mp4" | python -m json.tool
```

**Example with Python**

```python
import requests

with open("your_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/track",
        files={"file": ("video.mp4", f, "video/mp4")},
    )

data = response.json()
print(f"Tracked {data['num_tracks']} objects across {data['num_frames']} frames")
for det in data["detections"][:5]:
    print(det)
```

---

## Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn api:app --reload --port 8000
```

---

## Repository Structure

```
.
├── api.py                  # FastAPI application
├── Dockerfile              # Container definition
├── requirements.txt        # Pinned dependencies
├── best_model/
│   ├── tracker.py          # ByteTrack + NSA Kalman + Re-ID implementation
│   ├── object_detector.py  # Faster R-CNN wrapper
│   ├── data_track.py       # MOT16 dataset loader
│   └── utils.py            # Evaluation & visualisation helpers
└── other_files/
    ├── model_exec.ipynb    # Training / evaluation notebook
    ├── finetuning_crowdhuman.ipynb
    ├── mot_eval.py         # Standalone MOT metrics script
    └── make_video.py       # Result visualisation
```

---

## Dataset — MOT16

| | Value |
|---|---|
| Sequences | 14 (7 train / 7 test) |
| Total frames | 11,235 |
| Bounding boxes | ~300,000 |
| Pedestrian tracks | ~1,300 |
| Challenge | Occlusion, crowding, varying density |

---

## Tech Stack

Python · PyTorch · torchvision · FastAPI · OpenCV  
Faster R-CNN · ByteTrack · Kalman Filter · ResNet-18 Re-ID · py-motmetrics

---

## Authors

Eric Ayestaran · Marcel Hofmann · Unax Murua  
MSc Deep Learning in Audio, Video and Image Signal Processing — UAM
