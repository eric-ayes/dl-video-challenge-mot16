"""
FastAPI service for the MOT16 multi-object tracking pipeline.

Accepts a video file and returns per-frame tracking results in JSON.
The pipeline runs Faster R-CNN detection + ByteTrack + NSA Kalman + ResNet-18 Re-ID.
"""

import io
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# Make best_model importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))
from best_model.object_detector import FRCNN_FPN
from best_model.tracker import Tracker

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MOT16 Tracking API",
    description=(
        "Multi-Object Tracking pipeline: Faster R-CNN (ResNet-50 + FPN) detection, "
        "ByteTrack two-stage association, NSA Kalman filtering, and ResNet-18 Re-ID. "
        "Achieves MOTA 68.8% on MOT16 (up from 26.1% baseline)."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_detector: FRCNN_FPN | None = None


def get_detector() -> FRCNN_FPN:
    global _detector
    if _detector is None:
        _detector = FRCNN_FPN(num_classes=2)
        _detector.eval()
        _detector.to(DEVICE)
    return _detector


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class Detection(BaseModel):
    track_id: int
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class TrackingResponse(BaseModel):
    video_name: str
    num_frames: int
    num_tracks: int
    processing_time_s: float
    detections: List[Detection]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_TRANSFORMS = T.Compose([T.ToTensor()])


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR numpy frame (from OpenCV) to normalised RGB tensor."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return IMG_TRANSFORMS(pil_img).unsqueeze(0)  # [1, C, H, W]


def run_tracker(video_path: str) -> dict:
    """Run full tracking pipeline on a video file. Returns results dict."""
    detector = get_detector()
    tracker = Tracker(obj_detect=detector)
    tracker.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    num_frames = 0
    t0 = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        img_tensor = frame_to_tensor(frame_bgr).to(DEVICE)
        blob = {"img": img_tensor}

        with torch.no_grad():
            tracker.step(blob)

        num_frames += 1

    cap.release()
    elapsed = time.perf_counter() - t0

    raw_results = tracker.get_results()  # {track_id: {frame_idx: [x1,y1,x2,y2,score]}}

    detections: List[Detection] = []
    for track_id, frames in raw_results.items():
        for frame_idx, bbox_score in frames.items():
            detections.append(
                Detection(
                    track_id=int(track_id),
                    frame=int(frame_idx),
                    x1=float(bbox_score[0]),
                    y1=float(bbox_score[1]),
                    x2=float(bbox_score[2]),
                    y2=float(bbox_score[3]),
                    score=float(bbox_score[4]),
                )
            )

    detections.sort(key=lambda d: (d.frame, d.track_id))

    return {
        "num_frames": num_frames,
        "num_tracks": len(raw_results),
        "elapsed": elapsed,
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/track", response_model=TrackingResponse)
async def track_video(file: UploadFile = File(...)):
    """
    Upload a video and receive per-frame multi-object tracking results.

    - Accepted formats: mp4, avi, mov, mkv
    - Returns one detection entry per (track_id, frame) pair
    """
    allowed_suffixes = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Use one of {sorted(allowed_suffixes)}.",
        )

    # Write upload to a temp file so OpenCV can open it
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = run_tracker(tmp_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return TrackingResponse(
        video_name=file.filename or "unknown",
        num_frames=result["num_frames"],
        num_tracks=result["num_tracks"],
        processing_time_s=round(result["elapsed"], 3),
        detections=result["detections"],
    )


@app.get("/")
def root():
    return {
        "message": "MOT16 Tracking API — POST a video to /track",
        "docs": "/docs",
    }
