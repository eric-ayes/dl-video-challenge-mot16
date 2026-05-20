# Video Challenge: Multi-Object Tracking on MOT16

Master's course project — Deep Learning for Video Processing (UAM)  
Authors: Eric Ayestaran Guillorme · Marcel Hofmann · Unax Murua Urizarbarrena

## Overview

Multi-object tracking (MOT) system developed for the MOT16 benchmark. Starting from a baseline tracker (ResNet50 + FPN), we designed and implemented a full tracking pipeline combining ByteTrack, NSA Kalman Filtering, Re-Identification, and linear interpolation post-processing — improving MOTA from **26.1% to 68.8%**.

## Results

| | MOTA ↑ | MOTP ↑ | IDF1 ↑ | Precision ↑ | Recall ↑ | FP ↓ | FN ↓ |
|--|--|--|--|--|--|--|--|
| **Baseline** | 26.1% | 0.111 | 47.2% | 66.9% | 52.4% | 29,113 | 53,405 |
| **Ours** | **68.8%** | **0.109** | **67.1%** | **94.6%** | **73.9%** | **4,707** | **29,275** |

## Demo Videos (MOT16-09)

| Version | Video |
|---------|-------|
| Best approach | https://youtu.be/4KkDkTVgfUA |
| Finetuned model | https://youtu.be/i-wtiM7GUUw |

## Dataset — MOT16

- 14 videos (7 train / 7 test), 11,235 images total
- ~300,000 bounding boxes across ~1,300 pedestrian tracks
- Street-level pedestrian sequences with occlusion and crowding

## Architecture

### Detection
Baseline: Faster R-CNN with ResNet50 backbone and Feature Pyramid Network (FPN).

Attempted detection improvements (discarded):
- Fine-tuning on CrowdHuman dataset
- Sliding window approach (too slow, negligible gain)
- Hyperparameter tuning (confidence thresholds, proposal count)

The main gains came from the **tracking pipeline**, not detection.

### Tracking Pipeline

**1. ByteTrack**  
Instead of discarding low-confidence detections (< 0.6), ByteTrack retains detections with scores between 0.1 and 0.6 to maintain existing tracks during occlusion. This prevents trajectory fragmentation when people walk behind obstacles.

**2. NSA Kalman Filter (Noise Scale Association)**  
Extends the standard Kalman Filter by scaling uncertainty with detector confidence `(1.0 − score)`. When the detector is unsure, the system relies more on the object's predicted momentum from previous frames.

**3. Re-Identification (ResNet-18)**  
To maintain consistent IDs after occlusion, Re-ID embeddings from a ResNet-18 model are combined with spatial IoU in a weighted cost function:

```
cost = 0.4 × IoU + 0.6 × AppearanceDistance
```

Appearance distance (60% weight) is prioritized over spatial overlap as it is more reliable for identity matching.

**4. Post-Processing: Linear Interpolation**  
If a track ID disappears and reappears within 10 frames, missing bounding boxes are filled using linear interpolation of intermediate coordinates, recovering detections lost during brief occlusions.

## Tech Stack

- Python · PyTorch
- Faster R-CNN + FPN (torchvision)
- ByteTrack · Kalman Filter · ResNet-18 Re-ID
- MOT16 benchmark · py-motmetrics
- Google Colab

## Authors

Eric Ayestaran, Marcel Hofmann, Unax Murua  
MSc Deep Learning in Audio, Video and Image Signal Processing, UAM
