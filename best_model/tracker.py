# from typing import Optional

# import motmetrics as mm
# import numpy as np
# import torch
# import torch.nn.functional as F
# from filterpy.kalman import KalmanFilter
# from scipy.optimize import linear_sum_assignment
# from sympy import det
# from tracker.reid_extractor import ReIDExtractor
# from tracker.utils import (  # appearance_cost,
#     box_iou_matrix,
#     gallery_appearance_cost,
#     motion_cost,
# )


# class Tracker:
# 	"""
# 	The main tracking file, here is where magic happens.
# 	"""
# 	def __init__(self, obj_detect, reid_extractor: Optional[ReIDExtractor] = None):
# 		"""
# 		Initialization of the Tracker class.

# 		:param obj_detect: The object detector to be used for tracking.
# 		"""
# 		self.obj_detect = obj_detect
# 		self.reid_extractor = reid_extractor

# 		self.tracks: list[Track]= []
# 		self.track_num = 0
# 		self.im_index = 0
# 		self.results = {}

# 		self.mot_accum = None

# 	def reset(self, hard=True):
# 		"""
# 		Reset the tracker to its initial state (no tracks anymore).
		
# 		:param hard: If True, reset all internal variables.
# 		"""
# 		self.tracks = []

# 		if hard:
# 			self.track_num = 0
# 			self.results = {}
# 			self.im_index = 0

# 	def add(self, new_boxes, new_scores, new_embeds=None):
# 		"""
# 		Initializes new Track objects and saves them.
		
# 		:param new_boxes: The boxes for the new tracks.
# 		:param new_scores: The scores for the new tracks.
# 		:param new_embeds: The appearance embeddings for the new tracks.
# 		"""
# 		num_new = len(new_boxes)
# 		for i in range(num_new):
# 			self.tracks.append(Track(
# 				self.track_num + i,
# 				new_boxes[i],
# 				new_scores[i],
# 				new_embeds[i] if new_embeds is not None else None
# 			))
# 		self.track_num += num_new

# 	def get_pos(self):
# 		"""
# 		Get the positions of all active tracks.
# 		"""
# 		if len(self.tracks) == 1:
# 			box = self.tracks[0].box
# 		elif len(self.tracks) > 1:
# 			box = torch.stack([t.box for t in self.tracks], 0)
# 		else:
# 			box = torch.zeros(0).cuda()
# 		return box

# 	def data_association(
# 		self,
# 		frame: dict, 
# 		boxes: torch.Tensor, 
# 		scores: torch.Tensor, 
# 		alpha: float = 0.4,
# 		beta: float = 0.4,
# 		gamma: float = 0.2,
# 		iou_thresh: float = 0.3
# 	):
# 		"""
# 		Perform data association between current detections and existing tracks.
		
# 		:param self: Beschreibung
# 		:param dict frame: A dict containing the image and additional information.
# 		:param torch.Tensor boxes: The detected boxes
# 		:param torch.Tensor scores: The detection scores.
# 		:param float alpha: The influence of IoU in the association cost.
# 		:param float beta: The influence of appearance in the association cost.
# 		:param float gamma: The influence of motion in the association cost.
# 		:param float iou_thresh: The IoU threshold for matching, prohibits physically impossible matches.
# 		"""
# 		if len(self.tracks) == 0:
# 			if self.reid_extractor: # if ReID extractor is available, use it
# 				embeds = self.reid_extractor.extract_embeddings(frame, boxes)
# 			else: # else use simple average pooling embeddings
# 				embeds = self.extract_embeddings(frame, boxes)		

# 			self.add(boxes, scores, embeds)
# 			return

# 		# Predict track positions
# 		for t in self.tracks:
# 			t.predict()

# 		track_boxes = torch.stack([t.bbox for t in self.tracks])
# 		# det_embeds = self.extract_embeddings(frame, boxes)
# 		if self.reid_extractor:
# 			det_embeds = self.reid_extractor.extract_embeddings(frame, boxes)
# 		else:
# 			det_embeds = self.extract_embeddings(frame, boxes)
# 		# det_embeds: (box_count, emb_dim)
# 		track_embeds = []
# 		for t in self.tracks:
# 			if len(t.gallery) > 0:
# 				track_embeds.append(torch.stack(t.gallery).mean(dim=0))
# 			else:
# 				track_embeds.append(torch.zeros_like(det_embeds[0]))
# 		# track_embeds = torch.stack([t.appearance for t in self.tracks])

# 		iou = box_iou_matrix(track_boxes, boxes)
# 		# app = appearance_cost(track_embeds, det_embeds)
# 		app = gallery_appearance_cost(self.tracks, det_embeds)
# 		mot = motion_cost(self.tracks, boxes)

# 		mot = torch.exp(-0.5 * mot)

# 		# print("iou device:", iou.device if torch.is_tensor(iou) else "numpy")
# 		# print("app device:", app.device)
# 		# print("mot device:", mot.device if torch.is_tensor(mot) else "numpy")

# 		app = app.to(iou.device)

# 		# the higher the better
# 		# association = alpha * iou + beta * app + gamma * mot
		
# 		association = alpha * iou + beta * (1 - app) + gamma * mot

# 		cost = 1 - association
# 		cost = cost.cpu().numpy()
# 		# print("Cost: ", cost)
# 		row_idx, col_idx = linear_sum_assignment(cost)

# 		matched_tracks = set()
# 		matched_dets = set()

# 		for r, c in zip(row_idx, col_idx):
# 			if iou[r, c] < iou_thresh: # avoid physically impossible matches by introducing IoU threshold
# 				continue
# 			self.tracks[r].update(boxes[c], scores[c], det_embeds[c])
# 			matched_tracks.add(r)
# 			matched_dets.add(c)

# 		# for all non-matched detections, create new tracks
# 		for i in range(len(boxes)):
# 			if i not in matched_dets:
# 				self.tracks.append(
# 					Track(self.track_num, boxes[i], scores[i], det_embeds[i])
# 				)
# 				self.track_num += 1

# 		# Optional: remove dead tracks
# 		self.tracks = [t for i, t in enumerate(self.tracks)
# 					if i in matched_tracks or t.time_since_update < 5]
		
# 		# print(f"Frame {self.im_index}: {len(matched_dets)} matches, {len(boxes) - len(matched_dets)} new tracks, {len(self.tracks)} total tracks.")

	
# 	# def extract_embeddings(self, frame: dict, boxes: torch.Tensor) -> torch.Tensor:
# 	# 	"""
# 	# 	Very simple appearance embedding. Uses adaptive average pooling on the cropped image region.

# 	# 	:param dict frame: The frame blob containing the image information.
# 	# 	:param torch.Tensor boxes: The detected boxes.

# 	# 	:return torch.Tensor: The extracted embeddings.
# 	# 	"""
# 	# 	embeds = []
# 	# 	img = frame['img'] # comes with batch dimension: [1, C, H, W]

# 	# 	# print(f"Image size: {img.shape}")

# 	# 	for box in boxes:
# 	# 		x1, y1, x2, y2 = box.int()
# 	# 		crop = img[:, :, y1:y2, x1:x2] # cut in height and width dimensions
# 	# 		if crop.numel() == 0: # if empty crop, return zero embedding
# 	# 			embeds.append(torch.zeros(128, device=box.device))
# 	# 		else: # if non-empty crop, return average pooled embedding
# 	# 			embeds.append(F.adaptive_avg_pool2d(crop, (1, 1)).flatten())
# 	# 			# embedding = self.reid_extractor.extract_embeddings(frame, box.unsqueeze(0))
# 	# 			# embeds.append(embedding.squeeze(0))
# 	# 		# print(f"Crop size: {crop.shape}, Embed size: {embeds[-1].shape}")
# 	# 	return torch.stack(embeds)

# 	def extract_embeddings(self, frame: dict, boxes: torch.Tensor) -> torch.Tensor:
# 		"""
# 		Very simple appearance embedding. Uses adaptive average pooling on the cropped image region.

# 		:param dict frame: The frame blob containing the image information.
# 		:param torch.Tensor boxes: The detected boxes.

# 		:return torch.Tensor: The extracted embeddings.
# 		"""
# 		img = frame['img']  # [1, C, H, W]
# 		device = img.device

# 		if boxes is None or len(boxes) == 0:
# 			return torch.empty((0, img.shape[1]), device=device)

# 		embeds = []
# 		for box in boxes:
# 			x1, y1, x2, y2 = box.int()
# 			crop = img[:, :, y1:y2, x1:x2]
# 			if crop.numel() == 0:
# 				embeds.append(torch.zeros(img.shape[1], device=device))
# 			else:
# 				embeds.append(F.adaptive_avg_pool2d(crop, (1,1)).flatten())

# 		return torch.stack(embeds)

# 	def step(self, frame):
# 		"""
# 		This function should be called every timestep to perform simple tracking with a blob containing the image information.

# 		:param frame: The frame blob containing the image information.
# 		"""
# 		# object detection
# 		boxes, scores = self.obj_detect.detect(frame['img'])

# 		self.data_association(frame, boxes, scores)

# 		# results
# 		for t in self.tracks:
# 			if t.id not in self.results.keys():
# 				self.results[t.id] = {}
# 			self.results[t.id][self.im_index] = np.concatenate([t.bbox.cpu().numpy(), np.array([t.score])])

# 		self.im_index += 1

# 	def step_sahi(self, frame, tile_size: int = 640, overlap: float = 0.2, score_thresh: float = 0.5, cross_tile_iou: float = 0.5):
# 		"""
# 		This function should be called every timestep to perform SAHI-based tracking with a blob containing the image information.

# 		:param frame: The frame blob containing the image information.:
# 		:param int tile_size: The width/height of each square tile.
# 		:param float overlap: Fraction of overlap between tiles (0–1). Prevents objects from being cut at tile edges.
# 		:param float score_thresh: Minimum detection score per tile.
# 		:param float cross_tile_iou: NMS threshold for merging boxes across tiles.
# 		"""
# 		# object detection
# 		boxes, scores = self.obj_detect.detect_sahi(frame['img'], tile_size, overlap, score_thresh, cross_tile_iou)

# 		self.data_association(frame, boxes, scores)

# 		# results
# 		for t in self.tracks:
# 			if t.id not in self.results.keys():
# 				self.results[t.id] = {}
# 			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

# 		self.im_index += 1

# 	def get_results(self):
# 		"""
# 		Return the tracking results.
# 		"""
# 		return self.results


# class Track(object):
# 	"""
# 	Class that contains all necessary information for every individual track.
# 	"""
# 	def __init__(self, track_id, box, score, embedding=None, max_gallery=10):
# 		self.id = track_id
# 		self.score = score
# 		self.time_since_update = 0
# 		self.age = 1

# 		# self.embedding = embedding  # ReID embedding (D,)
# 		self.gallery = []
# 		if embedding is not None:
# 			self.gallery.append(embedding)
# 		self.max_gallery = max_gallery
		
# 		self.kf = KalmanFilter(dim_x=7, dim_z=4)
# 		self.kf.F = np.array([
#             [1,0,0,0,1,0,0],
#             [0,1,0,0,0,1,0],
#             [0,0,1,0,0,0,1],
#             [0,0,0,1,0,0,0],
#             [0,0,0,0,1,0,0],
#             [0,0,0,0,0,1,0],
#             [0,0,0,0,0,0,1]
#         ])
# 		self.kf.H = np.array([
#             [1,0,0,0,0,0,0],
#             [0,1,0,0,0,0,0],
#             [0,0,1,0,0,0,0],
#             [0,0,0,1,0,0,0]
#         ])
		
# 		self.kf.P[4:, 4:] *= 1000.
# 		self.kf.P *= 10.
# 		self.kf.R[2:, 2:] *= 10.
# 		self.kf.Q[4:, 4:] *= 0.01
# 		self.kf.x[:4] = self.box_to_z(box)
	
# 	@property
# 	def bbox(self):
# 		"""Return the current bounding box in [x1, y1, x2, y2] format."""
# 		return self.z_to_box(self.kf.x)  # convert state to bounding box format

# 	def box_to_z(self, box):
# 		"""
# 		Convert bounding box [x1, y1, x2, y2] to Kalman filter state [x, y, s, r] where x,y is the center of the box, s is the scale/area, and r is the aspect ratio.

# 		:param box: The bounding box.

# 		:return np.array: The converted state.
# 		"""
# 		w = max(box[2] - box[0], 1e-6)
# 		h = max(box[3] - box[1], 1e-6)
# 		x = box[0] + w/2.
# 		y = box[1] + h/2.
# 		s = w * h
# 		r = w / h
# 		return np.array([x, y, s, r]).reshape((4, 1))

# 	def z_to_box(self, x) -> torch.Tensor:
# 		"""
# 		Convert Kalman filter state [x, y, s, r] to bounding box [x1, y1, x2, y2].

# 		:param x: The Kalman filter state.

# 		:return torch.Tensor: The converted bounding box.
# 		"""
# 		s = max(x[2].item(), 1e-6)  # clamp area to small positive to avoid NaNs
# 		r = max(x[3].item(), 1e-6)  # clamp aspect ratio to small positive to avoid NaNs
# 		w = np.sqrt(s * r)
# 		h = s / w

# 		box_np = np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.], dtype=np.float32)

# 		return torch.from_numpy(box_np).reshape((4))

# 	def update_embedding(self, new_embedding):
# 		"""
# 		Update the track's appearance embedding.

# 		:param new_embedding: The new appearance embedding.
# 		"""
# 		# self.embedding = new_embedding
# 		if new_embedding is not None:
# 			self.gallery.append(new_embedding)
# 			if len(self.gallery) > self.max_gallery:
# 				self.gallery.pop(0)  # remove oldest

# 	def predict(self):
# 		"""
# 		Predict the new position of the track using Kalman filter.
# 		"""
# 		self.kf.predict()
# 		self.age += 1
# 		self.time_since_update += 1

# 	def update(self, box, score, embedding):
# 		"""
# 		Update the track with a new detection.

# 		:param box: The new bounding box.
# 		:param score: The new detection score.
# 		:param appearance: The new appearance embedding.
# 		"""
# 		self.kf.update(self.box_to_z(box))
# 		self.score = float(score)
# 		# self.embedding = embedding
# 		self.time_since_update = 0

from collections import deque

import motmetrics as mm
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# =========================
# Tracker
# =========================

class Tracker:
    def __init__(self, obj_detect):
        self.obj_detect = obj_detect
        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        # --- ByteTrack parameters ---
        self.max_age = 30
        self.min_hits = 2
        self.high_thr = 0.6
        self.low_thr = 0.1

        # --- ReID model ---
        resnet = models.resnet18(pretrained=True)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.encoder.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

        self.reid_transform = T.Compose([
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def reset(self, hard=True):
        self.tracks = []
        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    # -------------------------
    # Feature extraction
    # -------------------------
    def _get_features(self, frame_img, boxes):
        if len(boxes) == 0:
            return None

        img_tensor = frame_img.clone().detach()
        if img_tensor.ndim == 4:
            img_tensor = img_tensor.squeeze(0)

        pil_img = T.ToPILImage()(img_tensor.cpu())
        features = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                features.append(torch.zeros(512))
                continue

            crop = pil_img.crop((x1, y1, x2, y2))

            with torch.no_grad():
                feat = self.encoder(
                    self.reid_transform(crop)
                    .unsqueeze(0)
                    .to(self.device)
                )

            features.append(feat.cpu().view(-1))

        return torch.stack(features)

    # -------------------------
    # Data association
    # -------------------------
    def data_association(self, boxes, scores, frame_img):
        for t in self.tracks:
            t.predict()

        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
        inds_high = scores_np >= self.high_thr
        inds_low = (scores_np > self.low_thr) & (scores_np < self.high_thr)

        boxes_high = boxes[inds_high]
        scores_high = scores[inds_high]
        features_high = self._get_features(frame_img, boxes_high)

        boxes_low = boxes[inds_low]
        scores_low = scores[inds_low]

        matched_tracks_idx = []
        assigned_det_high = []

        # ---------- High confidence association (IoU + ReID) ----------
        if self.tracks and features_high is not None:
            tr_boxes = np.stack([t.get_mot_box() for t in self.tracks])
            det_boxes_high = np.stack([
                np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]])
                for b in boxes_high.cpu().numpy()
            ])

            tr_feats = torch.stack([t.get_gallery_feature() for t in self.tracks])

            iou_d = mm.distances.iou_matrix(
                tr_boxes, det_boxes_high, max_iou=0.5
            )

            dot = torch.mm(tr_feats, features_high.t())
            norm_tr = torch.norm(tr_feats, dim=1, keepdim=True)
            norm_det = torch.norm(features_high, dim=1, keepdim=True)
            sim = dot / (torch.mm(norm_tr, norm_det.t()) + 1e-6)

            app_d = np.nan_to_num((1.0 - sim.numpy()), nan=1.0)

            cost = 0.4 * iou_d + 0.6 * app_d
            cost = np.nan_to_num(cost, nan=1000.0)

            row, col = linear_sum_assignment(cost)

            for r, c in zip(row, col):
                if cost[r, c] < 0.6:
                    self.tracks[r].update(
                        boxes_high[c],
                        features_high[c],
                        scores_high[c].item()
                    )
                    matched_tracks_idx.append(r)
                    assigned_det_high.append(c)

        # ---------- Low confidence association (IoU only) ----------
        unmatched_tracks_idx = [
            i for i in range(len(self.tracks))
            if i not in matched_tracks_idx
        ]

        if unmatched_tracks_idx and len(boxes_low) > 0:
            tr_boxes_low = np.stack([
                self.tracks[i].get_mot_box()
                for i in unmatched_tracks_idx
            ])
            det_boxes_low = np.stack([
                np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]])
                for b in boxes_low.cpu().numpy()
            ])

            iou_d_low = mm.distances.iou_matrix(
                tr_boxes_low, det_boxes_low, max_iou=0.5
            )
            iou_d_low = np.nan_to_num(iou_d_low, nan=1.0)

            row_l, col_l = linear_sum_assignment(iou_d_low)

            for r, c in zip(row_l, col_l):
                if iou_d_low[r, c] < 0.7:
                    real_idx = unmatched_tracks_idx[r]
                    self.tracks[real_idx].update(
                        boxes_low[c], None, scores_low[c].item()
                    )
                    matched_tracks_idx.append(real_idx)

        # ---------- Track management ----------
        self.tracks = [
            t for i, t in enumerate(self.tracks)
            if i in matched_tracks_idx or t.time_since_update < self.max_age
        ]

        unassigned_high = [
            i for i in range(len(boxes_high))
            if i not in assigned_det_high
        ]

        for i in unassigned_high:
            self.add(boxes_high[i], scores_high[i], features_high[i])

    def add(self, box, score, feat):
        self.tracks.append(
            Track(box, score, feat, self.track_num)
        )
        self.track_num += 1

    # -------------------------
    # Step
    # -------------------------
    def step(self, frame):
        boxes, scores = self.obj_detect.detect(frame["img"])
        self.data_association(boxes, scores, frame["img"])

        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                if t.id not in self.results:
                    self.results[t.id] = {}
                self.results[t.id][self.im_index] = np.concatenate(
                    [t.box.numpy(), [t.score]]
                )

        self.im_index += 1

    # -------------------------
    # Results
    # -------------------------
    def get_results(self):
        results_final = {}

        for t_id, frames in self.results.items():
            if len(frames) < 2:
                results_final[t_id] = frames
                continue

            f_indices = sorted(frames.keys())
            new_frames = frames.copy()

            for i in range(len(f_indices) - 1):
                f1, f2 = f_indices[i], f_indices[i + 1]
                gap = f2 - f1

                if 1 < gap <= 10:
                    b1, b2 = frames[f1][:4], frames[f2][:4]
                    s1, s2 = frames[f1][4], frames[f2][4]

                    for step in range(1, gap):
                        alpha = step / gap
                        b_interp = (1 - alpha) * b1 + alpha * b2
                        s_interp = (1 - alpha) * s1 + alpha * s2
                        new_frames[f1 + step] = np.concatenate(
                            [b_interp, [s_interp]]
                        )

            results_final[t_id] = new_frames

        return results_final


# =========================
# Track
# =========================

class Track:
    def __init__(self, box, score, features, track_id, gallery_size=30):
        self.id = track_id
        self.score = score
        self.time_since_update = 0
        self.hits = 1

        # ---- ReID gallery ----
        self.gallery = deque(maxlen=gallery_size)
        if features is not None:
            self.gallery.append(features.clone())

        self.features = features.clone() if features is not None else None

        # ---- Kalman Filter ----
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.base_R = self.kf.R.copy()
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

        self.kf.x[:4] = self.box_to_z(box)
        self.box = box

    # -------------------------
    # Appearance
    # -------------------------
    def get_gallery_feature(self):
        if len(self.gallery) == 0:
            return self.features
        return torch.stack(list(self.gallery)).mean(dim=0)

    # -------------------------
    # Geometry
    # -------------------------
    def box_to_z(self, b):
        w = max(1e-2, b[2] - b[0])
        h = max(1e-2, b[3] - b[1])
        return np.array([
            b[0] + w / 2.,
            b[1] + h / 2.,
            w * h,
            w / h
        ]).reshape((4, 1))

    def z_to_box(self, x):
        w = np.sqrt(np.maximum(0, x[2] * x[3]))
        h = x[2] / np.maximum(1e-6, w)
        return torch.tensor([
            x[0] - w / 2.,
            x[1] - h / 2.,
            x[0] + w / 2.,
            x[1] + h / 2.
        ], dtype=torch.float32).flatten()

    def get_mot_box(self):
        b = self.box.numpy()
        return np.array([
            b[0],
            b[1],
            max(1e-2, b[2] - b[0]),
            max(1e-2, b[3] - b[1])
        ])

    # -------------------------
    # Kalman
    # -------------------------
    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        self.box = self.z_to_box(self.kf.x)

    def update(self, box, feat, score):
        self.time_since_update = 0
        self.hits += 1
        self.score = score

        scale = (1.0 - score)
        self.kf.R = self.base_R * scale
        self.kf.update(self.box_to_z(box))
        self.box = self.z_to_box(self.kf.x)

        if feat is not None:
            self.gallery.append(feat.clone())
            if self.features is None:
                self.features = feat.clone()
            else:
                self.features = 0.9 * self.features + 0.1 * feat
