#########################################
# Still ugly file with helper functions #
#########################################

import os
import random
from collections import defaultdict
from os import path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import torch
from cycler import cycler as cy
from torchvision.transforms import functional as F
from tqdm.auto import tqdm

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

def plot_sequence(tracks, db, first_n_frames=None):
    """
    Plots a whole sequence.

    :param dict tracks: The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
    :param torch.utils.data.Dataset db: The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
    :param int first_n_frames: If given, only the first n frames are plotted.
    """

    # print("[*] Plotting whole sequence to {}".format(output_dir))

    # if not osp.exists(output_dir):
    # 	os.makedirs(output_dir)

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        img = v['img'].mul(255).permute(1, 2, 0).byte().numpy()
        width, height, _ = img.shape

        dpi = 96
        fig, ax = plt.subplots(1, dpi=dpi)
        fig.set_size_inches(width / dpi, height / dpi)
        ax.set_axis_off()
        ax.imshow(img)

        for j, t in tracks.items():
            if i in t.keys():
                t_i = t[i]
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=1.0, **styles[j]
                    ))

                ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                            color=styles[j]['ec'], weight='bold', fontsize=6, ha='center', va='center')

        plt.axis('off')
        # plt.tight_layout()
        plt.show()
        # plt.savefig(im_output, dpi=100)
        # plt.close()

        if first_n_frames is not None and first_n_frames == i:
            break


def get_mot_accum(results, seq):
    """
    Get the MOTAccumulator for a given sequence and tracking results.

    :param results: The tracking results in the form {track_id: {frame: np.array([x1,y1,x2,y2,score]), ...}, ...}
    :param seq: The sequence data in the form of a MOT_Sequence object.
    """
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    """
    Evaluate multiple MOTAccumulators and print the results.

    :param accums: List of MOTAccumulators to evaluate.
    :param names: List of names corresponding to the accumulators.
    :param generate_overall: If True, generate overall summary across all accumulators.
    """
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(str_summary)


def evaluate_obj_detect(model, data_loader):
    """
    Evaluate an object detection model on a given data loader.

    :param model: The object detection model to evaluate.
    :param data_loader: The data loader providing the evaluation data.
    """

    model.eval()
    device = list(model.parameters())[0].device
    results = {}
    for imgs, targets in tqdm(data_loader):
        imgs = [img.to(device) for img in imgs]

        with torch.inference_mode(): # torch.no_grad()
            preds = model(imgs)

        for pred, target in zip(preds, targets):
            results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(),
                                                  'scores': pred['scores'].cpu()}

    data_loader.dataset.print_eval(results)


def _flip_coco_person_keypoints(kps, width):
    """
    Flip person keypoints from COCO format.

    :param kps: Keypoints to flip.
    :param width: Width of the image.
    """
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    """
    Class for composing several transforms together.
    """
    def __init__(self, transforms):
        """
        Initialization of the Compose object.
        
        :param transforms: List of transformations to compose."""
        self.transforms = transforms

    def __call__(self, image, target):
        """
        Call the composed transformations on the image and target.

        :param image: The image to transform.
        :param target: The target to transform.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def obj_detect_transforms(train: bool) -> Compose:
    """
    Get the transformations for the object detector.
    
    :param bool train: If True, return the training transformations, else the evaluation ones.

    :return Compose: A `Compose` object containing the transformations.
    """

    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def box_iou_matrix(boxes1, boxes2):
    """
    Compute the IoU matrix between two sets of boxes.

    :param boxes1: First set of boxes.
    :param boxes2: Second set of boxes.

    :return: IoU matrix in format `[num_tracks, num_detections]`.
    """
    # boxes: [N, 4], [M, 4]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

# def appearance_cost(track_embeds, det_embeds):
#     """
#     Compute the appearance cost matrix between track embeddings and detection embeddings.
    
#     :param track_embeds: Track embeddings.
#     :param det_embeds: Detection embeddings.

#     :return torch.Tensor: Appearance cost matrix in format `[num_tracks, num_detections]`.
#     """
#     track_embeds = torch.nn.functional.normalize(track_embeds, dim=1)
#     det_embeds = torch.nn.functional.normalize(det_embeds, dim=1)
#     return torch.matmul(track_embeds, det_embeds.t())

# def appearance_cost(track_embeds, det_embeds):
#     """
#     Cosine distance matrix: [num_tracks, num_dets]
#     """
#     track_embeds = track_embeds / track_embeds.norm(dim=1, keepdim=True)
#     det_embeds = det_embeds / det_embeds.norm(dim=1, keepdim=True)

#     return 1.0 - torch.mm(track_embeds, det_embeds.t())

def gallery_appearance_cost(tracks, det_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Vectorized appearance cost matrix using track galleries.

    :param list[Track] tracks: list of Track objects, each with appearance_gallery `[G_i, D]`
    :param det_embeddings: `[N, D]` tensor of detection embeddings

    :return: `[num_tracks, N]` cost matrix (cosine distance)
    """
    device = det_embeddings.device
    num_tracks = len(tracks)
    num_dets, D = det_embeddings.shape

    # Prepare a tensor to store costs
    cost_matrix = torch.zeros((num_tracks, num_dets), device=device)

    # Loop over tracks (still unavoidable because gallery sizes may differ)
    for i, track in enumerate(tracks):
        if len(track.gallery) == 0:
            # If no gallery, set maximum cost
            cost_matrix[i] = 1.0
            continue

        gallery = torch.stack(track.gallery).to(device)  # [G_i, D]
        gallery = torch.nn.functional.normalize(gallery, dim=1) # normalize
        dets = torch.nn.functional.normalize(det_embeddings, dim=1) # normalize [N, D]

        # cosine similarity: [G_i, N] = gallery @ dets.T
        sims = torch.mm(gallery, dets.t())
        # take **max similarity** → min distance
        cost_matrix[i] = 1.0 - sims.max(dim=0).values

    return cost_matrix

# def motion_cost(tracks, det_boxes):
#     """
#     Compute the motion cost matrix between predicted track boxes and detection boxes.

#     :param tracks: List of Track objects.
#     :param det_boxes: Detection boxes.

#     :return torch.Tensor: Motion cost matrix in format `[num_tracks, num_detections]`.
#     """
#     pred_boxes = torch.stack([t.box + t.velocity for t in tracks])
#     return box_iou_matrix(pred_boxes, det_boxes)


def motion_cost(tracks, det_boxes):
    """
    Compute Mahalanobis distance between predicted track positions and detection boxes.
    
    :param list[Track] tracks: list of Track objects
    :param det_boxes: torch tensor or array of detection boxes (N, 4)

    :return: torch tensor of shape [num_tracks, num_detections]
    """
    costs = []

    for t in tracks:
        mean = t.kf.x[:4]  # shape (4,1)
        S = t.kf.S[:4, :4] + 1e-6 * np.eye(4)  # only consider position+size
        S_inv = np.linalg.inv(S)

        row = []
        for det in det_boxes:
            z = t.box_to_z(det) # shape (4,1)
            diff = z - mean # shape (4,1)
            d = float(diff.T @ S_inv @ diff)  # convert 1x1 array to scalar
            row.append(d)
        costs.append(row)

    return torch.tensor(costs, dtype=torch.float32)