import torch
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image, to_tensor
from ultralytics import YOLO

# class FRCNN_FPN(FasterRCNN):
#     """
#     The Faster RCNN Feature Pyramid Network used for Multi Object Detection.
#     """
#     def __init__(self, num_classes: int, nms_thresh: float=0.5, **kwargs):
#         """
#         Initialization of new FRCNN FPN for MOT.
        
#         :param int num_classes: Amount of possible classes for object classification.
#         :param float nms_thresh: Threshold for Non Max Suppression, higher threshold leads to fewer predictions.
#         :param kwargs: Additional arguments that are passed to the FasterRCNN class.
#         """
#         backbone = resnet_fpn_backbone('resnet50', False)
#         super(FRCNN_FPN, self).__init__(backbone, num_classes, **kwargs)

#         self.roi_heads.nms_thresh = nms_thresh
class FRCNN_FPN(FasterRCNN):
    def __init__(
        self,
        num_classes: int,
        box_nms_thresh: float = 0.5,
        **kwargs,
    ):
        backbone = resnet_fpn_backbone("resnet50", pretrained=False)

        print(kwargs)

        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            box_nms_thresh=box_nms_thresh,
            **kwargs,
        )

    def detect(self, img):
        """
        Perform object detection on one image.
    
        :param img: The image the object detection should be performed on.
        """
        #loading image to the CUDA or CPU device
        device = list(self.parameters())[0].device
        img = img.to(device)
        
        #inference for image 'img'
        detections = self(img)[0]
        
        #further output details can be found at https://pytorch.org/vision/0.8/models.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()
    
    def detect_sahi(self, img, tile_size: int = 640, overlap: float = 0.2, score_thresh: float = 0.5, cross_tile_iou: float = 0.5):
        """
        SAHI (Slicing Aided Hyper Inference)-style detection with overlapping tiles. Move a sliding window over the image, detect objects in each tile, then merge results with NMS.

        :param img: Image to be analyzed, can be a PyTorch tensor or a PIL image, feed in only one image at a time for logical order of track.
        :param int tile_size: The width/height of each square tile.
        :param float overlap: Fraction of overlap between tiles (0–1). Prevents objects from being cut at tile edges.
        :param float score_thresh: Minimum detection score per tile.
        :param float cross_tile_iou: NMS threshold for merging boxes across tiles.
        """
        device = list(self.parameters())[0].device

        # handle batch dimension
        if isinstance(img, torch.Tensor):
            if img.ndim == 4:
                # remove batch dim
                img_tensor = img.squeeze(0)
            else:
                img_tensor = img
            img_pil = to_pil_image(img_tensor.cpu())
        else:
            img_pil = img

        # slice image
        w, h = img_pil.size
        # calculate stride, the step size made in the image moving horizontally and vertically
        stride = int(tile_size * (1 - overlap))
        all_boxes, all_scores = [], []

        for y in range(0, h, stride): # one vertical step at a time
            for x in range(0, w, stride):  # one horizontal step at a time
                x1, y1 = x, y
                x2, y2 = min(x + tile_size, w), min(y + tile_size, h) # ensure to not go out of image bounds
                tile = img_pil.crop((x1, y1, x2, y2))
                tile_tensor = to_tensor(tile).unsqueeze(0).to(device)

                # with torch.no_grad():
                # feed image tensor into model
                outputs = self(tile_tensor)[0]

                boxes = outputs['boxes']
                scores = outputs['scores']

                mask = scores >= score_thresh
                boxes, scores = boxes[mask], scores[mask]

                if len(boxes) == 0:
                    continue

                # shift boxes back to original image coordinates
                boxes[:, 0] += x1
                boxes[:, 1] += y1
                boxes[:, 2] += x1
                boxes[:, 3] += y1

                all_boxes.append(boxes)
                all_scores.append(scores)

        # merge boxes using NMS
        if len(all_boxes) == 0:
            return torch.empty((0, 4)), torch.empty((0,))
        # concatenate all boxes and scores from all tiles
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        # do universal NMS to remove duplicates
        keep = nms(all_boxes, all_scores, cross_tile_iou)
        return all_boxes[keep].cpu(), all_scores[keep].cpu()
    

class YOLODetector:
    """
    YOLO-based pedestrian detector with the same interface as FRCNN_FPN.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_thresh: float = 0.15,
        iou_thresh: float = 0.5,
        device: str | None = None,
    ):
        """
        :param model_path: Path to YOLO model weights.
        :param conf_thresh: Confidence threshold (lower = higher recall).
        :param iou_thresh: IoU threshold for NMS.
        :param device: 'cuda', 'cpu', or None (auto).
        """
        self.model = YOLO(model_path)

        if device is not None:
            self.model.to(device)

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # COCO person class = 0
        self.person_class = 0

    @torch.inference_mode()
    def detect(self, img: torch.Tensor):
        """
        Perform object detection on one image.

        :param img: Tensor [1, C, H, W] in range [0, 1]
        :return: boxes [N, 4], scores [N]
        """
        device = img.device
        orig_h, orig_w = img.shape[2], img.shape[3]

        # YOLO expects 640x640 inputs
        img_pad = F.interpolate(img, size=(640, 640), mode='bilinear', align_corners=False)

        # YOLO expects [B, C, H, W] or numpy
        results = self.model(
            img_pad,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=[self.person_class],
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device)
            )

        boxes = results.boxes.xyxy.to(device)
        scores = results.boxes.conf.to(device)
        
        # boxes are predicted in 640x640, re-scale boxes to original image size
        scale_x = orig_w / 640
        scale_y = orig_h / 640

        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2

        return boxes, scores