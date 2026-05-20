import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img):
        #loading image to the CUDA or CPU device
        device = list(self.parameters())[0].device
        img = img.to(device)

        #inference for image 'img'
        detections = self(img)[0]
        
        #further output details can be found at https://pytorch.org/vision/0.8/models.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()
