import torch
import torchreid
import torchvision.transforms as T


class ReIDExtractor:
    def __init__(self, device):
        self.device = device

        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.model.to(device).eval()

        self.transform = T.Compose([
            T.Resize((256, 128)),  # (H, W) for ReID
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.inference_mode()
    def extract_embeddings(self, frame: dict, boxes: torch.Tensor) -> torch.Tensor:
        """
        Extract ReID embeddings using OSNet.
        """
        img = frame['img'][0]  # remove batch dim -> [C, H, W]
        embeds = []

        if boxes.numel() == 0:
            return torch.empty((0, 512), device=self.device)

        for box in boxes:
            x1, y1, x2, y2 = box.int()
            crop = img[:, y1:y2, x1:x2]

            if crop.numel() == 0:
                embeds.append(torch.zeros(512, device=self.device))
                continue

            crop = crop.float() / 255.0
            crop = self.transform(crop)
            crop = crop.unsqueeze(0).to(self.device)

            feat = self.model(crop)
            feat = torch.nn.functional.normalize(feat, dim=1)

            embeds.append(feat.squeeze(0))

        return torch.stack(embeds)