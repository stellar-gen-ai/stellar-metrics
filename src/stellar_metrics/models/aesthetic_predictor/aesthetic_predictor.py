from pathlib import Path

import torch
from PIL import Image
from torch import nn

from stellar_metrics.models.clip.clip import CLIP

# Aesthetic model weights path
# https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth


class AestheticPredictorHead(nn.Module):
    """
    Multi-layer perceptron model. Used to predict the aesthetic score of an
    image based on its CLIP embedding.
    https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, image_embeds):
        return self.layers(image_embeds)


class AestheticPredictor:
    def __init__(
        self,
        device,
        dtype=torch.float16,
    ) -> None:
        self.dtype = dtype
        self.device = device
        # the aesthetic model expects ViT-L/14 clip as a backbone
        self.aesthetic_model = AestheticPredictorHead().to(device, dtype)

        self.clip_aes = CLIP(
            device=device,
            dtype=dtype,
            model_name="openai/clip-vit-large-patch14",
        )
        aesthetic_model_path = (
            Path(__file__).parent / "sac+logos+ava1-l14-linearMSE.pth"
        )
        self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_path))
        self.aesthetic_model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        syn_img: list[Image.Image],
    ):
        normalized_image_embeds = self.clip_aes(syn_img, None)["image_embeds"]
        # image_embeds seem to already be normalized
        # https://github.com/huggingface/transformers/blob/6da93f5580e109fad5f7b523cf2b6e8a5bafb623/src/transformers/models/clip/modeling_clip.py#L1165 # noqa: E501
        return self.aesthetic_model(normalized_image_embeds).flatten().cpu().numpy()
