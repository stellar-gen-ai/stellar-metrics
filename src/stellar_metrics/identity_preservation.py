import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stellar_metrics.models import FaceDector


class IdentityPreservation:
    def __init__(
        self, device=torch.device("cuda"), face_confidence_threshold: float = 0.9
    ):
        self.device = device
        self.face_detector = FaceDector(
            device=self.device, margin=face_confidence_threshold
        )

    def get_subjects_face_embeds(
        self, subject_image: list[Image.Image]
    ) -> torch.Tensor:
        return torch.tensor(
            [
                sorted(
                    self.face_detector(subject_image),
                    key=lambda d: d["face_confidence"],
                    reverse=True,
                )[0]["embedding"]
            ],
            device=self.device,
        )

    def get_embeds(self, image: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.face_detector(image)
        if embeds is None:
            return []
        return embeds

    def __call__(
        self,
        og_img: list[Image.Image],
        syn_img: list[Image.Image],
    ):
        scores = []
        for _og_img, _syn_img in zip(og_img, syn_img):
            syn_embed = self.get_embeds(_syn_img)
            og_embed = self.get_embeds(_og_img)
            if len(syn_embed) == 0 or len(og_embed) == 0:
                scores.append(0)
                continue
            # get similarity of each face of the generated images with their
            # corresponding subject face
            similarity = F.cosine_similarity(syn_embed.unsqueeze(1), og_embed, dim=-1)
            scores.append(similarity.max().cpu().numpy())

        return np.array(scores)
