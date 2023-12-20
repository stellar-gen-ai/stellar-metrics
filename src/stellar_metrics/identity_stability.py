import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stellar_metrics.identity_preservation import IdentityPreservation


class IdentityStability:
    def __init__(self, device=torch.device("cuda")):
        self.device = device
        self.backbone = IdentityPreservation(device=device)

    def get_subjects_face_embeds(
        self, subject_image: list[Image.Image]
    ) -> torch.Tensor:
        return torch.tensor(
            [
                sorted(
                    self.deepface_model(subject_image),
                    key=lambda d: d["face_confidence"],
                    reverse=True,
                )[0]["embedding"]
            ],
            device=self.device,
        )

    def get_embeds(
        self, images: list[Image.Image], face_confidence_threshold: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images_faces = [self.deepface_model(image) for image in images]
        filtered_images_faces = [
            list(
                filter(
                    lambda d: d["face_confidence"] > face_confidence_threshold,
                    image_faces,
                )
            )
            for image_faces in images_faces
        ]

        faces_per_image = [len(image_faces) for image_faces in filtered_images_faces]

        return torch.concat(
            [
                torch.tensor(
                    list(map(lambda d: d["embedding"], image_faces)),
                    device=self.device,
                )
                for image_faces in filtered_images_faces
            ],
        ), torch.tensor(
            faces_per_image,
            dtype=torch.long,
            device=self.device,
        )

    def __call__(
        self,
        aux_imgs: list[Image.Image],
        syn_img: list[Image.Image],
    ):
        scores = []
        for _aux_imgs, _syn_img in zip(aux_imgs, syn_img):
            score = min(
                self.backbone(og_img=_aux_imgs, syn_img=[_syn_img] * len(_aux_imgs))
            )
            scores.append(score)
        return np.array(scores)
