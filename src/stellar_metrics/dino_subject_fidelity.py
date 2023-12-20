from PIL import Image
from torch.nn import functional as F

from stellar_metrics.models import DINO


class DINOFidelity:
    def __init__(self, device, dtype) -> None:
        self.dtype = dtype
        self.device = device
        self.dino = DINO(device=device, dtype=dtype)

    def __call__(
        self,
        og_img: list[Image.Image],
        syn_img: list[Image.Image],
    ):
        dino_outputs = self.dino(og_img, syn_img)
        return {
            "dino_fidelity": (
                F.cosine_similarity(
                    dino_outputs["image_embeds"],
                    dino_outputs["subject_image_embeds"],
                )
                .cpu()
                .numpy()
            ),
        }
