import torch
from dreamsim import dreamsim
from PIL import Image


class DreamsimDistance:
    def __init__(self, device) -> None:
        self.device = device
        self.model, self.preprocess = dreamsim(
            pretrained=True, device=device, cache_dir="./.dreamsim_cache"
        )
        self.model.eval()

    def __call__(
        self,
        syn_img: list[Image.Image],
        og_img: list[Image.Image],
    ):
        return {
            "dreamsim_distance": (
                self.model(
                    torch.concatenate([
                        self.preprocess(image).to(self.device) for image in syn_img
                    ]),
                    torch.concatenate([
                        self.preprocess(image).to(self.device) for image in og_img
                    ]),
                )
                .cpu()
                .numpy()
            )
        }
