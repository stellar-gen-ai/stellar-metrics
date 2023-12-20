import numpy as np
from ImageReward import load as ir_load
from PIL import Image


class ImageReward:
    def __init__(self, device) -> None:
        self.model = ir_load("ImageReward-v1.0", device)

    def __call__(
        self,
        syn_img: list[Image.Image],
        prompt: list[str],
    ):
        return {
            "image_reward": np.array([
                self.model.score(prompt, image)
                for image, prompt in zip(syn_img, prompt)
            ])
        }
