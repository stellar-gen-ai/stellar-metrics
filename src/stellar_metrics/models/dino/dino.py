import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


class DINO:
    def __init__(self, device, dtype) -> None:
        self.dtype = dtype
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
        self.model = ViTModel.from_pretrained("facebook/dino-vits16").to(
            device=device, dtype=dtype
        )
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        subject_images: list[Image.Image],
        images: list[Image.Image],
    ):
        subject_inputs = self.processor(images=subject_images, return_tensors="pt").to(
            self.device
        )
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        concatenated_inputs = {
            key: torch.cat([inputs[key], subject_inputs[key]]) for key in inputs
        }
        concatenated_inputs["pixel_values"] = concatenated_inputs["pixel_values"].to(
            dtype=self.dtype
        )

        # NOTE: split this to two forward passes if this throws OOM
        outputs = self.model(**concatenated_inputs)
        image_embeds, subject_image_embeds = outputs.last_hidden_state[:, 0].chunk(2)

        return {
            "image_embeds": image_embeds,
            "subject_image_embeds": subject_image_embeds,
        }
