import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer


class CLIP:
    def __init__(
        self,
        device,
        dtype,
        model_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name).to(
            device=device, dtype=dtype
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model.eval()

    def prepare_images(self, images: list[Image.Image]):
        return torch.stack([self.image_transforms()(image) for image in images]).to(
            dtype=self.dtype, device=self.device
        )

    @torch.inference_mode()
    def __call__(
        self,
        images: list[Image.Image],
        prompt: list[str] | None,
    ):
        inputs = {}
        if prompt is None:
            prompt = [""] * len(images)

        inputs = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        pixel_values = torch.from_numpy(
            np.stack(self.image_processor(images)["pixel_values"])
        ).to(self.device)
        outputs = self.model(pixel_values=pixel_values, **inputs)

        return {
            "text_embeds": outputs.text_embeds,
            "image_embeds": outputs.image_embeds,
        }
