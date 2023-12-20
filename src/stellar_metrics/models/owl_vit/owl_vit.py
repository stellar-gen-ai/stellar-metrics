import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


class OwlViT:
    def __init__(self, device, dtype) -> None:
        self.dtype = dtype
        self.device = device
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-large-patch14"
        ).to(device=device, dtype=dtype)
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")

        self.model.eval()

    def __call__(
        self,
        images: list[Image.Image],
        wanted_labels: list[list[str]],
    ):
        inputs = self.processor(
            images=images,
            text=wanted_labels,
            return_tensors="pt",
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        results = self.model(**inputs)

        heights, widths = list(zip(*[image.size for image in images]))
        heights = torch.tensor(heights, device=self.device)
        widths = torch.tensor(widths, device=self.device)

        results = self.processor.post_process_object_detection(
            results,
            threshold=0.0,
            target_sizes=torch.stack([heights, widths], 1),
        )
        for result in results:
            result["boxes"] = result["boxes"].long()

        return results
