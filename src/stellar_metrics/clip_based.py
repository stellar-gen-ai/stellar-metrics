from PIL import Image
from torch.nn import functional as F

from stellar_metrics.models import CLIP

META_CLIP = [
    "facebook/metaclip-b32-400m",
    "facebook/metaclip-h14-fullcc2.5b",
    "facebook/metaclip-b32-fullcc2.5b",
    "facebook/metaclip-l14-400m",
    "facebook/metaclip-b16-fullcc2.5b",
    "facebook/metaclip-l14-fullcc2.5b",
    "facebook/metaclip-b16-400m",
]
LAION_CLIP = [
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
]
LAION_DATA_COMP_CLIP = [
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
]
OPENAI_CLIP = [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14-336",
]

CLIP_MAIN = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


class CLIPMetrics:
    def __init__(self, device, dtype, model_name) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.clip_large = CLIP(device=device, dtype=dtype, model_name=model_name)

    def __call__(
        self,
        og_img: list[Image.Image],
        syn_img: list[Image.Image],
        prompt: list[str],
    ):
        clip_outputs = self.clip_large(syn_img, prompt)
        subject_outputs = self.clip_large(og_img, None)
        image_embeds = clip_outputs["image_embeds"]

        return {
            "clip_t": (
                F.cosine_similarity(clip_outputs["text_embeds"], image_embeds)
                .cpu()
                .numpy()
            ),
            "clip_n": (
                F.cosine_similarity(
                    clip_outputs["text_embeds"], subject_outputs["image_embeds"]
                )
                .cpu()
                .numpy()
            ),
            "clip_i": (
                F.cosine_similarity(subject_outputs["image_embeds"], image_embeds)
                .cpu()
                .numpy()
            ),
        }
