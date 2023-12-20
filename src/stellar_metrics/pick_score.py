import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor

PICK_SCORE_PRETRAINED_PATH = "yuvalkirstain/PickScore_v1"
PICK_SCORE_PROCESSOR_PATH = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class PickScore:
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.model = (
            AutoModel.from_pretrained(PICK_SCORE_PRETRAINED_PATH).eval().to(device)
        )
        self.preprocessor = AutoProcessor.from_pretrained(PICK_SCORE_PROCESSOR_PATH)

    def __call__(
        self,
        syn_img: list[Image.Image],
        prompt: list[str],
    ):
        syn_img = self.preprocessor(
            images=syn_img,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        prompt = self.preprocessor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_embs = self.model.get_image_features(**syn_img)
            text_embs = self.model.get_text_features(**prompt)

            assert image_embs.shape == text_embs.shape
            assert image_embs.dim() == 2

            score = F.cosine_similarity(text_embs, image_embs).cpu().numpy()

        return {"pick_score": score}
