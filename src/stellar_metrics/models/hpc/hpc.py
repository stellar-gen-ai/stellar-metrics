import clip
import torch
from PIL import Image
from torch.nn import functional as F


class HumanPreferenceClassifier:
    def __init__(self, checkpoint_dir, device="cuda"):
        # pathsd = f"/home/{os.getlogin()}/.hpc/weights/"
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.model.load_state_dict(torch.load(checkpoint_dir)["state_dict"])

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image], prompts: list[str]):
        images = torch.cat(
            [
                self.preprocess(image).unsqueeze(0).to(device=self.device)
                for image in images
            ],
            dim=0,
        )
        text = clip.tokenize(prompts).to(self.device)

        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return {
            "human_preference_classifier": (
                F.cosine_similarity(image_features, text_features).cpu().numpy()
            )
        }
