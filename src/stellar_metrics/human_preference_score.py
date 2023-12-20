from pathlib import Path

import hpsv2
import numpy as np
from PIL import Image

from stellar_metrics.models import HumanPreferenceClassifier


class HumanPreferenceScore:
    def __init__(self, device) -> None:
        self.hpc = HumanPreferenceClassifier(
            checkpoint_dir=Path(__file__).parent.parent.parent
            / "checkpoints"
            / "hpc.pt",
            device=device,
        )

    def __call__(
        self,
        syn_img: list[Image.Image],
        prompt: list[str],
    ):
        return {
            "human_preference_score_v1": self.hpc(syn_img, prompt)[
                "human_preference_classifier"
            ],
            "human_preference_score_v2": np.array([
                hpsv2.score(image, prompt)[0] for image, prompt in zip(syn_img, prompt)
            ]),
        }
