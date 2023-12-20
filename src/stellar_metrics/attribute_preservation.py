import functools
import logging
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from stellar_metrics.models import FaceDector
from stellar_metrics.syn_dataset import SynImageDataset
from stellar_metrics.utils import signature

IMAGE_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
])


CACHE_DIR = Path.home() / ".cache" / "stellar" / "attr"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_features(pickle_files: list[Path]):
    return np.stack([pickle.load(f.open("rb")) for f in pickle_files])


def image_process(img: np.ndarray | Image.Image):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8"))
    img.resize((224, 224), resample=Image.Resampling.BICUBIC)
    return IMAGE_TRANSFORM(img)


class ImageListDataset(Dataset):
    def __init__(
        self,
        image_files: list[Path],
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        self.image_files = sorted([Path(f) for f in image_files], key=lambda x: str(x))
        if transform is None:
            transform = functools.partial(image_process)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])
        return {
            "save_name": signature(self.image_files[index]),
            "img": self.transform(img),
        }


class AttributePreservation:
    def __init__(
        self,
        device: str | None = None,
        cache_dir: Path | None = None,
        batch_size: int = 16,
        clip_model="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if cache_dir is None:
            cache_dir = CACHE_DIR
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.clip_model = clip_model

        self.attributes = [
            "Young",
            "Male",
            "High_Cheekbones",
            "Pointy_Nose",
            "Narrow_Eyes",
            "Double_Chin",
            "Big_Lips",
            "Big_Nose",
        ]

        assert batch_size > 1, "Must provide a batch-size larger than 1"
        self.batch_size = batch_size
        self.face_dector = FaceDector(device=self.device)

    def signature(self, file: Path):
        return signature(file, metadata=self.clip_model)

    @torch.no_grad()
    def _get_features(self, image: list[Path] | Path):
        model = CLIPModel.from_pretrained(self.clip_model).to(self.device)
        model.eval()
        processor = AutoProcessor.from_pretrained(self.clip_model)

        if isinstance(image, Path):
            image = [image]
        pickle_files = [(self.cache_dir / f"{self.signature(img)}.pt") for img in image]
        missing_images = [
            img
            for pickle_file, img in zip(pickle_files, image)
            if not pickle_file.exists()
        ]
        if len(missing_images) == 0:
            return _load_features(pickle_files)

        ds = ImageListDataset(
            missing_images,
            transform=lambda x: processor(images=x, return_tensors="pt")[
                "pixel_values"
            ].squeeze(),
        )
        dl = DataLoader(ds, batch_size=self.batch_size)
        for batch in tqdm(dl):
            features = model.get_image_features(
                pixel_values=batch["img"].to(self.device)
            )
            for i, name in enumerate(batch["save_name"]):
                pickle.dump(
                    features[i].detach().cpu().numpy(),
                    (self.cache_dir / f"{name}.pt").open("wb"),
                )
        return _load_features(pickle_files)

    def _extract_faces(self, images: list[Path]) -> list[np.ndarray]:
        face_embeds = []
        for image in images:
            im = Image.open(image)
            filtered_faces = self.face_dector(im)

            if filtered_faces is None:
                face_embeds.append(None)
                continue
            largest_face = filtered_faces[0]
            face_embeds.append(largest_face)

        return face_embeds

    def __call__(
        self,
        dataset: SynImageDataset,
        clean=False,
    ) -> pd.DataFrame:
        og_images = dataset.og_images
        syn_images = dataset.syn_images
        attributes = dataset.attributes

        df = pd.DataFrame({"name": dataset.save_names}).set_index("name")
        syn_vectors = self._extract_faces(syn_images)
        og_vectors = self._extract_faces(og_images)

        probe_dir = self.cache_dir / dataset.uid / "probes"
        probe_dir.mkdir(exist_ok=True, parents=True)
        split = "og"
        ys = []
        for i, attr in enumerate(dataset.attribute_names):
            if attr not in self.attributes:
                continue

            _y = attributes[:, i]
            ys.append(_y)
            probe_p: Path = probe_dir / f"{attr}_{split}.lr"

            _y = attributes[:, i]
            df[f"{attr}_target"] = _y
            df[f"{attr}_pred"] = _y * -1
            if len(np.unique(_y)) == 1:
                continue
            if probe_p.exists() and not clean:
                lr = pickle.load(probe_p.open("rb"))
            else:
                x_train = np.array(list(filter(lambda x: x is not None, og_vectors)))
                lr = LogisticRegression().fit(x_train, _y)
                pickle.dump(lr, probe_p.open("wb"))

            valid_test = list(
                filter(lambda x: x[1] is not None, zip(dataset.save_names, syn_vectors))
            )
            x_test = np.array([v for name, v in valid_test])
            names = [name for name, v in valid_test]
            y_pred_test = lr.predict(x_test)
            df.loc[names, f"{attr}_pred"] = y_pred_test

        return df.reset_index().sort_values("name")
