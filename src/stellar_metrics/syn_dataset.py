import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _cache_uid(images: list[Path]) -> str:
    """
    return a cache unique-id from a list of Paths that
    can be used for caching.

    Parameters
    ----------
    images : list[Path]
        List of images to calculate a unique hash on.
    """
    paths = ",".join(sorted(map(str, images), key=lambda x: x))
    return hashlib.md5(paths.encode("utf-8")).hexdigest()


class SynImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path | str,
        stellar_dataset_root: Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.syn_images = sorted(self.data_root.glob("*.png"), key=lambda x: x.stem)
        metadata = [
            json.load(p.open("rb"))
            for p in sorted(self.data_root.glob("*.json"), key=lambda x: x.stem)
        ]
        if len(metadata) == 0:
            raise RuntimeError(f"Empty data directory {data_root}")
        _metadata = metadata[0]
        if stellar_dataset_root is None:
            stellar_dataset_root = Path(_metadata["image_path"]).parent.parent
        self.stellar_dataset_root = stellar_dataset_root
        self.attribute_names: list[str] = list(_metadata["attributes"].keys())
        self.attributes = np.array([
            [_metadata["attributes"][k] for k in self.attribute_names]
            for _metadata in metadata
        ])
        self.detectables: list[list[str]] = [
            _metadata["detectables"] for _metadata in metadata
        ]

        self.save_names: list[str] = [_metadata["save_name"] for _metadata in metadata]
        self.prompts: list[str] = [_metadata["prompt"] for _metadata in metadata]
        self.og_images = [
            self.stellar_dataset_root.joinpath(
                *Path(_metadata["image_path"])._parts[-2:]
            )
            for _metadata in metadata
        ]
        self.aux_images = []
        for og_img in self.og_images:
            aux_images = list(og_img.parent.glob("*.jpg"))
            self.aux_images.append(aux_images)

    def __len__(self):
        return len(self.syn_images)

    def __getitem__(self, index):
        img = Image.open(self.syn_images[index]).convert("RGB")
        og_img = Image.open(self.og_images[index]).convert("RGB")
        attributes = self.attributes[index]
        detectables = self.detectables[index]
        save_name = self.save_names[index]
        prompt = self.prompts[index].format("a person")
        aux_imgs = [
            Image.open(aux_img).convert("RGB") for aux_img in self.aux_images[index]
        ]
        return {
            "syn_img": img,
            "og_img": og_img,
            "aux_imgs": aux_imgs,
            "attributes": attributes,
            "detectables": detectables,
            "name": save_name,
            "prompt": prompt,
        }

    @property
    def uid(self):
        return _cache_uid(self.og_images) + _cache_uid(self.syn_images)
