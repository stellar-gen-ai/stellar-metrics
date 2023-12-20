import hashlib
import io
from pathlib import Path

import torch
from PIL import Image


def get_bytes(obj, block_size=10_000):
    if isinstance(obj, Image.Image):
        img_byte_arr = io.BytesIO()
        obj.save(img_byte_arr, format="PNG")
        data = img_byte_arr.getvalue()[:block_size]
    elif isinstance(obj, Path):
        with obj.open("rb") as f:
            data = f.read(block_size)
    else:
        raise NotImplementedError
    return data


def signature(obj: Path | Image.Image, metadata: str = "", block_size=10_000):
    md5 = hashlib.md5()
    data = get_bytes(obj, block_size=block_size)

    md5.update(data + metadata.encode("utf-8"))
    return md5.hexdigest()


# List of dicts to dict of lists
def ld_to_dl(
    LD: list[dict],
    different_keys: str | bool = False,
    stack_tensors: bool = True,
):
    if not different_keys:
        ddict = {key: [dic[key] for dic in LD] for key in LD[0]}
    elif different_keys == "intersection":
        k_intersection = set.intersection(*map(set, LD))
        ddict = {key: [dic[key] for dic in LD] for key in k_intersection}
    elif different_keys == "union":
        k_union = set.union(*map(set, LD))
        ddict = {key: [dic[key] for dic in LD if key in dic] for key in k_union}
    else:
        raise ValueError(
            "different_keys must be False, 'intersection' or 'union', got"
            f" {different_keys}"
        )

    if stack_tensors:
        for key, value in ddict.items():
            if torch.is_tensor(value[0]):
                ddict[key] = torch.stack(value)

    return ddict
