import copy
import inspect
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from stellar_metrics import (
    AestheticPredictor,
    AttributePreservation,
    CLIPMetrics,
    DINOFidelity,
    DreamsimDistance,
    HumanPreferenceScore,
    IdentityPreservation,
    ImageReward,
    ObjectFaithfulness,
    PickScore,
    IdentityStability,
)
from stellar_metrics.clip_based import (
    CLIP_MAIN,
)
from stellar_metrics.syn_dataset import SynImageDataset

assets_folder = Path(__file__).parent / "assets"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dataset():
    return SynImageDataset(
        assets_folder / "stellar_net", assets_folder / "mock_stellar_dataset"
    )


def _data_batch():
    batch = defaultdict(list)
    for d in dataset():
        for k, v in d.items():
            batch[k].append(v)
    return dict(batch)


@pytest.fixture()
def data_batch():
    return _data_batch()


def test_object_faithfulness(data_batch):
    metric = ObjectFaithfulness(DEVICE, torch.float32)
    results = metric(
        data_batch["syn_img"],
        data_batch["detectables"],
        return_predictions=False,
    )
    shuffled_detectables = copy.deepcopy(data_batch["detectables"])
    np.random.seed(0)
    np.random.shuffle(shuffled_detectables)
    results_random = metric(
        data_batch["syn_img"], shuffled_detectables, return_predictions=False
    )
    assert results.mean() > results_random.mean()


def test_clip(data_batch):
    shuffled_prompts = copy.deepcopy(data_batch["prompt"])
    del data_batch["detectables"]
    np.random.seed(0)
    np.random.shuffle(shuffled_prompts)

    metric = CLIPMetrics(DEVICE, torch.float16, model_name=CLIP_MAIN)

    kwargs = list(inspect.signature(metric).parameters.keys())
    _data_batch = {k: data_batch[k] for k in kwargs}
    metric_proper = metric(**_data_batch)
    metric_random = metric(**_data_batch | dict(prompt=shuffled_prompts))
    assert metric_proper["clip_t"].mean() > metric_random["clip_t"].mean()
    assert metric_proper["clip_i"].mean() == metric_random["clip_i"].mean()


def test_pick_score(data_batch):
    metric = PickScore(DEVICE)
    results = metric(data_batch["og_img"], data_batch["prompt"])["pick_score"]
    shuffled_prompts = copy.deepcopy(data_batch["prompt"])
    np.random.seed(0)
    np.random.shuffle(shuffled_prompts)
    results_random = metric(data_batch["og_img"], shuffled_prompts)["pick_score"]
    # NOTE pick score is as good as random -> 🗑️
    assert results.mean() < results_random.mean()


def test_attribute_preservation(tmp_path: Path):
    """
    Reference Table:
        Young_pred  Young_target  is_train         name  Male_pred  Male_target
    0        -1.0            -1       1.0      000-0-1       -1.0           -1
    1        -1.0            -1       1.0      000-0-2       -1.0            1
    2        -1.0            -1       0.0  199-1-19991       -1.0            1
    3        -1.0             1       1.0  199-1-19992       -1.0           -1
    4        -1.0            -1       1.0  199-1-19993       -1.0            1
    5        -1.0             1       1.0  199-1-19998       -1.0           -1
    6        -1.0             1       0.0  199-1-19999       -1.0           -1

    """
    calculator = AttributePreservation(device=DEVICE)
    ds = dataset()
    df = calculator(ds)
    assert len(df) == len(ds)
    assert all([name in df["name"].values.tolist() for name in ds.save_names])


def test_aes_score(data_batch):
    metric = AestheticPredictor(DEVICE)
    results_syn = metric(data_batch["syn_img"])
    results_og = metric(data_batch["og_img"])
    assert results_syn.mean() > results_og.mean()


def test_identity_stability(data_batch):
    metric = IdentityStability(DEVICE)
    results = metric(aux_imgs=data_batch["aux_imgs"], syn_img=data_batch["syn_img"])
    shuffled_aux_imgs = copy.deepcopy(data_batch["aux_imgs"])
    np.random.seed(0)
    np.random.shuffle(shuffled_aux_imgs)
    results_random = metric(shuffled_aux_imgs, data_batch["syn_img"])
    assert results.mean() > results_random.mean()


def test_identity_preservation(data_batch):
    metric = IdentityPreservation(DEVICE)
    results = metric(data_batch["og_img"], data_batch["syn_img"])
    shuffled_og_imgs = copy.deepcopy(data_batch["og_img"])
    np.random.seed(0)
    np.random.shuffle(shuffled_og_imgs)
    results_random = metric(shuffled_og_imgs, data_batch["syn_img"])
    assert results.mean() > results_random.mean()


def test_image_reward(data_batch):
    prompts = [
        "a photo of a young woman",
        "a photo of a man",
        "a photo of a dog",
    ]
    metric = ImageReward(DEVICE)
    results = metric(syn_img=[data_batch["og_img"][0]] * len(prompts), prompt=prompts)[
        "image_reward"
    ]

    assert results[0] > results[1] > results[2]


def test_human_preference_score(data_batch):
    prompts = [
        "a photo of a young woman",
        "a photo of a man",
        "a photo of a dog",
    ]
    metric = HumanPreferenceScore(DEVICE)
    results = metric(syn_img=[data_batch["og_img"][0]] * len(prompts), prompt=prompts)
    hpsv1 = results["human_preference_score_v1"]
    hpsv2 = results["human_preference_score_v2"]

    assert hpsv2[0] > hpsv2[1]
    assert hpsv2[0] > hpsv2[2]
    assert hpsv1[0] > hpsv1[1]
    assert hpsv1[0] > hpsv1[2]


def test_dino_fid(data_batch):
    image = data_batch["og_img"][0]
    subject_image = data_batch["syn_img"][0]
    images = [image, subject_image, image, subject_image]
    subject_images = [image, subject_image, subject_image, image]

    metric = DINOFidelity(
        DEVICE,
        torch.float16,
    )
    dino_fid = metric(syn_img=images, og_img=subject_images)["dino_fidelity"]

    assert dino_fid[0] > dino_fid[2]
    assert np.isclose(dino_fid[2], dino_fid[3], atol=1e-2)
    assert np.isclose(dino_fid[0], dino_fid[1], atol=1e-2)
    assert np.isclose(dino_fid[0], 1, atol=1e-2)


def test_dreamsim_dist(data_batch):
    image = data_batch["og_img"][0]
    subject_image = data_batch["syn_img"][0]

    images = [image, subject_image, image, subject_image]
    subject_images = [image, subject_image, subject_image, image]

    metric = DreamsimDistance(DEVICE)
    dreamsim_dist = metric(syn_img=images, og_img=subject_images)["dreamsim_distance"]

    assert np.isclose(dreamsim_dist[2], dreamsim_dist[3], atol=1e-2)
    assert np.isclose(dreamsim_dist[0], dreamsim_dist[1], atol=1e-2)
    assert np.isclose(dreamsim_dist[0], 0, atol=1e-2)
