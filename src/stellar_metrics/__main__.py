import argparse
import inspect
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import tqdm

from stellar_metrics import AttributePreservation, CLIPMetrics, METRICS
from stellar_metrics.clip_based import (
    CLIP_MAIN,
    LAION_CLIP,
    LAION_DATA_COMP_CLIP,
    META_CLIP,
    OPENAI_CLIP,
)
from stellar_metrics.syn_dataset import SynImageDataset


def batch_collate(samples: list[dict], args: list[str]):
    batch = defaultdict(list)
    for sample in samples:
        for k, v in sample.items():
            if k in args:
                batch[k].append(v)
    return batch


def run(
    metric: str,
    stellar_path: Path,
    syn_path: Path,
    save_dir: Path | None = None,
    clip_version: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    device: str | None = None,
    batch_size: int = 16,
):
    if save_dir is None:
        save_dir = Path("/tmp/stellar-metrics-tmp")
    dataset = SynImageDataset(syn_path, stellar_path)
    save_path = save_dir / dataset.uid
    save_path.mkdir(exist_ok=True, parents=True)
    if metric == "aps":
        # This metric requires to be run over the entire dataset.
        metric_class = AttributePreservation(
            device=device, clip_model=clip_version, batch_size=batch_size
        )
        df_syn = metric_class(dataset, save_dir)

        metric_name = f"attribute_preservation_{clip_version}"
        metric_name = "_".join(metric_name.split("/"))
        df_syn.to_csv(save_path / "attribute_preservation_syn.csv")
        return df_syn
    elif metric == "clip":
        metric_obj = CLIPMetrics(device, model_name=clip_version, dtype=torch.float16)
        metric_name = f"clip_{clip_version}"
        metric_name = "_".join(metric_name.split("/"))
    elif metric in METRICS:
        kwargs = {"device": device, "dtype": torch.float16}
        metric_name, metric_class = METRICS[metric]
        init_parameters = list(inspect.signature(metric_class).parameters)
        metric_obj = metric_class(**{
            k: v for k, v in kwargs.items() if k in init_parameters
        })
    else:
        raise NotImplementedError
    results = []
    n_batches = math.ceil(len(dataset) / batch_size)
    parameters = list(inspect.signature(metric_obj).parameters)
    for i in tqdm.tqdm(range(n_batches)):
        samples = [
            dataset[k]
            for k in range(i * batch_size, (i + 1) * batch_size)
            if k < len(dataset)
        ]
        if len(samples) == 0:
            break
        batch = batch_collate(samples, parameters)
        out = metric_obj(**batch)
        if not isinstance(out, dict):
            out = {metric_name: out}
        keys = list(out.keys())
        for sample, *v in zip(samples, *out.values()):
            res = {"name": sample["name"]}
            for j, _v in enumerate(v):
                res[keys[j]] = _v
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(save_path / f"{metric_name}.csv")
    return df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=METRICS,
        help=f"Calculate the identity {list(METRICS.keys())} preservation metric.",
    )

    args.add_argument(
        "--clip-version",
        type=str,
        required=False,
        default=CLIP_MAIN,
        choices=LAION_CLIP + META_CLIP + OPENAI_CLIP + LAION_DATA_COMP_CLIP,
        help="Version of clip model to use for calculating the metric",
    )
    args.add_argument(
        "--stellar-path",
        type=Path,
        required=False,
        help=(
            "Path to the stellar dataset used as a reference to evaluate the generated"
            " images."
        ),
    )

    args.add_argument(
        "--batch-size",
        type=int,
        default=16,
        required=False,
        help="Batch size to use during inference",
    )

    args.add_argument(
        "--device",
        choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        default="cuda" if torch.cuda.is_available() else "cpu",
        required=False,
        help="The device to use for inference.",
    )
    args.add_argument(
        "--syn-path",
        type=Path,
        help=(
            "Path to the synthetic images to be evaluated. Must be"
            " named identical to the original reference image in"
            " `stellar-path/imgs`"
        ),
        required=True,
    )
    args.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Path to save the results.",
        required=True,
    )
    kwargs = vars(args.parse_args())
    run(**kwargs)
