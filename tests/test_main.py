from pathlib import Path

import numpy as np
import torch
from PIL import Image

from stellar_metrics import METRICS
from stellar_metrics.__main__ import run
from stellar_metrics.analysis.__main__ import make_table
from stellar_metrics.syn_dataset import SynImageDataset
from stellar_metrics.analysis.utils import read_res

data_root = Path(__file__).parent / "assets" / "stellar_net"
results_dir = Path(__file__).parent / "assets" / "save_dir"
stellar_data_root = Path(__file__).parent / "assets" / "mock_stellar_dataset"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rand_attributes():
    return np.random.randint(0, 1, len(data_root.glob("*")) // 2)


def test_dataset():
    ds = SynImageDataset(data_root=data_root, stellar_dataset_root=stellar_data_root)
    assert len(ds) == 7
    og_0 = np.array(ds[0]["og_img"])
    og_1 = np.array(ds[1]["og_img"])
    assert np.isclose(og_0, og_1).all()
    syn_img_0 = Image.open(data_root / "000-0-00.png")
    syn_img_1 = Image.open(data_root / "000-0-01.png")
    assert np.isclose(np.array(ds[0]["syn_img"]), syn_img_0).all()
    assert np.isclose(np.array(ds[1]["syn_img"]), syn_img_1).all()


def test_run_main(tmp_path: Path):
    for metric in ["clip", "aps", "ips", "goa", "sis"]:
        df = run(
            metric=metric,
            stellar_path=stellar_data_root,
            syn_path=data_root,
            save_dir=tmp_path,
            device=DEVICE,
            batch_size=2,
        )
        df.drop("name", axis=1, inplace=True)
        ref_df = read_res(results_dir, METRICS[metric][0])
        ref_df.drop("name", axis=1, inplace=True)
        assert np.isclose(df, ref_df, rtol=0.001).all()


def test_run_analysis():
    df = make_table(save_dir=results_dir, save=False)
    assert df.to_markdown() == (results_dir / "res.md").read_text()
