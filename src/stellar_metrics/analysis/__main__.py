import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stellar_metrics.analysis.utils import attribute_pres, identity_score, some_scores

UP = "($\\uparrow$)"
DOWN = "($\\downarrow$)"


def make_bold(x, is_max):
    top = -1 if is_max else 0
    best = np.argsort(x.values)[top]
    x.iloc[best] = "\\textbf{" + f"{x.iloc[best]:.3f}" + "}"
    return x


def make_table(save_dir: Path, save: bool = True):
    table = []
    table += attribute_pres(save_dir)
    table += some_scores(save_dir, "object")
    table += some_scores(save_dir, "pick_score")
    table += some_scores(save_dir, "clip")
    table += some_scores(save_dir, "aesth")
    table += some_scores(save_dir, "dino")
    table += some_scores(save_dir, "human_pref")
    table += some_scores(save_dir, "image_reward")
    table += some_scores(save_dir, "dreamsim")
    table += identity_score(save_dir)

    df = pd.DataFrame(table).set_index("metric").astype(float)

    if save:
        save_path = save_dir / f"res_{int(time.time())}.md"

        df.to_markdown(save_path, index=True)

    return df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--save-dir",
        type=Path,
    )
    kwargs = vars(args.parse_args())
    make_table(**kwargs)
