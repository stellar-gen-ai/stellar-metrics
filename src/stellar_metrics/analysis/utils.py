from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def read_res(results_dir: Path, name: str) -> pd.DataFrame:
    res = list(results_dir.rglob(f"{name}*.csv"))
    if len(res) != 1:
        raise ValueError(f"Improperly configured results in {results_dir} for {name}*.")
    df = pd.read_csv(res[0], index_col=0)
    return df


def parse_exclusion(results_dir: Path) -> pd.Series:
    """
    Finds which images failed to follow the prompt, as in there
    is insignificant increase in CLIP_t score as compared to a naive
    CLIP_n

    Parameters
    ----------
    results_dir : Path
        The directory where the clip results are included.

    Returns
    -------
    pd.Series
        _description_

    Raises
    ------
    Exception
        _description_
    """
    try:
        clip_df = read_res(results_dir, "clip").set_index("name")
    except ValueError:
        raise Exception(
            "CLIP results are required for making the table. "
            "i.e. 'python -m stellar_metrics --metric clip'"
        )
    return (clip_df["clip_n"] + clip_df["clip_n"].std() * 2) < clip_df["clip_t"]


def attribute_pres(save_dir: Path):
    try:
        method_df = read_res(save_dir, "attribute_preservation").set_index("name")
    except:
        return []

    if len(method_df) > 0:
        preds = sorted([c for c in method_df.columns if c.endswith("_pred")])
        targets = sorted([c for c in method_df.columns if c.endswith("_target")])

        mask = parse_exclusion(save_dir)
        auc_scores = []
        for p_c, t_c in zip(preds, targets):
            if p_c.strip("_pred") in t_c and method_df[t_c].nunique() > 1:
                method_df[p_c][~mask] *= -1
                auc_scores.append(roc_auc_score(method_df[t_c], method_df[p_c]))

        attr_preservation = np.mean(auc_scores)
    else:
        attr_preservation = 0
    return [{"metric": "attr", "value": attr_preservation}]


def some_scores(
    results_dir: Path,
    score_name: str,
):
    try:
        method_df = read_res(results_dir, score_name)
    except:
        return []

    val = method_df.drop("name", axis=1).mean()
    std = method_df.drop("name", axis=1).std()
    table = []
    for k, v in val.to_dict().items():
        table.append({
            "metric": k,
            "value": float(v),
            "std": float(std.loc[k]),
        })
    return table


def _identity_score(results_dir: Path, metric_name: str):
    method_df = read_res(results_dir, metric_name).set_index("name")

    mask = parse_exclusion(results_dir)
    mask = method_df.join(mask.to_frame(), lsuffix="_")[0]
    method_df.loc[
        ~mask.values,
        metric_name,
    ] = 0

    return method_df


def identity_score(results_dir: Path):
    table = []

    for metric_name in ["identity_preservation", "identity_stability"]:
        method_df = _identity_score(
            results_dir=results_dir,
            metric_name=metric_name,
        )
        val = method_df.mean()
        std = method_df.std()
        for k, v in val.to_dict().items():
            table.append({"metric": k, "value": v, "std": std.loc[k]})
    return table
