"""Inner validation partitions of an outer-training set, and the outer-test guard.

Everything here sees only outer-training rows. `assert_outer_test_isolated`
is called by every fit entry point that is given the unit's outer-test IDs.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import LeakageError, sha256_ids

ID_COL = "patient_id"


def assert_outer_test_isolated(outer_test_ids, **id_sets):
    """Raise if any fit/validation ID set touches the outer-test set."""
    if outer_test_ids is None:
        return
    test = set(map(str, outer_test_ids))
    for name, ids in id_sets.items():
        if ids is None:
            continue
        hit = set(map(str, ids)) & test
        if hit:
            raise LeakageError(f"{len(hit)} outer-test patient(s) in the {name} rows, "
                               f"e.g. {sorted(hit)[:3]}")


def _try_split(idx, strata, fraction, seed, min_size):
    counts = pd.Series(strata).value_counts()
    if (counts < min_size).any():
        return None
    try:
        return train_test_split(idx, test_size=fraction, random_state=seed, stratify=strata)
    except ValueError:
        return None


def inner_split(frame, task, cfg, seed, outer_test_ids=None):
    """Split outer-training rows into inner-train / inner-validation.

    Diagnostic: stratified by mets_liver. Prognostic: os_event x tertile of os_months
    (cut points from these outer-training rows), falling back to os_event alone,
    then to an unstratified split, each fallback recorded.
    Returns (inner_train_frame, inner_val_frame, record).
    """
    iv = cfg["inner_validation"]
    assert_outer_test_isolated(outer_test_ids, outer_training=frame[ID_COL])
    idx = np.arange(len(frame))
    attempts = []
    if task == "diagnostic":
        attempts.append(("mets_liver", frame["mets_liver"].astype(int).to_numpy()))
    else:
        cuts = np.quantile(frame["os_months"], [1 / 3, 2 / 3])
        tert = np.digitize(frame["os_months"], cuts)
        ev = frame["os_event"].astype(int).to_numpy()
        attempts.append(("os_event_x_os_months_tertile",
                         np.array([f"{a}_{b}" for a, b in zip(ev, tert)])))
        attempts.append(("os_event", ev))
    attempts.append(("unstratified", None))
    record = {"fraction": iv["fraction"], "seed": seed, "attempted": []}
    for name, strata in attempts:
        record["attempted"].append(name)
        if strata is None:
            tr, va = train_test_split(idx, test_size=iv["fraction"], random_state=seed)
        else:
            res = _try_split(idx, strata, iv["fraction"], seed, iv["min_stratum_size"])
            if res is None:
                continue
            tr, va = res
        record["stratification"] = name
        break
    if task == "prognostic":
        record["tertile_cut_points_from_outer_training"] = [float(c) for c in cuts]
    tr_df = frame.iloc[np.sort(tr)].reset_index(drop=True)
    va_df = frame.iloc[np.sort(va)].reset_index(drop=True)
    record.update(n_inner_train=int(len(tr_df)), n_inner_val=int(len(va_df)),
                  inner_train_ids_sha256=sha256_ids(tr_df[ID_COL]),
                  inner_val_ids_sha256=sha256_ids(va_df[ID_COL]))
    return tr_df, va_df, record


def attach_embeddings(frame, embeddings, cfg, task):
    """Join a clinical frame's patient IDs to the saved embedding table (inner join)."""
    cols = [c for c in embeddings.columns if c.startswith(tuple(cfg["embedding_prefixes"][task]))]
    out = frame[[ID_COL] + [c for c in frame.columns if c in ("mets_liver", "os_months",
                                                                "os_event")]].merge(
        embeddings[[ID_COL] + cols], on=ID_COL, how="inner", validate="one_to_one")
    if len(out) != len(frame):
        raise ValueError(f"{len(frame) - len(out)} row(s) have no saved embedding (synthetic rows "
                         f"never do: embedding mode is defined for real patients only)")
    return out
