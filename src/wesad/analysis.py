"""Normalization and effect-size helpers used on the window-level feature table."""
import re

import numpy as np
import pandas as pd


def subject_order(subjects):
    """Sort participant IDs naturally (S2, S3, ..., S10) rather than as text (S10, S11, S2)."""
    key = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]
    return sorted(pd.unique(pd.Series(subjects)), key=key)


def baseline_normalize(df, features, by="subject"):
    """Add <feature>_bz columns: (value - participant's resting mean) / participant's SD over all windows.

    Resting reference = windows with condition == "baseline" when that column exists and every participant
    has baseline windows; otherwise all non-stress windows (stress == 0) are used. The SD uses all of the
    participant's windows, which needs no labels. Returns (df_with_bz_columns, description of the reference).
    """
    if "condition" in df.columns and (df["condition"] == "baseline").groupby(df[by]).any().all():
        ref_mask, ref_name = df["condition"] == "baseline", "baseline windows"
    else:
        ref_mask, ref_name = df["stress"] == 0, "non-stress windows (no baseline label available)"
    mu = df[ref_mask].groupby(by)[features].mean().reindex(df[by]).to_numpy()
    sd = df.groupby(by)[features].std().replace(0, np.nan).reindex(df[by]).to_numpy()  # SD 0 -> NaN, not inf
    z = pd.DataFrame((df[features].to_numpy() - mu) / sd, index=df.index, columns=[f"{f}_bz" for f in features])
    return pd.concat([df.drop(columns=z.columns, errors="ignore"), z], axis=1), ref_name


def cohens_d(x1, x0):
    """Cohen's d for x1 vs x0 using the pooled SD; NaN if either group has fewer than 2 values."""
    x1, x0 = pd.Series(x1).dropna(), pd.Series(x0).dropna()
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    pooled = np.sqrt(((len(x1) - 1) * x1.var() + (len(x0) - 1) * x0.var()) / (len(x1) + len(x0) - 2))
    return (x1.mean() - x0.mean()) / pooled if pooled > 0 else np.nan


def effect_sizes(df, features):
    """Per-participant Cohen's d (participants x features) and a summary ranked by |median d|.

    Descriptive effect sizes only (no p-values): windows are not independent, so window-level tests are invalid.
    Needs <feature>_bz columns from baseline_normalize for the pooled normalized d.
    """
    order = subject_order(df["subject"])
    per_subject = pd.DataFrame({f: {s: cohens_d(g.loc[g.stress == 1, f], g.loc[g.stress == 0, f])
                                    for s, g in df.groupby("subject")} for f in features}).reindex(order)
    rows = []
    for f in features:
        d = per_subject[f].dropna()
        st, non = df.loc[df.stress == 1, f"{f}_bz"].dropna(), df.loc[df.stress == 0, f"{f}_bz"].dropna()
        rows.append({
            "feature": f,
            "median within-participant d": d.median(),
            "same direction (n/N)": f"{int((np.sign(d) == np.sign(d.median())).sum())}/{len(d)}",
            "pooled raw d": cohens_d(df.loc[df.stress == 1, f], df.loc[df.stress == 0, f]),
            "pooled normalized d": cohens_d(st, non),
        })
    table = pd.DataFrame(rows).set_index("feature")
    ranked = table["median within-participant d"].abs().sort_values(ascending=False).index
    return per_subject[ranked], table.loc[ranked]
