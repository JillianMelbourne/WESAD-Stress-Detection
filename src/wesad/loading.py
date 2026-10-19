"""Loading and validating data: raw WESAD participant files and window-level feature tables."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHANNELS, FS_CHEST, FS_WRIST, LABEL_NAMES, SUBJECTS, PipelineConfig

# Columns that describe a window rather than measure physiology (never used as predictors)
META_COLUMNS = ["subject", "t_start_min", "condition", "stress", "label_purity"]


class DataValidationError(ValueError):
    """Raised when a raw file or feature table fails validation; the message lists every problem."""


def _fail(name, problems):
    raise DataValidationError(f"{name} failed validation:\n  - " + "\n  - ".join(problems))


# =============================================================================
# Raw WESAD files
# =============================================================================
def raw_available(sid, cfg: PipelineConfig) -> bool:
    """True if the raw file (pickle or compact .npz copy) for participant `sid` is present."""
    folder = cfg.data_dir / sid
    return (folder / f"{sid}.pkl").exists() or (folder / f"{sid}_compact.npz").exists()


def validate_subject(sid, data, cfg: PipelineConfig):
    """Check one participant's raw data. Raises DataValidationError listing every problem found."""
    for key in ("signal", "label"):
        if key not in data:
            _fail(sid, [f"file has no '{key}' entry; is this a WESAD pickle?"])

    problems = []
    label = np.asarray(data["label"]).ravel()
    duration = len(label) / FS_CHEST
    for device, chans in CHANNELS.items():
        for ch in chans:
            if ch not in data["signal"].get(device, {}):
                problems.append(f"missing {device} {ch}")
                continue
            arr = np.asarray(data["signal"][device][ch])
            fs = FS_CHEST if device == "chest" else FS_WRIST[ch]
            if abs(len(arr) / fs - duration) > 1:            # allow < 1 s rounding at the end
                problems.append(f"{device} {ch} lasts {len(arr) / fs:,.1f} s but the label lasts {duration:,.1f} s")
            n_nan = int(np.isnan(arr).sum())
            if n_nan:
                problems.append(f"{device} {ch} has {n_nan:,} NaN values")
            elif np.std(arr) == 0:
                problems.append(f"{device} {ch} is flat (sensor off?)")
    unknown = set(np.unique(label).astype(int)) - set(LABEL_NAMES)
    if unknown:
        problems.append(f"unknown label codes {sorted(unknown)}")
    if not np.isin(label, list(cfg.target_map)).any():
        problems.append("no samples with a label used by the target (baseline, stress or amusement)")
    if problems:
        _fail(sid, problems)


def load_subject(sid, cfg: PipelineConfig):
    """Load and validate one participant. Returns dict(subject, chest, wrist, label).

    Reads data/raw/<sid>/<sid>.pkl. If a compact copy <sid>_compact.npz (same arrays as float32) sits next
    to it, that is read instead because it loads several times faster.
    """
    if sid not in SUBJECTS:
        raise ValueError(f"Unknown participant '{sid}'. Valid IDs: {', '.join(SUBJECTS)}")
    folder = cfg.data_dir / sid
    pkl, npz = folder / f"{sid}.pkl", folder / f"{sid}_compact.npz"
    if npz.exists():
        with np.load(npz) as z:
            data = {"label": z["label"],
                    "signal": {dev: {ch: z[f"{dev}_{ch}"] for ch in chans} for dev, chans in CHANNELS.items()}}
    elif pkl.exists():
        with open(pkl, "rb") as f:
            data = pickle.load(f, encoding="latin1")       # saved with Python 2, so latin1 is needed
    else:
        raise FileNotFoundError(f"No raw data for {sid}: expected {pkl}. Download WESAD from UCI and unzip "
                                f"it so each participant has its own folder in {cfg.data_dir}/.")
    validate_subject(sid, data, cfg)
    return {"subject": sid, "chest": data["signal"]["chest"], "wrist": data["signal"]["wrist"],
            "label": np.asarray(data["label"]).ravel().astype(int)}


# =============================================================================
# Window-level feature tables (the pipeline's own output, or a user's upload)
# =============================================================================
def validate_feature_table(df, min_subjects=2, min_windows_per_class=5, max_missing=0.5):
    """Check and clean a window-level feature table. Returns (clean_df, feature_columns, notes).

    Required: a `subject` column, a binary `stress` column (0/1) and at least one numeric feature column.
    Optional: `condition` (enables baseline normalization against true baseline windows).
    Cleaning (reported in `notes`, never silent): rows with a missing target are dropped, and feature
    columns that are more than `max_missing` empty are dropped. Anything that cannot be fixed raises
    DataValidationError listing every problem.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        _fail("Feature table", ["the table is empty or could not be read as a table"])
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    problems, notes = [], []

    missing = [c for c in ("subject", "stress") if c not in df.columns]
    if missing:
        _fail("Feature table", [f"missing required column(s): {missing}. Found: {list(df.columns)[:12]}..."])

    # target: drop rows without a label, then require exactly 0/1
    n_na = int(df["stress"].isna().sum())
    if n_na:
        df = df[df["stress"].notna()]
        notes.append(f"dropped {n_na} rows with a missing `stress` value")
    values = set(pd.unique(df["stress"]))
    if not values <= {0, 1, 0.0, 1.0, True, False}:
        problems.append(f"`stress` must contain only 0 and 1; found {sorted(map(str, values))[:6]}")
    else:
        df["stress"] = df["stress"].astype(int)
        counts = df["stress"].value_counts()
        for cls in (0, 1):
            if counts.get(cls, 0) < min_windows_per_class:
                problems.append(f"only {counts.get(cls, 0)} windows with stress = {cls} "
                                f"(need at least {min_windows_per_class})")

    df["subject"] = df["subject"].astype(str)
    if df["subject"].nunique() < min_subjects:
        problems.append(f"{df['subject'].nunique()} participant(s); need at least {min_subjects} "
                        "to evaluate on held-out people")

    # features: numeric columns that are not metadata, quality flags or QC values
    candidates = [c for c in df.columns if c not in META_COLUMNS and not c.startswith(("flag_", "qc_"))]
    non_numeric = [c for c in candidates if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        notes.append(f"ignored non-numeric column(s): {non_numeric}")
    features = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_bz")]
    too_empty = [c for c in features if df[c].isna().mean() > max_missing]
    if too_empty:
        notes.append(f"dropped feature(s) more than {max_missing:.0%} missing: {too_empty}")
        features = [c for c in features if c not in too_empty]
    infinite = [c for c in features if np.isinf(df[c]).any()]
    if infinite:
        df[infinite] = df[infinite].replace([np.inf, -np.inf], np.nan)
        notes.append(f"replaced infinite values with NaN in: {infinite}")
    if not features:
        problems.append("no usable numeric feature columns")

    if problems:
        _fail("Feature table", problems)
    return df.reset_index(drop=True), features, notes


def profile_table(df, features):
    """One row per feature: type, % missing, and summary statistics."""
    prof = pd.DataFrame({
        "dtype": df[features].dtypes.astype(str),
        "% missing": 100 * df[features].isna().mean(),
        "mean": df[features].mean(),
        "sd": df[features].std(),
        "min": df[features].min(),
        "max": df[features].max(),
    })
    prof.index.name = "feature"
    return prof


def upload_feature_table(dest_dir=Path("data/uploads")):
    """Ask the user to pick a CSV from their computer (Google Colab only). Returns the saved path."""
    try:
        from google.colab import files                       # only exists inside Colab
    except ImportError:
        raise RuntimeError("Uploading through the browser only works in Google Colab. Outside Colab, put your "
                           "CSV in data/uploads/ and set UPLOAD_PATH in the run-settings cell.") from None
    uploaded = files.upload()                               # opens the browser file picker
    if not uploaded:
        raise RuntimeError("No file was uploaded.")
    if len(uploaded) > 1:
        print(f"More than one file uploaded; using the first ({next(iter(uploaded))}).")
    name, content = next(iter(uploaded.items()))
    if not name.lower().endswith(".csv"):
        raise DataValidationError(f"'{name}' is not a .csv file. Upload a comma-separated table.")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name
    path.write_bytes(content)
    return path


# =============================================================================
# Summaries of one raw recording
# =============================================================================
def describe_recording(subj):
    """One row per channel: shape, sampling rate, duration and NaN count."""
    from .config import UNITS
    rows = []
    for device in ("chest", "wrist"):
        for ch, arr in subj[device].items():
            arr = np.asarray(arr)
            fs = FS_CHEST if device == "chest" else FS_WRIST[ch]
            rows.append({"device": device, "channel": ch, "shape": str(arr.shape), "fs_Hz": fs,
                         "duration_min": len(arr) / fs / 60, "units": UNITS[ch], "n_NaN": int(np.isnan(arr).sum())})
    label = subj["label"]
    rows.append({"device": "label", "channel": "label", "shape": str(label.shape), "fs_Hz": FS_CHEST,
                 "duration_min": len(label) / FS_CHEST / 60, "units": "code", "n_NaN": 0})
    return pd.DataFrame(rows)


def target_summary(label, cfg: PipelineConfig):
    """Minutes and share of the recording in each condition, and how each maps to the binary target."""
    counts = pd.Series(label).value_counts().sort_index()
    return pd.DataFrame({
        "condition": [LABEL_NAMES[int(k)] for k in counts.index],
        "minutes": counts.values / FS_CHEST / 60,
        "% of recording": 100 * counts.values / len(label),
        "binary target": [f"{cfg.target_map[k]} ({'stress' if cfg.target_map[k] else 'non-stress'})"
                          if k in cfg.target_map else "excluded" for k in counts.index],
    }, index=counts.index.rename("label"))
