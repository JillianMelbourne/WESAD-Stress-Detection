"""From raw WESAD signals to one row per 60-second window.

preprocess()          filter each whole recording once
window_features()     predictors for one window
extract_windows()     slide the window over one participant, keep labelled windows, add quality flags
build_feature_table() run all participants and save the table with the settings used
"""
import json

import numpy as np
import pandas as pd
from scipy import signal

from .config import EXCLUDE_FEATURES, FS_CHEST, LABEL_NAMES, SUBJECTS, PipelineConfig
from .loading import DataValidationError, load_subject

# Predictor columns produced by window_features(), in table order
FEATURE_COLUMNS = [
    "chest_hr_bpm", "chest_rmssd_ms", "wrist_hr_bpm",
    "chest_eda_uS", "chest_eda_slope_uS_per_min", "wrist_eda_uS", "wrist_scr_count",
    "resp_rate_bpm", "resp_interval_sd_s",
    "chest_temp_C", "wrist_temp_C", "wrist_temp_slope_C_per_min",
    "emg_rms_mV", "chest_acc_sd_g", "wrist_acc_sd_g",
]
FLAG_COLUMNS = ["flag_hr_missing", "flag_hr_disagree", "flag_ecg_artifact", "flag_temp_range", "flag_eda_contact"]


# ---- small signal helpers ------------------------------------------------------
def sos(kind, cut, fs, order=4):
    """Second-order sections for a Butterworth filter. kind: 'low', 'high' or 'band'."""
    return signal.butter(order, cut, btype=kind, fs=fs, output="sos")


def beats_to_hr(peaks, fs, cfg: PipelineConfig):
    """Heart rate (bpm), RMSSD (ms) and share of rejected beat intervals from beat positions.
    Intervals outside cfg.hr_range_bpm are rejected; with fewer than cfg.min_beats left, HR and RMSSD are NaN."""
    ibi = np.diff(peaks) / fs                                    # seconds between beats
    ok = (ibi > 60 / cfg.hr_range_bpm[1]) & (ibi < 60 / cfg.hr_range_bpm[0])
    rejected = 1 - ok.mean() if len(ibi) else 1.0
    ibi = ibi[ok]
    if len(ibi) < cfg.min_beats:
        return np.nan, np.nan, rejected
    return 60 / ibi.mean(), 1000 * np.sqrt(np.mean(np.diff(ibi) ** 2)), rejected


def slope_per_min(x, fs):
    """Linear trend of x in units per minute."""
    return np.polyfit(np.arange(len(x)) / fs / 60, x, 1)[0]


# ---- pipeline steps ----------------------------------------------------------------
def preprocess(subj, cfg: PipelineConfig):
    """Filter each whole recording once (zero-phase). Returns {name: (signal, sampling rate in Hz)}."""
    chest, wrist = subj["chest"], subj["wrist"]
    ecg = signal.sosfiltfilt(sos("band", cfg.ecg_band, FS_CHEST), np.ravel(chest["ECG"]))
    # QRS energy: squared derivative smoothed with a 150 ms moving average (105 samples at 700 Hz)
    ecg_energy = np.convolve(np.diff(ecg, prepend=ecg[0]) ** 2, np.ones(105) / 105, mode="same")
    # respiration: low-pass 5 Hz, downsample 700 -> 10 Hz, band-pass to breathing frequencies
    resp = signal.sosfiltfilt(sos("low", 5, FS_CHEST), np.ravel(chest["Resp"]))[::70]
    resp = signal.sosfiltfilt(sos("band", cfg.resp_band, 10, order=2), resp)
    return {
        "ecg_energy": (ecg_energy, FS_CHEST),
        "bvp":        (signal.sosfiltfilt(sos("band", cfg.bvp_band, 64, order=2), np.ravel(wrist["BVP"])), 64),
        "resp":       (resp, 10),
        "chest_eda":  (signal.sosfiltfilt(sos("low", cfg.eda_lowpass, FS_CHEST), np.ravel(chest["EDA"]))[::175], 4),
        "wrist_eda":  (np.ravel(wrist["EDA"]).astype(float), 4),
        "chest_temp": (np.ravel(chest["Temp"])[::175].astype(float), 4),
        "wrist_temp": (np.ravel(wrist["TEMP"]).astype(float), 4),
        "emg":        (signal.sosfiltfilt(sos("high", cfg.emg_highpass, FS_CHEST), np.ravel(chest["EMG"])), FS_CHEST),
        "acc_chest":  (np.linalg.norm(np.asarray(chest["ACC"], float), axis=1), FS_CHEST),
        "acc_wrist":  (np.linalg.norm(np.asarray(wrist["ACC"], float) / 64, axis=1), 32),
    }


def cut(sig, name, t0, window_s):
    """The window_s-second slice of signal `name` starting at t0 seconds."""
    x, fs = sig[name]
    return x[int(t0 * fs): int((t0 + window_s) * fs)]


def window_features(sig, t0, cfg: PipelineConfig):
    """All predictors for the window starting at t0 seconds, plus two beat-rejection rates used for QC."""
    w = cfg.window_s
    max_hr = cfg.hr_range_bpm[1]
    e = cut(sig, "ecg_energy", t0, w)
    pk, _ = signal.find_peaks(e, distance=int(60 / max_hr * FS_CHEST), height=0.3 * np.percentile(e, 99))
    b = cut(sig, "bvp", t0, w)
    bpk, _ = signal.find_peaks(b, distance=int(60 / max_hr * 64), prominence=0.3 * b.std())
    r = cut(sig, "resp", t0, w)
    rpk, _ = signal.find_peaks(r, distance=15, prominence=0.3 * r.std())
    breath_iv = np.diff(rpk) / 10
    we, ce, wt = cut(sig, "wrist_eda", t0, w), cut(sig, "chest_eda", t0, w), cut(sig, "wrist_temp", t0, w)
    scr, _ = signal.find_peaks(we, prominence=0.01, distance=4)
    hr_c, rmssd_c, rej_c = beats_to_hr(pk, FS_CHEST, cfg)
    hr_w, _, rej_w = beats_to_hr(bpk, 64, cfg)       # wrist RMSSD is not used: 64 Hz pulse timing is too coarse
    return {
        "chest_hr_bpm": hr_c, "chest_rmssd_ms": rmssd_c, "wrist_hr_bpm": hr_w,
        "chest_eda_uS": ce.mean(), "chest_eda_slope_uS_per_min": slope_per_min(ce, 4),
        "wrist_eda_uS": we.mean(), "wrist_scr_count": len(scr),
        "resp_rate_bpm": len(rpk) * 60 / w,
        "resp_interval_sd_s": breath_iv.std() if len(breath_iv) >= 3 else np.nan,
        "chest_temp_C": cut(sig, "chest_temp", t0, w).mean(),
        "wrist_temp_C": wt.mean(), "wrist_temp_slope_C_per_min": slope_per_min(wt, 4),
        "emg_rms_mV": np.sqrt(np.mean(cut(sig, "emg", t0, w) ** 2)),
        "chest_acc_sd_g": cut(sig, "acc_chest", t0, w).std(),
        "wrist_acc_sd_g": cut(sig, "acc_wrist", t0, w).std(),
        "qc_chest_ibi_rejected": rej_c, "qc_wrist_ibi_rejected": rej_w,
    }


def window_label(lab):
    """Most common label in a window and the share of samples that carry it."""
    values, counts = np.unique(lab, return_counts=True)
    i = counts.argmax()
    return int(values[i]), counts[i] / len(lab)


def add_quality_flags(win, cfg: PipelineConfig):
    """Add one boolean column per quality rule, plus flag_any."""
    def temp_ok(col):                                    # NaN = excluded channel, not flagged
        return win[col].between(*cfg.temp_range_c) | win[col].isna()
    flags = pd.DataFrame({
        "flag_hr_missing":   win[["chest_hr_bpm", "wrist_hr_bpm"]].isna().any(axis=1),
        "flag_hr_disagree":  (win["chest_hr_bpm"] - win["wrist_hr_bpm"]).abs() > cfg.hr_agreement_bpm,
        "flag_ecg_artifact": win["qc_chest_ibi_rejected"] > cfg.max_ibi_rejected,
        "flag_temp_range":   ~(temp_ok("wrist_temp_C") & temp_ok("chest_temp_C")),
        "flag_eda_contact":  win["wrist_eda_uS"] < cfg.min_wrist_eda_us,
    }, index=win.index)
    flags["flag_any"] = flags.any(axis=1)
    return pd.concat([win, flags], axis=1)


def extract_windows(subj, cfg: PipelineConfig, sig=None):
    """One row per usable window for one participant, with predictors and quality flags."""
    sig = preprocess(subj, cfg) if sig is None else sig
    label, sid = subj["label"], subj["subject"]
    rows = []
    for t0 in np.arange(0, len(label) / FS_CHEST - cfg.window_s + 1e-9, cfg.step_s):
        lab, purity = window_label(label[int(t0 * FS_CHEST): int((t0 + cfg.window_s) * FS_CHEST)])
        if lab not in cfg.target_map or purity < cfg.min_label_purity:
            continue                                     # transition, meditation, ignore, or mixed window
        rows.append({"subject": sid, "t_start_min": t0 / 60, "condition": LABEL_NAMES[lab],
                     "stress": cfg.target_map[lab], "label_purity": purity, **window_features(sig, t0, cfg)})
    if not rows:
        raise DataValidationError(f"{sid}: no windows passed the label rules; check target_map and min_label_purity")
    win = pd.DataFrame(rows)
    for col in EXCLUDE_FEATURES.get(sid, []):           # readme says this channel is unreliable
        win[col] = np.nan
    return add_quality_flags(win, cfg)


def build_feature_table(cfg: PipelineConfig, subjects=SUBJECTS):
    """Run the pipeline on every participant, one at a time, and save the table plus the settings used.

    A participant whose raw file is missing or fails validation is skipped with a message.
    Returns the combined table.
    """
    tables, skipped = [], {}
    for sid in subjects:
        subj = None
        try:
            subj = load_subject(sid, cfg)
            tables.append(extract_windows(subj, cfg))
            print(f"  {sid}: {len(tables[-1])} windows")
        except (FileNotFoundError, DataValidationError) as err:
            skipped[sid] = str(err)
            print(f"  {sid}: SKIPPED - {err}")
        finally:
            del subj                                     # free ~1 GB before the next participant
    if not tables:
        raise FileNotFoundError(f"No participant could be processed from {cfg.data_dir.resolve()}.")
    feats = pd.concat(tables, ignore_index=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cfg.feature_table, index=False)
    cfg.feature_table.with_suffix(".json").write_text(json.dumps(cfg.settings(), indent=2))
    print(f"Saved {len(feats):,} windows from {len(tables)} participants to {cfg.feature_table}"
          + (f" (skipped {list(skipped)})" if skipped else ""))
    return feats


def saved_settings_match(cfg: PipelineConfig):
    """True/False if the saved table's settings match cfg; None if no settings file exists."""
    meta = cfg.feature_table.with_suffix(".json")
    if not meta.exists():
        return None
    return json.loads(meta.read_text()) == json.loads(json.dumps(cfg.settings()))
