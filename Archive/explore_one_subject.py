"""
Plot one WESAD subject's chest and wrist signals on a shared time axis.

Run in Colab or Jupyter as a cell (or `python explore_one_subject.py`).
Change SUBJECT and DATA_DIR below.

The key idea: the .pkl arrays have no time column. Time is
    row number / sampling rate
and every array's row 0 is the same moment (the authors synced the devices).
So we NEVER delete rows from the raw arrays. We build a time axis for each
signal first, and select data by time (or by a label mask applied to the
700 Hz chest/label arrays) afterwards.
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SUBJECT = "S2"
DATA_DIR = Path("data/raw")          # folder that contains S2/S2.pkl, S3/S3.pkl, ...

FS_CHEST = 700                       # all chest signals and the labels
FS_WRIST = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
LABEL_NAMES = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
LABEL_COLORS = {1: "#9db4c0", 2: "#e07a5f", 3: "#81b29a", 4: "#c9b6e4"}

# ---------------------------------------------------------------- load
with open(DATA_DIR / SUBJECT / f"{SUBJECT}.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")      # files were saved with Python 2

chest = data["signal"]["chest"]
wrist = data["signal"]["wrist"]
label = np.asarray(data["label"]).ravel()

# ---------------------------------------------------------------- time axes
def time_axis(n_rows, fs):
    """Seconds since the start of the synced recording for each row."""
    return np.arange(n_rows) / fs

t_chest = time_axis(len(label), FS_CHEST)                       # also the label's time axis
t_wrist = {k: time_axis(len(v), FS_WRIST[k]) for k, v in wrist.items()}

print(f"{SUBJECT}: chest {len(label):,} rows / 700 Hz = {t_chest[-1] / 60:.1f} min")
for k, v in wrist.items():
    print(f"        wrist {k:<4} {len(v):>9,} rows / {FS_WRIST[k]:>2} Hz = {t_wrist[k][-1] / 60:.1f} min")

# Dropping unwanted labels the safe way: a mask, applied to the signal AND its time axis together.
# The rows you keep still carry their true time, so nothing shifts.
keep = np.isin(label, [1, 2, 3])
print(f"Rows with label 1-3: {keep.mean():.0%} of the recording "
      f"(e.g. ECG kept = chest['ECG'][keep], with times t_chest[keep])")

# Label periods as (label, start_s, end_s), used to shade the plots
change = np.flatnonzero(np.diff(label) != 0) + 1
starts, ends = np.r_[0, change], np.r_[change, len(label)]
periods = [(int(label[a]), t_chest[a], t_chest[b - 1]) for a, b in zip(starts, ends)
           if int(label[a]) in LABEL_COLORS]

def shade(ax):
    for lab, a, b in periods:
        ax.axvspan(a / 60, b / 60, color=LABEL_COLORS[lab], alpha=0.35, lw=0)

def every_nth(x, fs, display_fs):
    """Thin out samples for plotting only (a 100-min ECG has 4 million points)."""
    step = max(1, int(fs // display_fs))
    return x[::step], step

# ---------------------------------------------------------------- figure 1: whole session
acc_wrist = np.linalg.norm(np.asarray(wrist["ACC"], float) / 64, axis=1)   # 1/64 g -> g
acc_chest = np.linalg.norm(np.asarray(chest["ACC"], float), axis=1)

panels = [  # (title, signal, sampling rate, display rate)
    ("Chest EDA (µS)",         np.ravel(chest["EDA"]),  FS_CHEST, 4),
    ("Wrist EDA (µS)",         np.ravel(wrist["EDA"]),  4,        4),
    ("Chest respiration (%)",  np.ravel(chest["Resp"]), FS_CHEST, 10),
    ("Wrist skin temp (°C)",   np.ravel(wrist["TEMP"]), 4,        4),
    ("Chest ACC (g)",          acc_chest,               FS_CHEST, 4),
    ("Wrist ACC (g)",          acc_wrist,               32,       4),
]
fig, axes = plt.subplots(len(panels), 1, figsize=(13, 11), sharex=True)
for ax, (title, y, fs, dfs) in zip(axes, panels):
    y_plot, step = every_nth(y, fs, dfs)
    t_plot = time_axis(len(y), fs)[::step]
    ax.plot(t_plot / 60, y_plot, lw=0.6, color="#333333")
    shade(ax)
    ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in LABEL_COLORS.values()]
axes[0].legend(handles, [LABEL_NAMES[k] for k in LABEL_COLORS], ncol=4, frameon=False,
               loc="lower left", bbox_to_anchor=(0, 1.02))
axes[-1].set_xlabel("Minutes since start of synced recording (unshaded = label 0 or 5-7)")
fig.suptitle(f"{SUBJECT}: whole session, chest and wrist on one time axis", y=0.995)
fig.tight_layout()
plt.show()

# ---------------------------------------------------------------- figure 2: 10-second zoom
# Pick a moment by TIME (1 minute into baseline and 1 minute into stress), then convert
# to rows separately for each signal. Heartbeats in the chest ECG should line up with
# pulses in the wrist BVP (each pulse arrives a fraction of a second after its beat).
def window(x, fs, t0, seconds):
    a, b = int(t0 * fs), int((t0 + seconds) * fs)
    return time_axis(len(x), fs)[a:b], np.ravel(x)[a:b]

ZOOM_S = 10
fig, axes = plt.subplots(2, 2, figsize=(13, 5), sharex="col")
for col, lab in enumerate([1, 2]):
    t0 = t_chest[np.argmax(label == lab)] + 60          # first row with this label, + 60 s
    for row, (name, x, fs) in enumerate([("Chest ECG (mV)", chest["ECG"], FS_CHEST),
                                         ("Wrist BVP (a.u.)", wrist["BVP"], FS_WRIST["BVP"])]):
        t, y = window(x, fs, t0, ZOOM_S)
        ax = axes[row, col]
        ax.plot(t, y, lw=0.8, color=LABEL_COLORS[lab] if row else "#333333")
        ax.set_ylabel(name if col == 0 else "")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, col].set_title(f"{LABEL_NAMES[lab]}: {t0 / 60:.1f}-{(t0 + ZOOM_S) / 60:.1f} min")
    axes[1, col].set_xlabel("Seconds since start of synced recording")
fig.suptitle(f"{SUBJECT}: the same 10 seconds seen by both devices", y=1.0)
fig.tight_layout()
plt.show()
