"""Settings for the WESAD stress pipeline.

Everything that changes the feature table lives in `PipelineConfig`, so a run is fully described by one
object. Fixed facts about the dataset (sampling rates, channel names, label codes) are module constants.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---- Fixed facts about WESAD -------------------------------------------------
SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
            "S11", "S13", "S14", "S15", "S16", "S17"]          # S1 and S12 were dropped by the authors
FS_CHEST = 700                                                  # sampling rate of chest sensors (Hz)
FS_WRIST = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}          # sampling rate of wrist sensors (Hz)
CHANNELS = {"chest": ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"],
            "wrist": ["ACC", "BVP", "EDA", "TEMP"]}
UNITS = {"ACC": "g", "ECG": "mV", "EMG": "mV", "EDA": "µS", "Temp": "°C", "Resp": "%",
         "BVP": "a.u.", "TEMP": "°C"}
LABEL_NAMES = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation",
               5: "ignore", 6: "ignore", 7: "ignore"}

# ---- Plot styling --------------------------------------------------------------
COLORS = {1: "#9db4c0", 2: "#e07a5f", 3: "#81b29a", 4: "#c9b6e4"}   # by WESAD label code
CONDITION_ORDER = ["baseline", "amusement", "stress"]

# ---- Known problems from each participant's readme -----------------------------
# Features listed here are set to NaN for that participant.
EXCLUDE_FEATURES = {
    "S2":  ["chest_temp_C"],   # "RespiBAN temperature sensor was not fully attached"
    "S17": ["chest_temp_C"],   # same note
}
# Notes that do not invalidate a signal but matter when reading results.
README_NOTES = {
    "S3":  "Baseline recorded in a sunny workplace (skin temperature).",
    "S5":  "May have fallen asleep during the first meditation (not used here).",
    "S6":  "Reported the TSST interview was not really stressful.",
    "S8":  "Stressful day before the study; felt cold in the stress room (skin temperature).",
    "S9":  "Reported feeling ill on the study day.",
    "S15": "Did not believe the TSST cover story.",
    "S16": "Felt cold in the stress room (skin temperature).",
}


@dataclass
class PipelineConfig:
    """All settings that affect the feature table. Change a value, then rebuild from raw data."""
    # paths
    data_dir: Path = Path("data/raw")              # one folder per participant: data/raw/S2/S2.pkl, ...
    processed_dir: Path = Path("data/processed")   # feature tables are written here
    fig_dir: Path = Path("figures")
    save_figures: bool = True

    # windowing
    window_s: int = 60                 # window length (s)
    step_s: int = 30                   # step between window starts (s); 30 = 50% overlap
    min_label_purity: float = 1.0      # share of a window's samples that must carry one label

    # target: WESAD label -> binary stress; every other label is excluded
    target_map: dict = field(default_factory=lambda: {1: 0, 3: 0, 2: 1})

    # filters (Hz)
    ecg_band: tuple = (5, 15)          # QRS energy
    bvp_band: tuple = (0.7, 3.5)       # 42-210 bpm pulse
    resp_band: tuple = (0.1, 0.7)      # 6-42 breaths/min
    eda_lowpass: float = 1.0
    emg_highpass: float = 20           # minimizes motion artifacts 

    # quality thresholds
    hr_range_bpm: tuple = (40, 180)    # beat intervals outside this range are rejected
    min_beats: int = 10                # fewer valid beats in a window -> HR/HRV set to NaN
    hr_agreement_bpm: float = 10       # chest vs wrist HR gap that flags a likely BVP motion artifact
    max_ibi_rejected: float = 0.2      # > 20% of chest beat intervals rejected -> ECG artifact flag
    temp_range_c: tuple = (25, 40)     # plausible skin temperature
    min_wrist_eda_us: float = 0.05     # wrist EDA this close to zero suggests lost electrode contact

    def __post_init__(self):
        self.data_dir, self.processed_dir, self.fig_dir = map(Path, (self.data_dir, self.processed_dir, self.fig_dir))
        if self.window_s <= 0 or self.step_s <= 0:
            raise ValueError("window_s and step_s must be positive")
        if not 0 < self.min_label_purity <= 1:
            raise ValueError("min_label_purity must be in (0, 1]")
        if set(self.target_map.values()) != {0, 1}:
            raise ValueError("target_map must map WESAD labels to both 0 and 1")

    @property
    def feature_table(self) -> Path:
        return self.processed_dir / "wesad_windows.csv"

    def settings(self) -> dict:
        """JSON-friendly settings that change the feature table (paths and display options excluded)."""
        skip = {"data_dir", "processed_dir", "fig_dir", "save_figures"}
        out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items() if k not in skip}
        out["target_map"] = {str(k): v for k, v in self.target_map.items()}
        out["exclude_features"] = EXCLUDE_FEATURES
        return out
