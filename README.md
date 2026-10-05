# WESAD Stress Detection: Dataset Selection and Project Planning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JillianMelbourne/WESAD-Stress-Detection/blob/main/WESAD_dataset_exploration.ipynb)

**Jillian Melbourne** · HSE.751 (Dartmouth Health Data Science), Fall 2026 · Week 3: Dataset Selection and Final Project Planning Lab

## Purpose
This repository is the starting point for my final course project: a supervised binary classifier that detects acute psychosocial stress from wearable physiological signals. This first stage selects and justifies the dataset, turns raw sensor streams into an analysis table, explores and quality-checks it, and lays out the modelling plan.

## Why this dataset
My background is in signal processing for deep brain stimulation and wearables, so this dataset feels like familiar territory with new tools. Stress is largely invisible and hard to measure beyond self-report, yet it has profound health effects, from cardiovascular disease to mental health. Wearables offer one of the most promising routes to measuring it passively and continuously. The goal is a complete pipeline, from raw signal processing through model evaluation.

## Dataset
**WESAD (Wearable Stress and Affect Detection)**, Schmidt et al., 2018.
UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>

15 participants wore a chest sensor (RespiBAN: ECG, EDA, EMG, respiration, temperature, 3-axis acceleration, all at 700 Hz) and a wrist sensor (Empatica E4: blood volume pulse 64 Hz, 3-axis acceleration 32 Hz, EDA 4 Hz, skin temperature 4 Hz) during a lab protocol with baseline, amusement, stress (Trier Social Stress Test), and meditation conditions.

The raw data (~2.5 GB) are **not stored in this repository**. The notebook downloads them from UCI automatically.

### How the project criteria are met
| Requirement | WESAD |
|---|---|
| Binary classification | `stress` = 1 for the TSST stress condition, 0 for baseline + amusement (the authors' binary task) |
| ≥ 300 observations | ~1,000 single-condition 60-second windows across 15 subjects (exact count printed in Section 8 of the notebook) |
| ≥ 10 predictors | 32 engineered features per window (18 chest, 14 wrist) |

## What the notebook does
1. **Data selection:** dataset rationale, criteria check, backup dataset (UCI Mushroom).
2. **Import and raw structure:** downloads and extracts the per-subject pickles, prints the nested signal structure, sampling rates, units, and label coding.
3. **Feature engineering:** aligns the two devices in time, cuts 60-s windows (30-s step), keeps only windows with a single condition, and extracts heart rate/HRV (ECG, BVP), electrodermal activity, respiration, temperature, EMG, and movement features.
4. **Descriptive statistics:** table structure, target distribution, windows per subject, feature summaries, class-wise means with Cohen's *d*.
5. **Visualizations (5):** raw signals by condition, class balance per subject, key features by class, per-subject variability, feature correlation matrix.
6. **Data quality:** raw NaNs, device alignment, label coding (transient/ignored samples), windowing losses, feature missingness, implausible values, outliers; discussion of preprocessing and challenges.
7. **Project planning:** prediction problem, target, leave-one-subject-out validation, candidate models, and evaluation metrics (primary: macro F1; secondary: AUROC, balanced accuracy, sensitivity/specificity).
8. **Requirements check:** assertions that the analysis table meets the project criteria, plus session info.

## Repository contents
| Path | Description |
|---|---|
| `WESAD_dataset_exploration.ipynb` | Main notebook (run in Google Colab) |
| `requirements.txt` | Package versions for running locally |
| `figures/` | Figures written by the notebook |
| `data/processed/wesad_window_features.csv` | Window-level feature table written by the notebook (one row per 60-s window) |
| `README.md` | This file |

## How to run

### Option A: Google Colab (recommended)
1. Click the **Open in Colab** badge above.
2. Choose **Runtime → Run all**. No GPU is needed.
3. The first run downloads ~2.5 GB from UCI and extracts the 15 subject files (a few minutes), then builds the feature table (a few more minutes).
4. *Optional:* set `USE_DRIVE_CACHE = True` in Section 2.1 to keep the extracted data (~5 GB) in Google Drive, so later runs skip the download.

### Option B: Run locally
```bash
git clone https://github.com/JillianMelbourne/WESAD-Stress-Detection.git
cd WESAD-Stress-Detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook WESAD_dataset_exploration.ipynb
```
Data download to `data/raw/` (ignored by git). If you already have WESAD, point the notebook at it with `export WESAD_DATA_DIR=/path/to/WESAD` before launching Jupyter. Any folder containing `S2/S2.pkl` … `S17/S17.pkl` works.

### Notes on reproducibility
- Run the notebook top to bottom. Later cells depend on objects created earlier (`df`, `FEATURES`, the QC tables).
- Feature extraction is deterministic; the only random step (jitter in one plot) is seeded with `RANDOM_SEED = 42`.
- Feature extraction holds one subject in memory at a time (~1 GB peak), well within a standard Colab runtime.

## Software
| Package | Version |
|---|---|
| Python | 3.11 or newer |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scipy | 1.16.3 |
| matplotlib | 3.10.0 |

All are preinstalled in Google Colab. The notebook prints the versions it runs with in its first cell.

## Limitations (summary)
- Only 15 participants: windows from the same person are correlated, so evaluation must be subject-wise.
- Lab protocol: stress always involved speaking and standing, so some signal changes may reflect the task, not stress itself.
- The wrist BVP signal is motion-sensitive and coarser than chest ECG.
- A small, homogeneous sample of university students; results will not transfer directly to free-living settings.

## References
- Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI 2018*:400–408. https://doi.org/10.1145/3242969.3242985
- Kirschbaum C, Pirke KM, Hellhammer DH. The 'Trier Social Stress Test'. *Neuropsychobiology*. 1993;28(1–2):76–81.

WESAD is distributed by UCI under CC BY 4.0 for research use. Please cite Schmidt et al. (2018) when using it.
