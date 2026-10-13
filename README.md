# WESAD Stress Detection: Dataset Selection, Exploratory Analysis and Feature Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nHk-bvfX9Q8kEBYpN2tS5-TOoTffDOKc?usp=sharing)

**Jillian Melbourne** · HSE.751 (Dartmouth Health Data Science), Fall 2026 · Week 3: Dataset Selection and Final Project Planning Lab

> **Scope note.** WESAD has 15 participants (S2–S11, S13–S17). The raw-signal exploration (sections 4–7 and Figures 1–4 of the notebook) looks at **one participant, S2**, and should not be read as a statement about the whole cohort. The feature pipeline then runs on **all 15 participants** (sections 8, 9.2 and 11, Figures 5–7), and its output is saved in `data/processed/`.

## Purpose
This repository is the starting point for my final course project: a supervised binary classifier that detects acute psychosocial stress from wearable physiological signals. This stage selects and justifies the dataset, explores one participant (S2) in depth (raw structure, target variable, predictors, descriptive statistics, visualizations, data quality), and builds the window-level feature table for all 15 participants that the classifier will use.

## Why this dataset
My background is in signal processing for deep brain stimulation and wearables, so this dataset feels like familiar territory with new tools. Stress is largely invisible and hard to measure beyond self-report, yet it has profound health effects, from cardiovascular disease to mental health. Wearables offer one of the most promising routes to measuring it passively and continuously.

## Dataset
**WESAD (Wearable Stress and Affect Detection)**, Schmidt et al., 2018.
UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>

15 participants wore a chest sensor (RespiBAN: ECG, EDA, EMG, respiration, temperature, 3-axis acceleration, all at 700 Hz) and a wrist sensor (Empatica E4: blood volume pulse 64 Hz, 3-axis acceleration 32 Hz, EDA 4 Hz, skin temperature 4 Hz) during a lab protocol with baseline, amusement, stress (Trier Social Stress Test), and meditation conditions.

The raw data (~2.5 GB for all participants) are **not stored in this repository**. To run the notebook you only need one file, `S2/S2.pkl` (~930 MB): the cohort sections read the feature table already saved in `data/processed/`. All 15 pickles are needed only to rebuild that table.

## What the notebook does
`WESAD_updated_exploration.ipynb`:
1. **Dataset selection:** rationale for choosing WESAD.
2. **Dataset overview and project plan.**
3. **Setup and configuration:** one cell holds every setting (paths, window length, filter bands, quality thresholds, readme-based exclusions), plus the functions that load and validate a participant.
4. **Dataset structure (S2):** structure of the S2 pickle, sampling rates, units.
5. **Target variable (S2):** binary `stress` (TSST stress = 1; baseline + amusement = 0; transitions, meditation and ignored periods excluded).
6. **Feature pipeline:** `preprocess`, `window_features` and `extract_windows` turn raw signals into 15 predictors per 60-second window (30-second step).
7. **Data validation and quality flags** for each window.
8. **Cohort feature table:** the pipeline run on all 15 participants, cached to `data/processed/`, with a per-participant summary of windows, quality flags and readme notes.
9. **Descriptive statistics** for S2 and for the cohort.
10. **Exploratory visualizations (S2):** aligned multi-sensor session, ECG vs. BVP snippets, window features over time, feature distributions by condition.
11. **Baseline normalization and effect sizes (all participants):** features re-expressed relative to each person's baseline, and Cohen's d per feature and per participant.
12. **Data quality notes** and 13. **Summary.**

## Pipeline improvements (October 2026)
Five changes made after the first S2-only exploration:

1. **Modular, multi-subject pipeline.** The single feature-extraction cell was split into functions (`load_subject`, `preprocess`, `window_features`, `extract_windows`, `build_feature_table`) and now runs on all 15 participants, one at a time to limit memory. The result (1,015 windows) is cached in `data/processed/wesad_windows.csv` with the settings used to build it (`wesad_windows.json`); the cache is rebuilt automatically if a setting changes, or on request with `REBUILD_FEATURES = True`.
2. **Data validation and quality flags.** Each raw file is checked on load (all channels present, durations match the label, no NaNs, no flat channels, known label codes) and a missing or invalid participant is skipped with a clear message. Each window gets flags for missing HR, chest–wrist HR disagreement (likely BVP motion artifact), ECG artifact, implausible skin temperature, and lost wrist-EDA contact. Chest temperature is excluded for S2 and S17, whose readmes say the sensor was not fully attached, and the other readme notes are shown alongside the results.
3. **New features and baseline normalization.** Added EMG amplitude (previously loaded but unused), breath-to-breath variability and wrist temperature slope. Every feature also gets a baseline-normalized copy (`*_bz`: distance from the participant's own baseline mean, in units of their SD), which removes between-person offsets such as differing resting heart rates. Normalized features are saved in `data/processed/wesad_windows_normalized.csv`. Wrist RMSSD was tested and left out because 64 Hz pulse-peak timing made it unreliable.
4. **Effect-size summary.** A table and two figures rank features by Cohen's d (stress vs. non-stress) within each participant and pooled, and show how consistent each feature is across participants (Figures 6 and 7).
5. **Organization and reproducibility.** Numbered sections that match the table of contents, a single configuration cell, every figure saved to `figures/` automatically, and package versions printed at the top of the run.

Optional: if a file `<subject>_compact.npz` (the same arrays as float32) sits next to a participant's pickle, `load_subject` reads it instead, which is several times faster. These files are in `data/raw/` and are not tracked by git.

## How to run in Google Colab

The notebook runs in Colab with just two uploads: one participant's raw file (S2, for the single-subject sections) and the cohort feature table (for the all-participant sections). The raw data are too large for GitHub, so you upload them yourself. Colab deletes uploaded files when the session ends, so repeat the uploads each new session.

1. **Open the notebook.** Click the **Open In Colab** badge above, or go to <https://colab.research.google.com/drive/1nHk-bvfX9Q8kEBYpN2tS5-TOoTffDOKc?usp=sharing>. To keep your own editable copy, choose **File → Save a copy in Drive**.
2. **Get the files.**
   - **`S2.pkl`** (~930 MB): download the WESAD zip from the [UCI page](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection), unzip it, and find `WESAD/S2/S2.pkl`.
   - **`wesad_windows.csv` and `wesad_windows.json`**: download both from this repository's [`data/processed/`](data/processed/) folder. They hold the 60-second window features for all 15 participants, so you don't need the other 14 raw files (~13 GB).
3. **Upload them.** In Colab, open the **Files** panel (folder icon on the left). Create the folders shown below (right-click → **New folder**), then drag each file into its folder:
   ```
   data/
   ├── raw/S2/S2.pkl
   └── processed/
       ├── wesad_windows.csv
       └── wesad_windows.json
   ```
   Wait until each upload's progress circle disappears. You need both feature-table files: the `.json` records the settings the table was built with, and if it is missing the notebook tries to rebuild the table from raw data.
4. **Run it.** Choose **Runtime → Run all**. The full run takes about a minute. Two printed lines confirm it is working:
   - Section 3: `Loaded S2: passed validation`
   - Section 8: `Loaded cached feature table data/processed/wesad_windows.csv (1,015 windows)`

### Troubleshooting
| What you see | What it means | Fix |
|---|---|---|
| `FileNotFoundError: No data for S2` | `S2.pkl` is not at `data/raw/S2/S2.pkl` | Check the folder names (case matters) and that the upload finished |
| `S3: SKIPPED ...` and similar lines in section 8 | The feature table was not found, so the notebook tried to rebuild it from raw data | Upload `wesad_windows.csv` **and** `wesad_windows.json` to `data/processed/`, then run again |
| `Settings changed since the cache was built -> rebuilding` | A setting in the configuration cell differs from the one the table was built with | Undo the change, or rebuild from raw data (below) |
| Session crashes while loading | Colab ran out of memory | **Runtime → Restart session**, then **Run all** again |

### Rebuilding the feature table from raw data (optional)
Only needed if you change a pipeline setting (window length, filters, quality thresholds). It uses all 15 raw files (~14 GB), so keep them in Google Drive instead of uploading them every session:
1. Upload the unzipped `WESAD` folder to **My Drive**.
2. Add a new cell above the configuration cell and run it:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
3. In the configuration cell, set `DATA_DIR = Path("/content/drive/MyDrive/WESAD")` and `REBUILD_FEATURES = True`.
4. **Runtime → Run all.** Expect a minute or two per participant, mostly spent reading files from Drive. The new table is written to `data/processed/`. Download it from the **Files** panel if you want to keep it.

## Running locally (alternative)
```bash
git clone https://github.com/JillianMelbourne/WESAD-Stress-Detection.git
cd WESAD-Stress-Detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook WESAD_updated_exploration.ipynb
```
Place `S2.pkl` at `data/raw/S2/S2.pkl` (this folder is ignored by git), or change `DATA_DIR` in the notebook's configuration cell. To rebuild the cohort feature table, put every participant's folder in `data/raw/` and set `REBUILD_FEATURES = True`.

## Repository contents
| Path | Description |
|---|---|
| `WESAD_updated_exploration.ipynb` | Main notebook: S2 deep dive and the feature pipeline for all 15 participants |
| `data/raw/` | Raw WESAD files (not tracked by git; only `S2/S2.pkl` is needed unless rebuilding features) |
| `data/processed/` | Window-feature tables for all 15 participants (`wesad_windows.csv`, `wesad_windows_normalized.csv`) and the settings used (`wesad_windows.json`) |
| `data/data_documentation/` | Dataset documentation from the authors (`wesad_readme.pdf`) |
| `figures/` | Figures 1–7, saved automatically when the notebook runs |
| `Archive/` | Earlier multi-subject notebook and single-subject plotting script, kept for reference |
| `requirements.txt` | Package versions for running locally |
| `README.md` | This file |

## Software
| Package | Version |
|---|---|
| Python | 3.11 or newer |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scipy | 1.16.3 |
| matplotlib | 3.10.0 |

All are preinstalled in Google Colab.

## Limitations (summary)
- **The raw-signal exploration covers one participant (S2).** The cohort feature table and effect sizes cover all 15, but no classifier has been trained or evaluated yet.
- Within-participant effect sizes are inflated by overlapping windows and stable within-condition signals; pooled effect sizes are the better guide to performance on new people.
- Only 15 participants in the full dataset: windows from the same person are correlated, so later modelling must evaluate subject-wise.
- Lab protocol: stress always involved speaking and standing, so some signal changes may reflect the task, not stress itself.
- The wrist BVP signal is motion-sensitive and coarser than chest ECG.
- A small, homogeneous sample of university students; results will not transfer directly to free-living settings.

## References
- Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI 2018*:400–408. https://doi.org/10.1145/3242969.3242985
- Kirschbaum C, Pirke KM, Hellhammer DH. The 'Trier Social Stress Test'. *Neuropsychobiology*. 1993;28(1–2):76–81.

WESAD is distributed by UCI under CC BY 4.0 for research use. Please cite Schmidt et al. (2018) when using it.