# WESAD Stress Detection: Dataset Selection, Exploratory Analysis and Feature Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JillianMelbourne/WESAD-Stress-Detection/blob/main/WESAD_stress_pipeline.ipynb)

**Shared Colab notebook:** <https://colab.research.google.com/drive/1UJU_u2qojZmoUMo3Pse6NRSlskIUc7TJ?usp=sharing>

**Jillian Melbourne** · HSE.751 (Dartmouth Health Data Science), Fall 2026 · Week 4: Final Project Progress Update Lab

> **Scope note.** WESAD has 15 participants (S2–S11, S13–S17). The raw-signal deep dive (section 5 and Figures 1–4 of the notebook) looks at **one participant, S2**, and should not be read as a statement about the whole cohort. The feature pipeline and all later analyses cover **all 15 participants** (sections 6–13, Figures 5–21), using the feature table saved in `data/processed/`.

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
`WESAD_stress_pipeline.ipynb` is the main notebook. It focuses on decisions and results; the reusable code lives in `src/wesad/` (see [Project structure](#project-structure)).

1. **Dataset selection** and 2. **Dataset overview and project plan.**
3. **Setup and run settings:** fetches the project code if needed, and a settings form (data source, deep-dive participant, figure saving).
4. **Data input and validation:** loads the chosen feature table, validates it, profiles every feature and confirms success.
5. **Single-participant deep dive (raw signals):** recording structure, target variable and Figures 1–4 for one participant (S2 by default). Runs only when that participant's raw file is present.
6. **Feature pipeline:** how raw signals become 15 predictors per 60-second window, and the quality flags.
7. **Cohort overview and data quality:** windows, quality flags and readme notes for each participant.
8. **Descriptive statistics and feature distributions:** overlaid histograms, box plots and a scatterplot matrix by condition (Figures 5–7).
9. **Baseline normalization** (Figure 8).
10. **Effect sizes** (Figures 9 and 10).
11. **Inferential tests:** participant-level Wilcoxon signed-rank and sign tests with multiple-testing correction (Figure 11); collinearity (Figure 12); conditional (fixed-effects) logistic regression with assumption checks, including the Yeo–Johnson decision, VIF, linearity and a non-overlapping-windows sensitivity refit (Figure 13).
12. **Model evaluation:** a leave-one-participant-out validation function with nested, participant-grouped tuning; a pre-declared primary metric (F1); a majority-class baseline, logistic regression, elastic-net logistic regression, a decision tree and an RBF-kernel SVM; automatic best-model selection (Figure 14); and what each model learned: elastic-net coefficients and lasso path, the tree diagram, and Gini and permutation importances (Figures 15–17).
13. **Deep learning:** a multi-layer perceptron (neural network) on the same splits, its training curve, the final model selection across all models (Figures 18–19), and confusion matrices and overlaid ROC curves for every model (Figures 20–21).
14. **Data quality notes** and 15. **Summary and next steps.**

### Run settings
The settings cell in section 3 shows as a form in Colab. The defaults run without any input.

| Setting | What it does |
|---|---|
| `DATA_SOURCE` | **Use saved feature table** (default): `data/processed/wesad_windows.csv`, 1,015 windows from all 15 participants. **Upload my own feature table**: opens a file picker (see [Using your own data](#using-your-own-data)). **Rebuild from raw WESAD files**: re-runs feature extraction on `data/raw/`; falls back to the saved table if no raw files are present. |
| `UPLOAD_PATH` | Outside Colab only: path to the CSV to analyze. |
| `DEEP_DIVE_SUBJECT` | Participant for section 5 (needs that participant's raw `.pkl`). |
| `SAVE_FIGURES` | Save every figure to `figures/`. |

Pipeline settings (window length, filter bands, quality thresholds) are fields of `PipelineConfig` in `src/wesad/config.py`, e.g. `PipelineConfig(window_s=30)`. Changing one requires **Rebuild from raw WESAD files**; the saved table records the settings it was built with (`wesad_windows.json`) and the notebook warns if they differ.

### Using your own data
Upload a CSV with one row per time window:

| Column | Required | Content |
|---|---|---|
| `subject` | yes | participant ID |
| `stress` | yes | 1 = stress, 0 = non-stress |
| any numeric columns | at least one | features (e.g., heart rate, skin conductance) |
| `condition` | optional | e.g. `baseline`, `stress`; enables normalization against true baseline windows |

The notebook checks the table and stops with a list of every problem if it cannot be used (missing columns, a target other than 0/1, fewer than two participants or five windows per class). Fixable issues are handled and reported: rows without a target and features more than half empty are dropped, and text columns are ignored.

## Pipeline improvements (October 2026)
Changes made after the first S2-only exploration:

1. **Modular, multi-subject pipeline.** The single feature-extraction cell was split into functions (`load_subject`, `preprocess`, `window_features`, `extract_windows`, `build_feature_table`) and now runs on all 15 participants, one at a time to limit memory. The result (1,015 windows) is cached in `data/processed/wesad_windows.csv` with the settings used to build it (`wesad_windows.json`); a rebuild from raw data is chosen in the notebook's run settings, and the notebook warns if the saved table's settings differ from the current ones.
2. **Data validation and quality flags.** Each raw file is checked on load (all channels present, durations match the label, no NaNs, no flat channels, known label codes) and a missing or invalid participant is skipped with a clear message. Each window gets flags for missing HR, chest–wrist HR disagreement (likely BVP motion artifact), ECG artifact, implausible skin temperature, and lost wrist-EDA contact. Chest temperature is excluded for S2 and S17, whose readmes say the sensor was not fully attached, and the other readme notes are shown alongside the results.
3. **New features and baseline normalization.** Added EMG amplitude (previously loaded but unused), breath-to-breath variability and wrist temperature slope. Every feature also gets a baseline-normalized copy (`*_bz`: distance from the participant's own baseline mean, in units of their SD), which removes between-person offsets such as differing resting heart rates. Normalized features are saved in `data/processed/wesad_windows_normalized.csv`. Wrist RMSSD was tested and left out because 64 Hz pulse-peak timing made it unreliable.
4. **Effect-size summary.** A table and two figures rank features by Cohen's d (stress vs. non-stress) within each participant and pooled, and show how consistent each feature is across participants (Figures 9 and 10).
5. **Organization and reproducibility.** Numbered sections that match the table of contents, a single configuration cell, every figure saved to `figures/` automatically, and package versions printed at the top of the run.

6. **Package structure and run settings.** The pipeline functions moved from the notebook into the `src/wesad/` package, with all settings in a single `PipelineConfig`. The notebook starts with a settings form to choose the data source (saved table, own upload, or rebuild from raw files), validates and profiles whatever it loads, and runs end to end with no uploads.

Optional: if a file `<subject>_compact.npz` (the same arrays as float32) sits next to a participant's pickle, `load_subject` reads it instead, which is several times faster. These files are in `data/raw/` and are not tracked by git.

## How to run in Google Colab

1. **Open the notebook:** use the shared Colab notebook at <https://colab.research.google.com/drive/1UJU_u2qojZmoUMo3Pse6NRSlskIUc7TJ?usp=sharing>, or click the **Open In Colab** badge above to open the latest version straight from GitHub.
2. **Run it:** choose **Runtime → Run all.** No uploads are needed: the first code cell downloads the project code and the saved feature table from this repository. The full run takes about 3–4 minutes, most of it the nested tuning of the elastic net in section 12.
3. **Check it worked:** section 4 prints `✅ Dataset loaded and validated ... 1,015 windows, 15 features, 15 participants`.

Section 5 (the raw-signal deep dive for one participant) is skipped unless that participant's raw file is present; everything else runs without it. To include it:
1. Download the WESAD zip from the [UCI page](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection), unzip it, and find `WESAD/S2/S2.pkl` (~930 MB).
2. Run the notebook once so the project folder exists, then in the **Files** panel (folder icon on the left) open `WESAD-Stress-Detection/data/raw/`, create a folder `S2`, and drag `S2.pkl` into it.
3. **Runtime → Run all** again.

Colab deletes uploaded files when the session ends.

### Troubleshooting
| What you see | What it means | Fix |
|---|---|---|
| `src/wesad was not found` | The downloaded copy of the repository is out of date | Delete the `WESAD-Stress-Detection` folder in the **Files** panel and run again |
| `Feature table failed validation` followed by a list | An uploaded table cannot be used as it is | Fix each listed problem in the CSV and upload it again |
| `Uploading through the browser only works in Google Colab` | You chose **Upload** while running outside Colab | Set `UPLOAD_PATH` to the CSV's location |
| `No raw WESAD files found ... falling back` | **Rebuild** was chosen but `data/raw/` is empty | Expected unless you have the raw data; the saved table is used instead |
| Session crashes during a rebuild | Colab ran out of memory | **Runtime → Restart session**, then **Run all** again |

### Rebuilding the feature table from raw data (optional)
Only needed after changing a pipeline setting. It uses all 15 raw files (~14 GB), so keep them in Google Drive rather than uploading them each session:
1. Upload the unzipped `WESAD` folder to **My Drive**.
2. Add a cell after the settings cell and run it:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   cfg = PipelineConfig(data_dir="/content/drive/MyDrive/WESAD")
   ```
3. Set `DATA_SOURCE` to **Rebuild from raw WESAD files** and run the notebook from section 4 onwards. Expect a minute or two per participant.

## Running locally (alternative)
```bash
git clone https://github.com/JillianMelbourne/WESAD-Stress-Detection.git
cd WESAD-Stress-Detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook WESAD_stress_pipeline.ipynb
```
Raw files, if you have them, go in `data/raw/<subject>/<subject>.pkl` (ignored by git).

## Project structure
```
WESAD-Stress-Detection/
├── WESAD_stress_pipeline.ipynb      main notebook (narrative, results, figures)
├── src/wesad/                       pipeline code, imported by the notebook
│   ├── config.py                    PipelineConfig: every setting in one place; fixed WESAD facts
│   ├── loading.py                   load/validate raw files and feature tables; profiling; upload
│   ├── features.py                  raw signals -> 60-s window features and quality flags
│   ├── analysis.py                  baseline normalization, effect sizes
│   └── plots.py                     figure helpers
├── data/
│   ├── raw/                         raw WESAD files (not tracked by git)
│   ├── processed/                   feature tables for all 15 participants + settings used
│   └── data_documentation/          dataset documentation from the authors
├── figures/                         Figures 1–21, saved when the notebook runs
├── WESAD_updated_exploration.ipynb  earlier single-notebook version, kept for reference
├── Archive/                         earlier exploration notebook and script
├── requirements.txt
└── README.md
```

## Software
| Package | Version |
|---|---|
| Python | 3.11 or newer |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scipy | 1.16.3 |
| matplotlib | 3.10.0 |
| seaborn | 0.13 or newer |
| scikit-learn | 1.3 or newer |
| statsmodels | 0.14 or newer |

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
