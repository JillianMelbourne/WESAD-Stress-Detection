# WESAD Stress Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JillianMelbourne/WESAD-Stress-Detection/blob/main/WESAD_stress_pipeline.ipynb)

**Shared Colab notebook:** <https://colab.research.google.com/drive/1UJU_u2qojZmoUMo3Pse6NRSlskIUc7TJ?usp=sharing>

An end-to-end, reproducible pipeline that detects acute psychological stress from wearable chest and wrist sensors. It turns raw physiological signals into 60-second window features, tests which signals change under stress, and compares supervised classifiers on participants the models have never seen.

**Author:** Jillian Melbourne

## Overview
Stress is hard to measure beyond self-report, yet it has major health effects, from cardiovascular disease to mental health. Wearable sensors could measure it passively and continuously. This project asks two questions:

1. **Which physiological signals change under acute stress, and which predict it independently?**
2. **How well can a classifier detect stress in a person it was not trained on?**

The analysis lives in one notebook, `WESAD_stress_pipeline.ipynb`. The reusable code (loading, feature extraction, normalization, plotting) lives in the `src/wesad/` package, so the notebook can focus on decisions and results.

## Dataset
**WESAD (Wearable Stress and Affect Detection)**, Schmidt et al., 2018.
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) (CC BY 4.0)

Fifteen adults (S2–S11, S13–S17) completed a lab protocol with baseline, amusement, stress and meditation conditions while wearing two devices:

| Device | Signals | Sampling rate |
|---|---|---|
| Chest (RespiBAN) | ECG, electrodermal activity (EDA), EMG, respiration, temperature, 3-axis acceleration | 700 Hz |
| Wrist (Empatica E4) | Blood volume pulse (BVP), 3-axis acceleration, EDA, skin temperature | 64, 32, 4 and 4 Hz |

The stress condition is the Trier Social Stress Test: a prepared speech and mental arithmetic in front of an evaluating panel.

**Target variable:** `stress` = 1 for stress windows (WESAD label 2), 0 for baseline and amusement windows (labels 1 and 3). Transitions, meditation and unlabelled periods are excluded.

**Observations:** 1,015 sixty-second windows (30-second step), about 68 per participant; about 30% are stress windows.

**Features (15 per window):**

| Feature | Source |
|---|---|
| Heart rate (chest and wrist), heart rate variability (RMSSD) | ECG, BVP |
| EDA level (chest and wrist), chest EDA slope, wrist skin conductance response count | EDA |
| Breathing rate, breath-to-breath variability | Respiration |
| Chest and wrist temperature, wrist temperature slope | Temperature |
| Muscle activity (EMG amplitude) | EMG |
| Movement (chest and wrist) | Accelerometer |

**Included in this repository:** the processed feature tables in `data/processed/` (`wesad_windows.csv`, its build settings in `wesad_windows.json`, and a baseline-normalized copy). The raw recordings (about 2.5 GB compressed) are **not** included. They are needed only to rebuild the feature table or to run the single-participant raw-signal section.

## What the pipeline does

| Step | Notebook section | What happens |
|---|---|---|
| 1. Data input and validation | 3–4 | Loads the saved feature table, a user-uploaded CSV, or rebuilds features from raw files; checks required columns, the 0/1 target, class sizes and participant count; profiles every feature |
| 2. Signal processing (optional) | 5–6 | Filters raw signals, detects heartbeats and breaths, cuts 60-second windows, computes 15 features and flags poor-quality windows (motion artifact, sensor contact loss, implausible values) |
| 3. Exploration | 7–8 | Data-quality summary per participant, descriptive statistics, histograms, box plots and a scatterplot matrix by condition |
| 4. Baseline normalization | 9 | Expresses each feature relative to the participant's own resting level, removing large between-person differences |
| 5. Effect sizes | 10 | Cohen's d within each participant and pooled |
| 6. Inferential statistics | 11 | Participant-level Wilcoxon signed-rank and sign tests (Holm and Benjamini–Hochberg corrected); Spearman correlation and VIF for collinearity; conditional (fixed-effects) logistic regression with assumption checks and a non-overlapping-windows sensitivity analysis |
| 7. Classification | 12–13 | Majority-class baseline, logistic regression, elastic-net logistic regression, decision tree, RBF-kernel SVM and a multi-layer perceptron, all evaluated with leave-one-participant-out cross-validation and nested, participant-grouped tuning |
| 8. Model selection and interpretation | 12–13 | Best model chosen by a pre-declared primary metric (F1 for the stress class); per-participant results, confusion matrices, ROC curves, coefficients, tree diagram and feature importances |

**The pipeline adapts to the data:** it drops mostly-missing features, removes one feature from each highly correlated pair, applies class weighting when stress is the minority class, adds a Yeo–Johnson transform when features are strongly skewed, and falls back to stratified cross-validation (with a warning) when there are too few participants for leave-one-participant-out.

## Key results

| Model | Mean F1 (± SD across participants) |
|---|---|
| **SVM (RBF kernel)** | **0.94 ± 0.09** |
| Elastic-net logistic regression | 0.88 ± 0.14 |
| Logistic regression | 0.87 ± 0.25 |
| Neural network (MLP) | 0.86 ± 0.25 |
| Decision tree | 0.82 ± 0.13 |
| Majority-class baseline | 0.00 |

- **Signals that change under stress:** heart rate, skin conductance responses, breathing irregularity and chest movement rose in all 15 participants; EDA rose in 14 of 15.
- **Independent predictors:** in the fixed-effects logistic regression, each 1-SD rise above a person's resting level multiplies the odds of stress by 15.0 for wrist heart rate, 14.0 for wrist EDA and 4.5 for breathing irregularity; cooler wrist temperature also predicts stress (OR 0.38).
- **Best model:** the SVM detects 92% of stress windows with a 1.4% false-alarm rate on unseen participants, and is the only model that works well for the one participant whose skin conductance fell under stress.

## How to use

### Option 1: Google Colab (recommended)
1. Open the [shared Colab notebook](https://colab.research.google.com/drive/1UJU_u2qojZmoUMo3Pse6NRSlskIUc7TJ?usp=sharing), or click the **Open In Colab** badge above to open the latest version from GitHub.
2. Choose **Runtime → Run all.** No uploads are needed: the first code cell downloads the project code and saved feature table from this repository. A full run takes about 3–4 minutes.
3. Section 4 confirms success with `✅ Dataset loaded and validated ... 1,015 windows, 15 features, 15 participants`.

### Option 2: Run locally
```bash
git clone https://github.com/JillianMelbourne/WESAD-Stress-Detection.git
cd WESAD-Stress-Detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook WESAD_stress_pipeline.ipynb
```
Then choose **Run all**.

### Run settings
The settings cell in section 3 appears as a form in Colab. The defaults run without any input.

| Setting | What it does |
|---|---|
| `DATA_SOURCE` | **Use saved feature table** (default): the 1,015-window table in `data/processed/`. **Upload my own feature table**: opens a file picker (see below). **Rebuild from raw WESAD files**: re-runs feature extraction on `data/raw/`; falls back to the saved table if no raw files are present. |
| `UPLOAD_PATH` | Outside Colab only: path to the CSV to analyze. |
| `DEEP_DIVE_SUBJECT` | Participant shown in the raw-signal section (needs that participant's raw file). |
| `SAVE_FIGURES` | Saves every figure to `figures/`. |

Pipeline settings (window length, filter bands, quality thresholds) are fields of `PipelineConfig` in `src/wesad/config.py`. Changing one requires **Rebuild from raw WESAD files**; the notebook warns if the saved table was built with different settings.

### Using your own data
Choose **Upload my own feature table** and provide a CSV with one row per time window:

| Column | Required | Content |
|---|---|---|
| `subject` | yes | Participant ID |
| `stress` | yes | 1 = stress, 0 = non-stress |
| Numeric feature columns | at least one | e.g. heart rate, skin conductance |
| `condition` | optional | e.g. `baseline`, `stress`; enables normalization against true baseline windows |

The notebook stops with a list of every problem if the table cannot be used (missing columns, a target other than 0/1, fewer than two participants or fewer than five windows per class). Fixable issues are handled and reported: rows without a target and mostly-empty features are dropped, and text columns are ignored.

### Including the raw-signal section (optional)
The single-participant deep dive (section 5) runs only when that participant's raw file is present. To include it:
1. Download the WESAD zip from the [UCI page](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) and unzip it.
2. Place `WESAD/S2/S2.pkl` (about 930 MB) at `data/raw/S2/S2.pkl`. In Colab, run the notebook once, then use the **Files** panel to create `WESAD-Stress-Detection/data/raw/S2/` and drag the file in.
3. Run all again.

### Rebuilding the feature table from raw data (optional)
Needed only after changing a pipeline setting. It uses all 15 raw files, so in Colab keep them in Google Drive:
1. Upload the unzipped `WESAD` folder to **My Drive**.
2. Add and run a cell after the settings cell:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   cfg = PipelineConfig(data_dir="/content/drive/MyDrive/WESAD")
   ```
3. Set `DATA_SOURCE` to **Rebuild from raw WESAD files** and run from section 4 onwards (about a minute or two per participant).

If a file `<subject>_compact.npz` (the same arrays stored as float32) sits next to a participant's pickle, it is loaded instead, which is several times faster.

### Troubleshooting
| What you see | What it means | Fix |
|---|---|---|
| `src/wesad was not found` | The downloaded copy of the repository is incomplete or out of date | In Colab, choose **Runtime → Disconnect and delete runtime**, then run all again |
| `Feature table failed validation` followed by a list | An uploaded table cannot be used as it is | Fix each listed problem in the CSV and upload it again |
| `Uploading through the browser only works in Google Colab` | **Upload** was chosen outside Colab | Set `UPLOAD_PATH` to the CSV's location |
| `No raw WESAD files found ... falling back` | **Rebuild** was chosen but `data/raw/` is empty | Expected without the raw data; the saved table is used instead |
| Session crashes during a rebuild | Colab ran out of memory | **Runtime → Restart session**, then run all again |

## Project structure
```
WESAD-Stress-Detection/
├── WESAD_stress_pipeline.ipynb      main notebook (narrative, results, figures)
├── src/wesad/                       pipeline code imported by the notebook
│   ├── config.py                    PipelineConfig: every setting in one place
│   ├── loading.py                   load and validate raw files and feature tables; profiling; upload
│   ├── features.py                  raw signals -> 60-second window features and quality flags
│   ├── analysis.py                  baseline normalization and effect sizes
│   └── plots.py                     figure helpers
├── data/
│   ├── raw/                         raw WESAD files (not tracked by git)
│   ├── processed/                   feature tables for all 15 participants + build settings
│   └── data_documentation/          dataset documentation from the WESAD authors
├── figures/                         Figures 1–21, saved when the notebook runs
├── Archive/                         earlier exploration notebooks and scripts
├── requirements.txt
└── README.md
```

## Requirements
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

All are preinstalled in Google Colab. Run `pip install -r requirements.txt` for a local setup.

## Limitations
- **Small sample.** Fifteen participants, mostly young university students; results may not transfer to other populations or to free-living settings.
- **Dependent windows.** Windows overlap by 50% and come from few people, so evaluation holds out whole participants and regression p-values are approximate.
- **Task confound.** The stress task involved standing and speaking, so some changes (movement, breathing irregularity) may reflect the task rather than stress itself.
- **Signal quality.** Wrist BVP is motion-sensitive; wrist heart-rate variability was unreliable and is not used.
- **Personal differences.** One participant's skin conductance fell under stress; such person-level differences suggest a deployed device would need brief personal calibration.

## References
- Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI 2018*:400–408. https://doi.org/10.1145/3242969.3242985
- Kirschbaum C, Pirke KM, Hellhammer DH. The 'Trier Social Stress Test'. *Neuropsychobiology*. 1993;28(1–2):76–81.

WESAD is distributed by UCI under CC BY 4.0. Please cite Schmidt et al. (2018) when using it.
