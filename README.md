# WESAD Stress Detection: Dataset Selection and Exploratory Data Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JillianMelbourne/WESAD-Stress-Detection/blob/main/WESAD_S2_exploration.ipynb)

**Jillian Melbourne** · HSE.751 (Dartmouth Health Data Science), Fall 2026 · Week 3: Dataset Selection and Final Project Planning Lab

> **Scope note: this exploratory data analysis looks at one participant only (S2).**
> WESAD has 15 participants (S2–S11, S13–S17). The notebook `WESAD_S2_exploration.ipynb` loads only `S2/S2.pkl`, so everything it shows (descriptive statistics, figures, data-quality notes) describes S2 alone and should not be read as a statement about the whole cohort. The single-subject analysis is a first pass to understand the data structure before scaling the pipeline to all participants.

## Purpose
This repository is the starting point for my final course project: a supervised binary classifier that detects acute psychosocial stress from wearable physiological signals. This first stage selects and justifies the dataset and explores one participant (S2) in depth: raw structure, target variable, predictors, descriptive statistics, visualizations, and data quality.

## Why this dataset
My background is in signal processing for deep brain stimulation and wearables, so this dataset feels like familiar territory with new tools. Stress is largely invisible and hard to measure beyond self-report, yet it has profound health effects, from cardiovascular disease to mental health. Wearables offer one of the most promising routes to measuring it passively and continuously.

## Dataset
**WESAD (Wearable Stress and Affect Detection)**, Schmidt et al., 2018.
UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>

15 participants wore a chest sensor (RespiBAN: ECG, EDA, EMG, respiration, temperature, 3-axis acceleration, all at 700 Hz) and a wrist sensor (Empatica E4: blood volume pulse 64 Hz, 3-axis acceleration 32 Hz, EDA 4 Hz, skin temperature 4 Hz) during a lab protocol with baseline, amusement, stress (Trier Social Stress Test), and meditation conditions.

The raw data (~2.5 GB for all participants) are **not stored in this repository**. For this S2 analysis you only need one file, `S2/S2.pkl` (~930 MB).

## What the notebook does (S2 only)
`WESAD_S2_exploration.ipynb`:
1. **Dataset selection:** rationale for choosing WESAD.
2. **Dataset exploration:** structure of the S2 pickle, sampling rates, units, label coding.
3. **Target variable:** binary `stress` (TSST stress = 1; baseline + amusement = 0; transitions, meditation and ignored periods excluded).
4. **Predictor variables:** chest and wrist signals, and 60-second windowed features (30-second step).
5. **Descriptive statistics:** raw-signal and window-feature summaries by condition.
6. **Exploratory visualizations:** aligned multi-sensor session, ECG vs. BVP snippets, window features over time, feature distributions by condition.
7. **Data quality notes** and 8. **Summary.**

## How to run in Google Colab

The data file is too large for GitHub, so you upload `S2.pkl` to Colab yourself.

1. **Get `S2.pkl`.** Download the WESAD zip from the [UCI page](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection), unzip it, and find `S2/S2.pkl` (~930 MB). This is the only file the notebook uses.
2. **Open the notebook in Colab.** Click the **Open In Colab** badge above, or go to <https://colab.research.google.com/github/JillianMelbourne/WESAD-Stress-Detection/blob/main/WESAD_S2_exploration.ipynb>.
3. **Upload `S2.pkl`.** In Colab, open the **Files** panel (folder icon on the left). Create the folders `data/raw/S2/` (right-click → **New folder**), then drag `S2.pkl` into `data/raw/S2/`. Wait for the upload to finish.
4. **Run it.** Choose **Runtime → Run all**. The first code cell should print `Loaded S2`.

Colab deletes uploaded files when the session ends, so you'll need to upload `S2.pkl` again next time.

## Running locally (alternative)
```bash
git clone https://github.com/JillianMelbourne/WESAD-Stress-Detection.git
cd WESAD-Stress-Detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook WESAD_S2_exploration.ipynb
```
Place `S2.pkl` at `data/raw/S2/S2.pkl` (this folder is ignored by git), or change `DATA_DIR` in the notebook's setup cell.

## Repository contents
| Path | Description |
|---|---|
| `WESAD_S2_exploration.ipynb` | Main notebook: exploratory analysis of participant S2 only |
| `data/raw/` | Raw WESAD files (not tracked by git; only `S2/S2.pkl` is needed) |
| `data/data_documentation/` | Dataset documentation from the authors (`wesad_readme.pdf`) |
| `figures/` | Figures from the S2 exploration |
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
- **This analysis covers one participant (S2).** Results, feature distributions, and apparent stress effects may not generalize to the other 14 participants.
- Only 15 participants in the full dataset: windows from the same person are correlated, so later modelling must evaluate subject-wise.
- Lab protocol: stress always involved speaking and standing, so some signal changes may reflect the task, not stress itself.
- The wrist BVP signal is motion-sensitive and coarser than chest ECG.
- A small, homogeneous sample of university students; results will not transfer directly to free-living settings.

## References
- Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI 2018*:400–408. https://doi.org/10.1145/3242969.3242985
- Kirschbaum C, Pirke KM, Hellhammer DH. The 'Trier Social Stress Test'. *Neuropsychobiology*. 1993;28(1–2):76–81.

WESAD is distributed by UCI under CC BY 4.0 for research use. Please cite Schmidt et al. (2018) when using it.
