# SMS Spam Detection (SVM Notebook)

This repository contains a single Jupyter notebook that trains and evaluates an SMS spam classifier using a **Linear SVM** with combined **word-level and character-level TF-IDF** features.

The workflow is accuracy-focused and keeps useful raw-text signals (for example punctuation, numbers, capitalization, and repeated characters) instead of aggressively removing them.

## Project Overview

The notebook (`SMS_Spam_SVM.ipynb`) performs an end-to-end supervised text-classification workflow:

1. Load SMS dataset CSV (`spam.csv` or `SMS.csv`)
2. Normalize/validate expected columns and labels
3. Split data into stratified train/test sets
4. Build a TF-IDF + LinearSVC pipeline
5. Tune hyperparameters with 5-fold stratified cross-validation (`GridSearchCV`)
6. Evaluate on a held-out test set
7. Inspect misclassifications (optional)
8. Predict labels for new SMS messages

## Repository Contents

- `SMS_Spam_SVM.ipynb` — main notebook with full training/evaluation/prediction flow
- `README.md` — this guide

## Prerequisites

- Python 3 (notebook metadata uses `python3` kernel)
- Jupyter Notebook, JupyterLab, or Google Colab
- Python packages used in the notebook:
  - `scikit-learn`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `numpy`

> The notebook includes `!pip -q install scikit-learn pandas matplotlib seaborn` in an early cell.

## Installation / Setup

1. Open the repository directory.
2. Ensure your dataset file is available as either:
   - `spam.csv`, or
   - `SMS.csv`
3. Place that file in the same working directory where the notebook runs.
4. Launch the notebook:
   - Locally: open `SMS_Spam_SVM.ipynb` in Jupyter
   - Colab: upload/open the notebook there

## Dataset Expectations (Input)

The loader expects one of these column formats:

- Common UCI/Kaggle format: `v1` (label), `v2` (message)
- Already-normalized format: `label`, `message`

Additional behavior in the notebook:

- Keeps only `ham` and `spam` labels (after lowercasing/stripping)
- Drops rows with missing label/message
- Drops exact duplicate `(label, message)` rows
- Reads CSV with `encoding="latin-1"`

If columns do not match expected schema, the notebook raises a `ValueError`.

## How to Run

Run notebook cells top-to-bottom:

1. Install/import dependencies
2. Load and validate dataset
3. Train/test split (`test_size=0.20`, stratified, `random_state=42`)
4. Fit `GridSearchCV` over `LinearSVC` parameters:
   - `C`: `[0.1, 0.3, 1, 2, 5, 10]`
   - `class_weight`: `[None, "balanced"]`
5. Review best CV parameters/accuracy
6. Evaluate on held-out test set
7. (Optional) inspect misclassified messages
8. Use `predict_sms(...)` for ad-hoc predictions

## Outputs

After execution, the notebook prints/displays:

- Dataset shape and label distribution
- Train/test counts
- Best hyperparameters from grid search
- Best 5-fold CV accuracy
- Held-out test accuracy
- Classification report
- Confusion matrix heatmap
- Count/sample of misclassified messages
- Example predictions for sample SMS text

## Notes on Model/Data Files

- No pre-trained model file is committed in this repository.
- No dataset file is committed in this repository.
- The model is trained in-memory during notebook execution each time.
- If you want persistence, you would need to add explicit model serialization (not currently included).

## Limitations / Future Work

- Notebook-first workflow; no standalone training/inference script yet.
- Reported metrics depend entirely on the dataset file you provide and execution environment.
- No experiment tracking or model versioning is included.
- No automated test suite is present in this repository at the moment.
