# Machine Failure Analyzer Demo

A demo deployed on Streamlit for a machine-failure evaluation model. The included
model artifact enables local scoring. An Azure ML endpoint and Gemini chat is
also be enabled.

# Deployment Architecture
<img width="1380" height="602" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/d02a43d4-cd6a-431c-8cba-e1bd7a336cfb" />


## How to Deploy on Local

Deploy `app.py` from this repository using Python 3.12. `requirements.txt` is
the app-only dependency set. The app's public URL can be shared after setting
its visibility to public in Streamlit Cloud.

To enable the optional integrations, enter these names in Streamlit Cloud's
private Secrets settings (never in GitHub):

```toml
AZURE_ML_SCORING_URI = "YOUR-HTTPS-SCORING-URI"
AZURE_ML_ENDPOINT_KEY = "YOUR-ENDPOINT-KEY"
GEMINI_API_KEY = "YOUR-GEMINI-KEY"
GEMINI_MODEL = "gemini-3.5-flash-lite"
```

The dataset and model are synthetic research material. Alerts support review;
they are not observed failures or maintenance instructions. Uploaded CSV/Excel
rows outside the training ranges are excluded from scoring and reported.

## Colab notebook and reproducible data

[Open the final submission notebook in Google Colab](https://colab.research.google.com/github/CA-BijiteshKanrar/tsl-machine-failure-dashboard/blob/main/Machine_Failure_Prediction_Colab.ipynb).

### Evaluator quick start

1. Open the link above.
2. Choose **Runtime → Run all**. No Google Drive mount, API key, or manual dataset
   upload is required for the notebook analysis.
3. The setup cell shallow-clones this repository to
   `/content/tsl-machine-failure-dashboard` and reuses that clone if the cell is
   run again. The notebook reads the two bundled files from the clone:
   `data/train.csv` and `data/test.csv`.
4. The loader verifies both files against their recorded SHA-256 checksums before
   analysis. The expected shapes are 136,429 labelled training rows and 90,954
   unlabelled test rows.

### Dependency handling

The setup follows a Colab-safe installation pattern. It asks pip for bounded,
compatible versions without using `--force-reinstall`, replacing Colab's runtime,
or terminating the kernel:

```text
numpy>=1.26,<2.3       pandas==2.2.3        scipy>=1.12,<2
scikit-learn>=1.4,<2  xgboost>=2.1,<4      lightgbm>=4.5,<5
matplotlib>=3.8,<4    seaborn>=0.13,<1     joblib>=1.3,<2
```

### What the notebook demonstrates

- Dataset and business understanding, data-quality checks, EDA, and hypothesis
  testing.
- Leakage-safe preprocessing and feature engineering. Identifiers and recorded
  failure-type flags are excluded from model inputs.
- Class-imbalance-aware comparison of Logistic Regression, Random Forest,
  XGBoost, and LightGBM.
- Validation-only threshold selection using F2, followed by one locked-holdout
  evaluation with precision, recall, average precision, ROC AUC, and alert counts.
- Feature importance, fine-tuning experiments, stakeholder conclusions, and a
  Microsoft Azure deployment demonstration.

### Generated files

The final cells save the fitted model bundle and test predictions under the
temporary Colab `artifacts/` directory:

```text
artifacts/machine_failure_model.joblib
artifacts/submission.csv
artifacts/test_probabilities.csv
```

These files are **not downloaded automatically**. An evaluator can inspect the
displayed results in the notebook or download a required file manually from
Colab's Files panel. The test set is unlabelled, so the two CSVs contain model
predictions rather than test-performance measurements.

### Deployment demonstration

The companion public application demonstrates Azure ML endpoint scoring, CSV or
Excel batch evaluation, end-user visualizations, and Gemini-assisted explanations:
[Machine Failure Risk Analyzer](https://mfriskanalyzer.streamlit.app/).

Azure and Gemini credentials are stored only in private Streamlit deployment
secrets. They are not present in this repository or required to execute the Colab
analysis.
