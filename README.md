# Machine failure risk dashboard

Streamlit dashboard for a synthetic machine-failure research model. The included
model artifact enables local scoring. An Azure ML endpoint and Gemini chat can
also be enabled through **private Streamlit secrets**; keys are never requested
or rendered by the app.

## Streamlit Community Cloud

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

[Open the complete project notebook in Google Colab](https://colab.research.google.com/github/CA-BijiteshKanrar/tsl-machine-failure-dashboard/blob/main/Machine_Failure_Prediction_Colab.ipynb).

The versioned project inputs are stored in `data/train.csv` and `data/test.csv`.
In a fresh Colab runtime, the notebook downloads these files directly from this
repository and verifies their SHA-256 checksums. Local files are preferred when
available, and manual upload remains a fallback if GitHub cannot be reached.
