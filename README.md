# Machine Failure Analyzer Demo

A demo deployed on Streamlit for a machine-failure evaluation model. The included
model artifact enables local scoring. An Azure ML endpoint and Gemini chat is
also be enabled.

##Deployment Architecture
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

[Open the complete project notebook in Google Colab](https://colab.research.google.com/github/CA-BijiteshKanrar/tsl-machine-failure-dashboard/blob/main/Machine_Failure_Prediction_Colab.ipynb).

The versioned project inputs are stored in `data/train.csv` and `data/test.csv`.
In a fresh Colab runtime, the notebook shallow-clones this repository, reads the
bundled datasets, and verifies their SHA-256 checksums. Its dependency cell uses
compatible version ranges without force-reinstalling Colab's core stack. Local
files and manual upload remain fallback options.
