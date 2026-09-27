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
