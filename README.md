# EKPC Load Forecast (LSTM)

This repository contains a small example LSTM-based load forecasting project for EKPC (example dataset). It includes training code, a saved model, and a Streamlit app to run predictions.

Files
- `app.py` - Streamlit application to input last 3 hours usage and predict next-hour load using the saved LSTM model.
- `model_train.py` - Script to train the LSTM model from `ekpc_usage.csv` and save it as `ekpc_lstm_model.h5`.
- `ekpc_lstm_model.h5` - Saved trained LSTM model (HDF5). You can replace this with a newer model if desired.
- `ekpc_usage.csv` - Example dataset used for training (Datetime, EKPC_MW).
- `requirements.txt` - Python dependencies for the project.

Quick start (Windows PowerShell)

1) Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) (Optional) Retrain the model from the CSV (this will overwrite `ekpc_lstm_model.h5`):

```powershell
python model_train.py
```

4) Run the Streamlit app:

```powershell
streamlit run app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501) in your browser.

Notes and tips
- Model loading: The app attempts to load `ekpc_lstm_model.h5`. If you see compatibility issues between saved HDF5 models and your installed Keras/TensorFlow, retrain using `model_train.py` in the same environment, or re-save using the native Keras format (e.g. `model.save('my_model.keras')`).
- Python / TensorFlow compatibility: TensorFlow has strict OS/Python compatibility rules. If you encounter installation or import errors for TensorFlow, try using Python 3.10–3.11 in a virtual environment.
- GPU: If you want GPU support, install the appropriate tensorflow package for your GPU (or a matching wheel) according to TensorFlow's docs.

Troubleshooting
- "Could not deserialize..." or "AttributeError: 'NoneType' object has no attribute 'pop'": typically a model version mismatch. Retrain in your environment via `python model_train.py` and re-run the app.
- If `streamlit` is not found, ensure your venv is activated or install globally (not recommended).

Development & contributions
- Small, single-file project for demonstration. Contributions welcome: open an issue or PR with improvements (data preprocessing, model architecture, UI enhancements).

License
- This repository contains example code; include a license of your choice if you plan to publish. If unsure, add an appropriate LICENSE file (MIT, Apache-2.0, etc.).

Contact
- For questions about running the project, open an issue in the repo or contact the maintainer.
