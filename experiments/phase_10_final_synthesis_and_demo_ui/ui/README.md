# HDC System Demonstration

This directory contains a single-page, English-only Streamlit application for a local system demonstration. It presents two HDC tasks:

- Classification: HDC+OnlineHD Hybrid, dimension 5000.
- Proxy regression: COMMON_ENCODER_READOUT_BASELINE, common_ridge, dimension 10000, feature_k 50, levels 51, ridge alpha 0.01.

The application does not train, perform live inference, upload data, or contact external services. The interface presents the aligned anonymous records as `Record 0001` through `Record 0419`; internal provenance and integrity checks remain unchanged.

## Start locally

Install dependencies once:

```powershell
python -m pip install -r requirements.txt
```

Then run either:

```powershell
.\start_ui.ps1
```

or double-click `start_ui.bat`. The application binds only to `127.0.0.1:8501`. Both launchers stop with an English error if the port is already in use.

## Verification

```powershell
python -m pytest -q
```

Browser acceptance and bounded stress scripts require the local Streamlit server to be running at `http://127.0.0.1:8501`.
