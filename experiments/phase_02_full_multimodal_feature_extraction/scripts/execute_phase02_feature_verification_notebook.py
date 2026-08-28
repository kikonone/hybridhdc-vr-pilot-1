from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PHASE = ROOT / "experiments" / "phase_02_full_multimodal_feature_extraction"
NOTEBOOK_PATH = PHASE / "notebooks" / "Phase_02_Feature_Verification.ipynb"
RUNTIME_DIR = PHASE / "logs" / "jupyter_runtime_phase02_verification"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ["JUPYTER_RUNTIME_DIR"] = str(RUNTIME_DIR)
os.environ["JUPYTER_ALLOW_INSECURE_WRITES"] = "true"
os.environ["IPYTHONDIR"] = str(PHASE / "ipython")

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
)

try:
    client.execute()
except CellExecutionError:
    nbformat.write(notebook, NOTEBOOK_PATH)
    raise

nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Executed and saved: {NOTEBOOK_PATH}")
