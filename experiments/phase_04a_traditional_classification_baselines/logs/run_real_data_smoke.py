import json
import os
import sys
from pathlib import Path
import joblib
import sklearn

if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise RuntimeError("PYTHONNOUSERSITE must equal 1")
if Path(sys.executable).resolve() != Path(r"D:\Computer\anaconda3\python.exe").resolve():
    raise RuntimeError(f"Unexpected interpreter: {sys.executable}")

print("PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE"), flush=True)
print("Python executable:", sys.executable, flush=True)
print("Python version:", sys.version, flush=True)
print("scikit-learn:", sklearn.__version__, flush=True)
print("joblib:", joblib.__version__, flush=True)

ROOT = Path(__file__).resolve().parents[3]
runner = __import__("runpy").run_path(str(ROOT / "experiments/phase_04a_traditional_classification_baselines/logs/phase04a_runner.py"))
context = runner["setup"]()
result = runner["smoke"](context)
print(json.dumps(result, indent=2), flush=True)
