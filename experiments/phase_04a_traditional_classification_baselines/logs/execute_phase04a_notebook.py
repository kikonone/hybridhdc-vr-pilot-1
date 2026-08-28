from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "experiments" / "phase_04a_traditional_classification_baselines" / "Phase_04A_Classification_Baselines.ipynb"
notebook = nbformat.read(PATH, as_version=4)
client = NotebookClient(notebook, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(notebook, PATH)
print(PATH)
