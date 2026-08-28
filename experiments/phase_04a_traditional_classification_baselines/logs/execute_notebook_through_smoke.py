from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise RuntimeError("PYTHONNOUSERSITE must equal 1")
ROOT=Path(__file__).resolve().parents[3]
PATH=ROOT/'experiments/phase_04a_traditional_classification_baselines/Phase_04A_Classification_Baselines.ipynb'
nb=nbformat.read(PATH,as_version=4)
for cell in nb.cells:
    if cell.cell_type=='code' and 'import runpy' in cell.source:
        cell.source='import os, sys, sklearn, joblib\nprint("PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE"))\nprint("Python executable:", sys.executable)\nprint("Python version:", sys.version)\nprint("scikit-learn:", sklearn.__version__)\nprint("joblib:", joblib.__version__)\n'+cell.source
        break
client=NotebookClient(nb,timeout=3600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
with client.setup_kernel():
    for index,cell in enumerate(nb.cells[:17]):
        if cell.cell_type=='code': client.execute_cell(cell,index,store_history=True)
nbformat.write(nb,PATH)
print(PATH)
