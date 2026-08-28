"""Create and execute the Phase 00 project setup and data inventory notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = PROJECT_ROOT / "experiments" / "phase_00_project_setup" / "notebooks" / "00_project_setup_and_data_inventory.ipynb"


def markdown_cell(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    nb.cells = [
        markdown_cell(
            "# Phase 00: Project Setup and Data Inventory\n\n"
            "This notebook verifies the project path, checks the read-only source dataset folder, "
            "scans the dataset recursively, saves a file inventory, and prints candidate data files "
            "for Phase 01. It does not train any model."
        ),
        markdown_cell(
            "## 1. Project path check\n\n"
            "Confirm that the notebook is running from the expected project root."
        ),
        code_cell(
            "from pathlib import Path\n"
            "from datetime import datetime, timezone\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "EXPECTED_ROOT_NAME = 'hdc-vr-pilot'\n\n"
            "print(f'Project root: {PROJECT_ROOT}')\n"
            "print(f'Expected root name: {EXPECTED_ROOT_NAME}')\n"
            "print(f'Root name check: {PROJECT_ROOT.name == EXPECTED_ROOT_NAME}')\n"
            "print(f'Notebook executed at UTC: {datetime.now(timezone.utc).isoformat(timespec=\"seconds\")}')"
        ),
        markdown_cell(
            "## 2. Dataset folder check\n\n"
            "Verify that the required dataset folders exist. The dataset is treated as read-only."
        ),
        code_cell(
            "DATASET_ROOT = PROJECT_ROOT / 'vrdataset'\n"
            "REQUIRED_DATASET_DIRS = [\n"
            "    DATASET_ROOT,\n"
            "    DATASET_ROOT / 'dataPackage',\n"
            "    DATASET_ROOT / 'referenceDocuments',\n"
            "    DATASET_ROOT / 'starterCode',\n"
            "]\n\n"
            "for path in REQUIRED_DATASET_DIRS:\n"
            "    print(f'{path.relative_to(PROJECT_ROOT)} exists: {path.is_dir()}')\n\n"
            "if not all(path.is_dir() for path in REQUIRED_DATASET_DIRS):\n"
            "    missing = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_DATASET_DIRS if not path.is_dir()]\n"
            "    raise FileNotFoundError(f'Missing required dataset directories: {missing}')\n\n"
            "print('Dataset check passed. No files under vrdataset are modified by this notebook.')"
        ),
        markdown_cell(
            "## 3. Recursive scan of vrdataset\n\n"
            "Scan all files and directories under the source dataset to understand its scale."
        ),
        code_cell(
            "all_files = sorted(path for path in DATASET_ROOT.rglob('*') if path.is_file())\n"
            "all_dirs = sorted(path for path in DATASET_ROOT.rglob('*') if path.is_dir())\n"
            "total_size_bytes = sum(path.stat().st_size for path in all_files)\n\n"
            "print(f'Total directories under vrdataset: {len(all_dirs):,}')\n"
            "print(f'Total files under vrdataset: {len(all_files):,}')\n"
            "print(f'Total file size: {total_size_bytes / (1024 ** 2):,.2f} MiB')"
        ),
        markdown_cell(
            "## 4. File inventory for selected formats\n\n"
            "Create a file-level inventory for csv, xlsx, parquet, pkl, mat, npy, npz, txt, py, m, and ipynb files."
        ),
        code_cell(
            "TARGET_EXTENSIONS = {'.csv', '.xlsx', '.parquet', '.pkl', '.mat', '.npy', '.npz', '.txt', '.py', '.m', '.ipynb'}\n\n"
            "def dataset_area(path: Path) -> str:\n"
            "    relative = path.relative_to(DATASET_ROOT)\n"
            "    parts = relative.parts\n"
            "    if len(parts) >= 2 and parts[0] == 'starterCode' and parts[1] == 'data_feats':\n"
            "        return 'starterCode/data_feats'\n"
            "    return parts[0] if parts else 'vrdataset'\n\n"
            "records = []\n"
            "for path in all_files:\n"
            "    suffix = path.suffix.lower()\n"
            "    if suffix in TARGET_EXTENSIONS:\n"
            "        stat = path.stat()\n"
            "        records.append(\n"
            "            {\n"
            "                'relative_path': path.relative_to(PROJECT_ROOT).as_posix(),\n"
            "                'dataset_relative_path': path.relative_to(DATASET_ROOT).as_posix(),\n"
            "                'dataset_area': dataset_area(path),\n"
            "                'extension': suffix.lstrip('.'),\n"
            "                'file_name': path.name,\n"
            "                'size_bytes': stat.st_size,\n"
            "                'modified_time_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec='seconds'),\n"
            "            }\n"
            "        )\n\n"
            "inventory = pd.DataFrame.from_records(records).sort_values(['dataset_area', 'extension', 'relative_path']).reset_index(drop=True)\n"
            "print(f'Inventory rows: {len(inventory):,}')\n"
            "display(inventory.head(10))\n\n"
            "extension_counts = inventory['extension'].value_counts().sort_index().rename_axis('extension').reset_index(name='file_count')\n"
            "display(extension_counts)"
        ),
        markdown_cell(
            "## 5. Counts for key dataset areas\n\n"
            "Report separate file counts for dataPackage, starterCode, referenceDocuments, and starterCode/data_feats."
        ),
        code_cell(
            "COUNT_AREAS = {\n"
            "    'dataPackage': DATASET_ROOT / 'dataPackage',\n"
            "    'starterCode': DATASET_ROOT / 'starterCode',\n"
            "    'referenceDocuments': DATASET_ROOT / 'referenceDocuments',\n"
            "    'starterCode/data_feats': DATASET_ROOT / 'starterCode' / 'data_feats',\n"
            "}\n\n"
            "area_rows = []\n"
            "for area, area_path in COUNT_AREAS.items():\n"
            "    files = [path for path in area_path.rglob('*') if path.is_file()] if area_path.exists() else []\n"
            "    inventory_files = [path for path in files if path.suffix.lower() in TARGET_EXTENSIONS]\n"
            "    area_rows.append(\n"
            "        {\n"
            "            'area': area,\n"
            "            'exists': area_path.exists(),\n"
            "            'all_file_count': len(files),\n"
            "            'inventory_file_count': len(inventory_files),\n"
            "            'size_mib': round(sum(path.stat().st_size for path in files) / (1024 ** 2), 3),\n"
            "        }\n"
            "    )\n\n"
            "area_counts = pd.DataFrame(area_rows)\n"
            "display(area_counts)\n\n"
            "area_extension_counts = (\n"
            "    inventory[inventory['dataset_area'].isin(COUNT_AREAS.keys())]\n"
            "    .groupby(['dataset_area', 'extension'])\n"
            "    .size()\n"
            "    .reset_index(name='file_count')\n"
            "    .sort_values(['dataset_area', 'extension'])\n"
            ")\n"
            "display(area_extension_counts)"
        ),
        markdown_cell(
            "## 6. Save inventory\n\n"
            "Save the selected-format file inventory to the Phase 00 results folder."
        ),
        code_cell(
            "RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'phase_00_project_setup' / 'results'\n"
            "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "INVENTORY_PATH = RESULTS_DIR / 'data_inventory.csv'\n"
            "inventory.to_csv(INVENTORY_PATH, index=False)\n\n"
            "print(f'Saved inventory to: {INVENTORY_PATH.relative_to(PROJECT_ROOT)}')\n"
            "print(f'Saved rows: {len(inventory):,}')"
        ),
        markdown_cell(
            "## 7. Candidate data files for Phase 01\n\n"
            "Print likely data-bearing files to inspect during Phase 01. This list is only an inventory aid."
        ),
        code_cell(
            "DATA_EXTENSIONS = {'csv', 'xlsx', 'parquet', 'pkl', 'mat', 'npy', 'npz'}\n"
            "candidate_data = inventory[inventory['extension'].isin(DATA_EXTENSIONS)].copy()\n"
            "candidate_data = candidate_data[\n"
            "    candidate_data['relative_path'].str.contains('/dataPackage/')\n"
            "    | candidate_data['relative_path'].str.contains('/starterCode/data_feats/')\n"
            "]\n"
            "candidate_summary = (\n"
            "    candidate_data.groupby(['dataset_area', 'extension'])\n"
            "    .size()\n"
            "    .reset_index(name='file_count')\n"
            "    .sort_values(['dataset_area', 'extension'])\n"
            ")\n"
            "display(candidate_summary)\n\n"
            "priority_patterns = ['devSubjsFeatMat', 'perfmetric', 'ocuevts', 'feat-chunk']\n"
            "print('Priority candidate patterns:')\n"
            "for pattern in priority_patterns:\n"
            "    subset = candidate_data[candidate_data['file_name'].str.contains(pattern, case=False, regex=False)]\n"
            "    print(f'- {pattern}: {len(subset):,} files')\n\n"
            "print('\\nCandidate data file examples for Phase 01:')\n"
            "examples = candidate_data[\n"
            "    candidate_data['file_name'].str.contains('|'.join(priority_patterns), case=False, regex=True)\n"
            "][['relative_path', 'extension', 'size_bytes']].head(25)\n"
            "for row in examples.itertuples(index=False):\n"
            "    print(f'- {row.relative_path} ({row.extension}, {row.size_bytes} bytes)')"
        ),
        markdown_cell(
            "## Phase 00 note\n\n"
            "This notebook only prepares the project and inventories files. It does not perform labeling, feature engineering, or model training."
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Executed notebook saved to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
