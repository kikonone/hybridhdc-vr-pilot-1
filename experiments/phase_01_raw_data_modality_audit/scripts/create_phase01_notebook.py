from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path.cwd()
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase_01_raw_data_modality_audit"
NOTEBOOK_PATH = PHASE_DIR / "notebooks" / "01_raw_data_modality_audit.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
)

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Phase 01: Raw Data Audit and Modality Inventory

This notebook audits `vrdataset/dataPackage` before feature extraction. It treats `vrdataset` as read-only and saves all generated outputs under `experiments/phase_01_raw_data_modality_audit`.

This phase does not perform feature extraction, does not create the final modeling dataset, and does not train ML or HDC models."""
))

cells.append(nbf.v4.new_code_cell(
"""from pathlib import Path
import sys

START_DIR = Path.cwd().resolve()
PROJECT_ROOT = None
for candidate in [START_DIR, *START_DIR.parents]:
    if (candidate / "vrdataset" / "dataPackage").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Could not locate project root from {START_DIR}")

PHASE_DIR = PROJECT_ROOT / "experiments" / "phase_01_raw_data_modality_audit"
DATA_ROOT = PROJECT_ROOT / "vrdataset" / "dataPackage"

sys.dont_write_bytecode = True
sys.path.insert(0, str(PHASE_DIR / "scripts"))

print(f"Execution start directory: {START_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Read-only source data root: {DATA_ROOT}")
print(f"Output root: {PHASE_DIR}")
print(f"Source exists: {DATA_ROOT.exists()}")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Execute the Audit

The audit recursively scans the requested extensions, parses subject/session/difficulty/run identifiers, detects modality using filename keywords and safe CSV/MAT metadata inspection, checks readability, and writes all requested Phase 01 outputs."""
))

cells.append(nbf.v4.new_code_cell(
"""from IPython.display import display
from phase01_audit import run_audit

outputs = run_audit(PROJECT_ROOT)
inventory = outputs["inventory"]
run_availability = outputs["run_availability"]
summaries = outputs["summaries"]
report = outputs["report"]

print("Audit completed.")
print(f"Total files: {report['total_files']}")
print(f"Readable files: {report['total_readable_files']}")
print(f"Subjects: {report['total_subjects']}")
print(f"Sessions: {report['total_sessions']}")
print(f"Runs: {report['total_runs']}")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## File-Level Inventory Preview

The full file-level inventory is saved to `results/raw_file_inventory.csv`. The preview below shows parsed identifiers, detected modality, readability, and safe CSV preview metadata."""
))

cells.append(nbf.v4.new_code_cell(
"""preview_cols = [
    "relative_path", "extension", "subject_id", "session_id", "difficulty_level", "run_id",
    "detected_modality", "modality_detection_method", "is_readable",
    "row_count_if_csv", "column_count_if_csv", "first_columns_if_csv",
]
display(inventory[preview_cols].head(12))"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Difficulty Level Distribution

This table counts parsed run keys by difficulty level. Rest-task runs appear as `level-000`; ILS task difficulty levels appear as the workload levels used for later modeling."""
))

cells.append(nbf.v4.new_code_cell(
"""display(summaries["difficulty_distribution"])
print("Difficulty distribution figure saved to figures/difficulty_distribution.png")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Modality Availability Summary

The run-level table records whether each parsed run has eye tracking, ECG, EDA, EMG, respiration, head movement, X-Plane, control input, and performance data."""
))

cells.append(nbf.v4.new_code_cell(
"""display(summaries["modality_summary"])
display(summaries["modality_availability_summary"])
print("Modality availability figure saved to figures/modality_availability_bar.png")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Run-Level Modality Availability Preview

The complete run-level modality availability table is saved to `results/run_modality_availability.csv`."""
))

cells.append(nbf.v4.new_code_cell(
"""display(run_availability.head(12))"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Missing, Unknown, and Unreadable Files

Unknown files are intentionally not guessed. They are saved for manual review. Unreadable files, if any, are also saved separately."""
))

cells.append(nbf.v4.new_code_cell(
"""missing_summary = summaries["modality_availability_summary"][["modality", "missing_run_count", "run_coverage_percent"]]
display(missing_summary)

print(f"Unknown files requiring manual review: {report['unknown_file_count']}")
if report['unknown_file_count']:
    display(summaries["unknown_files"][["relative_path", "file_name", "modality_detection_method", "first_columns_if_csv"]].head(20))

print(f"Unreadable files: {report['unreadable_file_count']}")
if report['unreadable_file_count']:
    display(summaries["unreadable_files"][["relative_path", "file_name", "read_error"]].head(20))

print(f"Metadata/license files not modeling data: {report['metadata_or_license_file_count']}")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Complete Multimodal Runs and Phase 02 Feasibility

Strict complete runs require every requested modality, including control input. A second core definition excludes control input because no reliable control-input modality should be inferred unless manually confirmed."""
))

cells.append(nbf.v4.new_code_cell(
"""display(summaries["complete_runs_summary"])
print(f"Strict full multimodal feature extraction feasible: {report['full_multimodal_feature_extraction_feasible']}")
print(f"Core multimodal feature extraction feasible without control input: {report['core_multimodal_feature_extraction_feasible_without_control_input']}")
print(f"Manual modality mapping required before Phase 02: {report['manual_mapping_required_before_phase_02']}")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Saved Outputs

The following cell verifies that the requested Phase 01 artifacts were written under the experiment directory."""
))

cells.append(nbf.v4.new_code_cell(
"""expected_outputs = [
    "results/raw_file_inventory.csv",
    "results/run_modality_availability.csv",
    "results/modality_summary.csv",
    "results/unknown_files_for_review.csv",
    "results/unreadable_files.csv",
    "results/raw_data_audit_report.json",
    "tables/difficulty_distribution.csv",
    "tables/modality_availability_summary.csv",
    "tables/complete_runs_summary.csv",
    "figures/modality_availability_bar.png",
    "figures/difficulty_distribution.png",
    "logs/phase_01_log.txt",
    "README.md",
]
for rel in expected_outputs:
    path = PHASE_DIR / rel
    print(f"{rel}: {'OK' if path.exists() else 'MISSING'}")"""
))

nb["cells"] = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
