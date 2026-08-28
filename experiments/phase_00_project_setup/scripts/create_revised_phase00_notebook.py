"""Create and execute the revised Phase 00 project setup and inventory notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = PROJECT_ROOT / "experiments" / "phase_00_project_setup" / "notebooks" / "00_revised_project_setup_and_inventory.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    nb.cells = [
        md(
            """
            # Phase 00: Revised Project Setup and Data Inventory

            This notebook performs only Phase 00 work: path checks, source folder checks,
            recursive inventory, revised phase-structure recording, and setup reporting.
            It does not perform feature extraction, labeling, ML training, or HDC training.
            """
        ),
        md(
            """
            ## 1. Project path check

            Confirm the current working directory and prepare output locations under `experiments`.
            """
        ),
        code(
            """
            from pathlib import Path
            from datetime import datetime, timezone
            import json
            import pandas as pd
            from IPython.display import display

            PROJECT_ROOT = Path.cwd().resolve()
            EXPECTED_ROOT_NAME = "hdc-vr-pilot"
            DATASET_ROOT = PROJECT_ROOT / "vrdataset"
            DATA_PACKAGE = DATASET_ROOT / "dataPackage"
            STARTER_CODE = DATASET_ROOT / "starterCode"
            REFERENCE_DOCS = DATASET_ROOT / "referenceDocuments"

            PHASE00 = PROJECT_ROOT / "experiments" / "phase_00_project_setup"
            RESULTS_DIR = PHASE00 / "results"
            TABLES_DIR = PHASE00 / "tables"
            LOGS_DIR = PHASE00 / "logs"
            for directory in [RESULTS_DIR, TABLES_DIR, LOGS_DIR]:
                directory.mkdir(parents=True, exist_ok=True)

            executed_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(f"Current working directory: {PROJECT_ROOT}")
            print(f"Expected root name: {EXPECTED_ROOT_NAME}")
            print(f"Project root name check: {PROJECT_ROOT.name == EXPECTED_ROOT_NAME}")
            print(f"Notebook executed at UTC: {executed_at_utc}")
            """
        ),
        md(
            """
            ## 2. Required source folder checks

            Verify that `vrdataset`, `dataPackage`, `starterCode`, and `referenceDocuments` exist.
            The source dataset is read-only for this project.
            """
        ),
        code(
            """
            folder_checks = pd.DataFrame(
                [
                    {"folder": "vrdataset", "path": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(), "exists": DATASET_ROOT.is_dir()},
                    {"folder": "dataPackage", "path": DATA_PACKAGE.relative_to(PROJECT_ROOT).as_posix(), "exists": DATA_PACKAGE.is_dir()},
                    {"folder": "starterCode", "path": STARTER_CODE.relative_to(PROJECT_ROOT).as_posix(), "exists": STARTER_CODE.is_dir()},
                    {"folder": "referenceDocuments", "path": REFERENCE_DOCS.relative_to(PROJECT_ROOT).as_posix(), "exists": REFERENCE_DOCS.is_dir()},
                ]
            )
            display(folder_checks)
            print(f"Whether vrdataset exists: {DATASET_ROOT.is_dir()}")
            print(f"Whether dataPackage exists: {DATA_PACKAGE.is_dir()}")
            print(f"Whether starterCode exists: {STARTER_CODE.is_dir()}")
            print(f"Whether referenceDocuments exists: {REFERENCE_DOCS.is_dir()}")

            if not folder_checks["exists"].all():
                missing = folder_checks.loc[~folder_checks["exists"], "path"].tolist()
                raise FileNotFoundError(f"Missing required source folders: {missing}")
            print("Source folder checks passed. No files under vrdataset are modified by this notebook.")
            """
        ),
        md(
            """
            ## 3. Source role clarification

            `starterCode` is mainly an eye-tracking and oculomotor pilot reference. It can help
            interpret example processing choices, but it is not the final full multimodal dataset.

            `dataPackage` is the main source for the full multimodal thesis experiment. Later phases
            should audit and extract features from `vrdataset/dataPackage`, after Phase 01 documents
            available modalities, runs, subjects, levels, and quality constraints.
            """
        ),
        code(
            """
            source_roles = pd.DataFrame(
                [
                    {
                        "source": "vrdataset/starterCode",
                        "role": "Eye-tracking and oculomotor pilot reference",
                        "use_in_final_experiment": "Reference only; not the final full multimodal dataset",
                    },
                    {
                        "source": "vrdataset/dataPackage",
                        "role": "Main source for full multimodal experiment",
                        "use_in_final_experiment": "Primary source for Phase 01 audit and Phase 02 feature extraction",
                    },
                    {
                        "source": "vrdataset/referenceDocuments",
                        "role": "Documentation and data dictionary source",
                        "use_in_final_experiment": "Reference for schema, quality, and data interpretation",
                    },
                ]
            )
            display(source_roles)
            print("starterCode is treated as an eye-tracking and oculomotor pilot reference only.")
            print("dataPackage is treated as the main source for the full multimodal experiment.")
            """
        ),
        md(
            """
            ## 4. Recursive file inventory for `vrdataset`

            Scan all files under the read-only source folder and build a full inventory. The requested
            target extensions are: csv, mat, xdf, txt, npy, npz, pkl, parquet, py, m, and ipynb.
            """
        ),
        code(
            """
            TARGET_EXTENSIONS = ["csv", "mat", "xdf", "txt", "npy", "npz", "pkl", "parquet", "py", "m", "ipynb"]
            TARGET_EXTENSION_SET = {f".{ext}" for ext in TARGET_EXTENSIONS}

            def top_area(path: Path) -> str:
                rel = path.relative_to(DATASET_ROOT)
                parts = rel.parts
                if len(parts) >= 2 and parts[0] == "starterCode" and parts[1] == "data_feats":
                    return "starterCode/data_feats"
                return parts[0] if parts else "vrdataset"

            def folder_depth(path: Path, depth: int = 4) -> str:
                rel_parent = path.parent.relative_to(DATASET_ROOT)
                parts = rel_parent.parts[:depth]
                return "/".join(parts) if parts else "."

            all_files = sorted(path for path in DATASET_ROOT.rglob("*") if path.is_file())
            all_dirs = sorted(path for path in DATASET_ROOT.rglob("*") if path.is_dir())
            records = []
            for path in all_files:
                stat = path.stat()
                suffix = path.suffix.lower()
                extension = suffix.lstrip(".") if suffix else "[no_extension]"
                records.append(
                    {
                        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "dataset_relative_path": path.relative_to(DATASET_ROOT).as_posix(),
                        "file_name": path.name,
                        "extension": extension,
                        "is_target_extension": suffix in TARGET_EXTENSION_SET,
                        "dataset_area": top_area(path),
                        "analysis_folder": folder_depth(path),
                        "size_bytes": stat.st_size,
                        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                    }
                )

            data_inventory = pd.DataFrame.from_records(records).sort_values(["dataset_area", "extension", "relative_path"]).reset_index(drop=True)
            inventory_path = RESULTS_DIR / "data_inventory.csv"
            data_inventory.to_csv(inventory_path, index=False)

            print(f"Number of directories found under vrdataset: {len(all_dirs):,}")
            print(f"Number of files found: {len(data_inventory):,}")
            print(f"Inventory saved to: {inventory_path.relative_to(PROJECT_ROOT).as_posix()}")
            display(data_inventory.head(10))
            """
        ),
        md(
            """
            ## 5. File type summary and folder summary

            Summarize requested file extensions and the required source folders separately.
            """
        ),
        code(
            """
            observed_counts = data_inventory["extension"].value_counts().to_dict()
            file_type_summary = pd.DataFrame(
                [
                    {
                        "extension": ext,
                        "file_count": int(observed_counts.get(ext, 0)),
                        "total_size_bytes": int(data_inventory.loc[data_inventory["extension"] == ext, "size_bytes"].sum()),
                    }
                    for ext in TARGET_EXTENSIONS
                ]
            )
            file_type_summary["total_size_mib"] = (file_type_summary["total_size_bytes"] / (1024 ** 2)).round(3)
            file_type_summary.to_csv(TABLES_DIR / "file_type_summary.csv", index=False)

            folder_specs = {
                "vrdataset/dataPackage": DATA_PACKAGE,
                "vrdataset/starterCode": STARTER_CODE,
                "vrdataset/starterCode/data_feats": STARTER_CODE / "data_feats",
                "vrdataset/referenceDocuments": REFERENCE_DOCS,
            }
            folder_rows = []
            for label, folder in folder_specs.items():
                files = [path for path in folder.rglob("*") if path.is_file()] if folder.exists() else []
                target_files = [path for path in files if path.suffix.lower() in TARGET_EXTENSION_SET]
                folder_rows.append(
                    {
                        "folder": label,
                        "exists": folder.exists(),
                        "all_file_count": len(files),
                        "target_extension_file_count": len(target_files),
                        "total_size_bytes": sum(path.stat().st_size for path in files),
                        "total_size_mib": round(sum(path.stat().st_size for path in files) / (1024 ** 2), 3),
                    }
                )
            folder_summary = pd.DataFrame(folder_rows)
            folder_summary.to_csv(TABLES_DIR / "folder_summary.csv", index=False)

            print("File type summary:")
            display(file_type_summary)
            print("Folder summary:")
            display(folder_summary)
            """
        ),
        md(
            """
            ## 6. Top candidate folders for Phase 01

            Rank source folders by file count and size to guide the raw data modality audit.
            """
        ),
        code(
            """
            candidate_source = data_inventory[data_inventory["relative_path"].str.startswith("vrdataset/dataPackage/")].copy()
            top_candidate_folders = (
                candidate_source.groupby("analysis_folder", as_index=False)
                .agg(file_count=("relative_path", "count"), total_size_bytes=("size_bytes", "sum"))
                .sort_values(["file_count", "total_size_bytes"], ascending=False)
                .head(15)
            )
            top_candidate_folders["total_size_mib"] = (top_candidate_folders["total_size_bytes"] / (1024 ** 2)).round(3)
            print("Top candidate folders for Phase 01 raw modality audit:")
            display(top_candidate_folders)
            """
        ),
        md(
            """
            ## 7. Revised phase structure

            Save and display the revised 11-phase thesis workflow.
            """
        ),
        code(
            """
            phase_rows = [
                {"phase_index": 0, "phase_id": "phase_00_project_setup", "phase_title": "Project setup and data inventory", "status": "complete"},
                {"phase_index": 1, "phase_id": "phase_01_raw_data_modality_audit", "phase_title": "Raw data audit and modality inventory", "status": "not_started"},
                {"phase_index": 2, "phase_id": "phase_02_full_multimodal_feature_extraction", "phase_title": "Full multimodal feature extraction", "status": "not_started"},
                {"phase_index": 3, "phase_id": "phase_03_multimodal_dataset_labeling", "phase_title": "Multimodal dataset construction and four-class labeling", "status": "not_started"},
                {"phase_index": 4, "phase_id": "phase_04_ml_baseline_four_class", "phase_title": "Traditional ML four-class baseline", "status": "not_started"},
                {"phase_index": 5, "phase_id": "phase_05_basic_hdc_four_class", "phase_title": "Basic HDC four-class classification", "status": "not_started"},
                {"phase_index": 6, "phase_id": "phase_06_hdc_classifier_screening", "phase_title": "HDC classifier screening", "status": "not_started"},
                {"phase_index": 7, "phase_id": "phase_07_single_modality_contribution", "phase_title": "Single-modality contribution analysis", "status": "not_started"},
                {"phase_index": 8, "phase_id": "phase_08_multimodal_fusion", "phase_title": "Multimodal fusion strategy comparison", "status": "not_started"},
                {"phase_index": 9, "phase_id": "phase_09_robustness_missing_modality", "phase_title": "Robustness and missing-modality analysis", "status": "not_started"},
                {"phase_index": 10, "phase_id": "phase_10_onlinehd_lsl_simulation", "phase_title": "OnlineHD and LSL simulation", "status": "not_started"},
            ]
            revised_phase_structure = pd.DataFrame(phase_rows)
            phase_structure_path = RESULTS_DIR / "revised_phase_structure.csv"
            revised_phase_structure.to_csv(phase_structure_path, index=False)
            print(f"Revised phase structure saved to: {phase_structure_path.relative_to(PROJECT_ROOT).as_posix()}")
            display(revised_phase_structure)
            """
        ),
        md(
            """
            ## 8. Save setup report and log

            Save a JSON setup report and a plain-text Phase 00 log. These files record what was
            checked and where outputs were written.
            """
        ),
        code(
            """
            report = {
                "executed_at_utc": executed_at_utc,
                "project_root": str(PROJECT_ROOT),
                "data_safety": "vrdataset treated as read-only; generated outputs saved under experiments",
                "source_roles": {
                    "starterCode": "eye-tracking and oculomotor pilot reference only",
                    "dataPackage": "main source for full multimodal experiment",
                    "referenceDocuments": "documentation and data dictionary source",
                },
                "folder_checks": folder_checks.to_dict(orient="records"),
                "total_directories_under_vrdataset": len(all_dirs),
                "total_files_under_vrdataset": len(data_inventory),
                "target_extensions": TARGET_EXTENSIONS,
                "file_type_summary": file_type_summary.to_dict(orient="records"),
                "folder_summary": folder_summary.to_dict(orient="records"),
                "top_candidate_folders_for_phase_01": top_candidate_folders.to_dict(orient="records"),
                "outputs": {
                    "data_inventory": "experiments/phase_00_project_setup/results/data_inventory.csv",
                    "revised_phase_structure": "experiments/phase_00_project_setup/results/revised_phase_structure.csv",
                    "project_setup_report": "experiments/phase_00_project_setup/results/project_setup_report.json",
                    "file_type_summary": "experiments/phase_00_project_setup/tables/file_type_summary.csv",
                    "folder_summary": "experiments/phase_00_project_setup/tables/folder_summary.csv",
                    "phase_00_log": "experiments/phase_00_project_setup/logs/phase_00_log.txt",
                },
            }
            report_path = RESULTS_DIR / "project_setup_report.json"
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)

            log_lines = [
                "Phase 00 revised project setup and inventory log",
                f"Executed at UTC: {executed_at_utc}",
                f"Project root: {PROJECT_ROOT}",
                "Data safety: vrdataset treated as read-only",
                "starterCode role: eye-tracking and oculomotor pilot reference only",
                "dataPackage role: main source for full multimodal experiment",
                f"Total files found under vrdataset: {len(data_inventory):,}",
                f"Data inventory: {inventory_path.relative_to(PROJECT_ROOT).as_posix()}",
                f"Revised phase structure: {phase_structure_path.relative_to(PROJECT_ROOT).as_posix()}",
                f"Project setup report: {report_path.relative_to(PROJECT_ROOT).as_posix()}",
                "No feature extraction, labeling, ML training, or HDC training was performed.",
            ]
            log_path = LOGS_DIR / "phase_00_log.txt"
            log_path.write_text(chr(10).join(log_lines) + chr(10), encoding="utf-8")

            print(f"Project setup report saved to: {report_path.relative_to(PROJECT_ROOT).as_posix()}")
            print(f"Phase 00 log saved to: {log_path.relative_to(PROJECT_ROOT).as_posix()}")
            print("Phase 00 complete. No feature extraction, labeling, ML training, or HDC training was performed.")
            """
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



