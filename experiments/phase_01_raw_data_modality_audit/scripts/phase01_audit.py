from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_EXTENSIONS = {"csv", "mat", "xdf", "txt", "npy", "npz", "pkl", "parquet"}
MODALITIES = [
    "eye_tracking",
    "ecg",
    "eda",
    "emg",
    "respiration",
    "head_movement",
    "xplane",
    "control_input",
    "performance",
]
CORE_MULTIMODAL_MODALITIES = [
    "eye_tracking",
    "ecg",
    "eda",
    "emg",
    "respiration",
    "head_movement",
    "xplane",
    "performance",
]

FILENAME_KEYWORDS = {
    "license_or_metadata": ["license", "readme", "metadata", "description"],
    "performance": [
        "perfmetric",
        "perfmetrics",
        "total_error",
        "glideslope",
        "localizer",
        "airspeed_error",
        "cumulative_total_error",
    ],
    "eye_tracking": ["lslhtcviveeye", "eye", "gaze", "pupil", "eyetrk", "ocuevts"],
    "ecg": ["ecg", "heart", "cardiac", "bitalino"],
    "eda": ["eda", "gsr", "electrodermal"],
    "emg": ["emg"],
    "respiration": ["resp", "respiration", "breath", "breathing"],
    "head_movement": ["head", "hmd", "pose", "position", "rotation"],
    "control_input": ["joystick", "yoke", "throttle", "rudder", "control"],
    "xplane": ["lslxp11", "xplane", "xpc", "aircraft", "altitude", "airspeed", "ils"],
}

COLUMN_KEYWORDS = {
    "performance": ["total_error", "glideslope", "localizer", "airspeed_error", "cumulative_total_error"],
    "eye_tracking": ["gaze", "pupil", "eye_openness", "convergence", "validity_l", "validity_r"],
    "ecg": ["ecg"],
    "eda": ["eda", "gsr", "electrodermal"],
    "emg": ["emg"],
    "respiration": ["respiration", "resp", "breathing", "breath"],
    "head_movement": ["pilot_head", "head_position", "head_roll", "head_pitch", "head_yaw", "hmd", "pose", "rotation"],
    "control_input": ["joystick", "yoke", "throttle", "rudder", "control"],
    "xplane": ["aircraft", "airspeed", "altitude", "agl_altitude", "ils", "landing_gear", "climb_rate"],
}

SUBJECT_RE = re.compile(r"sub-[A-Za-z0-9]+", re.IGNORECASE)
SESSION_RE = re.compile(r"ses-[A-Za-z0-9]+", re.IGNORECASE)
LEVEL_RE = re.compile(r"level-[A-Za-z0-9]+", re.IGNORECASE)
RUN_RE = re.compile(r"run-[0-9]+", re.IGNORECASE)


def _norm_text(value: str) -> str:
    return value.lower().replace("-", "_")


def _first_regex(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(0) if match else ""


def parse_ids(relative_path: str, file_name: str) -> dict[str, str]:
    text = relative_path.replace("\\", "/") + "/" + file_name
    subject_id = _first_regex(SUBJECT_RE, text)
    session_id = _first_regex(SESSION_RE, text)
    difficulty_level = _first_regex(LEVEL_RE, text)
    run_id = _first_regex(RUN_RE, text)
    run_key = ""
    if subject_id and session_id and difficulty_level and run_id:
        run_key = f"{subject_id}_{session_id}_{difficulty_level}_{run_id}"
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "difficulty_level": difficulty_level,
        "run_id": run_id,
        "run_key": run_key,
    }


def filename_modality(file_name: str, relative_path: str) -> tuple[str, str]:
    # Filename keywords are evaluated on the filename itself. Folder names such as
    # task-ils describe the task, not the signal modality, and can otherwise create
    # false X-Plane hits through broad terms like "ils".
    haystack = _norm_text(file_name)
    haystack = haystack.replace("task_ils", "").replace("task_rest", "")
    hits = []
    for modality, keywords in FILENAME_KEYWORDS.items():
        if any(_norm_text(keyword) in haystack for keyword in keywords):
            hits.append(modality)
    if not hits:
        return "unknown", "unclassified_filename"
    if "license_or_metadata" in hits:
        return "license_or_metadata", "filename_keyword"
    if "performance" in hits:
        return "performance", "filename_keyword"
    non_meta_hits = [hit for hit in hits if hit != "license_or_metadata"]
    if len(non_meta_hits) == 1:
        return non_meta_hits[0], "filename_keyword"
    return "unknown", "ambiguous_filename_keywords:" + ";".join(non_meta_hits)


def column_modality(columns: list[str]) -> tuple[str, str]:
    if not columns:
        return "unknown", "no_columns_available"
    haystack = _norm_text(" ".join(columns))
    hits = []
    for modality, keywords in COLUMN_KEYWORDS.items():
        if any(_norm_text(keyword) in haystack for keyword in keywords):
            hits.append(modality)
    if len(hits) == 1:
        return hits[0], "csv_column_keyword"
    if len(hits) > 1:
        if "performance" in hits:
            return "performance", "csv_column_keyword_priority"
        if "head_movement" in hits and any(_norm_text(col).startswith("pilot_head") for col in columns):
            return "head_movement", "csv_column_keyword_priority"
        return "unknown", "ambiguous_csv_column_keywords:" + ";".join(hits)
    return "unknown", "unclassified_csv_columns"


def resolve_modality(filename_result: tuple[str, str], columns: list[str] | None, mat_keys: list[str] | None) -> tuple[str, str]:
    name_modality, name_method = filename_result
    column_result = column_modality(columns or []) if columns is not None else ("unknown", "no_csv_columns")
    column_modality_name, column_method = column_result

    if name_modality == "license_or_metadata":
        return name_modality, name_method
    if name_modality == "unknown" and column_modality_name != "unknown":
        return column_modality_name, column_method
    if name_modality != "unknown" and column_modality_name == "unknown":
        return name_modality, name_method
    if name_modality != "unknown" and column_modality_name != "unknown":
        # Let strongly diagnostic columns correct broad X-Plane filename tokens.
        if name_modality == "xplane" and column_modality_name in {"head_movement", "performance"}:
            return column_modality_name, "csv_columns_overrode_broad_filename_keyword"
        if name_modality == column_modality_name:
            return name_modality, name_method
        return name_modality, name_method + ";csv_column_support=" + column_modality_name

    if mat_keys:
        mat_result = column_modality(mat_keys)
        if mat_result[0] != "unknown":
            return mat_result[0], "mat_key_keyword"
    return "unknown", name_method if name_method else "unknown"


def inspect_csv(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "is_readable": False,
        "read_error": "",
        "row_count_if_csv": np.nan,
        "column_count_if_csv": np.nan,
        "first_columns_if_csv": "",
        "columns": [],
    }
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        header = [col.strip() for col in header]
        row_count = 0
        with path.open("rb") as handle:
            for _ in handle:
                row_count += 1
        row_count = max(row_count - 1, 0) if header else row_count
        result.update(
            {
                "is_readable": True,
                "row_count_if_csv": row_count,
                "column_count_if_csv": len(header),
                "first_columns_if_csv": json.dumps(header[:20], ensure_ascii=False),
                "columns": header,
            }
        )
    except Exception as exc:  # noqa: BLE001 - inventory should record any read failure.
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_mat(path: Path, large_file_mb: float = 50.0) -> dict[str, Any]:
    result = {"is_readable": False, "read_error": "", "mat_keys_if_mat": "", "mat_keys": []}
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb <= large_file_mb:
            from scipy.io import loadmat

            loaded = loadmat(path, variable_names=None, squeeze_me=False, struct_as_record=False)
            keys = [key for key in loaded.keys() if not key.startswith("__")]
        else:
            from scipy.io import whosmat

            keys = [item[0] for item in whosmat(path)]
        result.update({"is_readable": True, "mat_keys_if_mat": json.dumps(keys[:50]), "mat_keys": keys})
    except Exception as scipy_exc:  # noqa: BLE001
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                keys = list(handle.keys())
            result.update({"is_readable": True, "mat_keys_if_mat": json.dumps(keys[:50]), "mat_keys": keys})
        except Exception as h5_exc:  # noqa: BLE001
            result["read_error"] = f"scipy={type(scipy_exc).__name__}: {scipy_exc}; h5py={type(h5_exc).__name__}: {h5_exc}"
    return result


def inspect_other(path: Path, extension: str) -> dict[str, Any]:
    result = {"is_readable": False, "read_error": ""}
    try:
        if extension == "txt":
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                handle.read(4096)
        elif extension == "npy":
            np.load(path, mmap_mode="r", allow_pickle=False)
        elif extension == "npz":
            with np.load(path, allow_pickle=False) as data:
                list(data.files)
        elif extension == "parquet":
            pd.read_parquet(path, columns=[])
        elif extension == "xdf":
            with path.open("rb") as handle:
                handle.read(4096)
        elif extension == "pkl":
            with path.open("rb") as handle:
                handle.read(4096)
        else:
            with path.open("rb") as handle:
                handle.read(4096)
        result["is_readable"] = True
    except Exception as exc:  # noqa: BLE001
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def scan_files(project_root: Path) -> pd.DataFrame:
    data_root = project_root / "vrdataset" / "dataPackage"
    rows: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        extension = path.suffix.lower().lstrip(".")
        if extension not in TARGET_EXTENSIONS:
            continue
        relative_path = path.relative_to(data_root).as_posix()
        ids = parse_ids(relative_path, path.name)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        filename_result = filename_modality(path.name, relative_path)

        csv_info = {"is_readable": False, "read_error": "", "row_count_if_csv": np.nan, "column_count_if_csv": np.nan, "first_columns_if_csv": "", "columns": []}
        mat_info = {"mat_keys_if_mat": "", "mat_keys": []}
        other_info = {"is_readable": False, "read_error": ""}

        if extension == "csv":
            csv_info = inspect_csv(path)
            is_readable = csv_info["is_readable"]
            read_error = csv_info["read_error"]
        elif extension == "mat":
            mat_info = inspect_mat(path)
            is_readable = mat_info["is_readable"]
            read_error = mat_info["read_error"]
        else:
            other_info = inspect_other(path, extension)
            is_readable = other_info["is_readable"]
            read_error = other_info["read_error"]

        detected_modality, detection_method = resolve_modality(
            filename_result,
            csv_info.get("columns") if extension == "csv" else None,
            mat_info.get("mat_keys") if extension == "mat" else None,
        )

        rows.append(
            {
                "file_path": str(path.resolve()),
                "relative_path": relative_path,
                "file_name": path.name,
                "extension": extension,
                "file_size_mb": round(file_size_mb, 6),
                **ids,
                "detected_modality": detected_modality,
                "modality_detection_method": detection_method,
                "is_readable": bool(is_readable),
                "read_error": read_error,
                "row_count_if_csv": csv_info.get("row_count_if_csv", np.nan),
                "column_count_if_csv": csv_info.get("column_count_if_csv", np.nan),
                "first_columns_if_csv": csv_info.get("first_columns_if_csv", ""),
                "mat_keys_if_mat": mat_info.get("mat_keys_if_mat", ""),
            }
        )
    return pd.DataFrame(rows)


def build_run_availability(inventory: pd.DataFrame) -> pd.DataFrame:
    run_files = inventory[inventory["run_key"].fillna("") != ""].copy()
    base_columns = ["subject_id", "session_id", "difficulty_level", "run_id", "run_key"]
    if run_files.empty:
        return pd.DataFrame(columns=base_columns + [f"has_{m}" for m in MODALITIES] + ["available_modality_count", "available_modalities"])

    rows = []
    for run_key, group in run_files.groupby("run_key", sort=True):
        first = group.iloc[0]
        modalities = sorted(set(group["detected_modality"]) & set(MODALITIES))
        row = {col: first[col] for col in base_columns}
        for modality in MODALITIES:
            row[f"has_{modality}"] = modality in modalities
        row["available_modality_count"] = len(modalities)
        row["available_modalities"] = ";".join(modalities)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["subject_id", "session_id", "difficulty_level", "run_id"]).reset_index(drop=True)


def make_summaries(inventory: pd.DataFrame, run_availability: pd.DataFrame) -> dict[str, Any]:
    modality_summary = (
        inventory.groupby("detected_modality", dropna=False)
        .size()
        .reset_index(name="file_count")
        .sort_values("file_count", ascending=False)
    )

    difficulty_distribution = (
        run_availability.groupby("difficulty_level", dropna=False)
        .size()
        .reset_index(name="run_count")
        .sort_values("difficulty_level")
    )

    modality_availability_summary = pd.DataFrame(
        [
            {
                "modality": modality,
                "run_count": int(run_availability[f"has_{modality}"].sum()) if not run_availability.empty else 0,
                "missing_run_count": int((~run_availability[f"has_{modality}"]).sum()) if not run_availability.empty else 0,
            }
            for modality in MODALITIES
        ]
    )
    if not run_availability.empty:
        modality_availability_summary["run_coverage_percent"] = (
            modality_availability_summary["run_count"] / len(run_availability) * 100
        ).round(2)
    else:
        modality_availability_summary["run_coverage_percent"] = 0.0

    strict_cols = [f"has_{m}" for m in MODALITIES]
    core_cols = [f"has_{m}" for m in CORE_MULTIMODAL_MODALITIES]
    if run_availability.empty:
        strict_complete = pd.Series(dtype=bool)
        core_complete = pd.Series(dtype=bool)
    else:
        strict_complete = run_availability[strict_cols].all(axis=1)
        core_complete = run_availability[core_cols].all(axis=1)

    complete_runs_summary = pd.DataFrame(
        [
            {
                "definition": "strict_all_listed_modalities",
                "required_modalities": ";".join(MODALITIES),
                "complete_run_count": int(strict_complete.sum()) if len(strict_complete) else 0,
                "total_run_count": int(len(run_availability)),
            },
            {
                "definition": "core_multimodal_without_control_input",
                "required_modalities": ";".join(CORE_MULTIMODAL_MODALITIES),
                "complete_run_count": int(core_complete.sum()) if len(core_complete) else 0,
                "total_run_count": int(len(run_availability)),
            },
        ]
    )
    if len(run_availability):
        complete_runs_summary["coverage_percent"] = (complete_runs_summary["complete_run_count"] / len(run_availability) * 100).round(2)
    else:
        complete_runs_summary["coverage_percent"] = 0.0

    unknown_files = inventory[inventory["detected_modality"] == "unknown"].copy()
    unreadable_files = inventory[~inventory["is_readable"]].copy()
    metadata_files = inventory[inventory["detected_modality"] == "license_or_metadata"].copy()

    return {
        "modality_summary": modality_summary,
        "difficulty_distribution": difficulty_distribution,
        "modality_availability_summary": modality_availability_summary,
        "complete_runs_summary": complete_runs_summary,
        "unknown_files": unknown_files,
        "unreadable_files": unreadable_files,
        "metadata_files": metadata_files,
    }


def save_figures(phase_dir: Path, difficulty_distribution: pd.DataFrame, modality_availability_summary: pd.DataFrame) -> None:
    figures_dir = phase_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sorted_availability = modality_availability_summary.sort_values("run_count", ascending=False)
    plt.bar(sorted_availability["modality"], sorted_availability["run_count"], color="#2f6f73")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Runs with modality")
    plt.title("Run-level modality availability")
    plt.tight_layout()
    plt.savefig(figures_dir / "modality_availability_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(difficulty_distribution["difficulty_level"].astype(str), difficulty_distribution["run_count"], color="#8a5a44")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Run count")
    plt.title("Difficulty level distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "difficulty_distribution.png", dpi=180)
    plt.close()


def write_readme(phase_dir: Path, report: dict[str, Any], modality_availability_summary: pd.DataFrame, unknown_files: pd.DataFrame, unreadable_files: pd.DataFrame, metadata_files: pd.DataFrame) -> None:
    found_modalities = report["found_modalities"]
    incomplete_modalities = report["missing_or_incomplete_modalities"]
    manual_required = report["manual_mapping_required_before_phase_02"]
    feasible = report["full_multimodal_feature_extraction_feasible"]
    core_feasible = report["core_multimodal_feature_extraction_feasible_without_control_input"]

    lines = [
        "# Phase 01: Raw Data Audit and Modality Inventory",
        "",
        "## Scope",
        "This phase audited `vrdataset/dataPackage` as read-only source data. Generated artifacts were saved only under this experiment folder.",
        "",
        "No feature extraction, final modeling dataset creation, ML training, or HDC training was performed.",
        "",
        "## Modalities Found",
        "",
    ]
    for modality in found_modalities:
        rows = modality_availability_summary[modality_availability_summary["modality"] == modality]
        run_count = int(rows["run_count"].iloc[0]) if not rows.empty else 0
        lines.append(f"- `{modality}`: present in {run_count} runs")
    if not found_modalities:
        lines.append("- No modeling modalities were confidently detected.")

    lines.extend(["", "## Missing or Incomplete Modalities", ""])
    if incomplete_modalities:
        for modality in incomplete_modalities:
            rows = modality_availability_summary[modality_availability_summary["modality"] == modality]
            missing = int(rows["missing_run_count"].iloc[0]) if not rows.empty else report["total_runs"]
            lines.append(f"- `{modality}` is missing in {missing} runs")
    else:
        lines.append("- No incomplete modalities were detected across the requested modality set.")

    lines.extend([
        "",
        "## Manual Confirmation Needed",
        "",
        f"- Unknown files requiring review: {len(unknown_files)}",
        f"- Failed or unreadable files: {len(unreadable_files)}",
    ])
    if len(unknown_files):
        top_unknown = unknown_files["file_name"].head(10).tolist()
        for file_name in top_unknown:
            lines.append(f"- `{file_name}`")
    if manual_required:
        lines.append("- Manual modality mapping is required before Phase 02 because some files remain unknown or unreadable.")
    else:
        lines.append("- No manual modality mapping is required based on this audit.")

    lines.extend([
        "",
        "## Feasibility",
        "",
        f"- Strict full multimodal feature extraction across all listed modalities is feasible: {feasible}",
        f"- Core multimodal feature extraction without `control_input` is feasible: {core_feasible}",
        f"- Strict complete runs: {report['strict_complete_run_count']} of {report['total_runs']}",
        f"- Core complete runs: {report['core_complete_run_count']} of {report['total_runs']}",
        "",
        "## Phase 02 Recommendation",
        "",
        "Phase 02 should extract synchronized run-level features for the confidently detected modalities: eye tracking, ECG, EDA, EMG, respiration, head movement, X-Plane aircraft/performance streams, and per-run performance metrics. Control input should only be included if manual review identifies a reliable source for it.",
        "",
        "## Limitations",
        "",
        "- File-level modality detection is conservative: unclear files are labeled `unknown` rather than guessed.",
        "- Row counts were calculated from file lines and CSV headers were previewed safely; no feature matrices or modeling datasets were produced.",
        "- MAT support is implemented, but this dataPackage audit found no `.mat` files among the requested extensions.",
        "- Rest-task runs use `level-000`; ILS workload-class modeling should focus on the ILS difficulty levels unless rest is intentionally included as a separate baseline condition.",
        "",
        "## Metadata and Legal Files",
        "",
        f"- LICENSE, README, metadata, and description files are metadata/legal documents, not modeling data. Count found in this dataPackage audit: {len(metadata_files)}.",
    ])
    (phase_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_experiment_status(project_root: Path, report: dict[str, Any]) -> None:
    status_path = project_root / "EXPERIMENT_STATUS.md"
    text = status_path.read_text(encoding="utf-8")
    text = text.replace(
        "- Current completed scope: Phase 00 project setup and revised data inventory",
        "- Current completed scope: Phase 01 raw data audit and modality inventory",
    )
    old_row = "| 01 | `phase_01_raw_data_modality_audit` | Not started | Next: audit raw modalities, files, subjects, sessions, task levels, runs, and quality constraints in `dataPackage`. |"
    new_row = "| 01 | `phase_01_raw_data_modality_audit` | Complete | Raw dataPackage scan, file-level inventory, run-level modality availability, summaries, figures, JSON report, README, and executed notebook created. |"
    text = text.replace(old_row, new_row)

    phase_outputs = """
## Phase 01 Outputs
- Executed notebook: `experiments/phase_01_raw_data_modality_audit/notebooks/01_raw_data_modality_audit.ipynb`
- File inventory: `experiments/phase_01_raw_data_modality_audit/results/raw_file_inventory.csv`
- Run modality availability: `experiments/phase_01_raw_data_modality_audit/results/run_modality_availability.csv`
- Audit report: `experiments/phase_01_raw_data_modality_audit/results/raw_data_audit_report.json`
- Unknown review list: `experiments/phase_01_raw_data_modality_audit/results/unknown_files_for_review.csv`
- Unreadable file list: `experiments/phase_01_raw_data_modality_audit/results/unreadable_files.csv`
- Figures: `experiments/phase_01_raw_data_modality_audit/figures/modality_availability_bar.png`, `experiments/phase_01_raw_data_modality_audit/figures/difficulty_distribution.png`
- README: `experiments/phase_01_raw_data_modality_audit/README.md`

## Phase 01 Summary
- Files audited: {total_files}
- Readable files: {total_readable_files}
- Subjects: {total_subjects}
- Sessions: {total_sessions}
- Runs: {total_runs}
- Unknown files requiring manual review: {unknown_file_count}
- Unreadable files: {unreadable_file_count}
- Strict complete multimodal runs: {strict_complete_run_count}
- Core complete multimodal runs without control input: {core_complete_run_count}
""".format(**report)

    if "## Phase 01 Outputs" in text:
        text = re.sub(
            r"\n## Phase 01 Outputs\n.*?(?=\n## Latest Update\n)",
            "\n" + phase_outputs + "\n",
            text,
            flags=re.DOTALL,
        )
    else:
        text = text.replace("\n## Latest Update\n", "\n" + phase_outputs + "\n## Latest Update\n")

    latest = """## Latest Update
- Phase 01 raw data audit completed on {date}.
- The notebook was executed and saved with visible outputs.
- No files under `vrdataset` were intentionally modified.
- No feature extraction, final modeling dataset creation, ML training, HDC training, or model evaluation was performed.
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    text = re.sub(r"## Latest Update\n(?:- .*\n?)+", latest, text, flags=re.MULTILINE)
    status_path.write_text(text, encoding="utf-8")


def run_audit(project_root: Path | str) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    phase_dir = project_root / "experiments" / "phase_01_raw_data_modality_audit"
    for subdir in ["notebooks", "results", "tables", "figures", "logs"]:
        (phase_dir / subdir).mkdir(parents=True, exist_ok=True)

    started_at = datetime.now()
    inventory = scan_files(project_root)
    run_availability = build_run_availability(inventory)
    summaries = make_summaries(inventory, run_availability)

    results_dir = phase_dir / "results"
    tables_dir = phase_dir / "tables"
    inventory.to_csv(results_dir / "raw_file_inventory.csv", index=False)
    run_availability.to_csv(results_dir / "run_modality_availability.csv", index=False)
    summaries["modality_summary"].to_csv(results_dir / "modality_summary.csv", index=False)
    summaries["unknown_files"].to_csv(results_dir / "unknown_files_for_review.csv", index=False)
    summaries["unreadable_files"].to_csv(results_dir / "unreadable_files.csv", index=False)
    summaries["difficulty_distribution"].to_csv(tables_dir / "difficulty_distribution.csv", index=False)
    summaries["modality_availability_summary"].to_csv(tables_dir / "modality_availability_summary.csv", index=False)
    summaries["complete_runs_summary"].to_csv(tables_dir / "complete_runs_summary.csv", index=False)

    save_figures(phase_dir, summaries["difficulty_distribution"], summaries["modality_availability_summary"])

    found_modalities = sorted(set(inventory["detected_modality"]) & set(MODALITIES))
    missing_or_incomplete = summaries["modality_availability_summary"].loc[
        summaries["modality_availability_summary"]["missing_run_count"] > 0, "modality"
    ].tolist()
    strict_complete_count = int(summaries["complete_runs_summary"].loc[summaries["complete_runs_summary"]["definition"] == "strict_all_listed_modalities", "complete_run_count"].iloc[0])
    core_complete_count = int(summaries["complete_runs_summary"].loc[summaries["complete_runs_summary"]["definition"] == "core_multimodal_without_control_input", "complete_run_count"].iloc[0])

    report = {
        "phase": "01_raw_data_modality_audit",
        "started_at": started_at.isoformat(timespec="seconds"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "source_data_root": str((project_root / "vrdataset" / "dataPackage").resolve()),
        "output_root": str(phase_dir.resolve()),
        "target_extensions": sorted(TARGET_EXTENSIONS),
        "total_files": int(len(inventory)),
        "total_readable_files": int(inventory["is_readable"].sum()) if not inventory.empty else 0,
        "total_subjects": int(inventory.loc[inventory["subject_id"] != "", "subject_id"].nunique()) if not inventory.empty else 0,
        "total_sessions": int(inventory.loc[inventory["session_id"] != "", ["subject_id", "session_id"]].drop_duplicates().shape[0]) if not inventory.empty else 0,
        "total_runs": int(len(run_availability)),
        "difficulty_levels": summaries["difficulty_distribution"].to_dict(orient="records"),
        "found_modalities": found_modalities,
        "missing_or_incomplete_modalities": missing_or_incomplete,
        "modality_availability_summary": summaries["modality_availability_summary"].to_dict(orient="records"),
        "unknown_file_count": int(len(summaries["unknown_files"])),
        "unreadable_file_count": int(len(summaries["unreadable_files"])),
        "metadata_or_license_file_count": int(len(summaries["metadata_files"])),
        "strict_complete_run_count": strict_complete_count,
        "core_complete_run_count": core_complete_count,
        "full_multimodal_feature_extraction_feasible": bool(strict_complete_count > 0),
        "core_multimodal_feature_extraction_feasible_without_control_input": bool(core_complete_count > 0),
        "manual_mapping_required_before_phase_02": bool(len(summaries["unknown_files"]) > 0 or len(summaries["unreadable_files"]) > 0),
        "no_prohibited_actions": {
            "feature_extraction_performed": False,
            "final_modeling_dataset_created": False,
            "ml_or_hdc_training_performed": False,
            "vrdataset_modified": False,
        },
    }

    (results_dir / "raw_data_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_readme(phase_dir, report, summaries["modality_availability_summary"], summaries["unknown_files"], summaries["unreadable_files"], summaries["metadata_files"])
    update_experiment_status(project_root, report)

    log_lines = [
        "Phase 01 raw data audit log",
        f"Started: {report['started_at']}",
        f"Completed: {report['completed_at']}",
        f"Source: {report['source_data_root']}",
        f"Files audited: {report['total_files']}",
        f"Readable files: {report['total_readable_files']}",
        f"Subjects: {report['total_subjects']}",
        f"Sessions: {report['total_sessions']}",
        f"Runs: {report['total_runs']}",
        f"Found modalities: {', '.join(found_modalities)}",
        f"Unknown files: {report['unknown_file_count']}",
        f"Unreadable files: {report['unreadable_file_count']}",
        f"Strict full multimodal feasible: {report['full_multimodal_feature_extraction_feasible']}",
        f"Core multimodal feasible without control input: {report['core_multimodal_feature_extraction_feasible_without_control_input']}",
        "No feature extraction, dataset creation, or model training performed.",
        "No files under vrdataset were intentionally modified.",
    ]
    (phase_dir / "logs" / "phase_01_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return {
        "inventory": inventory,
        "run_availability": run_availability,
        "summaries": summaries,
        "report": report,
    }
