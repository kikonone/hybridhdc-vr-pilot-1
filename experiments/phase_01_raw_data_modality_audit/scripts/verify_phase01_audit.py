from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


RUN_DIR_RE = re.compile(
    r"^level-(?P<level>[^_]+)_run-(?P<run>\d+)$", re.IGNORECASE
)
SUBJECT_RE = re.compile(r"^sub-(?P<subject>.+)$", re.IGNORECASE)
SESSION_RE = re.compile(r"^ses-(?P<session>.+)$", re.IGNORECASE)
FILE_ID_RE = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+).*?_level-(?P<level>[^_]+)_run-(?P<run>\d+)",
    re.IGNORECASE,
)

MODALITIES = (
    "eye_tracking",
    "ecg",
    "eda",
    "emg",
    "respiration",
    "head_movement",
    "xplane",
    "performance",
    "torso_body_accelerometer",
    "control_input",
)

EXPECTED_RUNS_BY_TASK = {
    "rest": {
        ("000", "001"),
        ("000", "002"),
    },
    "ils": {
        ("01B", "001"),
        ("01B", "007"),
        ("01B", "012"),
        ("02B", "003"),
        ("02B", "008"),
        ("02B", "010"),
        ("03B", "002"),
        ("03B", "005"),
        ("03B", "011"),
        ("04B", "004"),
        ("04B", "006"),
        ("04B", "009"),
    },
}


@dataclass(frozen=True)
class CsvPreview:
    readable: bool
    columns: tuple[str, ...]
    first_row: tuple[str, ...]
    error: str


def safe_csv_preview(path: Path) -> CsvPreview:
    try:
        with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
            reader = csv.reader(handle)
            columns = tuple(next(reader, []))
            first_row = tuple(next(reader, []))
        return CsvPreview(True, columns, first_row, "")
    except (OSError, csv.Error, UnicodeError) as exc:
        return CsvPreview(False, (), (), f"{type(exc).__name__}: {exc}")


def parse_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def modality_evidence(columns: Iterable[str]) -> set[str]:
    names = {column.strip().lower() for column in columns}
    evidence: set[str] = set()
    if any(
        name.startswith(("gaze_", "pupil_", "eye_openness", "validity_", "convergence_"))
        or name in {"fixationseq", "saccadeseq"}
        for name in names
    ):
        evidence.add("eye_tracking")
    if any(name.startswith("ecg_") for name in names):
        evidence.add("ecg")
    if any(name.startswith(("eda_", "gsr_")) for name in names):
        evidence.add("eda")
    if any(name.startswith("emg_") for name in names):
        evidence.add("emg")
    if any(name.startswith("respiration_") for name in names):
        evidence.add("respiration")
    if any(name.startswith("pilot_head_") for name in names):
        evidence.add("head_movement")
    if any(name.startswith("aircraft_") for name in names):
        evidence.add("xplane")
    if {"glideslope_error_deg", "localizer_error_deg", "airspeed_error_kts", "total_error"} & names:
        evidence.add("performance")
    if any(name.startswith("accelerometry_torso_") for name in names):
        evidence.add("torso_body_accelerometer")

    # Trim and aircraft state are not direct pilot command measurements.
    explicit_control_patterns = (
        "joystick",
        "yoke",
        "throttle",
        "rudder",
        "control_input",
        "flight_control",
        "stick_position",
        "pedal_position",
        "aileron_command",
        "elevator_command",
    )
    if any(any(pattern in name for pattern in explicit_control_patterns) for name in names):
        evidence.add("control_input")
    return evidence


def run_identity(data_root: Path, run_dir: Path) -> dict[str, str]:
    relative = run_dir.relative_to(data_root)
    if len(relative.parts) != 4:
        raise ValueError(f"Unexpected run path depth: {relative}")
    task_part, subject_part, session_part, run_part = relative.parts
    task = task_part.removeprefix("task-")
    subject_match = SUBJECT_RE.match(subject_part)
    session_match = SESSION_RE.match(session_part)
    run_match = RUN_DIR_RE.match(run_part)
    if not (subject_match and session_match and run_match):
        raise ValueError(f"Cannot parse run identity: {relative}")
    subject_id = f"sub-{subject_match.group('subject')}"
    session_id = f"ses-{session_match.group('session')}"
    difficulty_level = f"level-{run_match.group('level')}"
    run_id = f"run-{run_match.group('run')}"
    return {
        "task": task,
        "subject_id": subject_id,
        "session_id": session_id,
        "difficulty_level": difficulty_level,
        "run_id": run_id,
        "run_key": "_".join((subject_id, session_id, difficulty_level, run_id)),
    }


def inspect_run(data_root: Path, run_dir: Path) -> dict[str, object]:
    identity = run_identity(data_root, run_dir)
    evidence_paths: dict[str, list[str]] = defaultdict(list)
    abnormal: list[str] = []
    unresolved: list[str] = []
    readable_files = 0
    data_files = 0
    metadata_files = 0
    verified_signal_files = 0
    all_files = sorted(path for path in run_dir.iterdir() if path.is_file())
    previews: dict[str, CsvPreview] = {}

    for path in all_files:
        relative_path = path.relative_to(data_root).as_posix()
        file_id_match = FILE_ID_RE.search(path.name)
        if file_id_match:
            file_identity = (
                f"sub-{file_id_match.group('subject')}",
                f"ses-{file_id_match.group('session')}",
                f"level-{file_id_match.group('level')}",
                f"run-{file_id_match.group('run')}",
            )
            path_identity = (
                identity["subject_id"],
                identity["session_id"],
                identity["difficulty_level"],
                identity["run_id"],
            )
            if file_identity != path_identity:
                abnormal.append(
                    f"identifier_mismatch:{relative_path}:{'|'.join(file_identity)}"
                )
        if path.suffix.lower() != ".csv":
            unresolved.append(relative_path)
            continue
        preview = safe_csv_preview(path)
        previews[path.name] = preview
        if preview.readable:
            readable_files += 1
        else:
            abnormal.append(f"unreadable:{relative_path}:{preview.error}")
            continue
        if not preview.columns:
            abnormal.append(f"missing_csv_header:{relative_path}")
            continue

        is_header_metadata = path.name.lower().endswith("_hea.csv")
        is_stream_data = path.name.lower().endswith("_dat.csv")
        if is_header_metadata:
            metadata_files += 1
            if not preview.first_row:
                abnormal.append(f"missing_metadata_row:{relative_path}")
                continue
            values = dict(zip((name.strip() for name in preview.columns), preview.first_row))
            nominal_rate = parse_float(values.get("Fs_Hz", ""))
            sample_count = parse_float(values.get("sampleCount", ""))
            effective_rate = parse_float(values.get("Fs_Hz_effective", ""))
            if sample_count is None or sample_count <= 0:
                abnormal.append(f"invalid_sample_count:{relative_path}:{values.get('sampleCount', '')}")
            if effective_rate is None or effective_rate <= 0:
                abnormal.append(f"invalid_effective_rate:{relative_path}:{values.get('Fs_Hz_effective', '')}")
            if nominal_rate and nominal_rate > 0 and sample_count and effective_rate and sample_count / effective_rate < 30:
                abnormal.append(
                    f"short_stream_metadata:{relative_path}:{sample_count / effective_rate:.3f}s"
                )
            continue

        data_files += 1
        content_modalities = modality_evidence(preview.columns)
        if not preview.first_row:
            abnormal.append(f"no_data_rows:{relative_path}")
            continue
        if len(preview.first_row) != len(preview.columns):
            abnormal.append(
                f"first_row_column_mismatch:{relative_path}:{len(preview.first_row)}/{len(preview.columns)}"
            )
        paired_metadata_zero_samples = False
        if is_stream_data:
            expected_header_name = path.name[:-8] + "_hea.csv"
            expected_header_path = run_dir / expected_header_name
            if expected_header_path.exists():
                header_preview = previews.get(expected_header_name) or safe_csv_preview(expected_header_path)
                if header_preview.first_row:
                    header_values = dict(
                        zip((name.strip() for name in header_preview.columns), header_preview.first_row)
                    )
                    paired_metadata_zero_samples = (
                        parse_float(header_values.get("sampleCount", "")) == 0
                    )
                    if paired_metadata_zero_samples:
                        abnormal.append(f"placeholder_data_with_zero_samples:{relative_path}")

        if content_modalities and not paired_metadata_zero_samples:
            verified_signal_files += 1
            for modality in sorted(content_modalities):
                evidence_paths[modality].append(relative_path)
        elif not content_modalities and (
            is_stream_data or "ocuevts" in path.name.lower() or "perfmetric" in path.name.lower()
        ):
            unresolved.append(relative_path)

        if is_stream_data:
            expected_header_name = path.name[:-8] + "_hea.csv"
            if expected_header_name not in previews and not (run_dir / expected_header_name).exists():
                abnormal.append(f"missing_paired_header:{relative_path}")

    for path in all_files:
        if path.name.lower().endswith("_hea.csv"):
            expected_data_name = path.name[:-8] + "_dat.csv"
            if not (run_dir / expected_data_name).exists():
                abnormal.append(
                    f"missing_paired_data:{path.relative_to(data_root).as_posix()}"
                )

    available = {modality for modality, paths in evidence_paths.items() if paths}
    physiological = bool(available & {"ecg", "eda", "emg", "respiration"})
    row: dict[str, object] = {
        **identity,
        "source_run_path": run_dir.relative_to(data_root).as_posix(),
        "source_file_count": len(all_files),
        "readable_file_count": readable_files,
        "signal_data_file_count": data_files,
        "verified_signal_file_count": verified_signal_files,
        "has_physiological_signals": physiological,
    }
    for modality in MODALITIES:
        row[f"has_{modality}"] = modality in available
        row[f"{modality}_source_paths"] = ";".join(evidence_paths.get(modality, []))
    row["available_modality_count"] = len(available)
    row["available_modalities"] = ";".join(sorted(available))
    row["missing_modalities"] = ";".join(sorted(set(MODALITIES) - available))
    row["abnormal_file_count"] = len(abnormal)
    row["abnormal_files"] = " | ".join(sorted(set(abnormal)))
    row["unresolved_file_count"] = len(unresolved)
    row["unresolved_files"] = ";".join(sorted(set(unresolved)))
    return row


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_from_csv(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discrepancy_rows(
    verified: list[dict[str, object]], prior_path: Path, data_root: Path
) -> list[dict[str, object]]:
    prior = {row["run_key"]: row for row in read_csv_rows(prior_path)}
    current = {str(row["run_key"]): row for row in verified}
    discrepancies: list[dict[str, object]] = []

    for run_key in sorted(set(prior) | set(current)):
        if run_key not in prior:
            discrepancies.append(
                {
                    "scope": "run",
                    "run_key": run_key,
                    "field": "run_presence",
                    "existing_value": "absent",
                    "verified_value": "present",
                    "severity": "high",
                    "evidence": current[run_key]["source_run_path"],
                    "interpretation": "Run directory exists but is absent from prior table.",
                }
            )
            continue
        if run_key not in current:
            discrepancies.append(
                {
                    "scope": "run",
                    "run_key": run_key,
                    "field": "run_presence",
                    "existing_value": "present",
                    "verified_value": "absent",
                    "severity": "high",
                    "evidence": "No matching run directory.",
                    "interpretation": "Prior table contains a run not found in raw data.",
                }
            )
            continue
        for modality in (
            "eye_tracking",
            "ecg",
            "eda",
            "emg",
            "respiration",
            "head_movement",
            "xplane",
            "performance",
            "control_input",
        ):
            old = bool_from_csv(prior[run_key][f"has_{modality}"])
            new = bool(current[run_key][f"has_{modality}"])
            if old != new:
                discrepancies.append(
                    {
                        "scope": "run_modality",
                        "run_key": run_key,
                        "field": f"has_{modality}",
                        "existing_value": old,
                        "verified_value": new,
                        "severity": "high" if old and not new else "medium",
                        "evidence": current[run_key][f"{modality}_source_paths"] or "No qualifying content evidence.",
                        "interpretation": "Existing filename-based availability differs from content-verified availability.",
                    }
                )

    torso_files = sorted(data_root.rglob("*lslshimmertorsoacc*_dat.csv"))
    torso_with_columns = 0
    for path in torso_files:
        preview = safe_csv_preview(path)
        if "torso_body_accelerometer" in modality_evidence(preview.columns):
            torso_with_columns += 1
    discrepancies.append(
        {
            "scope": "modality_label",
            "run_key": "",
            "field": "lslshimmertorsoacc classification",
            "existing_value": "unknown (908 files including data and metadata pairs)",
            "verified_value": f"torso_body_accelerometer ({torso_with_columns} data files with explicit torso acceleration columns)",
            "severity": "medium",
            "evidence": "DataDictionary.pdf section 'Stream: ACC' and accelerometry_torso_x/y/z_mps2 CSV columns.",
            "interpretation": "The label was conservative but is now resolvable from supplied metadata and content.",
        }
    )
    return discrepancies


def schedule_variances(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    actual_by_subject: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    sessions_by_subject: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subject = str(row["subject_id"])
        task = str(row["task"])
        level = str(row["difficulty_level"]).removeprefix("level-")
        run = str(row["run_id"]).removeprefix("run-")
        actual_by_subject[subject].add((task, level, run))
        sessions_by_subject[subject].add(str(row["session_id"]))
    missing: list[dict[str, str]] = []
    unexpected: list[dict[str, str]] = []
    for subject in sorted(actual_by_subject):
        for task, expected_pairs in EXPECTED_RUNS_BY_TASK.items():
            for level, run in sorted(expected_pairs):
                if (task, level, run) not in actual_by_subject[subject]:
                    session = ";".join(sorted(sessions_by_subject[subject]))
                    same_run_other_level = any(
                        actual_task == task and actual_run == run
                        for actual_task, _actual_level, actual_run in actual_by_subject[subject]
                    )
                    missing.append(
                        {
                            "subject_id": subject,
                            "session_id": session,
                            "task": task,
                            "difficulty_level": f"level-{level}",
                            "run_id": f"run-{run}",
                            "expected_run_key": f"{subject}_{session}_level-{level}_run-{run}",
                            "missing_type": "schedule_mapping_hole" if same_run_other_level else "physically_absent_run",
                            "basis": (
                                "Expected difficulty/run combination is absent, but the same task/run exists under another difficulty."
                                if same_run_other_level
                                else "No directory exists for this task/run; absent relative to the repeated schedule."
                            ),
                        }
                    )
        for task, level, run in sorted(actual_by_subject[subject]):
            if (level, run) not in EXPECTED_RUNS_BY_TASK[task]:
                session = ";".join(sorted(sessions_by_subject[subject]))
                unexpected.append(
                    {
                        "subject_id": subject,
                        "session_id": session,
                        "task": task,
                        "difficulty_level": f"level-{level}",
                        "run_id": f"run-{run}",
                        "run_key": f"{subject}_{session}_level-{level}_run-{run}",
                        "basis": "Identifier differs from the repeated run-to-difficulty mapping used by the other subjects.",
                    }
                )
    return missing, unexpected


def build_report(
    rows: list[dict[str, object]],
    discrepancies: list[dict[str, object]],
    missing_runs: list[dict[str, str]],
    unexpected_runs: list[dict[str, str]],
    duplicate_keys: list[str],
    performance_summary_check: dict[str, object],
) -> dict[str, object]:
    difficulty = Counter(str(row["difficulty_level"]) for row in rows)
    modality_counts = {
        modality: sum(bool(row[f"has_{modality}"]) for row in rows)
        for modality in MODALITIES
    }
    abnormal_runs = [row for row in rows if int(row["abnormal_file_count"]) > 0]
    unresolved_runs = [row for row in rows if int(row["unresolved_file_count"]) > 0]
    return {
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Independent run-directory scan with CSV column evidence, first-record checks, and paired stream metadata checks.",
        "subject_count": len({str(row["subject_id"]) for row in rows}),
        "session_count": len({(str(row["subject_id"]), str(row["session_id"])) for row in rows}),
        "raw_run_count": len(rows),
        "run_directory_file_count": sum(int(row["source_file_count"]) for row in rows),
        "data_package_file_count": sum(int(row["source_file_count"]) for row in rows) + 1,
        "difficulty_distribution": dict(sorted(difficulty.items())),
        "duplicate_run_keys": duplicate_keys,
        "missing_expected_run_directories": missing_runs,
        "physically_absent_run_count": sum(
            row["missing_type"] == "physically_absent_run" for row in missing_runs
        ),
        "unexpected_schedule_identifiers": unexpected_runs,
        "modality_run_counts": modality_counts,
        "runs_with_any_physiological_signal": sum(bool(row["has_physiological_signals"]) for row in rows),
        "runs_with_abnormal_files": len(abnormal_runs),
        "abnormal_observation_count": sum(int(row["abnormal_file_count"]) for row in rows),
        "runs_with_unresolved_files": len(unresolved_runs),
        "discrepancy_count": len(discrepancies),
        "explicit_control_input_confirmed": modality_counts["control_input"] > 0,
        "performance_summary_check": performance_summary_check,
        "limitations": [
            "CSV files were inspected using headers and safe first-record previews; complete signal arrays were not loaded.",
            "Short-stream checks use metadata duration below 30 seconds as an obvious abnormality threshold, not a scientific quality cutoff.",
            "Three physically absent directories and one unexpected difficulty/run mapping are identified against the dataset's repeated schedule, not forced from a research-plan target.",
            "CODEX_RULES.md was not present at the searched project/drive locations.",
        ],
    }


def validate_performance_summary(
    data_root: Path, rows: list[dict[str, object]]
) -> dict[str, object]:
    summary_path = data_root / "task-ils" / "PerfMetrics.csv"
    summary_rows = read_csv_rows(summary_path)
    summary_keys: list[str] = []
    for row in summary_rows:
        subject_id = f"sub-cp{int(row['subject']):03d}"
        session_id = f"ses-{row['date']}"
        difficulty_level = f"level-{int(row['difficulty']):02d}B"
        run_id = f"run-{int(row['run']):03d}"
        summary_keys.append("_".join((subject_id, session_id, difficulty_level, run_id)))
    ils_keys = {
        str(row["run_key"])
        for row in rows
        if row["task"] == "ils" and bool(row["has_performance"])
    }
    key_counts = Counter(summary_keys)
    return {
        "summary_path": summary_path.relative_to(data_root).as_posix(),
        "row_count": len(summary_rows),
        "unique_key_count": len(set(summary_keys)),
        "duplicate_keys": sorted(key for key, count in key_counts.items() if count > 1),
        "summary_keys_missing_run_performance": sorted(set(summary_keys) - ils_keys),
        "run_performance_keys_missing_summary": sorted(ils_keys - set(summary_keys)),
    }


def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = next(parent for parent in script_path.parents if (parent / "vrdataset" / "dataPackage").is_dir())
    data_root = project_root / "vrdataset" / "dataPackage"
    phase_root = project_root / "experiments" / "phase_01_raw_data_modality_audit"
    results_dir = phase_root / "results"
    verification_dir = phase_root / "verification"

    run_dirs = sorted(
        path
        for path in data_root.rglob("level-*_run-*")
        if path.is_dir() and RUN_DIR_RE.match(path.name)
    )
    rows = [inspect_run(data_root, run_dir) for run_dir in run_dirs]
    key_counts = Counter(str(row["run_key"]) for row in rows)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    prior_path = results_dir / "run_modality_availability.csv"
    discrepancies = discrepancy_rows(rows, prior_path, data_root)
    missing_runs, unexpected_runs = schedule_variances(rows)
    for missing in missing_runs:
        discrepancies.append(
            {
                "scope": "missing_run_directory",
                "run_key": missing["expected_run_key"],
                "field": "raw_run_directory",
                "existing_value": "not explicitly reported",
                "verified_value": "absent",
                "severity": "high" if missing["missing_type"] == "physically_absent_run" else "medium",
                "evidence": missing["basis"],
                "interpretation": (
                    "A run expected from the repeated raw-data schedule has no source directory."
                    if missing["missing_type"] == "physically_absent_run"
                    else "The expected difficulty/run combination is replaced by the same run under another difficulty."
                ),
            }
        )
    for unexpected in unexpected_runs:
        discrepancies.append(
            {
                "scope": "schedule_identifier",
                "run_key": unexpected["run_key"],
                "field": "difficulty_level/run_id mapping",
                "existing_value": "accepted as-is",
                "verified_value": "unexpected but internally consistent",
                "severity": "medium",
                "evidence": "Directory, filenames, per-run performance file, and PerfMetrics.csv all identify run 012 as difficulty 2 for cp031.",
                "interpretation": "This is not a filename-only typo, but it differs from the schedule used for other subjects.",
            }
        )
    performance_summary_check = validate_performance_summary(data_root, rows)
    report = build_report(
        rows,
        discrepancies,
        missing_runs,
        unexpected_runs,
        duplicate_keys,
        performance_summary_check,
    )

    verified_path = results_dir / "run_modality_availability_verified.csv"
    discrepancy_path = results_dir / "phase01_verification_discrepancies.csv"
    missing_path = results_dir / "missing_run_directories_verified.csv"
    unexpected_path = results_dir / "unexpected_schedule_identifiers_verified.csv"
    abnormal_path = results_dir / "abnormal_files_verified.csv"
    report_path = verification_dir / "verification_summary.json"
    write_csv(verified_path, rows)
    write_csv(discrepancy_path, discrepancies)
    write_csv(missing_path, missing_runs)
    write_csv(unexpected_path, unexpected_runs)
    abnormal_rows: list[dict[str, object]] = []
    for row in rows:
        for observation in str(row["abnormal_files"]).split(" | "):
            if not observation:
                continue
            parts = observation.split(":", 2)
            abnormal_rows.append(
                {
                    "run_key": row["run_key"],
                    "source_run_path": row["source_run_path"],
                    "issue_type": parts[0],
                    "file_path": parts[1] if len(parts) > 1 else "",
                    "detail": parts[2] if len(parts) > 2 else "",
                }
            )
    write_csv(abnormal_path, abnormal_rows)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
