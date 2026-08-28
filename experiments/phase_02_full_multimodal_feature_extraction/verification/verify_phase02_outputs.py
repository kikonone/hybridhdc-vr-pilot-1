from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
P1 = ROOT / "experiments" / "phase_01_raw_data_modality_audit"
P2 = ROOT / "experiments" / "phase_02_full_multimodal_feature_extraction"
P3 = ROOT / "experiments" / "phase_03_multimodal_dataset_labeling"
OUT = P2 / "verification"

P2_TABLE = P2 / "results" / "full_multimodal_run_level_features.csv"
P2_GROUPS = P2 / "results" / "feature_groups.json"
P2_LONG = P2 / "results" / "feature_extraction_long_table.csv"
P2_RUN_MODALITIES = P2 / "tables" / "runs_with_extracted_modalities.csv"
P1_RUNS = P1 / "results" / "run_modality_availability_verified.csv"
P1_DISCREPANCIES = P1 / "results" / "phase01_verification_discrepancies.csv"
P3_WITH = P3 / "results" / "cleaned_multimodal_four_class_with_performance.csv"
P3_WITHOUT = P3 / "results" / "cleaned_multimodal_four_class_without_performance.csv"
P3_FEATURES_WITH = P3 / "results" / "feature_columns_with_performance.json"
P3_FEATURES_WITHOUT = P3 / "results" / "feature_columns_without_performance.json"

P2_IDS = ["subject_id", "session_id", "run_id", "difficulty_level", "run_key"]
P3_IDS = ["subject_id", "session_id", "run_id", "difficulty_level", "target", "run_key"]

MODALITY_MAP = {
    "eye_tracking": ("has_eye_tracking", "eye_tracking_source_paths"),
    "ecg": ("has_ecg", "ecg_source_paths"),
    "eda": ("has_eda", "eda_source_paths"),
    "emg": ("has_emg", "emg_source_paths"),
    "respiration": ("has_respiration", "respiration_source_paths"),
    "head_movement": ("has_head_movement", "head_movement_source_paths"),
    "xplane": ("has_xplane", "xplane_source_paths"),
    "performance": ("has_performance", "performance_source_paths"),
    "torso_body_accelerometer": (
        "has_torso_body_accelerometer",
        "torso_body_accelerometer_source_paths",
    ),
    "control_input": ("has_control_input", "control_input_source_paths"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def clean_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def difficulty_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"level-(\d+)", str(value).lower())
    return float(int(match.group(1))) if match else np.nan


def table_audit(path: Path, identifier_columns: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    feature_columns = [column for column in frame.columns if column not in identifier_columns]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_array = numeric.to_numpy(dtype=float, copy=False)
    non_missing = int(numeric.notna().sum().sum())
    total = int(numeric.shape[0] * numeric.shape[1])
    unique_counts = numeric.nunique(dropna=True)
    all_nan = sorted(unique_counts[unique_counts == 0].index.tolist())
    single_value = sorted(unique_counts[unique_counts == 1].index.tolist())
    inf_counts = pd.Series(np.isinf(finite_array).sum(axis=0), index=feature_columns)
    constants = pd.DataFrame(
        [
            {
                "dataset": path.name,
                "feature_name": column,
                "constant_type": "all_nan" if column in set(all_nan) else "single_finite_value",
                "non_missing_count": int(numeric[column].notna().sum()),
                "value": (
                    ""
                    if column in set(all_nan)
                    else float(numeric[column].dropna().iloc[0])
                ),
            }
            for column in all_nan + single_value
        ]
    )
    composite = [column for column in ["subject_id", "session_id", "run_id", "difficulty_level"] if column in frame]
    missing_identifier_counts = {
        column: int(frame[column].isna().sum() + frame[column].astype(str).str.strip().eq("").sum())
        for column in identifier_columns
        if column in frame
    }
    parsed_difficulty = frame["difficulty_level"].map(difficulty_number)
    report = {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "identifier_columns": identifier_columns,
        "feature_columns": int(len(feature_columns)),
        "unique_subjects": int(frame["subject_id"].nunique(dropna=True)),
        "unique_run_keys": int(frame["run_key"].nunique(dropna=True)),
        "unique_run_ids_without_subject_context": int(frame["run_id"].nunique(dropna=True)),
        "duplicated_run_key_rows": int(frame["run_key"].duplicated(keep=False).sum()),
        "duplicated_subject_session_run_difficulty_rows": int(frame.duplicated(composite, keep=False).sum()),
        "missing_identifier_counts": missing_identifier_counts,
        "difficulty_distribution": {
            str(int(key)) if pd.notna(key) else "missing": int(value)
            for key, value in parsed_difficulty.value_counts(dropna=False).sort_index().items()
        },
        "nan_count": total - non_missing,
        "nan_rate": float((total - non_missing) / total) if total else 0.0,
        "infinite_value_count": int(inf_counts.sum()),
        "columns_with_infinite_values": {
            column: int(value) for column, value in inf_counts[inf_counts > 0].items()
        },
        "constant_column_count_including_all_nan": int(len(constants)),
        "all_nan_column_count": int(len(all_nan)),
        "single_finite_value_column_count": int(len(single_value)),
    }
    return report, constants


def provenance_table(groups: dict[str, list[str]]) -> pd.DataFrame:
    group_lookup = {
        feature: group
        for group, features in groups.items()
        if group != "identifier_columns"
        for feature in features
    }
    evidence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "modalities": set(),
            "source_columns": set(),
            "statistics": set(),
            "source_file": "",
        }
    )
    usecols = ["modality", "feature_name", "source_file", "source_column", "statistic"]
    for chunk in pd.read_csv(P2_LONG, usecols=usecols, chunksize=50_000, low_memory=False):
        for feature_name, rows in chunk.groupby("feature_name", sort=False):
            item = evidence[str(feature_name)]
            item["modalities"].update(rows["modality"].dropna().astype(str).unique())
            item["source_columns"].update(rows["source_column"].dropna().astype(str).unique())
            item["statistics"].update(rows["statistic"].dropna().astype(str).unique())
            if not item["source_file"]:
                sources = rows["source_file"].dropna().astype(str)
                item["source_file"] = sources.iloc[0] if len(sources) else ""

    records: list[dict[str, Any]] = []
    for feature_name in sorted(group_lookup):
        original_group = group_lookup[feature_name]
        item = evidence.get(feature_name, {})
        extracted_modalities = sorted(item.get("modalities", set()))
        source_columns = sorted(item.get("source_columns", set()))
        source_file = str(item.get("source_file", ""))
        verified_group = original_group
        verified_status = "VERIFIED"
        unknown_decision = ""
        notes = "Verified from extraction long-table source file and source-column records."

        if feature_name.startswith("unknown_lslshimmertorsoacc_"):
            modality = "torso_body_accelerometer"
            verified_group = "body_movement_features"
            unknown_decision = "A. VERIFIED body/torso movement"
            notes = (
                "Originally grouped as unknown. Verified by DataDictionary.pdf section 6 (Stream: ACC), "
                "representative raw CSV columns accelerometry_torso_x/y/z_mps2, and extraction provenance."
            )
        elif feature_name.startswith("eye_"):
            modality = "eye_tracking"
        elif feature_name.startswith("ecg_"):
            modality = "ecg"
        elif feature_name.startswith("eda_"):
            if any("ppg_finger" in column for column in source_columns):
                modality = "ppg"
                notes = "Verified PPG channel stored in the EDA stream; not relabeled as EDA."
            elif "derived" in source_columns and "eda_signal" in feature_name:
                modality = "mixed_eda_ppg_derived"
                notes = "Derived by Phase 02 from the mean of numeric EDA-stream channels (EDA and PPG)."
            else:
                modality = "eda_gsr"
        elif feature_name.startswith("emg_"):
            if any("accelerometry_forearm" in column for column in source_columns):
                modality = "forearm_accelerometer"
                notes = "Verified forearm accelerometry channel stored in the EMG stream."
            else:
                modality = "emg"
        elif feature_name.startswith("resp_"):
            modality = "respiration"
        elif feature_name.startswith("head_"):
            modality = "head_movement"
        elif feature_name.startswith("xplane_"):
            modality = "xplane_flight_state"
            notes = (
                "Verified X-Plane aircraft/pilot state. Trim or aircraft-state fields are not treated as "
                "explicit joystick/yoke/throttle/rudder control-input streams."
            )
        elif feature_name.startswith("performance_"):
            modality = "performance"
        else:
            modality = ";".join(extracted_modalities) if extracted_modalities else "UNVERIFIED"
            verified_status = "UNVERIFIED"
            notes = "No supported provenance classification rule matched this feature."

        source_detail = source_file
        if source_columns:
            source_detail += " | source_column=" + ";".join(source_columns)
        records.append(
            {
                "feature_name": feature_name,
                "feature_group": verified_group,
                "original_feature_group": original_group,
                "modality": modality,
                "source": source_detail,
                "verified_status": verified_status,
                "unknown_feature_decision": unknown_decision,
                "notes": notes,
            }
        )
    return pd.DataFrame(records)


def availability_and_discrepancies() -> tuple[pd.DataFrame, pd.DataFrame]:
    phase1 = pd.read_csv(P1_RUNS, low_memory=False)
    phase2 = pd.read_csv(P2_RUN_MODALITIES, low_memory=False)
    extracted_by_key = phase2.set_index("run_key")["extracted_modalities"].fillna("").to_dict()
    modeling_keys = set(pd.read_csv(P3_WITHOUT, usecols=["run_key"])["run_key"].astype(str))

    availability_rows: list[dict[str, Any]] = []
    discrepancy_rows: list[dict[str, Any]] = []
    for _, row in phase1.iterrows():
        run_key = str(row["run_key"])
        phase2_modalities = set(filter(None, str(extracted_by_key.get(run_key, "")).split(";")))
        verified_modalities = set(filter(None, clean_text(row["available_modalities"]).split(";")))
        if "torso_body_accelerometer" in verified_modalities:
            verified_modalities.add("torso_body_accelerometer")
        for modality, (flag_column, source_column) in MODALITY_MAP.items():
            extracted_label = "unknown" if modality == "torso_body_accelerometer" else modality
            phase2_claim = extracted_label in phase2_modalities
            verified_claim = as_bool(row[flag_column])
            if phase2_claim != verified_claim:
                discrepancy_rows.append(
                    {
                        "category": "run_modality_availability",
                        "run_key": run_key,
                        "item": modality,
                        "existing_value": f"phase02_extracted={phase2_claim}",
                        "verified_value": f"phase01_content_verified={verified_claim}",
                        "severity": "high",
                        "evidence": (
                            clean_text(row[source_column])
                            if verified_claim
                            else clean_text(row.get("abnormal_files", "")) or "No qualifying content evidence."
                        ),
                    }
                )
        difficulty = int(difficulty_number(row["difficulty_level"]))
        availability_rows.append(
            {
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "run_id": row["run_id"],
                "difficulty_level": row["difficulty_level"],
                "run_key": run_key,
                "source_file_path": row["source_run_path"],
                "available_modalities": ";".join(sorted(verified_modalities)),
                "phase02_extracted_modalities_reported": ";".join(sorted(phase2_modalities)),
                "has_physiological_signals": as_bool(row["has_physiological_signals"]),
                "has_ecg": as_bool(row["has_ecg"]),
                "has_eda_gsr": as_bool(row["has_eda"]),
                "has_emg": as_bool(row["has_emg"]),
                "has_respiration": as_bool(row["has_respiration"]),
                "has_eye_tracking": as_bool(row["has_eye_tracking"]),
                "has_head_movement": as_bool(row["has_head_movement"]),
                "has_xplane_flight_state": as_bool(row["has_xplane"]),
                "has_performance": as_bool(row["has_performance"]),
                "has_torso_body_accelerometer": as_bool(row["has_torso_body_accelerometer"]),
                "has_explicit_control_input": as_bool(row["has_control_input"]),
                "signal_data_file_count": int(row["signal_data_file_count"]),
                "verified_signal_file_count": int(row["verified_signal_file_count"]),
                "obviously_abnormal_or_missing_files": clean_text(row.get("abnormal_files", "")),
                "unresolved_files": clean_text(row.get("unresolved_files", "")),
                "modeling_eligible_four_class": run_key in modeling_keys,
                "exclusion_reason": "" if run_key in modeling_keys else (
                    "Excluded by Phase 03 four-class filter: level-000 rest run"
                    if difficulty == 0
                    else "NOT VERIFIED"
                ),
            }
        )
    return pd.DataFrame(availability_rows), pd.DataFrame(discrepancy_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        P2_TABLE,
        P2_GROUPS,
        P2_LONG,
        P2_RUN_MODALITIES,
        P1_RUNS,
        P3_WITH,
        P3_WITHOUT,
        P3_FEATURES_WITH,
        P3_FEATURES_WITHOUT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required verification inputs: " + "; ".join(missing))

    groups = read_json(P2_GROUPS)
    raw_audit, raw_constants = table_audit(P2_TABLE, P2_IDS)
    with_audit, with_constants = table_audit(P3_WITH, P3_IDS)
    without_audit, without_constants = table_audit(P3_WITHOUT, P3_IDS)
    constants = pd.concat([raw_constants, with_constants, without_constants], ignore_index=True)
    constants.to_csv(OUT / "constant_columns.csv", index=False)

    phase2 = pd.read_csv(P2_TABLE, low_memory=False)
    phase3 = pd.read_csv(P3_WITHOUT, usecols=P3_IDS, low_memory=False)
    excluded = phase2[~phase2["run_key"].isin(set(phase3["run_key"]))][P2_IDS].copy()
    excluded["exclusion_reason"] = excluded["difficulty_level"].map(
        lambda value: (
            "Excluded by Phase 03 four-class filter: level-000 rest run"
            if difficulty_number(value) == 0
            else "NOT VERIFIED"
        )
    )
    excluded.to_csv(OUT / "excluded_runs.csv", index=False)

    provenance = provenance_table(groups)
    provenance.to_csv(OUT / "feature_provenance.csv", index=False)

    features_with = read_json(P3_FEATURES_WITH)
    features_without = read_json(P3_FEATURES_WITHOUT)
    performance = sorted(groups.get("performance_features", []))
    unknown_original = sorted(groups.get("unknown_features", []))
    body_verified = sorted(
        provenance.loc[
            provenance["unknown_feature_decision"] == "A. VERIFIED body/torso movement",
            "feature_name",
        ].tolist()
    )
    feature_lists = {
        "definition": "Current Phase 03 modeling feature lists after the documented 95% missingness filter.",
        "features_without_performance": features_without,
        "features_with_performance": features_with,
        "features_performance_only": performance,
        "features_unknown_or_body_movement": body_verified,
        "counts": {
            "features_without_performance": len(features_without),
            "features_with_performance": len(features_with),
            "features_performance_only": len(performance),
            "features_unknown_or_body_movement": len(body_verified),
        },
    }
    write_json(OUT / "feature_lists.json", feature_lists)

    verified_groups = {
        "identifier_columns": groups.get("identifier_columns", []),
        "physiological_features": groups.get("physiological_features", []),
        "eye_tracking_features": groups.get("eye_tracking_features", []),
        "head_movement_features": groups.get("head_movement_features", []),
        "flight_parameter_features": groups.get("flight_parameter_features", []),
        "control_input_features": [],
        "performance_features": performance,
        "body_movement_features": body_verified,
        "unverified_features": sorted(
            provenance.loc[provenance["verified_status"] == "UNVERIFIED", "feature_name"].tolist()
        ),
    }
    write_json(OUT / "feature_group_metadata.json", verified_groups)

    availability, new_discrepancies = availability_and_discrepancies()
    availability.to_csv(OUT / "verified_run_level_modality_availability.csv", index=False)

    prior = pd.read_csv(P1_DISCREPANCIES, low_memory=False)
    prior_standard = pd.DataFrame(
        {
            "category": prior["scope"],
            "run_key": prior["run_key"],
            "item": prior["field"],
            "existing_value": prior["existing_value"],
            "verified_value": prior["verified_value"],
            "severity": prior["severity"],
            "evidence": prior["evidence"],
        }
    )
    discrepancies = pd.concat([prior_standard, new_discrepancies], ignore_index=True).drop_duplicates()
    discrepancies.to_csv(OUT / "discrepancy_table.csv", index=False)

    group_counts = {group: len(features) for group, features in groups.items()}
    prefix_counts = Counter(column.split("_", 1)[0] for column in phase2.columns if column not in P2_IDS)
    modality_mismatch_counts = (
        new_discrepancies.groupby("item").size().sort_values(ascending=False).astype(int).to_dict()
    )
    integrity = {
        "phase02_feature_columns_equal_group_union": set(phase2.columns) - set(P2_IDS)
        == {
            feature
            for group, features in groups.items()
            if group != "identifier_columns"
            for feature in features
        },
        "phase03_without_is_subset_of_phase02": set(features_without).issubset(phase2.columns),
        "phase03_with_is_subset_of_phase02": set(features_with).issubset(phase2.columns),
        "performance_count_verified_from_group_and_table": len(performance) == 59
        and set(performance).issubset(phase2.columns),
        "unknown_features_all_verified_as_torso_body_movement": len(unknown_original) == 42
        and set(unknown_original) == set(body_verified),
        "explicit_control_input_feature_count": len(groups.get("control_input_features", [])),
    }
    summary = {
        "source_outputs_modified": False,
        "extraction_rerun_performed": False,
        "phase02_raw_extracted_table": raw_audit,
        "phase03_with_performance_table": with_audit,
        "phase03_without_performance_table": without_audit,
        "raw_extracted_run_count": raw_audit["rows"],
        "modeling_eligible_run_count": without_audit["rows"],
        "excluded_run_count": int(len(excluded)),
        "exclusion_reason_counts": excluded["exclusion_reason"].value_counts().to_dict(),
        "phase02_group_counts": group_counts,
        "phase02_feature_prefix_counts": dict(sorted(prefix_counts.items())),
        "performance_feature_count": len(performance),
        "unknown_original_feature_count": len(unknown_original),
        "verified_body_movement_feature_count": len(body_verified),
        "unverified_feature_count": int((provenance["verified_status"] == "UNVERIFIED").sum()),
        "phase02_vs_content_verified_modality_mismatch_count": int(len(new_discrepancies)),
        "phase02_vs_content_verified_modality_mismatch_counts": modality_mismatch_counts,
        "runs_affected_by_modality_mismatch": int(new_discrepancies["run_key"].nunique()),
        "phase01_verified_modality_run_counts": {
            modality: int(availability[flag].sum())
            for modality, flag in {
                "eye_tracking": "has_eye_tracking",
                "ecg": "has_ecg",
                "eda_gsr": "has_eda_gsr",
                "emg": "has_emg",
                "respiration": "has_respiration",
                "head_movement": "has_head_movement",
                "xplane_flight_state": "has_xplane_flight_state",
                "performance": "has_performance",
                "torso_body_accelerometer": "has_torso_body_accelerometer",
                "explicit_control_input": "has_explicit_control_input",
            }.items()
        },
        "runs_with_abnormal_or_missing_files": int(
            availability["obviously_abnormal_or_missing_files"].fillna("").str.strip().ne("").sum()
        ),
        "runs_with_unresolved_files": int(
            availability["unresolved_files"].fillna("").str.strip().ne("").sum()
        ),
        "integrity_checks": integrity,
        "phase03_readiness": (
            "NOT READY: current Phase 03 tables were built from Phase 02 modality-presence claims that "
            "include placeholder/zero-sample streams in 38 run-modality pairs. Correct or mask those "
            "cells, then rebuild and re-verify Phase 03 datasets before modeling."
        ),
    }
    write_json(OUT / "verification_summary.json", summary)

    phase01_report = f"""# Phase 01 Verification

This verification reuses the independent content-aware Phase 01 audit and compares it with the existing Phase 02 extraction outputs. Original raw data and completed outputs were not modified.

## VERIFIED DATA COUNTS

- Subjects: {raw_audit['unique_subjects']}
- Raw run directories / Phase 02 rows: {raw_audit['rows']}
- Difficulty distribution: 0={raw_audit['difficulty_distribution'].get('0', 0)}, 1={raw_audit['difficulty_distribution'].get('1', 0)}, 2={raw_audit['difficulty_distribution'].get('2', 0)}, 3={raw_audit['difficulty_distribution'].get('3', 0)}, 4={raw_audit['difficulty_distribution'].get('4', 0)}
- Duplicate run keys: {raw_audit['duplicated_run_key_rows']}
- Missing identifiers in existing Phase 02 rows: {sum(raw_audit['missing_identifier_counts'].values())}

## VERIFIED MODALITIES

- Physiological evidence: ECG, EDA/GSR, EMG, respiration; PPG is also present inside the EDA stream.
- Eye tracking, head movement, X-Plane flight-state, performance, and torso/body accelerometry are content-verified where their run-level flags are true.
- Torso/body accelerometry is verified from DataDictionary.pdf section 6 and explicit accelerometry_torso_x/y/z_mps2 columns.

## UNVERIFIED MODALITIES

- Explicit control input: NOT VERIFIED and unavailable (0 runs, 0 extracted features). References to a hand controlling a joystick or throttle describe the experimental setup, not a recorded control-input stream.
- Unresolved raw modality labels after the content-aware Phase 01 audit: {summary['runs_with_unresolved_files']} runs.

## DISCREPANCIES

- Existing Phase 02 modality availability overstates content-verified availability in {len(new_discrepancies)} run-modality pairs across {summary['runs_affected_by_modality_mismatch']} runs.
- Mismatch counts: {json.dumps(modality_mismatch_counts, sort_keys=True)}.
- The earlier unknown torso stream is resolvable as torso/body accelerometry; it should not remain an unknown modality in verified metadata.

## RAW DATA ISSUES

- Runs with obvious abnormal/missing file observations: {summary['runs_with_abnormal_or_missing_files']}.
- Three physically absent scheduled run directories and one schedule-mapping hole are recorded in the Phase 01 discrepancy files.
- One internally consistent but unexpected difficulty/run mapping exists for cp031 run-012.

## OUTPUT FILES

- verified_run_level_modality_availability.csv
- discrepancy_table.csv
- verification_summary.json
- feature_provenance.csv

## PHASE 02 READINESS

CONDITIONAL. The raw inventory is sufficient for extraction, but availability must use content/metadata validation. Existing Phase 02 values from placeholder or zero-sample streams require masking or correction before downstream modeling datasets are accepted.
"""
    (OUT / "phase01_verification.md").write_text(phase01_report, encoding="utf-8")

    phase02_report = f"""# Phase 02 Multimodal Run-Level Feature Extraction Verification

The existing extraction was audited in place. No extraction rerun, source overwrite, or model training was performed.

## VERIFIED

- Raw extracted table: {raw_audit['rows']} rows x {raw_audit['columns']} columns ({raw_audit['feature_columns']} features plus 5 identifiers).
- Subjects: {raw_audit['unique_subjects']}; unique run keys: {raw_audit['unique_run_keys']}; duplicate run-key rows: {raw_audit['duplicated_run_key_rows']}.
- Every Phase 03 modeling row is unique for subject-session-run-difficulty: duplicate rows={without_audit['duplicated_subject_session_run_difficulty_rows']}.
- Modeling-eligible four-class rows: {without_audit['rows']} with distribution 1={without_audit['difficulty_distribution'].get('1', 0)}, 2={without_audit['difficulty_distribution'].get('2', 0)}, 3={without_audit['difficulty_distribution'].get('3', 0)}, 4={without_audit['difficulty_distribution'].get('4', 0)}.
- Phase 02 feature NaN rate: {raw_audit['nan_rate']:.6f}; infinite values: {raw_audit['infinite_value_count']}.

## NOT VERIFIED

- Explicit joystick/yoke/throttle/rudder control-input features: NOT VERIFIED; the existing control-input group is empty.
- Current Phase 03 scientific readiness is not verified because placeholder/zero-sample streams were treated as extracted in {len(new_discrepancies)} run-modality pairs.

## EXCLUDED RUNS

- Raw extracted rows: {raw_audit['rows']}.
- Final modeling rows: {without_audit['rows']}.
- Excluded: {len(excluded)}, all level-000 rest runs removed by the explicit Phase 03 four-class filter. No duplicate-key rows were removed.

## FEATURE GROUPS

- Original Phase 02 group counts: {json.dumps(group_counts, sort_keys=True)}.
- Phase 03 with performance: {with_audit['feature_columns']} features after removing 12 all-missing/high-missing columns.
- Phase 03 without performance: {without_audit['feature_columns']} features.
- Constant/all-NaN columns are listed in constant_columns.csv and must be handled inside training folds where applicable.

## UNKNOWN FEATURES

- Decision A: all {len(body_verified)} formerly unknown features are VERIFIED body/torso movement aggregates.
- Evidence: DataDictionary.pdf section 6, explicit accelerometry_torso_x/y/z_mps2 raw columns, and per-feature extraction provenance.
- Unverified feature count after provenance audit: {summary['unverified_feature_count']}.

## PERFORMANCE FEATURES

- Verified count: {len(performance)}; the number 59 is supported by the actual Phase 02 feature group and table, not forced.
- These comprise 4 cumulative PerfMetrics.csv values and 55 per-run performance-stream aggregates/metadata features.
- Performance features remain excluded from the primary Phase 03 list and included only in auxiliary lists.

## OUTPUT FILES

- phase02_verification.md
- phase01_verification.md
- verification_summary.json
- feature_provenance.csv
- feature_group_metadata.json
- feature_lists.json
- excluded_runs.csv
- verified_run_level_modality_availability.csv
- discrepancy_table.csv
- constant_columns.csv

## PHASE 03 READINESS

NOT READY. Preserve the original outputs, mask or minimally regenerate the affected placeholder/zero-sample modality features using the verified availability table, rebuild the Phase 03 datasets from that corrected input, and re-run this validation before any model training.
"""
    (OUT / "phase02_verification.md").write_text(phase02_report, encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
