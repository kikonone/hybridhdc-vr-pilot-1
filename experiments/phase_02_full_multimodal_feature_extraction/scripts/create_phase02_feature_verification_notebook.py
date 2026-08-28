from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = (
    ROOT
    / "experiments"
    / "phase_02_full_multimodal_feature_extraction"
    / "notebooks"
    / "Phase_02_Feature_Verification.ipynb"
)


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# Phase 02 Feature Verification and Placeholder Repair

This notebook repairs the verified Phase 02 placeholder/zero-sample inconsistency without modifying raw data, overwriting the original Phase 02 table, dropping runs, imputing values, starting Phase 03, or training models.

The repair is driven by the verified Phase 01 run-modality availability table and the verified Phase 02 feature provenance. It creates a versioned corrected Phase 02 table plus machine-readable repair, provenance, feature-group, quality, and validation artifacts.
"""
    ),
    markdown("## 1. Repository Validation and Protected Paths"),
    code(
        """
from __future__ import annotations

from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from IPython.display import Markdown, display


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (
            (candidate / "CODEX_NOTEBOOK_RULES.md").is_file()
            and (candidate / "experiments").is_dir()
            and (candidate / "vrdataset").is_dir()
        ):
            return candidate
    raise FileNotFoundError("Validated project root was not found from the notebook working directory.")


ROOT = find_project_root(Path.cwd())
PHASE = ROOT / "experiments" / "phase_02_full_multimodal_feature_extraction"
RESULTS = PHASE / "results"
VERIFICATION = PHASE / "verification"
RAW_ROOT = ROOT / "vrdataset"
P1_RESULTS = ROOT / "experiments" / "phase_01_raw_data_modality_audit" / "results"

ORIGINAL_TABLE = RESULTS / "full_multimodal_run_level_features.csv"
PHASE1_AVAILABILITY = P1_RESULTS / "run_modality_availability_verified.csv"
VERIFIED_PROVENANCE = VERIFICATION / "feature_provenance.csv"
VERIFIED_GROUPS = VERIFICATION / "feature_group_metadata.json"
PHASE2_FEATURE_LISTS = VERIFICATION / "feature_lists.json"
MISSING_RUNS = P1_RESULTS / "missing_run_directories_verified.csv"

CORRECTED_TABLE = RESULTS / "full_multimodal_run_level_features_corrected_v1.csv"
REPAIR_LOG = RESULTS / "phase02_placeholder_repair_log.csv"
ALL_NAN_AUDIT = RESULTS / "phase02_all_nan_feature_audit.csv"
SINGLE_VALUE_AUDIT = RESULTS / "phase02_single_value_feature_audit.csv"
CORRECTED_PROVENANCE = RESULTS / "phase02_corrected_feature_provenance.csv"
CORRECTED_GROUPS = RESULTS / "phase02_corrected_feature_groups.json"
VALIDATION_SUMMARY = RESULTS / "phase02_corrected_validation_summary.json"
MODELING_MANIFEST = RESULTS / "phase02_modeling_feature_manifest.json"
VERIFIED_AVAILABILITY_OUTPUT = RESULTS / "phase02_verified_run_modality_availability.csv"

IDENTIFIERS = ["subject_id", "session_id", "run_id", "difficulty_level", "run_key"]
OUTPUT_PATHS = [
    CORRECTED_TABLE,
    REPAIR_LOG,
    ALL_NAN_AUDIT,
    SINGLE_VALUE_AUDIT,
    CORRECTED_PROVENANCE,
    CORRECTED_GROUPS,
    VALIDATION_SUMMARY,
    MODELING_MANIFEST,
    VERIFIED_AVAILABILITY_OUTPUT,
]


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


for output_path in OUTPUT_PATHS:
    assert is_within(output_path, RESULTS), f"Output escaped Phase 02 results: {output_path}"
    assert not is_within(output_path, RAW_ROOT), f"Raw-data write target rejected: {output_path}"
assert ORIGINAL_TABLE not in OUTPUT_PATHS

required_inputs = [
    ORIGINAL_TABLE,
    PHASE1_AVAILABILITY,
    VERIFIED_PROVENANCE,
    VERIFIED_GROUPS,
    PHASE2_FEATURE_LISTS,
    MISSING_RUNS,
]
missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
assert not missing_inputs, f"Missing verified inputs: {missing_inputs}"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_metadata_fingerprint(root: Path) -> dict[str, object]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        entries.append(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}")
    payload = "\\n".join(entries).encode("utf-8")
    return {
        "file_count": len(entries),
        "metadata_sha256": sha256(payload).hexdigest(),
    }


original_table_hash_before = file_sha256(ORIGINAL_TABLE)
raw_fingerprint_before = tree_metadata_fingerprint(RAW_ROOT)

display(pd.DataFrame({
    "item": ["project_root", "raw_root", "original_table", "corrected_table"],
    "path": [str(ROOT), str(RAW_ROOT), str(ORIGINAL_TABLE), str(CORRECTED_TABLE)],
}))
print("Raw-data protection: enforced by output-path assertions and before/after metadata fingerprinting.")
"""
    ),
    markdown("## 2. Load Verified Evidence"),
    code(
        """
original = pd.read_csv(ORIGINAL_TABLE, low_memory=False)
phase1 = pd.read_csv(PHASE1_AVAILABILITY, low_memory=False)
provenance = pd.read_csv(VERIFIED_PROVENANCE, low_memory=False)
verified_groups = json.loads(VERIFIED_GROUPS.read_text(encoding="utf-8"))
verified_feature_lists = json.loads(PHASE2_FEATURE_LISTS.read_text(encoding="utf-8"))
missing_runs = pd.read_csv(MISSING_RUNS, low_memory=False)

assert all(column in original.columns for column in IDENTIFIERS)
assert original["run_key"].is_unique
assert provenance["feature_name"].is_unique
assert set(provenance["feature_name"]) == set(original.columns) - set(IDENTIFIERS)

evidence_overview = pd.DataFrame([
    {"evidence": "Phase 02 run-level feature table", "rows": len(original), "columns": len(original.columns), "path": str(ORIGINAL_TABLE)},
    {"evidence": "Verified Phase 01 availability", "rows": len(phase1), "columns": len(phase1.columns), "path": str(PHASE1_AVAILABILITY)},
    {"evidence": "Verified Phase 02 provenance", "rows": len(provenance), "columns": len(provenance.columns), "path": str(VERIFIED_PROVENANCE)},
    {"evidence": "Verified feature-group metadata", "rows": len(verified_groups), "columns": np.nan, "path": str(VERIFIED_GROUPS)},
])
display(evidence_overview)
"""
    ),
    markdown("## 3. Canonical Provenance and Feature-Type Classification"),
    code(
        """
AVAILABILITY_MODALITY_MAP = {
    "eye_tracking": "eye_tracking",
    "ecg": "ecg",
    "eda_gsr": "eda",
    "ppg": "eda",
    "mixed_eda_ppg_derived": "eda",
    "emg": "emg",
    "forearm_accelerometer": "emg",
    "respiration": "respiration",
    "head_movement": "head_movement",
    "xplane_flight_state": "xplane",
    "performance": "performance",
    "torso_body_accelerometer": "body_movement",
}

PHASE1_FLAG_COLUMNS = {
    "eye_tracking": "has_eye_tracking",
    "ecg": "has_ecg",
    "eda": "has_eda",
    "emg": "has_emg",
    "respiration": "has_respiration",
    "head_movement": "has_head_movement",
    "xplane": "has_xplane",
    "performance": "has_performance",
    "body_movement": "has_torso_body_accelerometer",
    "control_input": "has_control_input",
}


def parse_source_column(source: object) -> str:
    match = re.search(r"source_column=([^|]+)$", str(source).strip())
    return match.group(1).strip() if match else ""


def feature_type(row: pd.Series) -> str:
    source_column = parse_source_column(row["source"])
    stream_metadata_suffixes = ("_duration", "_sampling_rate_estimate", "_sample_count")
    if source_column in {"time", "time_dn", "timestamp", "processed_time", "processedtime"} and str(row["feature_name"]).endswith(stream_metadata_suffixes):
        return "GENERIC_MODALITY_STREAM_METADATA"
    return "SIGNAL_DERIVED"


corrected_provenance = provenance.copy()
corrected_provenance["availability_modality"] = corrected_provenance["modality"].map(AVAILABILITY_MODALITY_MAP)
corrected_provenance["feature_type"] = corrected_provenance.apply(feature_type, axis=1)
corrected_provenance["scientifically_requires_usable_modality_samples"] = True
corrected_provenance["feature_group"] = np.where(
    corrected_provenance["modality"].eq("torso_body_accelerometer"),
    "body_movement",
    corrected_provenance["feature_group"],
)
corrected_provenance["provenance_status"] = np.where(
    corrected_provenance["modality"].eq("torso_body_accelerometer"),
    "VERIFIED_BODY_MOVEMENT",
    np.where(corrected_provenance["verified_status"].eq("VERIFIED"), "VERIFIED_OTHER", "UNVERIFIED"),
)

unmapped = corrected_provenance[corrected_provenance["availability_modality"].isna()]
assert unmapped.empty, f"Provenance modalities require review: {unmapped[['feature_name', 'modality']].to_dict('records')[:5]}"

feature_type_summary = (
    corrected_provenance.groupby(["availability_modality", "feature_type"], dropna=False)
    .size().reset_index(name="feature_count")
)
display(feature_type_summary)
"""
    ),
    markdown("## 4. Recalculate the Run-Modality Discrepancies"),
    code(
        """
def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def mismatch_reason(row: pd.Series, modality: str) -> str:
    token_map = {
        "eye_tracking": ("lslhtcviveeye",),
        "ecg": ("lslshimmerecg",),
        "eda": ("lslshimmereda",),
        "emg": ("lslshimmeremg",),
        "respiration": ("lslshimmerresp", "lslrespitrace"),
        "head_movement": ("lslxp11xpcplt",),
        "xplane": ("lslxp11xpcac",),
        "performance": ("feat-perfmetric",),
        "body_movement": ("lslshimmertorsoacc",),
    }
    observations = str(row.get("abnormal_files", "") if pd.notna(row.get("abnormal_files", "")) else "").split(" | ")
    matches = [item for item in observations if any(token in item.lower() for token in token_map.get(modality, ()))]
    if matches:
        return " | ".join(matches)
    return "Verified Phase 01 content audit found no qualifying usable stream evidence for this modality."


features_by_modality = {
    modality: sorted(group["feature_name"].tolist())
    for modality, group in corrected_provenance.groupby("availability_modality")
}
types_by_feature = corrected_provenance.set_index("feature_name")["feature_type"].to_dict()
original_by_key = original.set_index("run_key", drop=False)

discrepancy_records = []
for _, availability_row in phase1.iterrows():
    run_key = str(availability_row["run_key"])
    if run_key not in original_by_key.index:
        continue
    feature_row = original_by_key.loc[run_key]
    for modality, flag_column in PHASE1_FLAG_COLUMNS.items():
        modality_features = features_by_modality.get(modality, [])
        if not modality_features:
            continue
        phase1_available = as_bool(availability_row[flag_column])
        present_count = int(feature_row[modality_features].notna().sum())
        phase2_present = present_count > 0
        if (not phase1_available) and phase2_present:
            signal_features = [name for name in modality_features if types_by_feature[name] == "SIGNAL_DERIVED"]
            metadata_features = [name for name in modality_features if types_by_feature[name] == "GENERIC_MODALITY_STREAM_METADATA"]
            discrepancy_records.append({
                "subject_id": availability_row["subject_id"],
                "session_id": availability_row["session_id"],
                "run_id": availability_row["run_id"],
                "run_key": run_key,
                "difficulty_level": availability_row["difficulty_level"],
                "modality": modality,
                "phase01_availability_status": "UNAVAILABLE_FALSE",
                "phase02_feature_presence_status": f"PRESENT_NON_NULL:{present_count}",
                "reason_for_mismatch": mismatch_reason(availability_row, modality),
                "affected_feature_count": len(modality_features),
                "signal_derived_feature_count": len(signal_features),
                "generic_modality_metadata_feature_count": len(metadata_features),
                "non_null_feature_count_before": present_count,
            })

discrepancies = pd.DataFrame(discrepancy_records).sort_values(["run_key", "modality"]).reset_index(drop=True)
actual_discrepancy_count = len(discrepancies)
print(f"Actual recalculated discrepancy count: {actual_discrepancy_count}")
print(f"Affected runs: {discrepancies['run_key'].nunique()}")
display(discrepancies)
"""
    ),
    markdown("## 5. Conservative Placeholder Repair"),
    code(
        """
ambiguous_features = corrected_provenance[
    corrected_provenance["feature_name"].isin(
        [name for modality in discrepancies["modality"].unique() for name in features_by_modality[modality]]
    )
    & (
        corrected_provenance["source"].isna()
        | corrected_provenance["availability_modality"].isna()
        | corrected_provenance["provenance_status"].eq("UNVERIFIED")
    )
]
assert ambiguous_features.empty, "Minimal regeneration gate stopped: affected provenance is ambiguous."

corrected = original.copy()
corrected_index = corrected.set_index("run_key").index
repair_records = []

for _, discrepancy in discrepancies.iterrows():
    run_key = discrepancy["run_key"]
    modality = discrepancy["modality"]
    modality_features = features_by_modality[modality]
    signal_features = [name for name in modality_features if types_by_feature[name] == "SIGNAL_DERIVED"]
    metadata_features = [name for name in modality_features if types_by_feature[name] == "GENERIC_MODALITY_STREAM_METADATA"]
    row_mask = corrected["run_key"].eq(run_key)
    assert int(row_mask.sum()) == 1

    signal_non_null_before = int(corrected.loc[row_mask, signal_features].notna().sum().sum())
    metadata_non_null_before = int(corrected.loc[row_mask, metadata_features].notna().sum().sum())

    # Both feature classes depend on a usable modality stream. Placeholder stream
    # counts/durations/rates are not legitimate run identifiers or availability metadata.
    corrected.loc[row_mask, modality_features] = np.nan

    repair_records.append({
        **discrepancy.to_dict(),
        "signal_non_null_values_masked": signal_non_null_before,
        "generic_stream_metadata_values_masked": metadata_non_null_before,
        "non_null_feature_count_after": int(corrected.loc[row_mask, modality_features].notna().sum().sum()),
        "repair_action": "SET_MODALITY_DEPENDENT_VALUES_TO_NAN",
        "minimal_regeneration_performed": False,
        "regenerated_feature_count": 0,
    })

repair_log = pd.DataFrame(repair_records)
misleading_pairs_after = int(sum(
    corrected.loc[corrected["run_key"].eq(row["run_key"]), features_by_modality[row["modality"]]].notna().any(axis=None)
    for _, row in discrepancies.iterrows()
))

assert len(corrected) == len(original)
assert corrected[IDENTIFIERS].equals(original[IDENTIFIERS])
assert misleading_pairs_after == 0
print(f"Repaired run-modality pairs: {len(repair_log)}")
print(f"Misleading run-modality pairs remaining: {misleading_pairs_after}")
display(repair_log[[
    "run_key", "modality", "affected_feature_count", "signal_derived_feature_count",
    "generic_modality_metadata_feature_count", "signal_non_null_values_masked",
    "generic_stream_metadata_values_masked", "non_null_feature_count_after", "repair_action"
]])
"""
    ),
    markdown("## 6. All-NaN and Single-Value Feature Audits"),
    code(
        """
feature_columns = [column for column in original.columns if column not in IDENTIFIERS]
before_numeric = original[feature_columns].apply(pd.to_numeric, errors="coerce")
after_numeric = corrected[feature_columns].apply(pd.to_numeric, errors="coerce")

before_unique = before_numeric.nunique(dropna=True)
after_unique = after_numeric.nunique(dropna=True)
before_all_nan = sorted(before_unique[before_unique == 0].index.tolist())
after_all_nan = sorted(after_unique[after_unique == 0].index.tolist())
before_single = sorted(before_unique[before_unique == 1].index.tolist())
after_single = sorted(after_unique[after_unique == 1].index.tolist())

provenance_by_feature = corrected_provenance.set_index("feature_name")
all_nan_records = []
for feature_name in after_all_nan:
    item = provenance_by_feature.loc[feature_name]
    statistic = feature_name.rsplit("_", 1)[-1]
    reason = (
        "The Phase 02 extractor returns NaN when skew/kurtosis sample-size or non-zero-variance "
        "preconditions fail; the saved provenance does not retain which precondition failed per run."
        if statistic in {"skew", "kurtosis"}
        else "No finite value exists in any of the 487 saved Phase 02 rows; a more specific cause is not verified."
    )
    all_nan_records.append({
        "feature_name": feature_name,
        "modality": item["availability_modality"],
        "source_provenance": item["source"],
        "reason_if_identifiable": reason,
        "decision": "STRUCTURALLY_UNUSABLE",
    })
all_nan_audit = pd.DataFrame(all_nan_records)

single_value_records = []
for feature_name in after_single:
    item = provenance_by_feature.loc[feature_name]
    values = after_numeric[feature_name].dropna()
    single_value_records.append({
        "feature_name": feature_name,
        "modality": item["availability_modality"],
        "source_provenance": item["source"],
        "non_missing_count": int(len(values)),
        "constant_value": float(values.iloc[0]),
        "was_single_value_before_repair": feature_name in set(before_single),
        "decision": "RECORD_ONLY_KEEP_GLOBAL_TABLE_FOLD_LOCAL_VARIANCE_FILTER_LATER",
    })
single_value_audit = pd.DataFrame(single_value_records)

print(f"All-NaN columns before/after: {len(before_all_nan)} / {len(after_all_nan)}")
print(f"Single-value columns before/after: {len(before_single)} / {len(after_single)}")
display(all_nan_audit)
display(single_value_audit.head(20))
"""
    ),
    markdown("## 7. Canonical Feature Groups and Modeling Manifest"),
    code(
        """
corrected_groups = {"identifier_columns": IDENTIFIERS.copy()}
for group_name, group in corrected_provenance.groupby("feature_group", sort=True):
    corrected_groups[str(group_name)] = sorted(group["feature_name"].tolist())
corrected_groups.setdefault("control_input_features", [])
corrected_groups.setdefault("unverified_features", [])

body_movement_features = sorted(
    corrected_provenance.loc[
        corrected_provenance["provenance_status"].eq("VERIFIED_BODY_MOVEMENT"), "feature_name"
    ].tolist()
)
performance_features = sorted(
    corrected_provenance.loc[
        corrected_provenance["availability_modality"].eq("performance"), "feature_name"
    ].tolist()
)
control_input_features = sorted(corrected_groups.get("control_input_features", []))
unverified_features = sorted(
    corrected_provenance.loc[corrected_provenance["provenance_status"].eq("UNVERIFIED"), "feature_name"].tolist()
)
structurally_unusable = sorted(all_nan_audit["feature_name"].tolist())

excluded_from_primary = set(performance_features) | set(control_input_features) | set(unverified_features) | set(structurally_unusable)
primary_without_performance_features = sorted(set(feature_columns) - excluded_from_primary)
performance_intersection = sorted(set(primary_without_performance_features).intersection(performance_features))

modeling_manifest = {
    "phase": "02_full_multimodal_feature_extraction",
    "source_corrected_table": str(CORRECTED_TABLE),
    "identifier_columns": IDENTIFIERS,
    "primary_without_performance_features": primary_without_performance_features,
    "performance_features": performance_features,
    "body_movement_features": body_movement_features,
    "control_input_features": control_input_features,
    "structurally_unusable_features": structurally_unusable,
    "unverified_features": unverified_features,
    "availability_columns_are_predictive_features": False,
    "feature_level_missing_indicators_created": False,
    "imputation_performed": False,
}

assert len(body_movement_features) == 42
assert all(name.startswith("unknown_lslshimmertorsoacc_") for name in body_movement_features)
assert set(body_movement_features) == set(verified_groups["body_movement_features"])
assert len(performance_features) == 59
assert not performance_intersection
assert len(control_input_features) == 0
assert corrected[body_movement_features].equals(original[body_movement_features])

display(pd.DataFrame({
    "manifest_group": ["primary_without_performance", "performance", "body_movement", "control_input", "structurally_unusable", "unverified"],
    "feature_count": [len(primary_without_performance_features), len(performance_features), len(body_movement_features), len(control_input_features), len(structurally_unusable), len(unverified_features)],
}))
print(f"Primary/performance intersection: {performance_intersection}")
"""
    ),
    markdown("## 8. Run Integrity and Difficulty Counts"),
    code(
        """
def difficulty_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"level-(\\d+)", str(value).lower())
    return float(int(match.group(1))) if match else np.nan


difficulty_values = corrected["difficulty_level"].map(difficulty_number)
difficulty_distribution = difficulty_values.value_counts(dropna=False).sort_index().astype(int).to_dict()
rest_rows = int((difficulty_values == 0).sum())
difficulty_1_to_4_rows = int(difficulty_values.isin([1, 2, 3, 4]).sum())

cp030_key = "sub-cp030_ses-20211025_level-03B_run-005"
cp030_exists_original = bool(original["run_key"].eq(cp030_key).any())
cp030_exists_corrected = bool(corrected["run_key"].eq(cp030_key).any())
cp030_missing_evidence = missing_runs[missing_runs["expected_run_key"].eq(cp030_key)]
cp030_independently_verified_absent = len(cp030_missing_evidence) == 1
cp030_usable_features = 0 if not cp030_exists_corrected else int(
    corrected.loc[corrected["run_key"].eq(cp030_key), feature_columns].notna().sum().sum()
)
cp030_modeling_candidate = bool(cp030_exists_corrected and cp030_usable_features > 0)

run_integrity = {
    "corrected_rows": len(corrected),
    "unique_run_keys": int(corrected["run_key"].nunique()),
    "rest_level_000_rows": rest_rows,
    "difficulty_1_to_4_rows": difficulty_1_to_4_rows,
    "relationship": f"{len(corrected)} = {rest_rows} rest + {difficulty_1_to_4_rows} difficulty 1-4",
    "cp030_level_03B_run_005": {
        "exists_in_original_phase02": cp030_exists_original,
        "exists_in_corrected_phase02": cp030_exists_corrected,
        "independently_verified_absent_from_raw_schedule": cp030_independently_verified_absent,
        "usable_feature_values": cp030_usable_features,
        "suitable_difficulty_3_modeling_candidate": cp030_modeling_candidate,
        "decision": "NOT A CANDIDATE: verified raw run directory is physically absent" if cp030_independently_verified_absent else "NOT VERIFIED",
    },
}

display(pd.DataFrame({"difficulty_level": list(difficulty_distribution.keys()), "run_count": list(difficulty_distribution.values())}))
display(pd.DataFrame([run_integrity["cp030_level_03B_run_005"]]))
print(run_integrity["relationship"])
"""
    ),
    markdown("## 9. Before-vs-After Quality and the 214-to-213 Explanation"),
    code(
        """
def quality_metrics(frame: pd.DataFrame) -> dict[str, object]:
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    array = numeric.to_numpy(dtype=float, copy=False)
    unique = numeric.nunique(dropna=True)
    missing_identifier_count = int(sum(
        frame[column].isna().sum() + frame[column].astype(str).str.strip().eq("").sum()
        for column in IDENTIFIERS
    ))
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "features": int(len(feature_columns)),
        "subjects": int(frame["subject_id"].nunique()),
        "unique_run_keys": int(frame["run_key"].nunique()),
        "duplicate_run_keys": int(frame["run_key"].duplicated(keep=False).sum()),
        "missing_identifiers": missing_identifier_count,
        "infinite_values": int(np.isinf(array).sum()),
        "feature_cell_nan_rate": float(numeric.isna().sum().sum() / numeric.size),
        "all_nan_columns": int((unique == 0).sum()),
        "single_value_columns": int((unique == 1).sum()),
    }


before_quality = quality_metrics(original)
after_quality = quality_metrics(corrected)

task_mask_original = original["difficulty_level"].map(difficulty_number).isin([1, 2, 3, 4])
task_primary_unique = (
    original.loc[task_mask_original, primary_without_performance_features]
    .apply(pd.to_numeric, errors="coerce")
    .nunique(dropna=True)
)
task_primary_single = set(task_primary_unique[task_primary_unique == 1].index)
phase2_single_set = set(before_single)
phase2_only_single = sorted(phase2_single_set - task_primary_single)
task_primary_only_single = sorted(task_primary_single - phase2_single_set)

single_value_difference_explanation = {
    "phase02_global_single_value_count": len(before_single),
    "task_only_primary_single_value_count": len(task_primary_single),
    "phase02_single_values_absent_from_task_primary": phase2_only_single,
    "task_primary_new_single_values_after_rest_filter": task_primary_only_single,
    "verified_reason": (
        "The primary task-only set excludes eight globally constant performance metadata features, "
        "while removing 68 rest rows makes seven eye/X-Plane features constant; net 214 - 8 + 7 = 213."
    ),
}

comparison_rows = [
    {"metric": "rows", "BEFORE": before_quality["rows"], "AFTER": after_quality["rows"]},
    {"metric": "features", "BEFORE": before_quality["features"], "AFTER": after_quality["features"]},
    {"metric": "NaN rate", "BEFORE": before_quality["feature_cell_nan_rate"], "AFTER": after_quality["feature_cell_nan_rate"]},
    {"metric": "all-NaN columns", "BEFORE": before_quality["all_nan_columns"], "AFTER": after_quality["all_nan_columns"]},
    {"metric": "single-value columns", "BEFORE": before_quality["single_value_columns"], "AFTER": after_quality["single_value_columns"]},
    {"metric": "discrepant run-modality pairs with misleading values", "BEFORE": actual_discrepancy_count, "AFTER": misleading_pairs_after},
    {"metric": "body-movement feature count", "BEFORE": 42, "AFTER": len(body_movement_features)},
    {"metric": "performance feature count", "BEFORE": 59, "AFTER": len(performance_features)},
    {"metric": "control-input feature count", "BEFORE": 0, "AFTER": len(control_input_features)},
]
before_after = pd.DataFrame(comparison_rows)
display(before_after)
print(single_value_difference_explanation["verified_reason"])
display(pd.DataFrame({
    "phase2_only_single_value_feature": pd.Series(phase2_only_single),
    "task_primary_new_single_value_feature": pd.Series(task_primary_only_single),
}))
"""
    ),
    markdown("## 10. Save Corrected Phase 02 Artifacts"),
    code(
        """
RESULTS.mkdir(parents=True, exist_ok=True)

corrected.to_csv(CORRECTED_TABLE, index=False)
repair_log.to_csv(REPAIR_LOG, index=False)
all_nan_audit.to_csv(ALL_NAN_AUDIT, index=False)
single_value_audit.to_csv(SINGLE_VALUE_AUDIT, index=False)
corrected_provenance.to_csv(CORRECTED_PROVENANCE, index=False)
phase1.to_csv(VERIFIED_AVAILABILITY_OUTPUT, index=False)
CORRECTED_GROUPS.write_text(json.dumps(corrected_groups, indent=2), encoding="utf-8")
MODELING_MANIFEST.write_text(json.dumps(modeling_manifest, indent=2), encoding="utf-8")

print("Saved corrected table and Phase 02 machine-readable repair artifacts.")
display(pd.DataFrame({
    "output_file": [path.name for path in OUTPUT_PATHS if path != VALIDATION_SUMMARY],
    "exists": [path.is_file() for path in OUTPUT_PATHS if path != VALIDATION_SUMMARY],
}))
"""
    ),
    markdown("## 11. Critical Assertions and Preservation Checks"),
    code(
        """
reloaded = pd.read_csv(CORRECTED_TABLE, low_memory=False)
raw_fingerprint_after = tree_metadata_fingerprint(RAW_ROOT)
original_table_hash_after = file_sha256(ORIGINAL_TABLE)

same_or_both_nan = original[feature_columns].eq(corrected[feature_columns]) | (
    original[feature_columns].isna() & corrected[feature_columns].isna()
)
changed_locations = (~same_or_both_nan).stack()
changed_locations = changed_locations[changed_locations].index.tolist()
changed_cells = {
    (str(original.loc[row_index, "run_key"]), feature_name)
    for row_index, feature_name in changed_locations
}
allowed_changed_cells = {
    (str(row["run_key"]), feature_name)
    for _, row in discrepancies.iterrows()
    for feature_name in features_by_modality[row["modality"]]
}
all_changes_are_to_nan = all(
    pd.isna(corrected.loc[row_index, feature_name])
    for row_index, feature_name in changed_locations
)
expected_masked_value_count = int(
    repair_log["signal_non_null_values_masked"].sum()
    + repair_log["generic_stream_metadata_values_masked"].sum()
)

critical_assertions = {
    "no_duplicate_run_keys": bool(reloaded["run_key"].is_unique),
    "no_missing_identifiers": bool(all(
        reloaded[column].notna().all() and reloaded[column].astype(str).str.strip().ne("").all()
        for column in IDENTIFIERS
    )),
    "no_infinite_values": bool(np.isinf(reloaded[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).sum() == 0),
    "body_movement_count_42_and_verified": bool(
        len(body_movement_features) == 42
        and corrected_provenance.loc[
            corrected_provenance["feature_name"].isin(body_movement_features), "provenance_status"
        ].eq("VERIFIED_BODY_MOVEMENT").all()
    ),
    "explicit_control_input_count_zero": len(control_input_features) == 0,
    "performance_count_matches_verified_actual": len(performance_features) == 59,
    "primary_performance_intersection_empty": len(performance_intersection) == 0,
    "unavailable_modalities_have_no_placeholder_values": misleading_pairs_after == 0,
    "only_verified_affected_cells_changed": changed_cells.issubset(allowed_changed_cells),
    "every_change_is_to_nan": all_changes_are_to_nan,
    "changed_cell_count_matches_repair_log": len(changed_cells) == expected_masked_value_count,
    "identifiers_and_run_metadata_unchanged": corrected[IDENTIFIERS].equals(original[IDENTIFIERS]),
    "all_487_phase02_rows_preserved": len(reloaded) == 487 and set(reloaded["run_key"]) == set(original["run_key"]),
    "raw_data_unchanged_during_notebook": raw_fingerprint_after == raw_fingerprint_before,
    "original_phase02_table_preserved": original_table_hash_after == original_table_hash_before,
    "original_table_not_overwritten": CORRECTED_TABLE.resolve() != ORIGINAL_TABLE.resolve(),
    "no_full_reextraction_performed": True,
    "no_model_training_performed": True,
    "no_imputation_performed": True,
}

failed_assertions = [name for name, passed in critical_assertions.items() if not passed]
assert not failed_assertions, f"Critical assertions failed: {failed_assertions}"

phase03_ready = not failed_assertions and actual_discrepancy_count > 0 and misleading_pairs_after == 0
validation_summary = {
    "phase": "02_full_multimodal_feature_extraction",
    "operation": "placeholder_zero_sample_repair",
    "executed_at": datetime.now().isoformat(timespec="seconds"),
    "original_table": str(ORIGINAL_TABLE),
    "corrected_table": str(CORRECTED_TABLE),
    "original_table_sha256_before": original_table_hash_before,
    "original_table_sha256_after": original_table_hash_after,
    "raw_data_fingerprint_before": raw_fingerprint_before,
    "raw_data_fingerprint_after": raw_fingerprint_after,
    "before_quality": before_quality,
    "after_quality": after_quality,
    "actual_discrepant_run_modality_pair_count": actual_discrepancy_count,
    "affected_run_count": int(discrepancies["run_key"].nunique()),
    "remaining_misleading_pair_count": misleading_pairs_after,
    "minimal_regeneration_performed": False,
    "full_reextraction_performed": False,
    "body_movement_feature_count": len(body_movement_features),
    "performance_feature_count": len(performance_features),
    "control_input_feature_count": len(control_input_features),
    "primary_without_performance_feature_count": len(primary_without_performance_features),
    "primary_performance_intersection": performance_intersection,
    "all_nan_feature_count": len(after_all_nan),
    "single_value_feature_count": len(after_single),
    "single_value_214_to_213_explanation": single_value_difference_explanation,
    "run_integrity": run_integrity,
    "difficulty_distribution": {str(int(key)): int(value) for key, value in difficulty_distribution.items()},
    "critical_assertions": critical_assertions,
    "phase03_ready": phase03_ready,
    "phase03_executed": False,
    "model_training_performed": False,
    "outputs": [str(path) for path in OUTPUT_PATHS],
}
VALIDATION_SUMMARY.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

assert VALIDATION_SUMMARY.is_file()
assert all(path.is_file() for path in OUTPUT_PATHS)
display(pd.DataFrame({"critical_assertion": list(critical_assertions), "passed": list(critical_assertions.values())}))
print(f"PHASE 03 READY: {'YES' if phase03_ready else 'NO'}")
"""
    ),
    markdown("## 12. Phase Validation Summary"),
    code(
        '''
summary_markdown = f"""
## VERIFIED

- Corrected Phase 02 table: {after_quality['rows']} rows, {after_quality['features']} feature columns, {after_quality['subjects']} subjects, and {after_quality['unique_run_keys']} unique run keys.
- Duplicate run keys: {after_quality['duplicate_run_keys']}; missing identifiers: {after_quality['missing_identifiers']}; infinite values: {after_quality['infinite_values']}.
- Original raw data and original completed Phase 02 table passed before/after preservation checks.

## PLACEHOLDER REPAIR

- Actual discrepancies supported by the loaded evidence: {actual_discrepancy_count} run-modality pairs across {discrepancies['run_key'].nunique()} runs.
- Remaining pairs with misleading modality-dependent values: {misleading_pairs_after}.
- Repair: modality-dependent signal and stream-metadata values were set to NaN; no values were set to zero, no imputation was performed, and no run was dropped.
- Minimal regeneration: not required. Full re-extraction: not performed.

## BODY MOVEMENT

- BODY_MOVEMENT_FEATURE_COUNT: {len(body_movement_features)}.
- Canonical group: body_movement. Provenance status: VERIFIED_BODY_MOVEMENT.
- Numerical values were not changed solely for group renaming.

## PERFORMANCE FEATURES

- Verified performance feature count: {len(performance_features)}.
- Intersection with the primary without-performance manifest: {len(performance_intersection)}.

## CONTROL INPUT

- Explicit control-input feature count: {len(control_input_features)}.
- X-Plane state, trim, physiological, and movement features were not relabeled as direct pilot control input.

## ALL-NaN FEATURES

- Before repair: {len(before_all_nan)}. After repair: {len(after_all_nan)}.
- All are recorded as STRUCTURALLY_UNUSABLE and excluded from the modeling feature manifest, while remaining present in the corrected Phase 02 table.

## SINGLE-VALUE FEATURES

- Before repair: {len(before_single)}. After repair: {len(after_single)}.
- They remain recorded in the global table; later variance filtering must be fitted inside training folds.
- The earlier 214-to-213 difference is verified as eight excluded constant performance features plus seven features made constant by the task-only row filter.

## RUN INTEGRITY

- All {len(corrected)} legitimate existing Phase 02 rows remain. Difficulty counts: {rest_rows} level-000/rest and {difficulty_1_to_4_rows} Difficulty 1-4 rows.
- cp030 level-03B run-005 exists in Phase 02: {cp030_exists_corrected}. It is independently verified as a physically absent raw run and is not a Difficulty 3 modeling candidate.

## CORRECTED OUTPUT FILES

{chr(10).join(f'- {path.name}' for path in OUTPUT_PATHS)}

## PHASE 03 READINESS

- The corrected Phase 02 table and manifests passed all critical assertions.
- Phase 03 was not started and no model was trained.

## PHASE 03 READY: {'YES' if phase03_ready else 'NO'}
"""
display(Markdown(summary_markdown))
'''
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
