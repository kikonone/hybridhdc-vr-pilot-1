"""Fail-closed loaders for the UI-local anonymous frozen data package."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = UI_ROOT / "data"
DATA_FILE = DATA_ROOT / "frozen_dual_task_oof.csv"
MODEL_FILE = DATA_ROOT / "frozen_dual_task_model.json"
MANIFEST_FILE = DATA_ROOT / "demo_data_manifest.json"

EXPECTED_COLUMNS = (
    "demo_id", "fold", "true_difficulty", "predicted_difficulty", "classification_correct",
    "difficulty_1_cosine", "difficulty_2_cosine", "difficulty_3_cosine", "difficulty_4_cosine",
    "true_difficulty_score", "raw_frozen_prediction", "bounded_frozen_prediction", "absolute_error",
)


class DataContractError(RuntimeError):
    """Raised when the frozen local package cannot be trusted for display."""


def _integer(value: str, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"Invalid integer in {field}") from exc


def _finite_float(value: str, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"Invalid numeric value in {field}") from exc
    if not math.isfinite(converted):
        raise DataContractError(f"Non-finite numeric value in {field}")
    return converted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Invalid local package file: {path.name}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"Unexpected JSON root in {path.name}")
    return value


def load_model() -> dict:
    model = _json(MODEL_FILE)
    required = {"title", "classification", "regression", "records", "folds", "frozen_evaluation_seeds"}
    if set(model) != required:
        raise DataContractError("Model contract keys do not match the system schema")
    if model["classification"]["model"] != "HDC+OnlineHD Hybrid" or model["classification"]["dimension"] != 5000:
        raise DataContractError("Classification model contract mismatch")
    regression = model["regression"]
    if regression["model"] != "COMMON_ENCODER_READOUT_BASELINE" or regression["variant"] != "common_ridge":
        raise DataContractError("Regression model contract mismatch")
    if regression["dimension"] != 10000 or regression["feature_k"] != 50 or regression["levels"] != 51:
        raise DataContractError("Regression configuration mismatch")
    if float(regression["ridge_alpha"]) != 0.01:
        raise DataContractError("Regression ridge alpha mismatch")
    return model


def load_rows() -> list[dict[str, str]]:
    try:
        with DATA_FILE.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                raise DataContractError("Demo CSV schema mismatch")
            rows = list(reader)
    except OSError as exc:
        raise DataContractError("Demo data file is unavailable") from exc
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 419:
        raise DataContractError(f"Expected 419 aligned rows, found {len(rows)}")
    expected_ids = [f"DEMO-{index:04d}" for index in range(1, 420)]
    if [row["demo_id"] for row in rows] != expected_ids:
        raise DataContractError("Anonymous IDs are missing, duplicated, or unstable")
    if {_integer(row["fold"], "fold") for row in rows} != {1, 2, 3, 4, 5}:
        raise DataContractError("Fold coverage mismatch")
    forbidden = re.compile(r"run_key|subject|session|participant", re.IGNORECASE)
    if any(forbidden.search(column) for column in EXPECTED_COLUMNS):
        raise DataContractError("Research identifier leaked into presentation schema")
    for row in rows:
        if any(value is None or str(value).strip() == "" for value in row.values()):
            raise DataContractError(f"Missing value in {row.get('demo_id', 'unknown record')}")
        true_class = _integer(row["true_difficulty"], "true_difficulty")
        predicted_class = _integer(row["predicted_difficulty"], "predicted_difficulty")
        if true_class not in range(1, 5) or predicted_class not in range(1, 5):
            raise DataContractError("Classification value outside Difficulty 1-4")
        expected_correct = str(true_class == predicted_class).lower()
        if row["classification_correct"].lower() != expected_correct:
            raise DataContractError("Classification correctness flag mismatch")
        scores = [_finite_float(row[f"difficulty_{index}_cosine"], f"difficulty_{index}_cosine") for index in range(1, 5)]
        target = _finite_float(row["true_difficulty_score"], "true_difficulty_score")
        raw = _finite_float(row["raw_frozen_prediction"], "raw_frozen_prediction")
        bounded = _finite_float(row["bounded_frozen_prediction"], "bounded_frozen_prediction")
        absolute_error = _finite_float(row["absolute_error"], "absolute_error")
        if not 1.0 <= target <= 4.0 or not 1.0 <= bounded <= 4.0:
            raise DataContractError("Bounded regression value outside 1-4")
        if not math.isclose(absolute_error, abs(target - bounded), rel_tol=0.0, abs_tol=1e-12):
            raise DataContractError("Presentation-only absolute error mismatch")


def verify_package() -> tuple[bool, str]:
    try:
        manifest = _json(MANIFEST_FILE)
        if manifest.get("status") != "PASS" or manifest.get("scientific_transformations") != "NONE":
            raise DataContractError("Manifest status mismatch")
        if manifest.get("ui_clipping_executed") is not False:
            raise DataContractError("UI clipping contract mismatch")
        for entry in manifest.get("output_files", []):
            path = Path(entry["path"])
            if path.parent.resolve() != DATA_ROOT.resolve() or not path.is_file():
                raise DataContractError(f"Missing local package artifact: {path.name}")
            if path.stat().st_size != int(entry["size_bytes"]) or _sha256(path) != entry["sha256"]:
                raise DataContractError(f"Integrity mismatch: {path.name}")
        load_model()
        load_rows()
    except (DataContractError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc)
    return True, "Local demo package verified"
