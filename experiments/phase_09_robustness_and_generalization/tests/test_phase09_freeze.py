from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_phase09 import (  # noqa: E402
    EXPECTED, EXPECTED_FOLDS, EXPECTED_PRIMARY, FINAL_MANIFEST, FREEZE_FILE,
    build_manifest, freeze_payload, preflight, protected_baseline, read_json,
    verify_hash_map,
)
from verify_phase09_freeze import verify_freeze  # noqa: E402


def test_read_only_preflight_passes_before_freeze() -> None:
    if FREEZE_FILE.exists():
        assert verify_freeze()["status"] == "PASS"
    else:
        result = preflight(require_freeze_absent=True)
        assert result["status"] == "PASS", {key: value for key, value in result["checks"].items() if not value}
        assert result["actual"] == {
            "authorized_runs": 720, "completed_runs": 720, "raw_prediction_rows": 30168,
            "canonical_oof_rows": 10056, "missing_modality_rows": 8380, "loso_rows": 1676,
        }


def test_protected_baseline_has_no_hash_mismatch() -> None:
    baseline = protected_baseline()
    assert len(baseline) == 3641
    assert verify_hash_map(baseline) == []


def test_manifest_build_is_reproducible_and_has_no_duplicate_paths() -> None:
    if FREEZE_FILE.exists():
        first = read_json(FINAL_MANIFEST)
        second = read_json(FINAL_MANIFEST)
    else:
        checked = preflight(require_freeze_absent=True)
        first = build_manifest("2026-08-21T00:00:00+00:00", checked)
        second = build_manifest("2026-08-21T00:00:00+00:00", checked)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["duplicate_artifacts"] == []
    assert first["hash_mismatches"] == []
    assert first["self_hash_included"] is False


def test_freeze_payload_preserves_claim_boundaries() -> None:
    payload = freeze_payload("2026-08-21T00:00:00+00:00", "a" * 64)
    assert payload["status"] == "FROZEN"
    assert payload["generalization_boundaries"]["SUBJECT_GENERALIZATION"] == "EVALUATED_VIA_35_SUBJECT_LOSO"
    assert payload["generalization_boundaries"]["FLIGHT_GENERALIZABLE_BEHAVIOR_CLAIM"] == "INCONCLUSIVE_DUE_TO_METADATA"
    assert payload["model_retraining_during_freeze"] is False
    assert payload["predictions_regenerated_during_freeze"] is False
    assert payload["phase10_executed"] is False


def test_expected_primary_and_fold_hashes_are_frozen() -> None:
    assert EXPECTED_PRIMARY == "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
    assert EXPECTED_FOLDS == "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
    assert EXPECTED["runs"] == 720 and EXPECTED["canonical"] == 10056
