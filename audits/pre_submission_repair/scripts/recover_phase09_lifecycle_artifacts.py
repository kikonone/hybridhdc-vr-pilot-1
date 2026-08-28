from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PHASE09 = ROOT / "experiments" / "phase_09_robustness_and_generalization"
TRANSCRIPT = Path(
    r"C:\Users\76176\.codex\sessions\2026\08\21\rollout-2026-08-21T14-28-56-01a02302-1bf5-7313-a87c-9866a2590820.jsonl"
)
EARLIER_TRANSCRIPT = Path(
    r"C:\Users\76176\.codex\sessions\2026\08\21\rollout-2026-08-21T10-48-44-01a02238-8334-74e1-8178-585eee39d0f2.jsonl"
)
FREEZE_TIME = "2026-08-21T06:40:53.404435+00:00"
CONTRACT_TIME = "2026-08-21T06:32:58.402554+00:00"
EXPECTED = {
    "audits/phase09_executor_validation_audit.json": "405943648d5e305b5eea04eeea2332657eca227672486f1642bca44961c56462",
    "configs/phase09_contract_freeze.json": "a9d57f15a8b8c5b16fc019c81f7dba43795725da88d7083347f8a6ff93bd9b11",
    "configs/phase09_execution_manifest.json": "14f2448b03147f080951b6ab72b9173777c0060253dc7974ae0b688f8b660537",
    "configs/phase09_frozen_contract.json": "f908109261ab4ae1141e657ab8a1a08760a42d145aa9e6a8d1d2ec37c33e4c93",
}


def encoded(document: dict, *, crlf: bool = False) -> bytes:
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if crlf:
        text = text.replace("\n", "\r\n")
    return text.encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transcript_timestamps() -> list[str]:
    raw = TRANSCRIPT.read_text(encoding="utf-8")
    pattern = re.compile(
        r"validated_at_utc.{0,100}?"
        r"(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:\+00:00|Z))"
    )
    return list(dict.fromkeys(pattern.findall(raw)))


def original_executor_audit_bytes() -> tuple[bytes, list[str]]:
    """Recover the complete dry-run audit printed in the trusted session record."""
    source_timestamp = "2026-08-21T03:28:31.180966+00:00"
    final_timestamp = "2026-08-21T04:10:25.113958+00:00"
    for line in EARLIER_TRANSCRIPT.read_text(encoding="utf-8").splitlines():
        if source_timestamp not in line:
            continue
        event = json.loads(line)
        stdout = event.get("payload", {}).get("item", {}).get("stdout", "")
        if '"audit": "executor_validation"' not in stdout:
            continue
        document = json.loads(stdout)
        document["validated_at_utc"] = final_timestamp
        data = encoded(document, crlf=True)
        if digest(data) == EXPECTED["audits/phase09_executor_validation_audit.json"]:
            return data, [final_timestamp]
    raise RuntimeError("Trusted transcript did not contain the expected complete executor audit")


def candidates() -> tuple[dict[str, bytes], list[str]]:
    paths = {relative: PHASE09 / relative for relative in EXPECTED}

    frozen = json.loads(paths["configs/phase09_frozen_contract.json"].read_text(encoding="utf-8"))
    frozen["ready_for_execution"] = True

    contract = json.loads(paths["configs/phase09_contract_freeze.json"].read_text(encoding="utf-8"))
    contract.update(
        status="CONTRACT_FROZEN_NOT_TRAINED",
        frozen_at_utc=CONTRACT_TIME,
        ready_for_execution=True,
    )

    current_execution = json.loads(paths["configs/phase09_execution_manifest.json"].read_text(encoding="utf-8"))
    base_keys = [
        "phase", "status", "training_run_count", "duplicate_run_identifiers",
        "run_counts_by_protocol", "run_counts_by_model",
        "full_primary_reference_counted_as_training",
        "sudden_test_time_missingness_counted_as_training",
    ]
    execution = {key: current_execution[key] for key in base_keys}
    execution.update(
        completed_training_runs=720,
        raw_prediction_rows=30168,
        canonical_oof_rows=10056,
        ready_for_phase09_freeze=False,
        analysis_completed=True,
        phase09_freeze_executed=False,
        phase10_executed=False,
    )
    execution["training_runs"] = current_execution["training_runs"]
    execution["sources"] = current_execution["sources"]
    execution.update(
        phase09_frozen=True,
        ready_to_proceed_to_phase10=True,
        freeze_time_utc=FREEZE_TIME,
        model_retraining_during_freeze=False,
        predictions_regenerated_during_freeze=False,
        last_updated_utc=FREEZE_TIME,
    )

    audit_bytes, matched_timestamps = original_executor_audit_bytes()

    output = {
        "audits/phase09_executor_validation_audit.json": audit_bytes,
        "configs/phase09_contract_freeze.json": encoded(contract),
        "configs/phase09_execution_manifest.json": encoded(execution, crlf=True),
        "configs/phase09_frozen_contract.json": encoded(frozen),
    }
    return output, matched_timestamps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    recovered, matched_timestamps = candidates()
    verification = {
        relative: {
            "bytes": len(data),
            "sha256": digest(data),
            "expected_sha256": EXPECTED[relative],
            "match": digest(data) == EXPECTED[relative],
        }
        for relative, data in recovered.items()
    }
    result = {
        "status": "PASS" if all(row["match"] for row in verification.values()) else "FAIL",
        "method": "deterministic reconstruction from trusted Codex transcript, recorded patch order, and frozen-manifest hashes",
        "transcript": str(TRANSCRIPT),
        "executor_audit_matched_timestamps": matched_timestamps,
        "artifacts": verification,
        "restored": False,
    }
    if args.restore:
        if result["status"] != "PASS":
            raise RuntimeError(result)
        quarantine = ROOT / "audits" / "pre_submission_repair" / "quarantine" / "phase_09"
        quarantine.mkdir(parents=True, exist_ok=True)
        for relative, data in recovered.items():
            target = PHASE09 / relative
            backup = quarantine / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            target.write_bytes(data)
        result["restored"] = True
        result["quarantine"] = str(quarantine)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
