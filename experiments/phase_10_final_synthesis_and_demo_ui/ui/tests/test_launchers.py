"""Launcher conflict acceptance check. Run while the localhost server is active."""
from __future__ import annotations

import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UI = Path(__file__).resolve().parents[1]


def main() -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=10) as response:
        assert response.status == 200
    powershell = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(UI / "start_ui.ps1")],
        cwd=UI, capture_output=True, text=True, timeout=20,
    )
    batch = subprocess.run(
        ["cmd.exe", "/d", "/c", str(UI / "start_ui.bat")],
        cwd=UI, capture_output=True, text=True, timeout=20,
    )
    powershell_output = powershell.stdout + powershell.stderr
    batch_output = batch.stdout + batch.stderr
    assert powershell.returncode == 2, powershell_output
    assert batch.returncode == 2, batch_output
    assert "Port 8501 is already in use" in powershell_output
    assert "Port 8501 is already in use" in batch_output
    report = {
        "audit": "ui_launcher_acceptance", "status": "PASS",
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": "127.0.0.1", "port": 8501,
        "powershell_conflict_exit_code": powershell.returncode,
        "batch_conflict_exit_code": batch.returncode,
        "english_conflict_message": True,
    }
    (UI / "audits/ui_launcher_acceptance_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
