from __future__ import annotations

import json
import re
from pathlib import Path

UI = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    UI / "app.py",
    UI / "components/data_access.py",
    UI / "components/charts.py",
    UI / "components/sections.py",
    UI / "components/theme.py",
    UI / "assets/aviation_console.css",
    UI / "configs/ui_contract.json",
    UI / "data/frozen_dual_task_model.json",
    UI / "README.md",
    UI / "pages/README.md",
    UI / "start_ui.ps1",
    UI / "start_ui.bat",
]


def _runtime_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME_FILES)


def test_required_structure_exists() -> None:
    required = [
        "app.py", "requirements.txt", "README.md", "start_ui.bat", "start_ui.ps1",
        "components/data_access.py", "components/charts.py", "components/sections.py",
        "components/theme.py", "assets/aviation_console.css", "configs/ui_contract.json",
        "data/demo_data_manifest.json", "data/frozen_dual_task_oof.csv", "data/frozen_dual_task_model.json",
    ]
    assert all((UI / relative).is_file() for relative in required)


def test_single_page_and_exactly_two_tasks() -> None:
    contract = json.loads((UI / "configs/ui_contract.json").read_text(encoding="utf-8"))
    assert contract["single_page"] is True
    assert contract["allowed_tasks"] == ["Classification", "Regression"]
    assert list((UI / "pages").glob("*.py")) == []


def test_required_english_copy_is_present() -> None:
    source = _runtime_text()
    required = [
        "HDC Classification and", "Proxy-Regression Demonstration",
        "A local demonstration of classification and proxy-regression results. It does not perform live inference.",
        "Anonymous record", "Classification result", "Proxy-regression result",
        "HDC+OnlineHD Hybrid", "COMMON_ENCODER_READOUT_BASELINE", "common_ridge",
        "bounded difficulty-induced workload proxy regression",
        "cosine similarities", "not calibrated probabilities",
    ]
    assert all(token in source for token in required)


def test_audience_copy_omits_provenance_language_and_demo_prefix() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [UI / "app.py", UI / "components/sections.py", UI / "components/charts.py", UI / "assets/aviation_console.css"]
    )
    forbidden_visible_phrases = [
        "PHASE 10 · OFFLINE DEFENSE CONSOLE", "HDC // OOF", "<h1>Frozen",
        "Anonymous frozen OOF record", "Frozen classification replay", "Frozen proxy-regression replay",
        "Selected frozen model", "Selected frozen head", "RAW FROZEN PREDICTION",
        "BOUNDED FROZEN PREDICTION", "FROZEN CLASS SIMILARITY PROFILE",
    ]
    assert not {phrase for phrase in forbidden_visible_phrases if phrase in source}
    assert "removeprefix('DEMO-')" in source


def test_runtime_visible_material_is_english_only() -> None:
    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
    violations = {str(path.relative_to(UI)): cjk.findall(path.read_text(encoding="utf-8")) for path in RUNTIME_FILES}
    assert not {path: chars for path, chars in violations.items() if chars}


def test_offline_read_only_contract() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in [UI / "app.py", *(UI / "components").glob("*.py")])
    forbidden = [
        "requests.", "urllib.", "httpx.", "socket.", "st.file_uploader", "st.camera_input",
        ".fit(", ".predict(", "train_test_split", "subprocess", "os.system", "st.sidebar",
    ]
    assert not {token for token in forbidden if token in source}


def test_localhost_binding_and_english_port_conflict_behavior() -> None:
    powershell = (UI / "start_ui.ps1").read_text(encoding="utf-8")
    batch = (UI / "start_ui.bat").read_text(encoding="utf-8")
    for source in (powershell, batch):
        assert "--server.address 127.0.0.1" in source
        assert "Port" in source and "already in use" in source
        assert "0.0.0.0" not in source


def test_no_research_identifiers_in_presentation_dataset() -> None:
    header = (UI / "data/frozen_dual_task_oof.csv").read_text(encoding="utf-8").splitlines()[0].lower()
    assert not any(token in header for token in ("run_key", "subject", "session", "participant"))
