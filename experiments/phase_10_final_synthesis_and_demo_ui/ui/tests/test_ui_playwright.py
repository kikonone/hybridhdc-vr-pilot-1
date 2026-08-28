"""Actual Streamlit browser acceptance run. Execute while the local server is up."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

UI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI / "tests/.deps"))
from playwright.sync_api import Page, sync_playwright

URL = "http://127.0.0.1:8501"
REPORTS = UI / "tests/reports"
SHOTS = REPORTS / "screenshots"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def wait_for_app(page: Page) -> None:
    page.get_by_text("HDC Classification and", exact=False).wait_for(timeout=45_000)
    page.get_by_text("SYSTEM", exact=True).wait_for(timeout=15_000)
    page.get_by_text("READY", exact=True).wait_for(timeout=15_000)


def display_id(demo_id: str) -> str:
    return f"Record {demo_id.removeprefix('DEMO-')}"


def choose_record(page: Page, demo_id: str) -> None:
    label = display_id(demo_id)
    page.get_by_label("Anonymous record").click()
    page.keyboard.type(label)
    page.get_by_role("option", name=label, exact=True).click()
    page.locator(".sync-line").get_by_text(label, exact=True).wait_for(timeout=30_000)


def choose_task(page: Page, task: str) -> None:
    page.get_by_text(task, exact=True).first.click()
    heading = "Classification result" if task == "Classification" else "Proxy-regression result"
    page.get_by_text(heading, exact=True).wait_for(timeout=30_000)


def run() -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    page_errors: list[str] = []
    tracebacks: list[str] = []
    tested_records: set[str] = set()
    interactions = 0

    with sync_playwright() as playwright:
        launch = {"headless": True, "args": ["--disable-gpu", "--no-first-run"]}
        if EDGE.is_file():
            launch["executable_path"] = str(EDGE)
        browser = playwright.chromium.launch(**launch)
        for width, height in [(1366, 768), (1920, 1080)]:
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
            wait_for_app(page)

            page.get_by_text("Classification result", exact=True).wait_for(timeout=30_000)
            assert page.locator(".sync-line").get_by_text("Record 0001", exact=True).count() == 1
            tested_records.add("DEMO-0001")
            interactions += 1
            page.screenshot(path=str(SHOTS / f"classification_{width}x{height}.png"), full_page=True)

            choose_record(page, "DEMO-0210")
            tested_records.add("DEMO-0210")
            interactions += 1
            choose_task(page, "Regression")
            interactions += 1
            assert page.locator(".detail-grid").get_by_text("Record 0210", exact=True).count() == 1
            assert page.get_by_text("RAW PREDICTION", exact=True).count() == 1
            assert page.get_by_text("BOUNDED PREDICTION", exact=True).count() == 1
            page.screenshot(path=str(SHOTS / f"regression_{width}x{height}.png"), full_page=True)

            choose_record(page, "DEMO-0419")
            tested_records.add("DEMO-0419")
            interactions += 1
            assert page.locator(".detail-grid").get_by_text("Record 0419", exact=True).count() == 1
            choose_task(page, "Classification")
            interactions += 1
            assert page.locator(".sync-line").get_by_text("Record 0419", exact=True).count() == 1

            for _ in range(5):
                choose_task(page, "Regression")
                choose_task(page, "Classification")
                interactions += 2

            page.reload(wait_until="domcontentloaded", timeout=60_000)
            wait_for_app(page)
            interactions += 1
            body = page.locator("body").inner_text()
            for forbidden in ("Frozen", "OOF", "canonical", "PHASE 10", "DEMO-"):
                assert forbidden.lower() not in body.lower(), forbidden
            if "Traceback (most recent call last)" in body:
                tracebacks.append(f"{width}x{height}: traceback rendered after refresh")
            context.close()
        browser.close()

    expected_screenshots = [
        "classification_1366x768.png", "regression_1366x768.png",
        "classification_1920x1080.png", "regression_1920x1080.png",
    ]
    assert all((SHOTS / name).is_file() and (SHOTS / name).stat().st_size > 0 for name in expected_screenshots)
    assert tested_records == {"DEMO-0001", "DEMO-0210", "DEMO-0419"}
    assert not console_errors, console_errors
    assert not page_errors, page_errors
    assert not tracebacks, tracebacks

    report = {
        "audit": "ui_playwright_e2e", "status": "PASS",
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "browser": "Microsoft Edge via Playwright" if EDGE.is_file() else "Playwright Chromium",
        "url": URL,
        "viewports": [{"width": 1366, "height": 768}, {"width": 1920, "height": 1080}],
        "tasks": ["Classification", "Regression"],
        "records_checked": sorted(tested_records),
        "refreshes": 2, "rapid_task_switches": 20, "interactions": interactions,
        "same_demo_id_across_tasks": True,
        "console_errors": console_errors, "page_errors": page_errors, "tracebacks": tracebacks,
        "screenshots": [f"tests/reports/screenshots/{name}" for name in expected_screenshots],
    }
    (UI / "audits/ui_playwright_e2e_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
