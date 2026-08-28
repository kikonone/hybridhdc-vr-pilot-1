"""Bounded 25-session, 1000-request, 10-minute localhost stress and soak run."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

UI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI / "tests/.deps"))
import psutil
from playwright.sync_api import Page, sync_playwright

URL = "http://127.0.0.1:8501"
HEALTH = f"{URL}/_stcore/health"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
SESSION_COUNT = 25
HTTP_INTERACTIONS = 1000
UI_SWITCHES = 500
RELOADS = 100
SOAK_SECONDS = 600


def file_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((UI / "data").glob("*")) if path.is_file()
    }


def server_pid() -> int:
    if os.environ.get("HDC_UI_SERVER_PID"):
        return int(os.environ["HDC_UI_SERVER_PID"])
    for connection in psutil.net_connections(kind="tcp"):
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == 8501 and connection.pid:
            return int(connection.pid)
    raise RuntimeError("Could not identify the local Streamlit process on port 8501")


def tree_rss_mb(pid: int) -> float:
    process = psutil.Process(pid)
    processes = [process, *process.children(recursive=True)]
    return sum(item.memory_info().rss for item in processes if item.is_running()) / (1024 * 1024)


def timed_health(_: int) -> tuple[float, int]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(HEALTH, timeout=10) as response:
            status = response.status
            response.read()
    except Exception:
        return time.perf_counter() - started, 0
    return time.perf_counter() - started, status


def choose_record(page: Page, demo_id: str) -> None:
    label = f"Record {demo_id.removeprefix('DEMO-')}"
    page.get_by_label("Anonymous record").click()
    page.keyboard.type(label)
    page.get_by_role("option", name=label, exact=True).click()
    page.locator(".sync-line").get_by_text(label, exact=True).wait_for(timeout=30_000)


def choose_task(page: Page, task: str) -> None:
    page.get_by_text(task, exact=True).first.click()
    heading = "Classification result" if task == "Classification" else "Proxy-regression result"
    page.get_by_text(heading, exact=True).wait_for(timeout=30_000)


def main() -> dict:
    start_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    pid = server_pid()
    hashes_before = file_hashes()
    memory_samples = [{"elapsed_seconds": 0.0, "rss_mb": tree_rss_mb(pid)}]
    console_errors: list[str] = []
    page_errors: list[str] = []
    tracebacks: list[str] = []
    status_codes: list[int] = []
    http_latencies: list[float] = []
    task_switches = 0
    record_switches = 0
    reloads = 0

    with ThreadPoolExecutor(max_workers=SESSION_COUNT) as executor:
        for latency, status in executor.map(timed_health, range(HTTP_INTERACTIONS)):
            http_latencies.append(latency)
            status_codes.append(status)

    with sync_playwright() as playwright:
        launch = {"headless": True, "args": ["--disable-gpu", "--no-first-run"]}
        if EDGE.is_file():
            launch["executable_path"] = str(EDGE)
        browser = playwright.chromium.launch(**launch)
        contexts = [browser.new_context(viewport={"width": 1280, "height": 720}) for _ in range(SESSION_COUNT)]
        pages = []
        for context in contexts:
            page = context.new_page()
            page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
            page.get_by_text("Classification result", exact=True).wait_for(timeout=45_000)
            pages.append(page)

        for round_index in range(10):
            for page_index, page in enumerate(pages):
                demo_number = ((round_index * SESSION_COUNT + page_index) * 37) % 419 + 1
                choose_record(page, f"DEMO-{demo_number:04d}")
                record_switches += 1
                choose_task(page, "Regression" if round_index % 2 == 0 else "Classification")
                task_switches += 1
                if round_index in {1, 3, 5, 7}:
                    page.reload(wait_until="domcontentloaded", timeout=60_000)
                    page.get_by_text("HDC Classification and", exact=False).wait_for(timeout=45_000)
                    reloads += 1
                elapsed = time.monotonic() - started
                if elapsed - memory_samples[-1]["elapsed_seconds"] >= 60:
                    memory_samples.append({"elapsed_seconds": round(elapsed, 3), "rss_mb": tree_rss_mb(pid)})

        while time.monotonic() - started < SOAK_SECONDS:
            elapsed = time.monotonic() - started
            latency, status = timed_health(HTTP_INTERACTIONS + len(status_codes))
            http_latencies.append(latency)
            status_codes.append(status)
            if elapsed - memory_samples[-1]["elapsed_seconds"] >= 60:
                memory_samples.append({"elapsed_seconds": round(elapsed, 3), "rss_mb": tree_rss_mb(pid)})
            page = pages[len(status_codes) % SESSION_COUNT]
            body = page.locator("body").inner_text()
            if "Traceback (most recent call last)" in body:
                tracebacks.append("Traceback rendered during soak")
            page.wait_for_timeout(500)

        for context in contexts:
            context.close()
        browser.close()

    duration = time.monotonic() - started
    memory_samples.append({"elapsed_seconds": round(duration, 3), "rss_mb": tree_rss_mb(pid)})
    hashes_after = file_hashes()
    failures = sum(status == 0 or 500 <= status <= 599 for status in status_codes)
    growth_mb = memory_samples[-1]["rss_mb"] - memory_samples[0]["rss_mb"]
    abnormal_growth = growth_mb > max(512.0, memory_samples[0]["rss_mb"])
    assertions = {
        "duration_at_least_600_seconds": duration >= SOAK_SECONDS,
        "concurrent_sessions_at_least_25": len(pages) >= SESSION_COUNT,
        "http_interactions_at_least_1000": len(status_codes) >= HTTP_INTERACTIONS,
        "task_and_record_switches_at_least_500": task_switches + record_switches >= UI_SWITCHES,
        "reloads_at_least_100": reloads >= RELOADS,
        "zero_5xx_or_request_failures": failures == 0,
        "zero_console_errors": not console_errors,
        "zero_page_errors": not page_errors,
        "zero_tracebacks": not tracebacks,
        "zero_data_mutation": hashes_before == hashes_after,
        "no_abnormal_memory_growth": not abnormal_growth,
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    sorted_latencies = sorted(http_latencies)
    median_latency = statistics.median(sorted_latencies)
    percentile_index = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * .95))
    peak_memory_mb = max(sample["rss_mb"] for sample in memory_samples)
    report = {
        "audit": "ui_bounded_stress_test", "status": status,
        "started_at_utc": start_utc, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 3), "concurrent_browser_sessions": len(pages),
        "http_page_interactions": len(status_codes), "task_switches": task_switches,
        "record_switches": record_switches, "total_rapid_switches": task_switches + record_switches,
        "reloads": reloads, "http_failures_or_5xx": failures,
        "throughput_interactions_per_second": round(len(status_codes) / duration, 3),
        "error_rate": round(failures / len(status_codes), 8),
        "latency_ms": {
            "p50": round(median_latency * 1000, 3),
            "mean": round(statistics.mean(http_latencies) * 1000, 3),
            "p95": round(sorted_latencies[percentile_index] * 1000, 3),
            "max": round(max(http_latencies) * 1000, 3),
        },
        "console_errors": console_errors, "page_errors": page_errors, "tracebacks": tracebacks,
        "data_hashes_unchanged": hashes_before == hashes_after,
        "memory": {"samples": memory_samples, "peak_rss_mb": round(peak_memory_mb, 3),
                   "growth_mb": round(growth_mb, 3), "abnormal_growth": abnormal_growth},
        "assertions": assertions,
    }
    (UI / "audits/ui_stress_test_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    memory_audit = {
        "audit": "ui_memory_soak", "status": "PASS" if assertions["duration_at_least_600_seconds"] and not abnormal_growth else "FAIL",
        "duration_seconds": round(duration, 3), "server_pid": pid,
        "initial_rss_mb": round(memory_samples[0]["rss_mb"], 3),
        "final_rss_mb": round(memory_samples[-1]["rss_mb"], 3),
        "peak_rss_mb": round(peak_memory_mb, 3),
        "growth_mb": round(growth_mb, 3), "threshold_mb": round(max(512.0, memory_samples[0]["rss_mb"]), 3),
        "samples": memory_samples, "abnormal_growth": abnormal_growth,
    }
    (UI / "audits/ui_memory_soak_audit.json").write_text(json.dumps(memory_audit, indent=2) + "\n", encoding="utf-8")
    if status != "PASS":
        raise AssertionError(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
