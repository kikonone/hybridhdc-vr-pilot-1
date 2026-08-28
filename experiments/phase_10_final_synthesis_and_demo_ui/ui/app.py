"""Single-page local demonstration for the dual-task HDC system."""
from __future__ import annotations

import streamlit as st

from components.data_access import load_model, load_rows, verify_package
from components.sections import classification_demo, regression_demo
from components.theme import apply_theme

st.set_page_config(
    page_title="HDC System Demonstration",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

package_ok, package_message = verify_package()
if not package_ok:
    st.error(f"System data validation failed: {package_message}")
    st.stop()

model = load_model()
rows = load_rows()
row_by_id = {row["demo_id"]: row for row in rows}


def record_label(demo_id: str) -> str:
    return f"Record {demo_id.removeprefix('DEMO-')}"

st.markdown(
    "<header class='hero'>"
    "<h1>HDC Classification and<br><span>Proxy-Regression Demonstration</span></h1>"
    "<p>A local demonstration of classification and proxy-regression results. It does not perform live inference.</p>"
    "</header>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='integrity-rail'>"
    "<div><span>SYSTEM</span><strong>READY</strong></div>"
    "<div><span>RECORDS</span><strong>419 × 2</strong></div>"
    "<div><span>TASKS</span><strong>2</strong></div>"
    "<div><span>MODE</span><strong>LOCAL DEMO</strong></div>"
    "</div>",
    unsafe_allow_html=True,
)

control_left, control_right = st.columns([1, 1.15], gap="large")
with control_left:
    task = st.radio(
        "Demonstration task",
        ["Classification", "Regression"],
        horizontal=True,
        key="task_selector",
    )
with control_right:
    selected_id = st.selectbox(
        "Anonymous record",
        options=list(row_by_id),
        index=0,
        key="record_selector",
        format_func=record_label,
        help="Select a demonstration record. No data are uploaded or inferred.",
    )

st.markdown(
    f"<div class='sync-line'><span>ACTIVE RECORD</span><strong>{record_label(selected_id)}</strong>"
    "<small>The selected record is shared across both tasks.</small></div>",
    unsafe_allow_html=True,
)

selected_row = {**row_by_id[selected_id], "display_id": record_label(selected_id)}
if task == "Classification":
    classification_demo(model, selected_row)
else:
    regression_demo(model, selected_row)

st.markdown(
    "<footer class='demo-footer'><span>LOCAL SYSTEM DEMONSTRATION · NO UPLOADS</span>"
    "<p>This interface shows classification and bounded proxy-regression outputs. "
    "No live inference or training is performed.</p></footer>",
    unsafe_allow_html=True,
)
