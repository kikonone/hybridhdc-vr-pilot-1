"""The two audience-facing system demonstration task views."""
from __future__ import annotations

import html

import streamlit as st

from components.charts import classification_scores, regression_scale
from components.theme import detail_grid, model_strip


def _safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def classification_demo(model: dict, row: dict[str, str]) -> None:
    config = model["classification"]
    model_strip([
        ("Classification model", config["model"]),
        ("Hypervector", f"{config['dimension']:,}D"),
        ("Encoding", f"k={config['feature_k']} · {config['levels']} levels"),
        ("Output", "4 difficulty classes"),
    ])
    st.markdown("<div class='task-kicker'>CLASSIFICATION / FOUR-LEVEL DIFFICULTY</div>", unsafe_allow_html=True)
    st.markdown("## Classification result")
    st.caption("Select an anonymous record to view its classification output.")

    detail_grid([
        ("Anonymous record", row["display_id"], "record-id"),
        ("Outer fold", row["fold"], ""),
        ("True class", f"Difficulty {row['true_difficulty']}", ""),
        ("Predicted difficulty", f"Difficulty {row['predicted_difficulty']}", "accent"),
    ])
    correct = row["classification_correct"] == "true"
    status = "CORRECT" if correct else "INCORRECT"
    st.markdown(
        f"<div class='result-banner {'pass' if correct else 'miss'}'>"
        f"<span>CLASSIFICATION RESULT · {status}</span><strong>True Difficulty {_safe(row['true_difficulty'])} · "
        f"Predicted Difficulty {_safe(row['predicted_difficulty'])}</strong></div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(classification_scores(row), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='method-note'><b>Score semantics</b><span>The displayed class scores are cosine similarities, not calibrated probabilities. "
        "They do not sum to one.</span></div>",
        unsafe_allow_html=True,
    )


def regression_demo(model: dict, row: dict[str, str]) -> None:
    config = model["regression"]
    model_strip([
        ("Regression head", config["model"]),
        ("Variant", config["variant"]),
        ("Hypervector", f"{config['dimension']:,}D"),
        ("Readout", f"Ridge α={config['ridge_alpha']:.2f}"),
    ])
    st.markdown("<div class='task-kicker'>PROXY REGRESSION / BOUNDED 1–4 SCALE</div>", unsafe_allow_html=True)
    st.markdown("## Proxy-regression result")
    st.caption("This view presents bounded difficulty-induced workload proxy regression. It is not a direct physiological workload measurement.")

    detail_grid([
        ("Anonymous record", row["display_id"], "record-id"),
        ("Outer fold", row["fold"], ""),
        ("True difficulty score", f"{float(row['true_difficulty_score']):.2f}", ""),
        ("Absolute error", f"{float(row['absolute_error']):.3f}", "accent"),
    ])
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"<div class='prediction-panel'><span>RAW PREDICTION</span>"
            f"<strong>{float(row['raw_frozen_prediction']):.6f}</strong>"
            "<small>Unbounded model output</small></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"<div class='prediction-panel bounded'><span>BOUNDED PREDICTION</span>"
            f"<strong>{float(row['bounded_frozen_prediction']):.6f}</strong>"
            "<small>Output on the 1–4 demonstration scale</small></div>",
            unsafe_allow_html=True,
        )
    st.plotly_chart(regression_scale(row), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"<div class='method-note'><b>Model configuration</b><span>{_safe(config['model'])} · "
        f"dimension {_safe(config['dimension'])} · feature_k {_safe(config['feature_k'])} · "
        f"levels {_safe(config['levels'])} · {_safe(config['ridge_alpha_policy'])}.</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Absolute error is calculated for the selected record and shown for demonstration only.")
