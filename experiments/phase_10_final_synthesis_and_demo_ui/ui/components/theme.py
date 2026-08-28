"""Offline aviation-console visual primitives."""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st


def apply_theme() -> None:
    css = (Path(__file__).resolve().parents[1] / "assets/aviation_console.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def model_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f"<div class='model-cell'><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in items
    )
    st.markdown(f"<div class='model-strip'>{cells}</div>", unsafe_allow_html=True)


def detail_grid(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f"<div class='detail-card {html.escape(tone)}'><span>{html.escape(str(label))}</span>"
        f"<strong>{html.escape(str(value))}</strong></div>"
        for label, value, tone in items
    )
    st.markdown(f"<div class='detail-grid'>{cells}</div>", unsafe_allow_html=True)
