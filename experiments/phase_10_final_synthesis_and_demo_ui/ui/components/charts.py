"""Compact Plotly views for the system demonstration."""
from __future__ import annotations

import plotly.graph_objects as go

PANEL = "#0d171e"
GRID = "#263944"
TEXT = "#dce8ed"
CYAN = "#42d7df"
BLUE = "#3c86c7"
AMBER = "#f2b84b"
MUTED = "#718691"


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=46, r=24, t=58, b=42),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="Bahnschrift, Aptos, Segoe UI, sans-serif", color=TEXT, size=13),
        hoverlabel=dict(bgcolor="#152630", bordercolor=CYAN, font_color="#ffffff"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def classification_scores(row: dict[str, str]) -> go.Figure:
    values = [float(row[f"difficulty_{index}_cosine"]) for index in range(1, 5)]
    predicted = int(row["predicted_difficulty"])
    fig = go.Figure(go.Bar(
        x=[f"Difficulty {index}" for index in range(1, 5)],
        y=values,
        marker=dict(color=[CYAN if index == predicted else BLUE for index in range(1, 5)], line_width=0),
        text=[f"{value:.4f}" for value in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Cosine similarity: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(title=dict(text="CLASS SIMILARITY PROFILE", font=dict(size=14), x=0.025))
    fig.update_yaxes(title="Cosine similarity", range=[min(0.0, min(values) - 0.06), max(values) + 0.1])
    return _layout(fig, 335)


def regression_scale(row: dict[str, str]) -> go.Figure:
    target = float(row["true_difficulty_score"])
    bounded = float(row["bounded_frozen_prediction"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1, 4], y=[0, 0], mode="lines",
        line=dict(color=GRID, width=12), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[target], y=[0.09], mode="markers+text", name="True score",
        marker=dict(color=AMBER, size=18, symbol="diamond", line=dict(color=PANEL, width=2)),
        text=[f"TRUE  {target:.2f}"], textposition="top center",
        hovertemplate="True difficulty score: %{x:.6f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[bounded], y=[-0.09], mode="markers+text", name="Bounded prediction",
        marker=dict(color=CYAN, size=18, symbol="circle", line=dict(color=PANEL, width=2)),
        text=[f"BOUNDED  {bounded:.3f}"], textposition="bottom center",
        hovertemplate="Bounded prediction: %{x:.6f}<extra></extra>",
    ))
    fig.update_layout(title=dict(text="BOUNDED PROXY SCALE", font=dict(size=14), x=0.025))
    fig.update_xaxes(range=[0.85, 4.15], tickmode="array", tickvals=[1, 2, 3, 4],
                     ticktext=["Difficulty 1", "Difficulty 2", "Difficulty 3", "Difficulty 4"])
    fig.update_yaxes(range=[-0.33, 0.33], visible=False)
    return _layout(fig, 285)
