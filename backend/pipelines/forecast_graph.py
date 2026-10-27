"""Deterministic forecast graph.

This module owns only the prediction workflow. The conversational supervisor
and its tools live in ``agent/supervisor.py`` and ``agent/tools.py``.
"""

from datetime import datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from pipelines.nodes.feature_node import process as build_features
from pipelines.nodes.ml_node import process as run_ml
from pipelines.nodes.state import ForecastState
from ml_engine.ensemble import run_explanations


builder = StateGraph(ForecastState)
builder.add_node("feature_node", build_features)
builder.add_node("ml_node", run_ml)
builder.set_entry_point("feature_node")
builder.add_edge("feature_node", "ml_node")
builder.add_edge("ml_node", END)

forecast_graph = builder.compile()


def run_forecast(
    target_date: datetime,
    discount_pct: float = 0.0,
    fixed_discount: float = 0.0,
) -> dict:
    """Run the prediction graph and return its structured forecast result."""
    state = forecast_graph.invoke(
        {
            "target_date": target_date,
            "discount_pct": float(discount_pct),
            "fixed_discount": float(fixed_discount),
        }
    )
    return state["forecast_results"]


def run_explanation(
    target_date: datetime,
    discount_pct: float = 0.0,
    fixed_discount: float = 0.0,
) -> dict:
    """Build features and calculate SHAP for an existing forecast request."""
    feature_state = build_features(
        {
            "target_date": target_date,
            "discount_pct": float(discount_pct),
            "fixed_discount": float(fixed_discount),
        }
    )
    return run_explanations(
        feature_state["ml_features"],
        target_date,
        float(discount_pct),
        float(fixed_discount),
    )