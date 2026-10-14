"""Deterministic forecast graph.

This module owns only the prediction workflow. The conversational supervisor
and its tools live in ``supervisor.py`` and ``agent_tools.py``.
"""

from datetime import datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from pipelines.nodes.feature_node import process as build_features
from pipelines.nodes.ml_node import process as run_ml
from pipelines.nodes.state import ForecastState


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

# def export_graph_image(output_dir: Path | None = None) -> None:
#     """Vẽ forecast_graph và lưu ảnh PNG + mã Mermaid vào output_dir."""
#     output_dir = output_dir or Path.cwd()          # thư mục đang đứng khi chạy lệnh
#     drawable = forecast_graph.get_graph()

#     # 1. Mã Mermaid dạng text: không cần mạng, luôn thành công
#     mmd_path = output_dir / "forecast_graph.mmd"
#     mmd_path.write_text(drawable.draw_mermaid(), encoding="utf-8")
#     print(f"Đã lưu mã Mermaid: {mmd_path}")

#     # 2. Ảnh PNG: mặc định gọi dịch vụ mermaid.ink nên cần internet
#     try:
#         png_path = output_dir / "forecast_graph.png"
#         png_path.write_bytes(drawable.draw_mermaid_png())
#         print(f"Đã lưu ảnh: {png_path}")
#     except Exception as exc:
#         print(f"Không vẽ được PNG (kiểm tra mạng): {exc}")


# if __name__ == "__main__":
#     export_graph_image()