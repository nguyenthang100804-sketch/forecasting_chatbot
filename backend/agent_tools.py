import json
from datetime import datetime

from langchain_core.tools import tool

from graph import run_forecast
def _normalize_date(value: str) -> str:
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(
    "target_date phải có dạng YYYY-MM-DD, DD/MM/YYYY hoặc DD-MM-YYYY."
    )


@tool
def predict_revenue(
    target_date: str,
    discount_pct: float = 0.0,
    fixed_discount: float = 0.0,
) -> str:
    """Dự đoán doanh thu và COGS cho một ngày cụ thể.

    Chỉ gọi tool này khi người dùng thực sự yêu cầu dự đoán và đã cung cấp ngày.
    discount_pct là phần trăm giảm giá; fixed_discount là mức giảm cố định bằng VND.
    """
    normalized_date = _normalize_date(target_date)
    if discount_pct < 0 or fixed_discount < 0:
        raise ValueError("Mức giảm giá không được âm.")

    forecast_results = run_forecast(
        normalized_date,
        float(discount_pct),
        float(fixed_discount),
    )
    result = {
        "target_date": normalized_date,
        "discount_pct": float(discount_pct),
        "fixed_discount": float(fixed_discount),
        "forecast_results": forecast_results,
    }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


TOOLS = [predict_revenue]
