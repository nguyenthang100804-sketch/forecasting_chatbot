from typing import Any, TypedDict


class ForecastPayload(TypedDict):
    target_date: str
    discount_pct: float
    fixed_discount: float
    forecast_results: dict[str, Any]

