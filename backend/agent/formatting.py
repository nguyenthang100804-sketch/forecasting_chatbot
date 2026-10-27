import json
import re

from agent.types import ForecastPayload

FORECAST_MARKER_PATTERN = re.compile(r"\s*<!--FORECAST_DATA:.*?-->", re.DOTALL)


def _format_money(value: object) -> str:
    return f"{float(value):,.0f} VND"


def _format_factors(items: list[dict]) -> str:
    if not items:
        return "không có dữ liệu nổi bật"
    return "; ".join(
        f"{item['feature']} ({_format_money(item['impact'])})" for item in items
    )


def format_forecast_context(forecast: ForecastPayload) -> str:
    """Format one stored forecast for the language model."""
    results = forecast["forecast_results"]
    revenue = results["revenue"]
    cogs = results["cogs"]
    return "\n".join(
        [
            f"Ngày dự báo: {forecast['target_date']}.",
            f"Giảm giá theo phần trăm: {forecast['discount_pct']}%.",
            f"Giảm giá cố định (đơn vị mô hình): {forecast['fixed_discount']}.",
            f"Doanh thu dự kiến: {_format_money(revenue['prediction'])}.",
            f"Mức nền doanh thu: {_format_money(revenue.get('base_value', revenue['prediction']))}.",
            f"Yếu tố làm tăng doanh thu: {_format_factors(revenue.get('top_drivers', []))}.",
            f"Yếu tố làm giảm doanh thu: {_format_factors(revenue.get('top_barriers', []))}.",
            f"COGS dự kiến: {_format_money(cogs['prediction'])}.",
            f"Mức nền COGS: {_format_money(cogs.get('base_value', cogs['prediction']))}.",
            f"Yếu tố làm tăng COGS: {_format_factors(cogs.get('top_drivers', []))}.",
            f"Yếu tố làm giảm COGS: {_format_factors(cogs.get('top_barriers', []))}.",
        ]
    )


def format_tool_result_for_model(content: object) -> str:
    """Keep only the forecast facts needed for the final natural-language reply."""
    try:
        payload = json.loads(content) if isinstance(content, str) else content
        if not isinstance(payload, dict) or "forecast_results" not in payload:
            return str(content)
        return format_forecast_context(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return str(content)


def strip_forecast_marker(content: str) -> str:
    """Hide the frontend-only chart payload from later model requests."""
    return FORECAST_MARKER_PATTERN.sub("", content).strip()


def format_forecast_history(history: dict[str, list[ForecastPayload]]) -> str:
    if not history:
        return "\nChưa có kết quả dự báo nào trong conversation này.\n"

    entries = [
        format_forecast_context(forecast)
        for forecasts in history.values()
        for forecast in forecasts
    ]
    return (
        "\nCác kết quả dự báo đã có trong conversation này. "
        "Chỉ dùng chúng để giải thích, không gọi lại tool:\n"
        + "\n\n---\n\n".join(entries)
    )


def forecast_marker(forecast: ForecastPayload) -> str:
    return (
        "\n\n<!--FORECAST_DATA:"
        f"{json.dumps(forecast, ensure_ascii=False, separators=(',', ':'))}-->"
    )
