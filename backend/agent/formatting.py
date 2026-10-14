import json

from agent.types import ForecastPayload


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
            f"Mức nền doanh thu: {_format_money(revenue['base_value'])}.",
            f"Yếu tố làm tăng doanh thu: {_format_factors(revenue['top_drivers'])}.",
            f"Yếu tố làm giảm doanh thu: {_format_factors(revenue['top_barriers'])}.",
            f"COGS dự kiến: {_format_money(cogs['prediction'])}.",
            f"Mức nền COGS: {_format_money(cogs['base_value'])}.",
            f"Yếu tố làm tăng COGS: {_format_factors(cogs['top_drivers'])}.",
            f"Yếu tố làm giảm COGS: {_format_factors(cogs['top_barriers'])}.",
        ]
    )


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
