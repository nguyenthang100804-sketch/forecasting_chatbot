import json


def process(state):
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) != "tool":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if "forecast_results" in payload:
            return {
                "last_forecast": payload,
                "forecast_results": payload["forecast_results"],
                "target_date": payload["target_date"],
                "discount_pct": payload["discount_pct"],
                "fixed_discount": payload["fixed_discount"],
            }
    return {}
