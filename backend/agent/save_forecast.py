import json

from agent.types import ForecastPayload


def process(state):
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) != "tool":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if "forecast_results" in payload and {
            "target_date",
            "discount_pct",
            "fixed_discount",
        }.issubset(payload):
            forecast = ForecastPayload(
                target_date=payload["target_date"],
                discount_pct=float(payload["discount_pct"]),
                fixed_discount=float(payload["fixed_discount"]),
                forecast_results=payload["forecast_results"],
            )
            history = {
                key: list(value)
                for key, value in state.get("forecast_history", {}).items()
            }
            history.setdefault(forecast["target_date"], []).append(forecast)
            return {"forecast_history": history}
    return {}
