import pandas as pd
from pipelines.nodes.state import ForecastState
from ml_engine.ensemble import run_predictions

def process(state: ForecastState):
    """Machine Learning Node."""
    features = state.get("ml_features")
    if not features:
        raise ValueError("Lỗi ML Node: Chưa nhận được features.")
        
    results = run_predictions(
        features,
        state.get("target_date"),
        state.get("discount_pct", 0.0),
        state.get("fixed_discount", 0.0),
    )
    
    return {"forecast_results": results}
