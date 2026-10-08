import os
import pandas as pd
import lightgbm as lgb
from functools import lru_cache
from nodes.state import ForecastState
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("Đang nạp mô hình LightGBM...")
try:
    models = {
        'rev_p1': lgb.Booster(model_file=os.path.join(MODEL_DIR, 'model_rev_phase1.txt')),
        'rev_p2': lgb.Booster(model_file=os.path.join(MODEL_DIR, 'model_rev_phase2.txt')),
        'cogs_p1': lgb.Booster(model_file=os.path.join(MODEL_DIR, 'model_cogs_phase1.txt')),
        'cogs_p2': lgb.Booster(model_file=os.path.join(MODEL_DIR, 'model_cogs_phase2.txt'))
    }
except Exception as e:
    print(f"Lỗi nạp mô hình (Có thể chưa có file model): {e}")
    models = {}

# Import lại logic giải thích SHAP cũ
import sys
sys.path.append(BASE_DIR)
from ml_engine.explainer import run_inference_with_shap

# Cơ chế Caching
@lru_cache(maxsize=100)
def cached_inference(features_json_str: str) -> dict:
    """Caching: Tránh tính toán lại SHAP nếu truyền vào cùng tập feature."""
    features_dict = json.loads(features_json_str)
    df_features = pd.DataFrame([features_dict])
    
    # Ép kiểu Categorical lại vì qua JSON bị mất kiểu
    from nodes.feature_node import PROMO_SEASON_CATS, DISCOUNT_TYPE_CATS
    df_features['promo_season'] = df_features['promo_season'].astype(pd.CategoricalDtype(categories=PROMO_SEASON_CATS))
    df_features['discount_type'] = df_features['discount_type'].astype(pd.CategoricalDtype(categories=DISCOUNT_TYPE_CATS))
    
    return run_inference_with_shap(models, df_features, alpha=0.45)

def process(state: ForecastState):
    """Machine Learning Node."""
    features = state.get("ml_features")
    if not features:
        raise ValueError("Lỗi ML Node: Chưa nhận được features.")
        
    # Serialize dict -> string để dùng lru_cache
    features_json = json.dumps({k: (v if not pd.isna(v) else None) for k, v in features.items()})
    
    results = cached_inference(features_json)
    
    return {"forecast_results": results}
