import json
import os
from functools import lru_cache

import lightgbm as lgb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from ml_engine.features_shape_13_19 import build_features as build_shape_features
from ml_engine.final_model_shap import explain_final
from nodes.feature_node import DISCOUNT_TYPE_CATS, PROMO_SEASON_CATS

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODELS = {
    "revenue": (
        lgb.Booster(model_file=os.path.join(MODEL_DIR, "model_rev_phase1.txt")),
        lgb.Booster(model_file=os.path.join(MODEL_DIR, "model_rev_phase2.txt")),
        lgb.Booster(
            model_file=os.path.join(MODEL_DIR, "revenue_model_shape_13_19.txt")
        ),
    ),
    "cogs": (
        lgb.Booster(model_file=os.path.join(MODEL_DIR, "model_cogs_phase1.txt")),
        lgb.Booster(model_file=os.path.join(MODEL_DIR, "model_cogs_phase2.txt")),
        lgb.Booster(
            model_file=os.path.join(MODEL_DIR, "cogs_model_shape_13_19.txt")
        ),
    ),
}


def _two_period_frame(features):
    frame = pd.DataFrame([features])
    frame["promo_season"] = frame["promo_season"].astype(
        pd.CategoricalDtype(categories=PROMO_SEASON_CATS)
    )
    frame["discount_type"] = frame["discount_type"].astype(
        pd.CategoricalDtype(categories=DISCOUNT_TYPE_CATS)
    )
    return frame


def _shape_frame(target_date, discount_pct, fixed_discount):
    frame = build_shape_features(target_date)
    if discount_pct > 0 or fixed_discount > 0:
        frame.loc[:, "discount_max"] = max(discount_pct, fixed_discount)
        frame.loc[:, "promo_n"] = 1.0
    return frame


def _top_factors(contributions):
    drivers = [
        {"feature": item["feature"], "impact": float(item["money"])}
        for item in contributions
        if item["money"] > 0
    ][:3]
    barriers = [
        {"feature": item["feature"], "impact": float(item["money"])}
        for item in sorted(contributions, key=lambda item: item["money"])
        if item["money"] < 0
    ][:2]
    return drivers, barriers


@lru_cache(maxsize=100)
def _cached_ensemble(features_json, target_date, discount_pct, fixed_discount):
    features = json.loads(features_json)
    two_period = _two_period_frame(features)
    shape = _shape_frame(target_date, discount_pct, fixed_discount)
    output = {}

    for target, model_set in MODELS.items():
        result = explain_final(
            *model_set,
            two_period,
            target_date,
            "Revenue" if target == "revenue" else "COGS",
            x_shape=shape,
        )
        drivers, barriers = _top_factors(result["contributions"])
        output[target] = {
            **result,
            "top_drivers": drivers,
            "top_barriers": barriers,
        }
    return output


def run_ensemble(features, target_date, discount_pct=0.0, fixed_discount=0.0):
    """Chạy ensemble 0.5 Two-Period + 0.5 Shape và SHAP nhóm của handoff."""
    serializable = {
        key: (None if pd.isna(value) else value) for key, value in features.items()
    }
    return _cached_ensemble(
        json.dumps(serializable, sort_keys=True, default=str),
        target_date,
        float(discount_pct),
        float(fixed_discount),
    )
