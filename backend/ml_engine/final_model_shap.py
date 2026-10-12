"""
Dự báo + SHAP của MÔ HÌNH CUỐI cho một ngày.

    from final_model_shap import explain_final
    res = explain_final(p1, p2, m2, x_two_period, "2023-09-10", "Revenue")

Mô hình cuối:
    TwoPeriod = 0.45 · expm1(P1) + 0.55 · expm1(P2)
    Shape     = 0.80 · expm1(M2)
    Final     = 0.5 · TwoPeriod + 0.5 · Shape

Kết quả trả về là tiền (cùng đơn vị với dữ liệu gốc), và luôn đúng:
    base_value + tổng money = prediction
"""

import numpy as np
import pandas as pd

from .features_shape_13_19 import build_features

# ---------------------------------------------------------------------------
# Hằng số của mô hình cuối
# ---------------------------------------------------------------------------
W = 0.5                               # Final = W · TwoPeriod + (1 − W) · Shape
A_P1 = 0.45                           # TwoPeriod = A_P1 · P1 + (1 − A_P1) · P2
K = {"Revenue": 0.80, "COGS": 0.80}   # hệ số mức của Shape

# Cửa sổ Tết của Two-Period, tính theo days_to_tet của Shape (ngày − mồng 1 Tết).
# Trong cửa sổ: days_to_tet của Shape thuộc nhóm Tet; ngoài cửa sổ: thuộc nhóm Lunar calendar.
TET_WINDOW = (-30, 32)
LUNAR_FEATURE = "days_to_tet (ngoài cửa sổ Tết)"

# Đổi tên để cùng một khái niệm ở hai mô hình có cùng tên (SHAP cùng tên sẽ được cộng)
RENAME_TWO_PERIOD = {"promo_count_active": "promo_count"}
RENAME_SHAPE = {"promo_n": "promo_count"}
for _k in [1, 2, 3, 4]:
    RENAME_TWO_PERIOD[f"fourier_year_sin_{_k}"] = f"fourier_sin_{_k}"
    RENAME_TWO_PERIOD[f"fourier_year_cos_{_k}"] = f"fourier_cos_{_k}"
    RENAME_SHAPE[f"fourier_sin{_k}"] = f"fourier_sin_{_k}"
    RENAME_SHAPE[f"fourier_cos{_k}"] = f"fourier_cos_{_k}"

# 8 nhóm, theo tên sau khi đổi
GROUPS = {
    "Seasonality": ["fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2",
                    "fourier_sin_3", "fourier_cos_3", "fourier_sin_4", "fourier_cos_4",
                    "month", "doy", "quarter", "weekofyear"],
    "Lunar calendar": [LUNAR_FEATURE],
    "Tet": ["days_to_tet", "tet_before", "tet_after"],
    "Edge-of-month": ["dom", "is_som", "is_eom", "dom_reverse", "days_in_month"],
    "Weekly": ["dow", "is_weekend"],
    "Promotion": ["promo_active", "promo_count", "promo_season", "discount_type",
                  "promo_discount_pct", "promo_fixed_discount", "discount_max"],
    "Special day": ["is_special_day"],
    "Year-parity": ["is_odd_year"],
}


def group_of(feature):
    for group, features in GROUPS.items():
        if feature in features:
            return group
    return "Other"


# ---------------------------------------------------------------------------
# Các bước
# ---------------------------------------------------------------------------
def shap_one_row(booster, x):
    """SHAP (không gian log) của một booster cho 1 dòng.

    Trả: (dict feature → SHAP, base value log, dự báo log). base + tổng SHAP = dự báo log.
    """
    columns = booster.feature_name()
    contrib = booster.predict(x[columns], pred_contrib=True)[0]
    shap_values = {}
    for i, feature in enumerate(columns):
        shap_values[feature] = float(contrib[i])
    base = float(contrib[-1])
    log_pred = float(booster.predict(x[columns])[0])
    assert np.isclose(base + sum(shap_values.values()), log_pred), "SHAP không cộng tính"
    return shap_values, base, log_pred


def rename(shap_values, mapping):
    out = {}
    for feature, value in shap_values.items():
        out[mapping.get(feature, feature)] = value
    return out


def feature_value(feature, x_two_period, x_shape):
    """Giá trị của feature ngày đó (để hiển thị). Ưu tiên bảng Shape, không có thì lấy bảng Two-Period."""
    if feature == LUNAR_FEATURE:
        feature = "days_to_tet"
    for x, mapping in [(x_shape, RENAME_SHAPE), (x_two_period, RENAME_TWO_PERIOD)]:
        for original, new in mapping.items():
            if new == feature and original in x.columns:
                return to_python(x[original].iloc[0])
        if feature in x.columns:
            return to_python(x[feature].iloc[0])
    return None


def to_python(value):
    """Đổi giá trị numpy / category sang kiểu Python để dump JSON được."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, str) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------
def explain_final(p1, p2, m2, x_two_period, date, target, x_shape=None):
    """Dự báo và SHAP của mô hình cuối cho một ngày.

    Nhận:
        p1, p2        : 2 booster Two-Period (phase 1, phase 2) của target
        m2            : booster Shape 13–19 của target (models/revenue_model_shape_13_19.txt hoặc cogs_model_shape_13_19.txt)
        x_two_period  : DataFrame 1 dòng, feature Two-Period của ngày đó (giống cái đưa vào p1.predict)
        date          : "YYYY-MM-DD"
        target        : "Revenue" hoặc "COGS"
    Trả: dict
        prediction    : dự báo mô hình cuối
        base_value    : mức nền
        contributions : [{feature, group, value, money}], sắp theo |money| giảm dần
        groups        : [{group, money}], sắp theo |money| giảm dần
        (base_value + tổng money = prediction)
    """
    if x_shape is None:
        x_shape = build_features(date)

    # 1. SHAP của từng mô hình con (không gian log)
    shap_p1, base_p1, log_p1 = shap_one_row(p1, x_two_period)
    shap_p2, base_p2, log_p2 = shap_one_row(p2, x_two_period)
    shap_m2, base_m2, log_m2 = shap_one_row(m2, x_shape)

    # 2. Two-Period = 0.45 · P1 + 0.55 · P2 (trên SHAP)
    shap_two_period = {}
    for feature in shap_p1:
        shap_two_period[feature] = A_P1 * shap_p1[feature] + (1 - A_P1) * shap_p2[feature]
    base_two_period = A_P1 * base_p1 + (1 - A_P1) * base_p2
    log_two_period = A_P1 * log_p1 + (1 - A_P1) * log_p2

    # 3. Shape: days_to_tet ngoài cửa sổ Tết chuyển sang cột Lunar calendar
    days_to_tet = int(x_shape["days_to_tet"].iloc[0])
    if not (TET_WINDOW[0] <= days_to_tet <= TET_WINDOW[1]):
        shap_m2[LUNAR_FEATURE] = shap_m2.pop("days_to_tet")

    # 4. Đổi tên rồi ghép: 0.5 · Two-Period + 0.5 · Shape (cùng tên thì cộng)
    shap_two_period = rename(shap_two_period, RENAME_TWO_PERIOD)
    shap_m2 = rename(shap_m2, RENAME_SHAPE)
    shap_final = {}
    for feature, value in shap_two_period.items():
        shap_final[feature] = shap_final.get(feature, 0.0) + W * value
    for feature, value in shap_m2.items():
        shap_final[feature] = shap_final.get(feature, 0.0) + (1 - W) * value

    # Hệ số K của Shape đưa vào mức nền
    base_log = W * base_two_period + (1 - W) * base_m2 + (1 - W) * np.log(K[target])
    pred_log = W * log_two_period + (1 - W) * log_m2 + (1 - W) * np.log(K[target])
    total_shap = sum(shap_final.values())
    assert np.isclose(base_log + total_shap, pred_log), "SHAP mô hình cuối không cộng tính"

    # 5. Dự báo thật của mô hình cuối
    two_period = A_P1 * np.expm1(log_p1) + (1 - A_P1) * np.expm1(log_p2)
    shape = K[target] * np.expm1(log_m2)
    prediction = float(W * two_period + (1 - W) * shape)

    # 6. Quy SHAP ra tiền: cùng một hệ số cho mọi feature, sao cho nền + tổng = dự báo thật
    base_value = float(np.expm1(base_log))
    if abs(total_shap) < 1e-9:
        scale = 0.0
        base_value = prediction
    else:
        scale = (prediction - base_value) / total_shap

    contributions = []
    group_money = {}
    for feature, value in shap_final.items():
        money = value * scale
        group = group_of(feature)
        contributions.append({"feature": feature, "group": group,
                              "value": feature_value(feature, x_two_period, x_shape),
                              "money": money})
        group_money[group] = group_money.get(group, 0.0) + money
    contributions.sort(key=lambda item: abs(item["money"]), reverse=True)

    groups = []
    for group, money in group_money.items():
        groups.append({"group": group, "money": money})
    groups.sort(key=lambda item: abs(item["money"]), reverse=True)

    total_money = 0.0
    for item in contributions:
        total_money += item["money"]
    assert np.isclose(base_value + total_money, prediction), "Nền + đóng góp khác dự báo"

    return {"prediction": prediction, "base_value": base_value,
            "contributions": contributions, "groups": groups}
