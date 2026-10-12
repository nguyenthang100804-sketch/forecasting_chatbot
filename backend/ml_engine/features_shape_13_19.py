"""
Tạo feature cho mô hình Shape 13–19 từ MỘT ngày.

    from features_shape_13_19 import build_features
    x_shape = build_features("2023-09-10")      # DataFrame 1 dòng, 25 cột

Cách tính giống hệt notebook gốc (notebooks/shape_13_19.ipynb), chỉ khác:
- Ngày Tết lấy từ thư viện lunardate, nên dùng được cho năm bất kỳ (bản gốc ghi cứng tới 2024).
- Khuyến mãi năm >= 2023: chép lịch của năm cùng chẵn/lẻ (2021 cho năm lẻ, 2022 cho năm chẵn).
  Với 2023–2024 cách này trùng hệt bản gốc ("lịch 2021–2022 dời thêm 2 năm").
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lunardate import LunarDate

PROMO_FILE = Path(__file__).resolve().parent / "promotions.csv"

# Thứ tự cột đúng như lúc train (model.feature_name())
FEATURE_COLUMNS = [
    "days_to_tet", "tet_before", "tet_after", "is_special_day",
    "dow", "dom", "month", "quarter", "doy", "days_in_month", "dom_reverse",
    "is_weekend", "is_eom", "is_som", "is_odd_year",
    "fourier_sin1", "fourier_cos1", "fourier_sin2", "fourier_cos2",
    "fourier_sin3", "fourier_cos3", "fourier_sin4", "fourier_cos4",
    "discount_max", "promo_n",
]

# Ngày lễ bán lẻ theo dương lịch (tháng, ngày)
RETAIL_SPECIAL_DAYS = {(1, 1), (2, 14), (3, 8), (4, 30), (5, 1), (9, 2), (9, 9),
                       (10, 10), (10, 20), (11, 11), (12, 12), (12, 24), (12, 25)}

_promotions = None   # đọc file khuyến mãi một lần rồi giữ lại


# ---------------------------------------------------------------------------
# Tết
# ---------------------------------------------------------------------------
def tet_date(year):
    """Mồng 1 Tết (dương lịch) của một năm."""
    return pd.Timestamp(LunarDate(year, 1, 1).to_solar_date())


def days_to_nearest_tet(date):
    """Số ngày từ mồng 1 Tết gần nhất tới ngày date (âm = trước Tết, 0 = mồng 1).

    Xét Tết của năm trước, năm nay, năm sau; nếu cách đều hai Tết thì lấy Tết sớm hơn
    (giống np.argmin trong bản gốc).
    """
    best = None
    for year in [date.year - 1, date.year, date.year + 1]:
        diff = (date - tet_date(year)).days
        if best is None or abs(diff) < abs(best):
            best = diff
    return best


# ---------------------------------------------------------------------------
# Khuyến mãi
# ---------------------------------------------------------------------------
def load_promotions():
    """Đọc lịch khuyến mãi lịch sử (2013–2022) từ promotions.csv."""
    global _promotions
    if _promotions is None:
        _promotions = pd.read_csv(PROMO_FILE, parse_dates=["start_date", "end_date"])
    return _promotions


def promotions_until(year):
    """Lịch khuyến mãi lịch sử + lịch chiếu cho các năm 2023 → year.

    Năm lẻ chép các chương trình bắt đầu trong 2021, năm chẵn chép 2022,
    dời ngày bắt đầu/kết thúc đúng số năm chênh lệch.
    """
    history = load_promotions()
    parts = [history[["start_date", "end_date", "discount_value"]]]

    for target_year in range(2023, year + 1):
        if target_year % 2 == 1:
            source_year = 2021
        else:
            source_year = 2022
        shift = target_year - source_year

        source = history[history.start_date.dt.year == source_year]
        projected = pd.DataFrame({
            "start_date": source.start_date + pd.DateOffset(years=shift),
            "end_date": source.end_date + pd.DateOffset(years=shift),
            "discount_value": source.discount_value,
        })
        parts.append(projected)

    return pd.concat(parts, ignore_index=True)


def promo_features(date):
    """discount_max = mức giảm lớn nhất đang chạy, promo_n = số chương trình đang chạy (0 nếu không có)."""
    promos = promotions_until(date.year)
    running = promos[(promos.start_date <= date) & (promos.end_date >= date)]
    if len(running) == 0:
        return 0.0, 0.0
    return float(running.discount_value.max()), float(len(running))


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------
def build_features(date):
    """Nhận 1 ngày dạng "YYYY-MM-DD", trả DataFrame 1 dòng gồm 25 feature của mô hình Shape 13–19."""
    date = pd.Timestamp(date)
    row = {}

    # Tết
    days_to_tet = days_to_nearest_tet(date)
    row["days_to_tet"] = days_to_tet
    row["tet_before"] = int(-30 <= days_to_tet <= -1)
    row["tet_after"] = int(0 <= days_to_tet <= 14)
    row["is_special_day"] = int((date.month, date.day) in RETAIL_SPECIAL_DAYS)

    # Lịch
    row["dow"] = date.dayofweek
    row["dom"] = date.day
    row["month"] = date.month
    row["quarter"] = date.quarter
    row["doy"] = date.dayofyear
    row["days_in_month"] = date.days_in_month
    row["dom_reverse"] = date.days_in_month - date.day
    row["is_weekend"] = int(date.dayofweek >= 5)
    row["is_eom"] = int(row["dom_reverse"] <= 3)
    row["is_som"] = int(date.day <= 2)
    row["is_odd_year"] = date.year % 2
    for k in [1, 2, 3, 4]:
        angle = 2 * np.pi * k * date.dayofyear / 365.25
        row[f"fourier_sin{k}"] = np.sin(angle)
        row[f"fourier_cos{k}"] = np.cos(angle)

    # Khuyến mãi
    row["discount_max"], row["promo_n"] = promo_features(date)

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
