import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine.feature_generator import generate_features

def get_expected_promo(date_obj):
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    
    # Check Year-End Sale (Nov 18 to Jan 2)
    if (month == 11 and day >= 18) or (month == 12) or (month == 1 and day <= 2):
        return {"promo_season": "Year-End Sale", "promo_discount_pct": 20.0, "promo_fixed_discount": 0.0, "discount_type": "percentage"}
        
    # Check Spring Sale (Mar 18 to Apr 17)
    if (month == 3 and day >= 18) or (month == 4 and day <= 17):
        return {"promo_season": "Spring Sale", "promo_discount_pct": 12.0, "promo_fixed_discount": 0.0, "discount_type": "percentage"}
        
    # Check Mid-Year Sale (Jun 23 to Jul 22)
    if (month == 6 and day >= 23) or (month == 7 and day <= 22):
        return {"promo_season": "Mid-Year Sale", "promo_discount_pct": 18.0, "promo_fixed_discount": 0.0, "discount_type": "percentage"}
        
    # Check Odd-year Promos (Rural Special & Urban Blowout)
    if year % 2 != 0:
        # Rural Special (Jan 30 to Mar 1)
        if (month == 1 and day >= 30) or (month == 2) or (month == 3 and day == 1):
            return {"promo_season": "Rural Special", "promo_discount_pct": 15.0, "promo_fixed_discount": 0.0, "discount_type": "percentage"}
            
        # Overlap: Fall Launch starts Aug 30, Urban Blowout ends Sep 2
        is_fall_launch = (month == 8 and day >= 30) or (month == 9) or (month == 10 and day <= 1)
        is_urban_blowout = (month == 7 and day >= 30) or (month == 8) or (month == 9 and day <= 2)
        
        if is_fall_launch and is_urban_blowout:
            return {"promo_season": "Fall Launch|Urban Blowout", "promo_discount_pct": 10.0, "promo_fixed_discount": 50.0, "discount_type": "fixed|percentage"}
        elif is_urban_blowout:
            return {"promo_season": "Urban Blowout", "promo_discount_pct": 0.0, "promo_fixed_discount": 50.0, "discount_type": "fixed"}
        elif is_fall_launch:
            return {"promo_season": "Fall Launch", "promo_discount_pct": 10.0, "promo_fixed_discount": 0.0, "discount_type": "percentage"}
            
    # Check Fall Launch (Even years - No overlap)
    if (month == 8 and day >= 30) or (month == 9) or (month == 10 and day <= 1):
        return {"promo_season": "Fall Launch", "promo_discount_pct": 10.0, "promo_fixed_discount": 0.0, "discount_type": "percentage"}
        
    return {"promo_season": "NO_PROMO", "promo_discount_pct": 0.0, "promo_fixed_discount": 0.0, "discount_type": "none"}


def generate_store(start_date="2023-01-01", end_date="2030-12-31", output_path="calendar_features.parquet"):
    print(f"Đang sinh dữ liệu lịch từ {start_date} đến {end_date} bao gồm quy luật Promotion...")
    dates = pd.date_range(start=start_date, end=end_date)
    
    records = []
    for d in dates:
        d_str = d.strftime("%Y-%m-%d")
        
        # 1. Lấy thông tin calendar cơ bản (từ hàm cũ)
        df_row = generate_features(d_str, 0.0, 0.0)
        row_dict = df_row.iloc[0].to_dict()
        row_dict['target_date'] = d_str
        
        # 2. Bơm thông tin Promotion theo quy luật
        promo_info = get_expected_promo(d)
        row_dict.update(promo_info)
        
        # Cập nhật cờ active
        if promo_info['promo_season'] != "NO_PROMO":
            row_dict['promo_active'] = 1
            row_dict['promo_count_active'] = 1
        else:
            row_dict['promo_active'] = 0
            row_dict['promo_count_active'] = 0
            
        records.append(row_dict)
        
    df_store = pd.DataFrame(records)
    
    # Không drop các cột promo nữa, giữ lại để nạp vào cache
    df_store.to_parquet(output_path, index=False)
    print(f"Đã lưu thành công bảng feature tĩnh (có Promotion) tại {output_path} (Dung lượng: {os.path.getsize(output_path) / 1024:.2f} KB)")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(BASE_DIR, "calendar_features.parquet")
    generate_store(output_path=out_path)
