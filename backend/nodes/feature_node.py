import os
import pandas as pd
import numpy as np
from nodes.state import ForecastState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_PATH = os.path.join(BASE_DIR, "calendar_features.parquet")

if os.path.exists(STORE_PATH):
    feature_store_df = pd.read_parquet(STORE_PATH)
    feature_store_df.set_index('target_date', inplace=True)
else:
    feature_store_df = None
    print(f"WARNING: Không tìm thấy {STORE_PATH}. Vui lòng chạy scripts/generate_feature_store.py")

PROMO_SEASON_CATS = ['NO_PROMO', 'Rural Special', 'Spring Sale', 'Mid-Year Sale', 
                     'Urban Blowout', 'Fall Launch|Urban Blowout', 'Fall Launch', 'Year-End Sale']
DISCOUNT_TYPE_CATS = ['none', 'percentage', 'fixed', 'fixed|percentage']

def process(state: ForecastState):
    """Feature Node: Tra cứu và cho phép User chép đè Promotion nếu có."""
    target_date = state.get("target_date")
    user_discount_pct = state.get("discount_pct", 0.0)
    user_fixed_discount = state.get("fixed_discount", 0.0)
    
    if not target_date or feature_store_df is None:
        raise ValueError("Lỗi: Không có target_date hoặc Feature Store chưa được khởi tạo.")
        
    # 1. Tra cứu O(1) để lấy Base Features (ĐÃ BAO GỒM Default Promo nếu trùng lịch)
    if target_date in feature_store_df.index:
        base_features = feature_store_df.loc[target_date].to_dict()
    else:
        raise ValueError(f"Ngày {target_date} không có trong cơ sở dữ liệu.")
        
    # 2. Xử lý Chép đè (Override):
    # Nếu người dùng có chủ đích nhập % giảm giá hoặc fixed discount -> Chép đè lịch mặc định!
    if user_discount_pct > 0 or user_fixed_discount > 0:
        base_features['promo_active'] = 1
        base_features['promo_count_active'] = 1
        base_features['promo_discount_pct'] = float(user_discount_pct)
        base_features['promo_fixed_discount'] = float(user_fixed_discount)
        
        # Vì người dùng tạo Promo tùy chỉnh, ta không biết tên Season là gì -> ép về NaN
        base_features['promo_season'] = np.nan
        
        if user_discount_pct > 0 and user_fixed_discount > 0:
            base_features['discount_type'] = 'fixed|percentage'
        elif user_discount_pct > 0:
            base_features['discount_type'] = 'percentage'
        else:
            base_features['discount_type'] = 'fixed'
            
    # Nếu người dùng không nhập gì, ta sẽ DÙNG NGUYÊN BẢN CÁC CỘT PROMO ĐÃ LƯU TRONG PARQUET.
    
    # 3. Ép kiểu chuẩn cho LightGBM
    df = pd.DataFrame([base_features])
    df['promo_season'] = df['promo_season'].astype(pd.CategoricalDtype(categories=PROMO_SEASON_CATS))
    df['discount_type'] = df['discount_type'].astype(pd.CategoricalDtype(categories=DISCOUNT_TYPE_CATS))
    
    expected_cols = ['dow', 'dom', 'weekofyear', 'month', 'quarter', 'doy', 'is_weekend', 'is_som', 'is_eom',
                     'fourier_year_sin_1', 'fourier_year_cos_1', 'fourier_year_sin_2', 'fourier_year_cos_2',
                     'fourier_year_sin_3', 'fourier_year_cos_3', 'fourier_year_sin_4', 'fourier_year_cos_4',
                     'promo_active', 'promo_count_active', 'promo_season', 'discount_type', 'promo_discount_pct',
                     'promo_fixed_discount', 'is_special_day', 'tet_before', 'tet_after', 'days_to_tet']
                     
    final_dict = df[expected_cols].iloc[0].to_dict()
    
    return {"ml_features": final_dict}
