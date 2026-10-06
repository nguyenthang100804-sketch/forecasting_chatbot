import pandas as pd
import numpy as np

# Các category chuẩn xác lấy từ file training_features_v2.csv
PROMO_SEASON_CATS = ['NO_PROMO', 'Rural Special', 'Spring Sale', 'Mid-Year Sale', 
                     'Urban Blowout', 'Fall Launch|Urban Blowout', 'Fall Launch', 'Year-End Sale']
DISCOUNT_TYPE_CATS = ['none', 'percentage', 'fixed', 'fixed|percentage']

# Lịch các ngày lễ cố định
def get_fixed_holidays(year):
    return [
        f'{year}-01-01', f'{year}-02-14', f'{year}-03-08', f'{year}-04-30', 
        f'{year}-05-01', f'{year}-09-02', f'{year}-10-10', f'{year}-10-20', 
        f'{year}-10-31', f'{year}-11-11', f'{year}-12-12', f'{year}-12-25'
    ]

# Mốc thời gian Tết Âm Lịch (có thể thêm các năm tương lai)
TET_DATES = {
    2023: '2023-01-22', 2024: '2024-02-10', 2025: '2025-01-29', 2026: '2026-02-17', 2027: '2027-02-06'
}
HUNG_KINGS = {2023: '2023-04-29', 2024: '2024-04-18', 2025: '2025-04-07', 2026: '2026-04-26'}
MID_AUTUMN = {2023: '2023-09-29', 2024: '2024-09-17', 2025: '2025-10-06', 2026: '2026-09-25'}

def generate_features(target_date: str, discount_pct: float, fixed_discount: float) -> pd.DataFrame:
    """
    Biến đổi thông tin từ Intent Parser thành 1 dòng DataFrame chứa đúng các cột mà LightGBM yêu cầu.
    """
    date_obj = pd.to_datetime(target_date)
    year = date_obj.year
    
    # 1. Các biến thời gian cơ bản
    doy = date_obj.dayofyear
    is_weekend = 1 if date_obj.dayofweek >= 5 else 0
    
    features: dict[str, object] = {
        'dow': date_obj.dayofweek,
        'dom': date_obj.day,
        'weekofyear': date_obj.isocalendar().week,
        'month': date_obj.month,
        'quarter': date_obj.quarter,
        'doy': doy,
        'is_weekend': is_weekend,
        'is_som': int(date_obj.is_month_start),
        'is_eom': int(date_obj.is_month_end),
    }
    
    # 2. Fourier terms
    for k in range(1, 5):
        features[f'fourier_year_sin_{k}'] = np.sin(2 * np.pi * k * (doy - 1) / 365.25)
        features[f'fourier_year_cos_{k}'] = np.cos(2 * np.pi * k * (doy - 1) / 365.25)
        
    # 3. Thông tin khuyến mãi
    promo_active = 1 if (discount_pct > 0 or fixed_discount > 0) else 0
    features['promo_active'] = promo_active
    features['promo_count_active'] = 1 if promo_active else 0
    features['promo_discount_pct'] = float(discount_pct)
    features['promo_fixed_discount'] = float(fixed_discount)
    
    # Xác định loại giảm giá
    if discount_pct > 0 and fixed_discount > 0:
        features['discount_type'] = 'fixed|percentage'
    elif discount_pct > 0:
        features['discount_type'] = 'percentage'
    elif fixed_discount > 0:
        features['discount_type'] = 'fixed'
    else:
        features['discount_type'] = 'none'
        
    # Gán tên campaign tạm nếu có. Dùng np.nan (biến thiếu) để tránh lỗi Categorical trong Pandas
    # LightGBM hoàn toàn hiểu và xử lý tốt giá trị NaN cho categorical
    features['promo_season'] = np.nan if promo_active else 'NO_PROMO'
    
    # 4. Các biến ngày lễ & Tết
    holidays_str = get_fixed_holidays(year)
    if year in TET_DATES:
        m1 = pd.to_datetime(TET_DATES[year])
        tet_days = [m1, m1 + pd.Timedelta(days=1), m1 + pd.Timedelta(days=2), m1 - pd.Timedelta(days=1)]
        holidays_str.extend([d.strftime('%Y-%m-%d') for d in tet_days])
    if year in HUNG_KINGS: holidays_str.append(HUNG_KINGS[year])
    if year in MID_AUTUMN: holidays_str.append(MID_AUTUMN[year])
    
    features['is_special_day'] = 1 if target_date in holidays_str else 0
    
    # Khoảng thời gian quanh Tết
    tet_before = 0
    tet_after = 0
    days_to_tet = 0
    
    if year in TET_DATES:
        m1 = pd.to_datetime(TET_DATES[year])
        m3 = m1 + pd.Timedelta(days=2)
        before_start = m1 - pd.Timedelta(days=30)
        before_end   = m1 - pd.Timedelta(days=1)
        after_start  = m3 + pd.Timedelta(days=1)
        after_end    = m3 + pd.Timedelta(days=30)
        
        if before_start <= date_obj <= before_end:
            tet_before = 1
            days_to_tet = (m1 - date_obj).days
        elif m1 <= date_obj <= m3:
            days_to_tet = (m1 - date_obj).days
        elif after_start <= date_obj <= after_end:
            tet_after = 1
            days_to_tet = (m1 - date_obj).days
            
    features['tet_before'] = tet_before
    features['tet_after'] = tet_after
    features['days_to_tet'] = days_to_tet
    
    # Đóng gói thành DataFrame 1 dòng
    df = pd.DataFrame([features])
    
    # BẮT BUỘC ÉP KIỂU CATEGORY ĐÚNG NHƯ LÚC TRAIN (Tránh lỗi LightGBM)
    df['promo_season'] = df['promo_season'].astype(pd.CategoricalDtype(categories=PROMO_SEASON_CATS))
    df['discount_type'] = df['discount_type'].astype(pd.CategoricalDtype(categories=DISCOUNT_TYPE_CATS))
    
    # Sắp xếp đúng thứ tự cột như trong file csv
    expected_cols = ['dow', 'dom', 'weekofyear', 'month', 'quarter', 'doy', 'is_weekend', 'is_som', 'is_eom',
                     'fourier_year_sin_1', 'fourier_year_cos_1', 'fourier_year_sin_2', 'fourier_year_cos_2',
                     'fourier_year_sin_3', 'fourier_year_cos_3', 'fourier_year_sin_4', 'fourier_year_cos_4',
                     'promo_active', 'promo_count_active', 'promo_season', 'discount_type', 'promo_discount_pct',
                     'promo_fixed_discount', 'is_special_day', 'tet_before', 'tet_after', 'days_to_tet']
    
    return df[expected_cols]

if __name__ == "__main__":
    # Test thử 
    df_test = generate_features("2026-10-06", discount_pct=20.0, fixed_discount=0.0)
    print("Features sinh ra:")
    print(df_test.T)
