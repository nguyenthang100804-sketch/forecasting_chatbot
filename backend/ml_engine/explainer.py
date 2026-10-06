import numpy as np
import pandas as pd

def _transform_shap_to_original(shap_log, base_value_log):
    """
    Chuyển đổi SHAP values từ không gian Log (do lúc train dùng np.log1p) sang không gian thực (VND).
    Sử dụng phương pháp phân bổ tỷ lệ (Proportional Allocation).
    """
    pred_log = base_value_log + np.sum(shap_log)
    pred_orig = np.expm1(pred_log)
    base_orig = np.expm1(base_value_log)
    
    total_diff_orig = pred_orig - base_orig
    total_diff_log = np.sum(shap_log)
    
    if total_diff_log == 0 or np.isnan(total_diff_log):
        return np.zeros_like(shap_log), base_orig, pred_orig
        
    weight = total_diff_orig / total_diff_log
    shap_orig = shap_log * weight
    
    return shap_orig, base_orig, pred_orig

def _extract_top_factors(feature_names, shap_values, top_pos=3, top_neg=2):
    """Lọc ra Top các yếu tố ảnh hưởng Tích cực và Tiêu cực nhất"""
    factors = []
    for name, val in zip(feature_names, shap_values):
        if abs(val) > 1.0:  # Bỏ qua các yếu tố ảnh hưởng quá bé (< 1 VND)
            factors.append({"feature": name, "impact": float(val)})
            
    # Sắp xếp giảm dần theo giá trị
    factors = sorted(factors, key=lambda x: x["impact"], reverse=True)
    
    drivers = [f for f in factors if f["impact"] > 0][:top_pos]
    barriers = [f for f in factors if f["impact"] < 0][::-1][:top_neg] # Đảo ngược để lấy âm nhất
    
    return drivers, barriers

def run_inference_with_shap(models_dict, df_features, alpha=0.45):
    """
    Chạy dự đoán trên 4 mô hình, tính toán SHAP, blending (alpha), 
    và trả về format JSON chuẩn bị sẵn cho LLM Chuyên gia.
    """
    feature_names = list(df_features.columns)
    
    # ==========================
    # 1. DỰ ĐOÁN REVENUE
    # ==========================
    # Phase 1
    shap_mat_r1 = models_dict['rev_p1'].predict(df_features, pred_contrib=True)[0]
    shap_r1, base_r1, pred_r1 = _transform_shap_to_original(shap_mat_r1[:-1], shap_mat_r1[-1])
    
    # Phase 2
    shap_mat_r2 = models_dict['rev_p2'].predict(df_features, pred_contrib=True)[0]
    shap_r2, base_r2, pred_r2 = _transform_shap_to_original(shap_mat_r2[:-1], shap_mat_r2[-1])
    
    # Blending Revenue
    final_rev_pred = max(alpha * pred_r1 + (1 - alpha) * pred_r2, 0)
    final_rev_base = alpha * base_r1 + (1 - alpha) * base_r2
    final_rev_shap = alpha * shap_r1 + (1 - alpha) * shap_r2
    
    rev_drivers, rev_barriers = _extract_top_factors(feature_names, final_rev_shap)

    # ==========================
    # 2. DỰ ĐOÁN COGS
    # ==========================
    # Phase 1
    shap_mat_c1 = models_dict['cogs_p1'].predict(df_features, pred_contrib=True)[0]
    shap_c1, base_c1, pred_c1 = _transform_shap_to_original(shap_mat_c1[:-1], shap_mat_c1[-1])
    
    # Phase 2
    shap_mat_c2 = models_dict['cogs_p2'].predict(df_features, pred_contrib=True)[0]
    shap_c2, base_c2, pred_c2 = _transform_shap_to_original(shap_mat_c2[:-1], shap_mat_c2[-1])
    
    # Blending COGS
    final_cogs_pred = max(alpha * pred_c1 + (1 - alpha) * pred_c2, 0)
    final_cogs_base = alpha * base_c1 + (1 - alpha) * base_c2
    final_cogs_shap = alpha * shap_c1 + (1 - alpha) * shap_c2
    
    cogs_drivers, cogs_barriers = _extract_top_factors(feature_names, final_cogs_shap)
    
    # ==========================
    # 3. KẾT XUẤT JSON
    # ==========================
    return {
        "revenue": {
            "prediction": float(final_rev_pred),
            "base_value": float(final_rev_base),
            "top_drivers": rev_drivers,
            "top_barriers": rev_barriers
        },
        "cogs": {
            "prediction": float(final_cogs_pred),
            "base_value": float(final_cogs_base),
            "top_drivers": cogs_drivers,
            "top_barriers": cogs_barriers
        }
    }
