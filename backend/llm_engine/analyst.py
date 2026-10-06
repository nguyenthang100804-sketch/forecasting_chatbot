import json
from google import genai

def generate_analysis(intent_data: dict, forecast_results: dict) -> str:
    """
    Nhận kết quả từ Data Pipeline & XAI, biến nó thành đoạn văn bản phân tích
    tự nhiên thông qua LLM.
    """
    client = genai.Client()
    
    # 1. Trích xuất thông tin
    target_date = intent_data['target_date']
    discount = intent_data['discount_pct']
    fixed = intent_data['fixed_discount']
    
    rev_pred = forecast_results['revenue']['prediction']
    rev_drivers = forecast_results['revenue']['top_drivers']
    rev_barriers = forecast_results['revenue']['top_barriers']
    
    cogs_pred = forecast_results['cogs']['prediction']
    
    # Lợi nhuận gộp cơ bản
    gross_profit = rev_pred - cogs_pred
    
    # 2. Xây dựng Prompt cho Chuyên gia
    prompt = f"""
    Bạn là một Chuyên gia phân tích dữ liệu kinh doanh (Data Analyst).
    Dựa vào số liệu dự báo từ AI, hãy viết một đoạn văn ngắn gọn, chuyên nghiệp báo cáo kết quả.

    ==== SỐ LIỆU DỰ BÁO ====
    - Ngày dự kiến: {target_date}
    - Chương trình Khuyến mãi: Giảm {discount}%, Tiền mặt: {fixed} VND
    - Doanh thu dự kiến: {rev_pred:,.0f} VND
    - Giá vốn hàng bán (COGS): {cogs_pred:,.0f} VND
    - Lợi nhuận gộp ước tính: {gross_profit:,.0f} VND
    
    - TOP YẾU TỐ KÉO DOANH THU TĂNG (Động lực):
    {json.dumps(rev_drivers, ensure_ascii=False)}
    
    - TOP YẾU TỐ KÉO DOANH THU GIẢM (Rào cản):
    {json.dumps(rev_barriers, ensure_ascii=False)}

    ==== YÊU CẦU ====
    1. Viết 3-4 câu, giọng điệu chuyên nghiệp, ngắn gọn nhưng đầy đủ ý.
    2. Báo cáo rõ doanh thu, COGS và lợi nhuận gộp ước tính. Các con số nên được định dạng có dấu phẩy phân cách hàng nghìn và không có chữ số thập phân. Không đính kèm đơn vị tiền tệ (VND) trong câu văn.
    3. Giải thích tại sao có kết quả đó dựa vào TOP Động lực và Rào cản, không dùng tên biến để giải thích (Hãy dịch khéo léo tên biến kỹ thuật. VD: 'is_weekend' -> rơi vào cuối tuần, 'days_to_tet' -> cận Tết, 'promo_active' -> chạy khuyến mãi, 'month' -> tính mùa vụ của tháng).
    4. TUYỆT ĐỐI KHÔNG nhắc đến các từ kỹ thuật như "SHAP", "JSON", "Mô hình LightGBM", "Base value". Hãy đóng vai chuyên gia kinh doanh thực thụ.
    """
    
    # 3. Gọi LLM sinh văn bản (NLG)
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt
    )
    
    return response.text or ""
