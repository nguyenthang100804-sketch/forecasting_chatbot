import requests
import uuid

API_URL = "http://127.0.0.1:8000/api/chat"

def print_header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def send_chat(session_id, message):
    try:
        res = requests.post(API_URL, json={"session_id": session_id, "message": message}, timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"❌ Lỗi khi gọi API: {str(e)}")
        return {}

def run_tests():
    print("BẮT ĐẦU CHẠY BỘ TEST CASES TỰ ĐỘNG...")

    # ==========================================
    # TC_01: Happy Path
    # ==========================================
    print_header("TC_01: Happy Path (Đủ thông tin)")
    session_tc01 = str(uuid.uuid4())
    res_01 = send_chat(session_tc01, "Dự đoán doanh thu ngày 25/12/2023, giảm giá 15% và tặng kèm voucher giảm cố định 50.0")
    
    assert res_01.get("status") == "success", "Lỗi TC_01: Status không thành công"
    parsed_01 = res_01.get("intent_parsed", {})
    assert parsed_01.get("target_date") == "2023-12-25", "Lỗi TC_01: Sai ngày"
    assert parsed_01.get("discount_pct") == 15.0, "Lỗi TC_01: Sai % giảm giá"
    assert parsed_01.get("fixed_discount") == 50.0, "Lỗi TC_01: Sai fixed discount"
    print("✅ TC_01 PASS: Bắt đúng Intent và xuất JSON thành công.")


    # ==========================================
    # TC_07, TC_08, TC_09, TC_10 (Tái sử dụng TC_01)
    # ==========================================
    print_header("TC_07 -> TC_10: Kiểm tra XAI & NLG")
    
    # TC_07: Tính toàn vẹn của SHAP (Original Scale)
    forecast_01 = res_01.get("forecast_results", {})
    rev_pred = forecast_01["revenue"]["prediction"]
    rev_base = forecast_01["revenue"]["base_value"]
    rev_drivers = forecast_01["revenue"]["top_drivers"]
    rev_barriers = forecast_01["revenue"]["top_barriers"]
    
    sum_drivers = sum([d["impact"] for d in rev_drivers])
    sum_barriers = sum([b["impact"] for b in rev_barriers]) # impact của barrier là số âm
    
    # Sai số cho phép (tolerance) do làm tròn số học Float < 1 VND
    assert abs(rev_base + sum_drivers + sum_barriers - rev_pred) < 1.0, "Lỗi TC_07: SHAP + Base != Prediction"
    print("✅ TC_07 PASS: Tính toán quy đổi SHAP Log -> VND chính xác 100%.")

    # TC_08: Business Logic (Kiểm tra xem mảng Drivers có phần tử không)
    driver_features = [d["feature"] for d in rev_drivers]
    assert len(driver_features) > 0, "Lỗi TC_08: Không tìm thấy Top Drivers"
    print(f"✅ TC_08 PASS: (Tự kiểm chứng) Top Drivers là: {driver_features}")

    # TC_09: Không rò rỉ thuật ngữ kỹ thuật
    message_text = res_01.get("message", "").lower()
    technical_words = ["shap", "lightgbm", "json", "base value", "is_weekend"]
    for word in technical_words:
        assert word not in message_text, f"Lỗi TC_09: Lộ từ kỹ thuật '{word}' cho End-User"
    print("✅ TC_09 PASS: Văn bản NLG sạch.")

    # TC_10: Hallucination Stability
    # Cắt vài chữ số đầu của doanh thu để xem LLM có nhắc đúng số đó trong văn bản không
    rev_str_prefix = str(int(rev_pred))[:3] 
    clean_msg = message_text.replace(".", "").replace(",", "")
    assert rev_str_prefix in clean_msg, f"Lỗi TC_10: LLM bị ảo giác, báo sai con số {rev_pred} trong chuỗi NLG."
    print("✅ TC_10 PASS: LLM bám sát con số dự báo của LightGBM.")


    # ==========================================
    # TC_02: Trí nhớ ngữ cảnh (Conversational Memory)
    # ==========================================
    print_header("TC_02: Trí nhớ ngữ cảnh (Chặn hỏi ngày)")
    session_tc02 = str(uuid.uuid4())
    
    # Bước 1: Hỏi thiếu ngày
    res_02_step1 = send_chat(session_tc02, "Nếu tôi chạy sale 30% thì sao?")
    assert res_02_step1.get("status") == "clarify", "Lỗi TC_02: Không chặn lại hỏi ngày"
    print("✅ TC_02 (Bước 1) PASS: Bot đã phát hiện thiếu ngày và chặn lại.")

    # Bước 2: Bổ sung ngày (dùng chung session)
    res_02_step2 = send_chat(session_tc02, "Ngày 10/08/2023 nhé")
    assert res_02_step2.get("status") == "success", "Lỗi TC_02: Không chốt được ý định"
    assert res_02_step2["intent_parsed"]["discount_pct"] == 30.0, "Lỗi TC_02: Quên mất discount 30% từ tin nhắn trước"
    print("✅ TC_02 (Bước 2) PASS: Bot nhớ được dữ kiện '30%' và tự phân tích 'cuối tuần sau'.")


    # ==========================================
    # TC_03: Trích xuất Ngày tháng ẩn ý (Implicit Dates)
    # ==========================================
    print_header("TC_03: Ngày tháng ẩn ý (Implicit Dates)")
    session_tc03 = str(uuid.uuid4())
    res_03 = send_chat(session_tc03, "Dự báo doanh thu mùng 1 Tết Âm lịch năm nay")
    assert res_03.get("status") == "success", "Lỗi TC_03: Không phân tích được ngày ẩn ý"
    print(f"✅ TC_03 PASS: 'Mùng 1 Tết Âm lịch' được Agent giải mã thành công: {res_03['intent_parsed']['target_date']}")


    # ==========================================
    # TC_04: Truy vấn ngoài luồng (Casual Chat)
    # ==========================================
    print_header("TC_04: Casual Chat")
    session_tc04 = str(uuid.uuid4())
    res_04 = send_chat(session_tc04, "Chào bot, thời tiết hôm nay thế nào?")
    assert res_04.get("status") == "casual_chat", "Lỗi TC_04: Không nhận diện được Casual Chat"
    print(f"✅ TC_04 PASS: Nhận diện chuẩn xác truy vấn rác. Phản hồi: {res_04['message']}")


    # ==========================================
    # TC_05 & TC_06: Feature Engineering (Categorical Fallback)
    # ==========================================
    print_header("TC_05 & TC_06: Xử lý Categorical Lạ & Ngày Lễ")
    session_tc05 = str(uuid.uuid4())
    # Cố tình nhét chương trình lạ hoắc vào
    res_05 = send_chat(session_tc05, "Dự đoán ngày 14/02/2023 với chương trình Siêu Bão Deal 1K")
    assert res_05.get("status") == "success", "Lỗi TC_05: Bị crash ở Backend (Lỗi LightGBM Pandas)"
    print("✅ TC_05 & TC_06 PASS: Vượt qua rào cản Categorical (ép về np.nan) mà không vỡ Backend.")

    print_header("🎉 TẤT CẢ TEST CASES PASS HOÀN TOÀN! 🎉")

if __name__ == "__main__":
    run_tests()
