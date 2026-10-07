import os
import datetime
from pydantic import BaseModel, Field
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# 1. Định nghĩa Schema y hệt thiết kế của bạn
class IntentSchema(BaseModel):
    is_prediction_query: bool = Field(description="True nếu user yêu cầu dự đoán doanh thu/COGS. False nếu là giao tiếp bình thường.")
    target_date: Optional[str] = Field(description="Ngày mục tiêu cần dự đoán (YYYY-MM-DD). Trả về null nếu câu hỏi không đề cập thời gian.")
    discount_pct: float = Field(default=0.0, description="Phần trăm giảm giá (nếu có), ví dụ 'giảm 20%' -> 20.0. Mặc định 0.0.")
    fixed_discount: float = Field(default=0.0, description="Giảm giá tiền mặt cố định, ví dụ 'giảm 50 cành' -> 50000.0. Mặc định 0.0.")
    clarification_needed: Optional[str] = Field(description="Câu hỏi cho user nếu thiếu thông tin cần thiết cho dự đoán, bao gồm mục đích và ngày. Nếu đủ thì null.")

client = genai.Client()

def create_intent_chat_session():
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    # Khởi tạo một phiên Chat có trí nhớ
    chat = client.chats.create(
        model='gemini-3.6-flash',
        config=types.GenerateContentConfig(
            # Chuyển Prompt tĩnh vào System Instruction
            system_instruction=f"Bạn là hệ thống phân tích ý định. Hôm nay là ngày {today_str}. Chỉ được phép phân tích câu hỏi và trả về JSON theo yêu cầu, không giải thích gì thêm.",
            response_mime_type="application/json",
            response_schema=IntentSchema,
            temperature=0.1,
        )
    )
    return chat
# --- Chạy thử nghiệm Khả năng nhớ Ngữ cảnh ---
if __name__ == "__main__":
    # Khởi tạo phiên làm việc cho 1 User
    user_session_chat = create_intent_chat_session()
    
    print("User: Dự đoán giúp doanh thu nếu giảm giá 20%")
    # Lần 1: Gọi send_message
    res1 = user_session_chat.send_message("Dự đoán giúp doanh thu nếu giảm giá 20%")
    clarification_needed = getattr(res1.parsed, "clarification_needed", None)
    print(f"Cần hỏi lại: {clarification_needed}")
    
    print("\nUser: Cho ngày mai đi")
    # Lần 2: User trả lời cộc lốc, nhưng Chat vẫn nhớ 20%
    res2 = user_session_chat.send_message("Cho ngày mai đi")
    print("--- Kết quả cuối cùng ---")
    if isinstance(res2.parsed, IntentSchema):
        print(f"Ngày: {res2.parsed.target_date}")
        print(f"Giảm giá (%): {res2.parsed.discount_pct}")
