from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi của người dùng")

class ExtractedIntent(BaseModel):
    query_date: Optional[str] = Field(None, description="Ngày cần dự đoán theo định dạng YYYY-MM-DD")
    fixed_discount: float = Field(0, description="Số tiền giảm giá cố định (VND)")
    discount_pct: float = Field(0, description="Phần trăm giảm giá")
    
class ChatResponse(BaseModel):
    prediction_value: float
    analysis_text: str
    drivers: dict
