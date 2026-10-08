from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ForecastState(TypedDict):
    # Lịch sử hội thoại (CopilotKit tự động thêm)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Kết quả Regex/LLM Parsing
    target_date: Optional[str]
    discount_pct: Optional[float]
    fixed_discount: Optional[float]
    clarification_needed: Optional[str]
    is_prediction_query: Optional[bool]
    
    # Data truyền giữa các bước
    ml_features: Optional[dict]
    forecast_results: Optional[dict]
