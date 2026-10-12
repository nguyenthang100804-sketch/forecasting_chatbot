"""Conversational supervisor that decides whether to run the forecast graph."""

import json
import logging
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent_tools import TOOLS
from nodes.state import ForecastState
from save_forecast_function import process as save_forecast

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parent / ".env")

EXPLANATION_WORDS = ("giải thích", "tại sao", "vì sao", "chi tiết", "phân tích")
PREDICTION_WORDS = ("dự đoán", "dự báo", "doanh thu", "forecast", "predict")

SYSTEM_PROMPT = """
Bạn là trợ lý Forecast Chatbot bằng tiếng Việt.

Quy tắc:
- Trả lời các câu hỏi thông thường một cách tự nhiên, không gọi tool dự báo.
- Chỉ gọi tool predict_revenue khi người dùng thực sự yêu cầu dự đoán doanh thu/COGS.
- Nếu người dùng yêu cầu dự đoán nhưng chưa có ngày cụ thể, hãy hỏi lại ngày và không gọi tool.
- target_date khi gọi tool phải ở dạng YYYY-MM-DD. discount_pct là phần trăm, fixed_discount là VND.
- Nếu người dùng yêu cầu giải thích kết quả mà chưa có kết quả dự báo gần nhất, hãy nói rõ chưa có dữ liệu và hỏi họ muốn dự đoán ngày nào.
- Nếu có kết quả dự báo gần nhất trong context, hãy giải thích chính kết quả đó và không gọi lại tool.
- Không bịa số liệu. Khi trả kết quả dự báo, luôn nêu cả doanh thu và COGS.
"""


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _last_user_text(messages: Iterable[BaseMessage]) -> str:
    for message in reversed(list(messages)):
        if getattr(message, "type", None) == "human":
            return _message_text(message).lower()
    return ""


def _forecast_marker(forecast: dict) -> str:
    return f"\n\n<!--FORECAST_DATA:{json.dumps(forecast, ensure_ascii=False, separators=(',', ':'))}-->"


def _is_forecast_related(text: str) -> bool:
    return any(word in text for word in EXPLANATION_WORDS + PREDICTION_WORDS)


def agent_process(state: ForecastState):
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            "Thiếu GOOGLE_API_KEY hoặc GEMINI_API_KEY. "
            "Hãy cấu hình biến này trong backend/.env hoặc environment."
        )

    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    response = model.bind_tools(TOOLS).invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT
                + (
                    "\nKết quả dự báo gần nhất (chỉ dùng khi giải thích, không dự đoán lại):\n"
                    + json.dumps(state["last_forecast"], ensure_ascii=False)
                    if state.get("last_forecast")
                    else "\nChưa có kết quả dự báo nào trong conversation này.\n"
                )
            )
        ]
        + state.get("messages", [])
    )
    if not isinstance(response, AIMessage):
        raise TypeError("Agent phải trả về AIMessage.")

    user_text = _last_user_text(state.get("messages", []))
    if not response.tool_calls and state.get("last_forecast") and _is_forecast_related(user_text):
        response = AIMessage(
            content=f"{_message_text(response)}{_forecast_marker(state['last_forecast'])}",
            response_metadata=response.response_metadata,
        )
    return {"messages": [response]}


def route_agent(state: ForecastState):
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "tool_calls", None):
        return "tool_node"
    return END


builder = StateGraph(ForecastState)
builder.add_node("agent_node", agent_process)
builder.add_node("tool_node", ToolNode(TOOLS))
builder.add_node("save_forecast_node", save_forecast)
builder.set_entry_point("agent_node")
builder.add_conditional_edges("agent_node", route_agent)
builder.add_edge("tool_node", "save_forecast_node")
builder.add_edge("save_forecast_node", "agent_node")

supervisor_graph = builder.compile(checkpointer=InMemorySaver())
