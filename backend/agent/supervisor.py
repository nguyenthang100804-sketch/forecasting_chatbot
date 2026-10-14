"""Conversational supervisor that decides whether to run the forecast graph."""

import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.tools import TOOLS
from agent.formatting import (
    format_forecast_history,
    forecast_marker,
)
from agent.types import ForecastPayload
from agent.save_forecast import process as save_forecast
from pipelines.nodes.state import ForecastState

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EXPLANATION_WORDS = ("giải thích", "tại sao", "vì sao", "chi tiết", "phân tích")
PREDICTION_WORDS = ("dự đoán", "dự báo", "doanh thu", "forecast", "predict")

SYSTEM_PROMPT = """
Bạn là trợ lý Forecast Chatbot bằng tiếng Việt.

Quy tắc:
- Trả lời các câu hỏi thông thường một cách tự nhiên, không gọi tool dự báo.
- Chỉ gọi tool predict_revenue khi người dùng thực sự yêu cầu dự đoán doanh thu/COGS.
- Nếu người dùng yêu cầu dự đoán nhưng chưa có ngày cụ thể, hãy hỏi lại ngày và không gọi tool.
- target_date khi gọi tool phải ở dạng YYYY-MM-DD. discount_pct là phần trăm.
- fixed_discount dùng đúng đơn vị của dữ liệu mô hình (ví dụ 50), không quy đổi sang VND.
- Nếu người dùng yêu cầu giải thích kết quả mà chưa có kết quả dự báo tương ứng trong lịch sử, hãy nói rõ chưa có dữ liệu và hỏi họ muốn dự đoán ngày nào.
- Nếu kết quả cho ngày được hỏi đã có trong lịch sử, hãy dùng đúng kết quả đó và tuyệt đối không gọi lại tool.
- Nếu cùng một ngày có nhiều kịch bản giảm giá, hãy chọn kịch bản có mức giảm khớp với câu hỏi; nếu không xác định được, hãy hỏi lại.
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


def _latest_forecast(
    history: dict[str, list[ForecastPayload]],
) -> ForecastPayload | None:
    forecasts = [forecast for values in history.values() for forecast in values]
    return forecasts[-1] if forecasts else None


def _parse_requested_discount(text: str) -> tuple[float | None, float | None]:
    percent_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:%|phần\s*trăm|percent)", text
    )
    fixed_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|vnd|vnđ|đồng)\b", text
    )
    fixed_context_match = re.search(
        r"(?:giảm\s*)?(?:cố\s*định|fixed(?:\s+discount)?)\s*(?:là|:)?\s*"
        r"(\d+(?:[.,]\d+)?)\b",
        text,
    )

    discount_pct = (
        float(percent_match.group(1).replace(",", "."))
        if percent_match
        else None
    )
    fixed_discount = None
    if fixed_match or fixed_context_match:
        number_match = fixed_match or fixed_context_match
        fixed_discount = float(number_match.group(1).replace(",", "."))
    return discount_pct, fixed_discount


def _forecast_for_request(
    history: dict[str, list[ForecastPayload]], user_text: str
) -> ForecastPayload | None:
    date_matches = re.findall(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}[/-]\d{2}[/-]\d{4}\b", user_text
    )
    discount_pct, fixed_discount = _parse_requested_discount(user_text)
    candidates: list[ForecastPayload] = []
    for date_text in date_matches:
        for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                date_key = datetime.strptime(date_text, date_format).strftime("%Y-%m-%d")
            except ValueError:
                continue
            forecasts = history.get(date_key)
            if forecasts:
                candidates.extend(forecasts)
            break

    if not date_matches:
        candidates = [
            forecast for forecasts in history.values() for forecast in forecasts
        ]
    if discount_pct is not None:
        candidates = [
            forecast
            for forecast in candidates
            if math.isclose(forecast["discount_pct"], discount_pct)
        ]
    if fixed_discount is not None:
        candidates = [
            forecast
            for forecast in candidates
            if math.isclose(forecast["fixed_discount"], fixed_discount)
        ]
    return candidates[-1] if candidates else None


def _is_forecast_related(text: str) -> bool:
    return any(word in text for word in EXPLANATION_WORDS + PREDICTION_WORDS)


def _stream_model_response(model_runner, messages: list[BaseMessage]) -> AIMessage:
    """Stream model chunks while preserving the final tool-call message."""
    response_chunk: AIMessageChunk | None = None
    for chunk in model_runner.stream(messages):
        if not isinstance(chunk, AIMessageChunk):
            raise TypeError("Model phải trả về AIMessageChunk khi streaming.")
        response_chunk = chunk if response_chunk is None else response_chunk + chunk

    if response_chunk is None:
        raise RuntimeError("Model không trả về dữ liệu khi streaming.")

    return AIMessage(
        content=response_chunk.content,
        additional_kwargs=response_chunk.additional_kwargs,
        response_metadata=response_chunk.response_metadata,
        name=response_chunk.name,
        id=response_chunk.id,
        tool_calls=response_chunk.tool_calls,
        invalid_tool_calls=response_chunk.invalid_tool_calls,
        usage_metadata=response_chunk.usage_metadata,
    )


def agent_process(state: ForecastState):
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            "Thiếu GOOGLE_API_KEY hoặc GEMINI_API_KEY. "
            "Hãy cấu hình biến này trong backend/.env hoặc environment."
        )

    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    history = state.get("forecast_history", {})
    user_text = _last_user_text(state.get("messages", []))
    has_explanation_context = bool(history) and any(
        word in user_text for word in EXPLANATION_WORDS
    )
    model_runner = model if has_explanation_context else model.bind_tools(TOOLS)
    response = _stream_model_response(
        model_runner,
        [
            SystemMessage(
                content=SYSTEM_PROMPT
                + format_forecast_history(history)
            )
        ]
        + state.get("messages", []),
    )
    if not isinstance(response, AIMessage):
        raise TypeError("Agent phải trả về AIMessage.")

    requested_forecast = _forecast_for_request(history, user_text)
    if not response.tool_calls and requested_forecast and _is_forecast_related(user_text):
        response = AIMessage(
            content=f"{_message_text(response)}{forecast_marker(requested_forecast)}",
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
