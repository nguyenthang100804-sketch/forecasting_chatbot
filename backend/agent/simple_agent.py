"""Agent dự báo tối giản dùng factory ``create_agent`` của LangChain.

File này là phiên bản dễ đọc để học cách model gọi tool trong vòng lặp.
``supervisor.py`` vẫn giữ nguyên để tham khảo cách tự dựng StateGraph.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from agent.tools import TOOLS

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SYSTEM_PROMPT = """\
Bạn là trợ lý dự báo doanh thu bằng tiếng Việt.
- Trả lời câu hỏi thông thường mà không gọi tool.
- Chỉ gọi predict_revenue khi người dùng yêu cầu dự đoán và đã cung cấp ngày.
- Nếu thiếu ngày, hãy hỏi lại và không gọi tool.
- Khi trả kết quả, luôn nêu doanh thu và COGS.
- Không bịa số liệu; dùng kết quả từ tool để giải thích.
"""

simple_agent = create_agent(
    model=ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)
