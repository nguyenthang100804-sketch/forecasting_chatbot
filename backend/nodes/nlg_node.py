import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from nodes.state import ForecastState

logger = logging.getLogger(__name__)


def format_vnd(value):
    return f"{value:,.0f} VNĐ"


def extract_text_content(content):
    """Chuẩn hóa nội dung dạng text hoặc content blocks từ Gemini."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)

    return str(content)


def build_explanation_fallback(date, rev, cogs, drivers, barriers):
    driver_str = drivers[0]["feature"] if drivers else "các yếu tố tích cực trong ngày"
    barrier_str = barriers[0]["feature"] if barriers else "các yếu tố bất lợi trong ngày"
    return (
        f"Dự báo cho ngày {date} ghi nhận **Doanh thu đạt {format_vnd(rev)}** "
        f"và **COGS đạt {format_vnd(cogs)}**.\n\n"
        f"Kết quả chịu ảnh hưởng tích cực từ `{driver_str}`, trong khi "
        f"`{barrier_str}` là yếu tố gây áp lực lên kết quả dự báo."
    )


def process(state: ForecastState):
    """NLG Node: Sinh văn bản báo cáo hoặc Template."""
    results = state.get("forecast_results")
    if not results:
        return {}
        
    date = state.get("target_date")
    discount = state.get("discount_pct")
    fixed = state.get("fixed_discount")
    
    rev = results['revenue']['prediction']
    cogs = results['cogs']['prediction']
    drivers = results['revenue']['top_drivers']
    barriers = results['revenue']['top_barriers']
    
    # Kỹ thuật Templating: Tiết kiệm token nếu không có câu hỏi phụ
    messages = state.get("messages", [])
    human_messages = [
        message for message in messages
        if getattr(message, "type", None) == "human"
    ]
    last_msg = human_messages[-1].content.lower() if human_messages else ""
    
    # Nếu câu hỏi đơn giản (không hỏi "tại sao", "giải thích"), trả về mẫu câu tĩnh
    complex_words = ["tại sao", "giải thích", "chi tiết", "vì sao", "phân tích"]
    if not any(w in last_msg for w in complex_words):
        driver_str = drivers[0]['feature'] if drivers else "không có yếu tố nào rõ rệt"
        barrier_str = barriers[0]['feature'] if barriers else "không có yếu tố nào"
        
        reply = (f"Dự kiến ngày {date} (với mức giảm {discount}% và giảm {fixed} VNĐ):\n\n"
                 f"- **Doanh thu:** {format_vnd(rev)}\n"
                 f"- **COGS:** {format_vnd(cogs)}\n\n"
                 f"Yếu tố thúc đẩy chính là `{driver_str}`, trong khi `{barrier_str}` kéo giảm doanh thu.")

        from langchain_core.messages import AIMessage
        chart_marker = f"\n\n<!--FORECAST_DATA:{json.dumps(results, separators=(',', ':'))}-->"
        return {
            "final_reply": reply,
            "messages": [AIMessage(content=f"{reply}{chart_marker}")],
        }
        
    # Nếu câu hỏi phức tạp, gọi LLM chuyên sâu
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    
    prompt = f"""
    Bạn là chuyên gia phân tích doanh thu. Người dùng hỏi: {last_msg}
    Đây là số liệu kỹ thuật (SHAP) do mô hình AI xuất ra: {results}
    
    Hãy viết một đoạn văn ngắn gọn, chuyên nghiệp để phân tích vì sao doanh thu ngày {date} lại đạt mức đó.
    Phải đề cập rõ cả doanh thu ({format_vnd(rev)}) và COGS ({format_vnd(cogs)}).
    Tuyệt đối không dùng từ kỹ thuật như SHAP, LightGBM, JSON. Không chèn Markdown code block.
    """
    
    from langchain_core.messages import AIMessage
    # Truyền dữ liệu chart cùng một response thay vì tool call. Tool call sẽ
    # khiến runtime tiếp tục một lượt agent khác sau khi frontend xử lý action.
    chart_marker = f"\n\n<!--FORECAST_DATA:{json.dumps(results, separators=(',', ':'))}-->"
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        explanation = extract_text_content(res.content).strip()
        if not explanation:
            raise ValueError("LLM trả về nội dung giải thích rỗng")
    except Exception:
        logger.exception("Không thể gọi LLM để giải thích dự báo; dùng fallback cục bộ")
        explanation = build_explanation_fallback(date, rev, cogs, drivers, barriers)

    reply = f"{explanation}{chart_marker}"
    return {"final_reply": reply, "messages": [AIMessage(content=reply)]}
