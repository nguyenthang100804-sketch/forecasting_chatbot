import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

from dateutil.relativedelta import relativedelta  # có sẵn nếu từng cài dateparser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from nodes.state import ForecastState

load_dotenv()

VN_TZ = timezone(timedelta(hours=7))  # không cần tzdata trên Windows
PREDICT_KEYWORDS = ("dự báo", "dự đoán", "doanh thu", "forecast", "predict", "revenue")
CASUAL_WORDS = {"chào", "hello", "hi", "hey"}

ASK_DATE_MSG = "Bạn muốn dự báo cho ngày nào? Ví dụ: 25/12/2026 hoặc 'ngày mai'."
CASUAL_REPLY = "Chào bạn, mình là trợ lý dự báo. Bạn muốn dự báo cho thời điểm nào?"


class IntentSchema(BaseModel):
    is_prediction_query: bool = Field(description="True nếu hỏi về dự đoán doanh thu. False nếu chỉ giao tiếp.")
    target_date: Optional[str] = Field(default=None, description="Ngày mục tiêu cần dự đoán (YYYY-MM-DD).")
    discount_pct: float = Field(default=0.0, description="Phần trăm giảm giá (ví dụ 'giảm 20%'). Mặc định 0.0.")
    fixed_discount: float = Field(default=0.0, description="Giảm giá tiền mặt cố định (ví dụ 'giảm 50k'). Mặc định 0.0.")
    clarification_needed: Optional[str] = Field(default=None, description="Hỏi lại user nếu thiếu ngày tháng.")


# ---------- Discount ----------
def extract_discount(text: str) -> Tuple[float, float]:
    match_pct = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
    pct = float(match_pct.group(1).replace(",", ".")) if match_pct else 0.0

    # 50k, 100K, 50 nghìn, 50 ngàn (có \b để không khớp nhầm "5 km", "2 kg")
    match_fixed = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:k|nghìn|ngàn)\b", text, re.IGNORECASE)
    fixed = float(match_fixed.group(1).replace(",", ".")) * 1000 if match_fixed else 0.0
    return pct, fixed


# ---------- Date ----------
def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _date_without_year(d: int, m: int, today: date) -> Optional[date]:
    """Thiếu năm: lấy năm nay, nếu đã qua thì lấy năm sau (vì đang dự báo tương lai)."""
    cand = _safe_date(today.year, m, d)
    if cand and cand < today:
        cand = _safe_date(today.year + 1, m, d)
    return cand


def parse_target_date(text: str, today: date) -> Tuple[Optional[date], bool]:
    """Trả về (ngày, month_only). month_only=True nghĩa là user chỉ nói tháng, chưa có ngày cụ thể."""
    t = text.lower()

    # 1. ISO: 2026-12-25
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        return _safe_date(int(m[1]), int(m[2]), int(m[3])), False

    # 2. dd/mm/yyyy hoặc dd-mm-yyyy
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", t)
    if m:
        return _safe_date(int(m[3]), int(m[2]), int(m[1])), False

    # 3. "ngày 25 tháng 12 [năm 2026]"
    m = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s+năm\s+(\d{4}))?", t)
    if m:
        d, mo = int(m[1]), int(m[2])
        if m[3]:
            return _safe_date(int(m[3]), mo, d), False
        return _date_without_year(d, mo, today), False

    # 4. dd/mm
    m = re.search(r"\b(\d{1,2})/(\d{1,2})\b(?!/)", t)
    if m:
        return _date_without_year(int(m[1]), int(m[2]), today), False

    # 5. Tương đối
    if re.search(r"hôm nay|today", t):
        return today, False
    if re.search(r"ngày mai|tomorrow", t):
        return today + timedelta(days=1), False
    if re.search(r"ngày kia|ngày mốt", t):
        return today + timedelta(days=2), False
    m = re.search(r"(\d+)\s*ngày\s*(?:nữa|sau|tới)", t)
    if m:
        return today + timedelta(days=int(m[1])), False
    m = re.search(r"(\d+)\s*tuần\s*(?:nữa|sau|tới)", t)
    if m:
        return today + timedelta(weeks=int(m[1])), False
    if re.search(r"tuần sau|tuần tới|next week", t):
        return today + timedelta(weeks=1), False
    if re.search(r"tháng sau|tháng tới|next month", t):
        return today + relativedelta(months=1), False

    # 6. Chỉ có tháng ("tháng 12") -> chưa đủ để dự báo theo ngày
    if re.search(r"tháng\s+\d{1,2}\b", t):
        return None, True

    return None, False


# ---------- LLM fallback ----------
def _llm_fallback(messages, today: date) -> dict:
    print("Hybrid Parser thất bại, kích hoạt LLM Fallback...")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        structured_llm = llm.with_structured_output(IntentSchema)
        system_msg = f"Hôm nay là {today.isoformat()}. Hãy trích xuất ý định của người dùng."
        resp = structured_llm.invoke([{"role": "system", "content": system_msg}] + messages)
    except Exception as e:
        print(f"LLM fallback lỗi: {e}")
        return {
            "is_prediction_query": True,
            "target_date": None,
            "discount_pct": 0.0,
            "fixed_discount": 0.0,
            "clarification_needed": ASK_DATE_MSG,
        }

    # Kiểm tra định dạng ngày mà LLM trả về
    target = None
    if resp.target_date:
        try:
            target = datetime.strptime(resp.target_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            target = None

    clarification = resp.clarification_needed
    if resp.is_prediction_query and target is None and not clarification:
        clarification = ASK_DATE_MSG
    elif not resp.is_prediction_query and not clarification:
        clarification = CASUAL_REPLY

    return {
        "is_prediction_query": resp.is_prediction_query,
        "target_date": target,
        "discount_pct": resp.discount_pct,
        "fixed_discount": resp.fixed_discount,
        "clarification_needed": clarification,
    }


# ---------- Node ----------
def process(state: ForecastState):
    """Hybrid Parser: Regex trước, fallback sang LLM."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    human_messages = [
        message for message in messages
        if getattr(message, "type", None) == "human"
    ]
    if not human_messages:
        return {}

    last_msg = human_messages[-1].content
    text_lower = last_msg.lower()
    today = datetime.now(VN_TZ).date()

    # Chào hỏi xã giao: so khớp theo TỪ, không theo chuỗi con
    words = set(re.findall(r"\w+", text_lower))
    if len(words) < 5 and (words & CASUAL_WORDS or "tên gì" in text_lower):
        return {
            "is_prediction_query": False,
            "target_date": None,
            "clarification_needed": CASUAL_REPLY,
        }

    discount_pct, fixed_discount = extract_discount(last_msg)
    target, month_only = parse_target_date(last_msg, today)

    # Các câu hỏi tiếp nối như "giải thích kết quả" dùng lại ngày và khuyến mãi
    # của dự báo trước đó trong cùng thread.
    explanation_words = ("tại sao", "giải thích", "chi tiết", "vì sao", "phân tích")
    previous_target = state.get("target_date")
    if target is None and previous_target and any(word in text_lower for word in explanation_words):
        return {
            "is_prediction_query": True,
            "target_date": previous_target,
            "discount_pct": state.get("discount_pct", 0.0),
            "fixed_discount": state.get("fixed_discount", 0.0),
            "clarification_needed": None,
        }

    # Có ngày hợp lệ -> trả luôn, không tốn token
    if target is not None:
        return {
            "is_prediction_query": True,
            "target_date": target.strftime("%Y-%m-%d"),
            "discount_pct": discount_pct,
            "fixed_discount": fixed_discount,
            "clarification_needed": None,
        }

    # Chỉ nói tháng -> hỏi lại ngày cụ thể
    if month_only:
        return {
            "is_prediction_query": True,
            "target_date": None,
            "discount_pct": discount_pct,
            "fixed_discount": fixed_discount,
            "clarification_needed": "Bạn muốn dự báo cho ngày cụ thể nào trong tháng đó? Ví dụ: 15/12/2026.",
        }

    # Không bắt được ngày -> nhờ LLM (có cả lịch sử hội thoại)
    return _llm_fallback(messages, today)