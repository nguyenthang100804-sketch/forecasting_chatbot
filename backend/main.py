from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lightgbm as lgb
import os
import uuid

# Import 3 module của bạn
from llm_engine.intent_parser import create_intent_chat_session, IntentSchema
from ml_engine import feature_generator
from ml_engine.explainer import run_inference_with_shap

app: FastAPI = FastAPI(title="Revenue Prediction Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép Frontend từ Vercel gọi API
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from llm_engine.analyst import generate_analysis

models = {}
# Lưu trữ phiên chat của từng user để Agent có memory. Key là session_id (UUID), value là đối tượng ChatSession từ LLM Engine
chat_sessions = {}

@app.on_event("startup")
async def load_models():
    print("Đang nạp mô hình LightGBM vào RAM...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, "models")
    try:
        models['rev_p1'] = lgb.Booster(model_file=os.path.join(model_dir, 'model_rev_phase1.txt'))
        models['rev_p2'] = lgb.Booster(model_file=os.path.join(model_dir, 'model_rev_phase2.txt'))
        models['cogs_p1'] = lgb.Booster(model_file=os.path.join(model_dir, 'model_cogs_phase1.txt'))
        models['cogs_p2'] = lgb.Booster(model_file=os.path.join(model_dir, 'model_cogs_phase2.txt'))
        print("Đã nạp thành công 4 mô hình!")
    except Exception as e:
        print(f"Lỗi khi nạp mô hình: {e}")

# ==========================================
# 2. API ENDPOINTS
# ==========================================
class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/api/chat")
async def chat_with_agent(req: ChatRequest):
    # 1. Lấy hoặc tạo phiên làm việc (Session) cho user này
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = create_intent_chat_session()
    
    session = chat_sessions[req.session_id]
    
    # 2. BƯỚC 1: PHÂN TÍCH Ý ĐỊNH
    try:
        response = session.send_message(req.message)
        intent: IntentSchema = response.parsed
    except Exception as e:
        return {"status": "error", "message": f"Lỗi LLM: {str(e)}"}

    # Nếu người dùng chỉ đang giao tiếp bình thường (Chào hỏi...)
    if not intent.is_prediction_query:
        return {
            "status": "casual_chat",
            "message": "Chào bạn, mình là Agent dự đoán doanh thu. Bạn muốn dự đoán cho ngày nào?"
        }

    # Nếu đang hỏi dự đoán nhưng thiếu thông tin (vd: thiếu ngày)
    if intent.clarification_needed:
        return {
            "status": "clarify",
            "message": intent.clarification_needed
        }

    target_date = intent.target_date
    if not isinstance(target_date, str) or not target_date.strip():
        return {
            "status": "clarify",
            "message": "Vui lòng cung cấp ngày cần dự đoán."
        }

    # BƯỚC 2: DATA PIPELINE (Sinh đặc trưng từ JSON)
    try:
        df_features = feature_generator.generate_features(
            target_date=target_date,
            discount_pct=intent.discount_pct,
            fixed_discount=intent.fixed_discount
        )
    except Exception as e:
        return {"status": "error", "message": f"Lỗi Data Pipeline: {str(e)}"}

    # BƯỚC 3: CHẠY INFERENCE VÀ LẤY XAI (SHAP)
    try:
        result_json = run_inference_with_shap(models, df_features, alpha=0.45)
    except Exception as e:
        return {"status": "error", "message": f"Lỗi Model Inference: {str(e)}"}

    # BƯỚC 4: CHUYÊN GIA PHÂN TÍCH (NLG)
    try:
        final_answer = generate_analysis(
            intent_data={
                "target_date": intent.target_date,
                "discount_pct": intent.discount_pct,
                "fixed_discount": intent.fixed_discount
            },
            forecast_results=result_json
        )
    except Exception as e:
        return {"status": "error", "message": f"Lỗi LLM Analyst: {str(e)}"}

    # Trả về cả Text tự nhiên cho Chatbot và JSON Khổng lồ cho UI (Vẽ biểu đồ SHAP)
    return {
        "status": "success",
        "message": final_answer,
        "forecast_results": result_json,
        "intent_parsed": {
            "target_date": intent.target_date,
            "discount_pct": intent.discount_pct,
            "fixed_discount": intent.fixed_discount
        }
    }

@app.get("/")
def read_root():
    return {"status": "Backend đang hoạt động tốt!", "models_loaded": len(models) == 4}
