# Backend Forecast Chatbot

Tài liệu này mô tả cách backend của Forecast Chatbot tiếp nhận câu hỏi, phân tích ý định, tạo feature, chạy mô hình dự báo doanh thu/COGS, sinh phần giải thích và trả kết quả cho frontend.

## 1. Tổng quan kiến trúc

Backend được xây dựng trên các thành phần chính:

- **FastAPI**: cung cấp HTTP API.
- **AG-UI LangGraph**: chuyển LangGraph thành agent tương thích với CopilotKit.
- **LangGraph**: điều phối pipeline xử lý hội thoại.
- **LightGBM**: dự báo doanh thu và COGS.
- **SHAP prediction contributions**: tính các yếu tố tác động tích cực/tiêu cực đến dự báo.
- **Google Gemini**: chỉ dùng để sinh phần giải thích ngôn ngữ tự nhiên khi người dùng yêu cầu.
- **Parquet feature store**: lưu các feature lịch, ngày lễ và promotion mặc định.

Backend không có endpoint `/api/chat` riêng. Frontend gửi message qua endpoint AG-UI:

```text
POST /api/copilotkit
```

Endpoint này được đăng ký trong [main.py](./main.py).

## 2. Cấu trúc thư mục

```text
backend/
├── main.py                         # FastAPI app và AG-UI endpoint
├── graph.py                        # LangGraph workflow
├── requirements.txt                # Python dependencies
├── render.yaml                     # Cấu hình deploy Render
├── calendar_features.parquet       # Feature store theo ngày
├── models/
│   ├── model_rev_phase1.txt        # Model revenue giai đoạn 1
│   ├── model_rev_phase2.txt        # Model revenue giai đoạn 2
│   ├── model_cogs_phase1.txt       # Model COGS giai đoạn 1
│   └── model_cogs_phase2.txt       # Model COGS giai đoạn 2
├── nodes/
│   ├── state.py                    # Schema state của LangGraph
│   ├── intent_node.py              # Phân tích intent/ngày/khuyến mãi
│   ├── feature_node.py             # Tra cứu và hoàn thiện feature
│   ├── ml_node.py                  # Chạy LightGBM + SHAP
│   └── nlg_node.py                 # Tạo câu trả lời
├── ml_engine/
│   ├── feature_generator.py        # Sinh feature calendar
│   └── explainer.py                # Chuyển SHAP và chọn yếu tố ảnh hưởng
└── scripts/
    └── generate_feature_store.py   # Sinh calendar_features.parquet
```

## 3. Khởi động backend

### 3.1. Cài dependencies

Từ thư mục `backend`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Trên Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.2. Cấu hình biến môi trường

Khi cần dùng Gemini để giải thích, khai báo:

```text
GOOGLE_API_KEY=<your-google-api-key>
```

Không commit API key vào repository.

Nếu chỉ chạy câu dự báo thông thường, backend có thể trả template tĩnh và không cần gọi Gemini. Câu hỏi giải thích cần Gemini; nếu Gemini lỗi hoặc quá tải, backend dùng fallback cục bộ.

### 3.3. Chạy local

Từ thư mục `backend`:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Kiểm tra server:

```text
GET http://127.0.0.1:8000/
```

Response thành công:

```json
{
  "status": "Backend LangGraph + CopilotKit đang hoạt động tốt!"
}
```

Frontend thường proxy `/api/copilotkit` tới runtime đang chạy ở port được cấu hình trong `vite.config.js`.

## 4. Luồng xử lý một message

Workflow được khai báo trong [graph.py](./graph.py):

```text
AG-UI/CopilotKit request
          |
          v
    intent_node
       /    \
 fallback   feature_node
    |           |
   END       ml_node
                 |
                 v
              nlg_node
                 |
                 v
                END
```

### Bước 1: `intent_node`

File: [nodes/intent_node.py](./nodes/intent_node.py)

Node này:

1. Lấy **HumanMessage cuối cùng** trong lịch sử, không lấy tool/action message.
2. Nhận diện câu hỏi dự báo hay giao tiếp thông thường.
3. Trích xuất:
   - Ngày dự báo.
   - Phần trăm giảm giá.
   - Mức giảm tiền cố định.
4. Hỗ trợ nhiều dạng ngày:
   - `2026-12-25`
   - `25/12/2026`
   - `ngày 25 tháng 12 năm 2026`
   - `ngày mai`, `tuần sau`, `tháng sau`
5. Nếu câu hỏi là câu tiếp nối như “giải thích kết quả”, node sử dụng lại `target_date`, `discount_pct` và `fixed_discount` của state trước đó.
6. Nếu không parse được bằng regex, node gọi Gemini structured output để làm fallback intent parsing.

Nếu thiếu ngày hoặc không phải câu hỏi dự báo, node đặt `clarification_needed`. `route_intent()` sẽ chuyển request sang `fallback_node`.

### Bước 2: `fallback_node`

Nếu thiếu dữ liệu cần thiết, backend trả câu hỏi yêu cầu người dùng bổ sung ngày. Nếu người dùng chỉ chào hỏi hoặc hỏi ngoài phạm vi, backend trả lời rằng hệ thống hỗ trợ dự báo doanh thu.

Node này không chạy ML và không gọi Gemini.

### Bước 3: `feature_node`

File: [nodes/feature_node.py](./nodes/feature_node.py)

Node đọc `calendar_features.parquet` và tra cứu theo `target_date`.

Feature store đã chứa:

- Feature ngày trong tuần/tháng/quý.
- Fourier seasonality features.
- Ngày lễ và các feature liên quan đến Tết.
- Promotion mặc định theo lịch.

Nếu người dùng nhập promotion, promotion người dùng nhập sẽ **override** promotion mặc định trong feature store:

- Phần trăm: ví dụ `giảm 20%`.
- Tiền cố định: ví dụ `giảm 50k`.
- Có thể dùng cả hai.

Sau đó node ép kiểu `promo_season` và `discount_type` về categorical categories đúng với lúc train LightGBM, đồng thời sắp xếp đúng thứ tự 27 feature model yêu cầu.

Nếu ngày không tồn tại trong feature store, node phát sinh lỗi rõ ràng thay vì trả kết quả giả.

### Bước 4: `ml_node`

File: [nodes/ml_node.py](./nodes/ml_node.py)

Khi module được import, bốn LightGBM Booster được nạp từ `models/`:

| Mục tiêu | Phase 1 | Phase 2 |
|---|---|---|
| Doanh thu | `model_rev_phase1.txt` | `model_rev_phase2.txt` |
| COGS | `model_cogs_phase1.txt` | `model_cogs_phase2.txt` |

Node serialize feature thành JSON để dùng `lru_cache`. Cache có tối đa 100 bộ feature, giúp tránh chạy lại dự báo/SHAP cho cùng một input.

Kết quả được tính bởi `run_inference_with_shap()`:

1. Chạy prediction contribution của từng model.
2. Chuyển giá trị từ không gian `log1p` về VND.
3. Blend phase 1 và phase 2 với:

```text
alpha = 0.45
prediction = 0.45 * phase1 + 0.55 * phase2
```

4. Tách các yếu tố thành:
   - `top_drivers`: tác động dương.
   - `top_barriers`: tác động âm.

Output dạng khái quát:

```json
{
  "revenue": {
    "prediction": 3378387.0,
    "base_value": 3100000.0,
    "top_drivers": [
      {"feature": "dow", "impact": 527000.0}
    ],
    "top_barriers": [
      {"feature": "fourier_year_cos_1", "impact": -611000.0}
    ]
  },
  "cogs": {
    "prediction": 2781056.0,
    "base_value": 2400000.0,
    "top_drivers": [],
    "top_barriers": []
  }
}
```

### Bước 5: `nlg_node`

File: [nodes/nlg_node.py](./nodes/nlg_node.py)

Node xác định loại yêu cầu dựa trên HumanMessage cuối cùng:

#### Dự báo thông thường

Nếu không có các từ khóa như `giải thích`, `tại sao`, `vì sao`, `chi tiết`, `phân tích`, backend không gọi LLM. Backend dùng template tĩnh có:

- Doanh thu.
- COGS.
- Yếu tố thúc đẩy chính.
- Yếu tố kéo giảm chính.

#### Giải thích

Nếu là yêu cầu giải thích, backend:

1. Gửi prompt cùng số liệu revenue/COGS và các yếu tố SHAP tới Gemini.
2. Chuẩn hóa response text hoặc content blocks từ Gemini bằng `extract_text_content()`.
3. Gắn dữ liệu forecast vào response bằng marker nội bộ:

```text
<!--FORECAST_DATA:{...json...}-->
```

Frontend dùng marker này để vẽ chart; marker được loại bỏ trước khi render nội dung chat.

Biểu đồ **không gọi LLM riêng**. Chart dùng trực tiếp dữ liệu đã được tạo từ ML/SHAP.

#### Fallback khi Gemini lỗi

Nếu Gemini lỗi, bao gồm lỗi `503 Service Unavailable`, hoặc trả nội dung rỗng:

- Backend ghi stack trace bằng logger.
- Sinh phần giải thích ngắn bằng `build_explanation_fallback()`.
- Vẫn đính kèm `FORECAST_DATA`.

Vì vậy lỗi Gemini không làm mất dữ liệu biểu đồ.

## 5. State và memory hội thoại

Schema state nằm trong [nodes/state.py](./nodes/state.py):

```text
messages
target_date
discount_pct
fixed_discount
clarification_needed
is_prediction_query
ml_features
forecast_results
```

`messages` dùng `add_messages`, cho phép LangGraph nối lịch sử hội thoại.

Graph được compile với:

```python
InMemorySaver()
```

AG-UI sử dụng `thread_id` để tiếp tục cùng một conversation. Frontend hiện tạo một `thread_id` mới mỗi lần tải trang, nên F5 sẽ bắt đầu phiên mới. Trong cùng một lần mở trang, các câu hỏi tiếp theo vẫn dùng chung context.

`InMemorySaver` chỉ lưu trong process hiện tại:

- Restart backend sẽ mất memory.
- Nhiều worker/process không chia sẻ memory.
- Production nhiều instance nên thay bằng durable checkpointer như PostgreSQL hoặc Redis.

## 6. Tạo lại feature store

Feature store được sinh bởi:

```powershell
cd backend
python scripts/generate_feature_store.py
```

Script tạo dữ liệu mặc định từ `2023-01-01` đến `2030-12-31` và ghi vào:

```text
backend/calendar_features.parquet
```

Khi mở rộng phạm vi ngày, cần đảm bảo:

1. Feature store bao phủ ngày người dùng muốn dự báo.
2. Các model vẫn tương thích với danh sách feature.
3. Quy tắc promotion trong `get_expected_promo()` khớp với nghiệp vụ.

## 7. Cấu hình deploy Render

File [render.yaml](./render.yaml) cấu hình:

```yaml
runtime: python
rootDir: backend
buildCommand: pip install -r requirements.txt
startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Render cần biến môi trường:

```text
GOOGLE_API_KEY
```

Port phải lấy từ `$PORT` do Render cấp, không hard-code port production.

## 8. Lỗi thường gặp và cách kiểm tra

### Không tìm thấy feature store

Log:

```text
WARNING: Không tìm thấy .../calendar_features.parquet
```

Khắc phục:

```powershell
cd backend
python scripts/generate_feature_store.py
```

### Không nạp được model

Kiểm tra bốn file model trong `backend/models/`. Nếu model không nạp được, `ml_node` sẽ không thể dự báo và cần dừng deployment để sửa cấu hình/file model, không nên trả fallback giả.

### Gemini trả lỗi 503

Đây là lỗi tạm thời do model quá tải hoặc quota. Dự báo ML vẫn có thể chạy; phần giải thích dùng fallback cục bộ và chart vẫn dựa trên `forecast_results`.

### Ngày không có trong feature store

Kiểm tra `target_date` đã nằm trong khoảng ngày được tạo bởi `generate_feature_store.py` chưa.

### Kiểm tra nhanh syntax

```powershell
cd backend
.\venv\Scripts\python.exe -m py_compile `
  main.py graph.py `
  nodes\intent_node.py `
  nodes\feature_node.py `
  nodes\ml_node.py `
  nodes\nlg_node.py
```

## 9. Hợp đồng dữ liệu với frontend

Frontend nhận message text qua AG-UI/CopilotKit. Khi có marker `FORECAST_DATA`, frontend:

1. Parse JSON kết quả forecast.
2. Xóa marker khỏi text hiển thị.
3. Chỉ render chart khi đó là message giải thích.
4. Hiển thị các quick action dựa trên message forecast mới nhất.

Backend không phát `tool_call` để vẽ chart. Cách này tránh làm runtime chạy thêm một lượt agent và tránh gọi LLM lặp lại.

## 10. Các giới hạn hiện tại

- Checkpoint đang dùng memory trong process, chưa bền vững khi restart.
- CORS đang cho phép mọi origin (`allow_origins=["*"]`); production nên giới hạn theo domain frontend.
- Model và feature store được load lúc import module; thay model cần restart process.
- Cache ML là cache cục bộ từng process, không chia sẻ giữa các instance.
- LLM chỉ được dùng cho intent fallback và giải thích; dự báo số liệu không phụ thuộc vào LLM.

