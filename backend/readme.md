# Backend Forecast Chatbot

Backend dùng FastAPI, CopilotKit/AG-UI và LangGraph để xây dựng một agent có
khả năng:

- Trả lời các câu hỏi thông thường bằng Gemini.
- Chỉ gọi tool dự báo khi người dùng thực sự yêu cầu dự báo.
- Hỏi lại ngày nếu yêu cầu dự báo chưa có ngày cụ thể.
- Lưu kết quả dự báo gần nhất trong conversation state.
- Giải thích kết quả gần nhất mà không chạy lại prediction.

## 1. Kiến trúc runtime

```text
Frontend/CopilotKit
        |
        v
POST /api/copilotkit
        |
        v
LangGraphAGUIAgent
        |
        v
    supervisor_graph
        |
        v
    agent_process
      /       \
     /         \ có tool_call
    v           v
   END      ToolNode
                 |
                 v
       predict_revenue
                 |
                 v
          forecast_graph
           /         \
          v           v
   feature_node -> ml_node -> END
```

Supervisor là điểm vào của AG-UI. Agent tự quyết định:

| Loại message | Hành động |
|---|---|
| Câu hỏi thông thường | Trả lời trực tiếp, không chạy ML |
| Yêu cầu dự báo có ngày | Gọi `predict_revenue` |
| Yêu cầu dự báo thiếu ngày | Hỏi lại ngày, không gọi tool |
| Giải thích sau dự báo | Đọc `last_forecast`, không gọi lại tool |
| Dự đoán ngày khác | Gọi lại `predict_revenue` với ngày mới |

## 2. Cấu trúc file

```text
backend/
├── main.py                    # FastAPI và AG-UI endpoint
├── graph.py                   # Forecast Graph: feature -> ML
├── supervisor.py              # Agent giám sát và vòng tool-call
├── agent_tools.py             # Tool predict_revenue
├── save_forecast_function.py      # Lưu tool result vào state
├── requirements.txt
├── render.yaml
├── calendar_features.parquet  # Feature store theo ngày
├── models/
│   ├── model_rev_phase1.txt
│   ├── model_rev_phase2.txt
│   ├── model_rev_shape_13_19.txt
│   ├── model_cogs_phase1.txt
│   ├── model_cogs_phase2.txt
│   └── cogs_model_shape_13_19.txt
├── nodes/
│   ├── state.py               # ForecastState
│   ├── feature_node.py        # Tra cứu/chuẩn hóa feature
│   └── ml_node.py             # LightGBM + SHAP
├── ml_engine/
│   ├── feature_generator.py
│   └── explainer.py
└── scripts/
    └── generate_feature_store.py
```

`intent_node.py`, `fallback_node` và `nlg_node.py` đã được loại bỏ khỏi flow.
Việc quyết định intent và sinh câu trả lời hiện thuộc về Agent trong
`supervisor.py`. `graph.py` chỉ chạy workflow dự báo.

## 3. API entry point

Trong [main.py](./main.py), backend tạo `LangGraphAGUIAgent` từ
`supervisor_graph` và đăng ký:

```text
POST /api/copilotkit
```

Frontend không gọi endpoint dự đoán riêng. Mọi message đều đi qua agent.

Endpoint kiểm tra:

```text
GET /
```

Response:

```json
{
  "status": "Backend LangGraph + CopilotKit đang hoạt động tốt!"
}
```

## 4. Agent giám sát

File: [supervisor.py](./supervisor.py)

Agent dùng `ChatGoogleGenerativeAI` và bind tool:

```python
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
)
agent = model.bind_tools(TOOLS)
```

System prompt yêu cầu agent:

- Không gọi prediction cho câu hỏi QA thông thường.
- Chỉ gọi tool khi có yêu cầu dự báo.
- Không gọi tool khi yêu cầu dự báo chưa có ngày.
- Dùng `last_forecast` để giải thích.
- Không dự đoán lại khi người dùng chỉ yêu cầu giải thích.

Agent nhận toàn bộ `messages` trong thread cùng một system context. Khi có
`last_forecast`, kết quả đó được đưa vào context để agent giải thích chính
xác dự báo gần nhất.

### Câu hỏi không liên quan

Ví dụ:

```text
COGS nghĩa là gì?
```

Agent trả lời trực tiếp. Supervisor kết thúc; Forecast Graph không chạy.

### Dự báo thiếu ngày

Ví dụ:

```text
Hãy dự đoán doanh thu
```

Agent tự hỏi lại ngày. Vì chưa có `target_date`, agent không phát tool call.

### Dự báo có ngày

Ví dụ:

```text
Hãy dự đoán doanh thu cho ngày 10/09/2023, giảm 20%
```

Agent phát tool call tương đương:

```json
{
  "name": "predict_revenue",
  "arguments": {
    "target_date": "2023-09-10",
    "discount_pct": 20,
    "fixed_discount": 0
  }
}
```

### Giải thích

Ví dụ:

```text
Giải thích kết quả dự đoán vừa rồi
```

Nếu state có `last_forecast`, agent giải thích từ dữ liệu đó và không gọi
`predict_revenue` lần nữa. Nếu chưa có dự báo, agent thông báo chưa có kết quả
để giải thích và hỏi người dùng muốn dự đoán ngày nào.

## 5. Tool `predict_revenue`

File: [agent_tools.py](./agent_tools.py)

Tool nhận:

```python
predict_revenue(
    target_date: str,
    discount_pct: float = 0.0,
    fixed_discount: float = 0.0,
)
```

Tool thực hiện:

```text
Chuẩn hóa ngày
    ↓
graph.run_forecast() -> Forecast Graph (feature -> ML)
    ↓
ml_node.process() -> ensemble_engine
    ↓
Two-Period 0.5 + Shape 13-19 0.5
    ↓
SHAP theo 8 nhóm feature
    ↓
Trả structured JSON
```

`ensemble_engine.py` nạp một lần bốn model Two-Period của project và hai model
Shape trong `backend/models`. Hai bộ feature vẫn được giữ độc lập:
`calendar_features.parquet` phục vụ Two-Period, còn
`ml_engine/features_shape_13_19.py` tạo đúng 25 cột Shape theo thứ tự lúc
huấn luyện. Kết quả cuối dùng công thức:

```text
TwoPeriod = 0.45 * expm1(P1) + 0.55 * expm1(P2)
Shape = 0.8 * expm1(M2)
Final = 0.5 * TwoPeriod + 0.5 * Shape
```

SHAP được tính trong không gian log cho từng model, đổi tên feature tương ứng,
gộp thành các nhóm `Seasonality`, `Lunar calendar`, `Tet`, `Edge-of-month`,
`Weekly`, `Promotion`, `Special day` và `Year-parity`, sau đó quy đổi ra tiền.
Vì vậy `base_value + sum(contributions.money)` luôn bằng `prediction`, còn
frontend vẽ `groups` thay vì danh sách SHAP cũ.

Tool chấp nhận các dạng ngày:

- `YYYY-MM-DD`
- `DD/MM/YYYY`
- `DD-MM-YYYY`

Promotion âm bị từ chối. Promotion người dùng nhập sẽ override promotion mặc
định trong `calendar_features.parquet`.

Tool trả JSON dạng:

```json
{
  "target_date": "2023-09-10",
  "discount_pct": 20,
  "fixed_discount": 0,
  "forecast_results": {
    "revenue": {
      "prediction": 3378387.0,
      "base_value": 3100000.0,
      "top_drivers": [],
      "top_barriers": []
    },
    "cogs": {
      "prediction": 2781056.0,
      "base_value": 2400000.0,
      "top_drivers": [],
      "top_barriers": []
    }
  }
}
```

Tool chỉ trả dữ liệu, không tự gọi Gemini để viết câu trả lời.

## 6. Forecast Graph và tool call

File: [graph.py](./graph.py)

`graph.py` không chứa Agent hoặc ToolNode. Nó chỉ chạy pipeline dự báo:

```python
builder.add_node("feature_node", build_features)
builder.add_node("ml_node", run_ml)
```

Routing dựa trên `AIMessage.tool_calls`:

```text
supervisor.py
    |
    +-- QA/giải thích -------> trả lời trực tiếp
    |
    +-- yêu cầu dự báo ------> ToolNode
                                  |
                                  v
                         agent_tools.predict_revenue
                                  |
                                  v
                            graph.run_forecast
                                  |
                                  v
                        feature_node -> ml_node
```

Supervisor quay lại `agent_process` sau ToolNode để Agent nhận ToolMessage và
viết câu trả lời cuối cùng cho người dùng. Các câu QA thông thường chỉ chạy
một lượt agent; Forecast Graph chỉ chạy khi tool được gọi.

## 7. Lưu forecast vào memory

File: [save_forecast_node.py](./save_forecast_node.py)

Sau khi `ToolNode` thực thi tool, node này đọc `ToolMessage`, parse JSON và
cập nhật:

```python
{
    "last_forecast": payload,
    "forecast_results": payload["forecast_results"],
    "target_date": payload["target_date"],
    "discount_pct": payload["discount_pct"],
    "fixed_discount": payload["fixed_discount"],
}
```

`last_forecast` là dữ liệu agent dùng cho yêu cầu giải thích tiếp theo.
Prediction ngày mới sẽ ghi đè `last_forecast`; vì vậy “giải thích kết quả vừa
rồi” luôn giải thích dự báo mới nhất.

Khi gọi lại agent, `last_forecast` vẫn được giữ dưới dạng dữ liệu có cấu trúc
trong state nhưng được format thành bản tóm tắt tiếng Việt dễ đọc trước khi
đưa vào prompt Gemini. JSON chỉ tiếp tục được dùng cho `ToolMessage` và marker
`FORECAST_DATA` để frontend có thể parse dữ liệu biểu đồ.

Graph dùng:

```python
InMemorySaver()
```

Frontend tạo `thread_id` mới khi reload trang. Do đó:

- Các message trong cùng một lần mở trang dùng chung memory.
- F5 tạo conversation mới.
- Restart backend làm mất memory.
- Các process backend khác nhau không chia sẻ memory.

Production nhiều instance nên thay bằng durable checkpointer như PostgreSQL
hoặc Redis.

## 8. Prediction và SHAP

### Feature

File [nodes/feature_node.py](./nodes/feature_node.py) tra cứu
`calendar_features.parquet`, xử lý promotion override và ép kiểu categorical
đúng với model.

### Model

File [nodes/ml_node.py](./nodes/ml_node.py) nạp bốn LightGBM model:

| Mục tiêu | Phase 1 | Phase 2 |
|---|---|---|
| Revenue | `model_rev_phase1.txt` | `model_rev_phase2.txt` |
| COGS | `model_cogs_phase1.txt` | `model_cogs_phase2.txt` |

Prediction được blend:

```text
0.45 * phase1 + 0.55 * phase2
```

`ml_node` cache tối đa 100 bộ feature để tránh tính lại cùng một forecast.
SHAP contribution được xử lý trong [ml_engine/explainer.py](./ml_engine/explainer.py),
bao gồm chuyển đổi từ không gian `log1p` về VND và chọn drivers/barriers.

## 9. Chart data

Agent node gắn dữ liệu forecast vào response cuối bằng marker:

```text
<!--FORECAST_DATA:{...json...}-->
```

Frontend parse marker để:

1. Nhận biết kết quả forecast.
2. Hiển thị nút `Dự đoán ngày khác`.
3. Hiển thị nút `Giải thích kết quả dự đoán`.
4. Vẽ chart khi response là kết quả giải thích.
5. Xóa marker trước khi render text.

Marker chỉ được thêm vào response cuối, không dùng frontend tool call để vẽ
chart. Vì vậy chart không tạo thêm một vòng agent và không gây gọi LLM lặp.

## 10. Chạy local

Từ thư mục `backend`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GOOGLE_API_KEY="<your-google-api-key>"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="<your-google-api-key>"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## 11. Tạo lại feature store

```powershell
cd backend
python scripts/generate_feature_store.py
```

Script mặc định tạo feature từ `2023-01-01` đến `2030-12-31` trong:

```text
backend/calendar_features.parquet
```

Ngày ngoài phạm vi feature store sẽ làm tool báo lỗi thay vì trả kết quả giả.

## 12. Deploy Render

`render.yaml` dùng:

```yaml
rootDir: backend
buildCommand: pip install -r requirements.txt
startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Cần cấu hình biến môi trường:

```text
GOOGLE_API_KEY
```

Không commit API key vào repository.

## 13. Kiểm tra nhanh

```powershell
cd backend
.\venv\Scripts\python.exe -m py_compile `
  main.py graph.py agent_tools.py `
  save_forecast_function.py nodes\state.py `
  nodes\feature_node.py nodes\ml_node.py
```

Chạy tool trực tiếp:

```powershell
.\venv\Scripts\python.exe -c "from agent_tools import predict_revenue; print(predict_revenue.invoke({'target_date': '2023-09-10'}))"
```

## 14. Giới hạn hiện tại

- Gemini vẫn là bộ não quyết định intent, nên chất lượng tool calling phụ
  thuộc model và system prompt.
- `InMemorySaver` không bền vững khi restart hoặc scale nhiều process.
- CORS trong `main.py` đang cho phép mọi origin; production nên giới hạn theo
  domain frontend.
- Model và feature store được load lúc import module.
- `GOOGLE_API_KEY` là bắt buộc để agent trả lời và quyết định tool.
