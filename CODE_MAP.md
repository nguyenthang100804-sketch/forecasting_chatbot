# Bản đồ mã nguồn Forecast Chatbot

Tài liệu này là mục lục tra cứu nhanh khi kiểm thử, báo lỗi hoặc viết báo
cáo. Các dòng được ghi theo vị trí hiện tại của mã nguồn; khi mã thay đổi,
hãy tìm lại tên hàm/symbol trong file tương ứng.

## 1. Luồng tổng quát

```text
Frontend
  App.jsx
    -> AgentChatPage.jsx
      -> useChat.js
        -> CopilotKit /api/copilotkit

Backend
  main.py
    -> supervisor.py (Agent giám sát hội thoại)
      -> ToolNode
        -> agent_tools.py
          -> graph.py (Forecast Graph)
            -> feature_node.py
            -> ml_node.py
              -> ensemble_engine.py
                -> features_shape_13_19.py
                -> final_model_shap.py
        -> save_forecast_node.py
```

## 2. Tra nhanh theo chức năng

| Chức năng cần kiểm tra | File / symbol chính | Vai trò |
|---|---|---|
| Khởi động API | [backend/main.py](./backend/main.py#L1) | Tạo FastAPI và expose endpoint CopilotKit |
| Forecast Graph | [backend/graph.py](./backend/graph.py#L12) `forecast_graph` | Chỉ chạy feature → ML, không biết Agent/Tools |
| API chạy dự báo | [backend/graph.py](./backend/graph.py#L24) `run_forecast` | Interface công khai để tool gọi graph |
| Agent giám sát | [backend/supervisor.py](./backend/supervisor.py#L61) `agent_process` | Quyết định QA, gọi tool hoặc giải thích |
| Vòng supervisor | [backend/supervisor.py](./backend/supervisor.py#L104) `supervisor_graph` | Chứa Agent, ToolNode và lưu forecast |
| Prompt/rule của agent | [backend/supervisor.py](./backend/supervisor.py#L20) `SYSTEM_PROMPT` | Quy định QA, dự báo thiếu ngày, giải thích và không dự đoán lại |
| Nhận diện message từ Gemini | [backend/supervisor.py](./backend/supervisor.py#L42) `_message_text` | Chuẩn hóa content dạng chuỗi hoặc content block |
| Gắn dữ liệu cho biểu đồ | [backend/supervisor.py](./backend/supervisor.py#L62) `_forecast_marker` | Tạo marker `FORECAST_DATA` trong phản hồi |
| Công cụ dự đoán | [backend/agent_tools.py](./backend/agent_tools.py#L22) `predict_revenue` | Chuẩn hóa ngày, kiểm tra promotion và gọi pipeline ML |
| Chuẩn hóa ngày | [backend/agent_tools.py](./backend/agent_tools.py#L10) `_normalize_date` | Hỗ trợ `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY` |
| Trạng thái hội thoại | [backend/nodes/state.py](./backend/nodes/state.py#L5) `ForecastState` | Khai báo messages, feature, forecast và last forecast |
| Lưu kết quả tool vào memory | [backend/save_forecast_node.py](./backend/save_forecast_node.py#L4) `process` | Đọc ToolMessage và cập nhật `last_forecast` |
| Feature Two-Period | [backend/nodes/feature_node.py](./backend/nodes/feature_node.py#L20) `process` | Đọc feature store và áp dụng promotion override |
| Feature store | [backend/calendar_features.parquet](./backend/calendar_features.parquet) | Bảng feature phục vụ model Two-Period |
| Chạy ML ensemble | [backend/ensemble_engine.py](./backend/ensemble_engine.py#L90) `run_ensemble` | Kết hợp Two-Period và Shape cho Revenue/COGS |
| Node gọi ML | [backend/nodes/ml_node.py](./backend/nodes/ml_node.py#L5) `process` | Nhận `ml_features`, trả `forecast_results` |
| Feature Shape | [backend/ml_engine/features_shape_13_19.py](./backend/ml_engine/features_shape_13_19.py#L110) `build_features` | Tạo 25 feature Shape cho một ngày |
| SHAP model cuối | [backend/ml_engine/final_model_shap.py](./backend/ml_engine/final_model_shap.py#L118) `explain_final` | Gộp SHAP và quy đổi đóng góp sang tiền |
| Nhóm SHAP | [backend/ml_engine/final_model_shap.py](./backend/ml_engine/final_model_shap.py#L42) `GROUPS` | 8 nhóm: Seasonality, Tet, Promotion... |
| SHAP từng model | [backend/ml_engine/final_model_shap.py](./backend/ml_engine/final_model_shap.py#L68) `shap_one_row` | Lấy base value và contribution từ LightGBM |
| Cache ensemble | [backend/ensemble_engine.py](./backend/ensemble_engine.py#L67) `_cached_ensemble` | Tránh nạp/tính lại kết quả giống nhau |
| Model files | [backend/models/](./backend/models/) | Hai phase Two-Period và hai model Shape |
| ID conversation khi F5 | [frontend/src/App.jsx](./frontend/src/App.jsx#L9) `threadId` | Tạo UUID mới khi khởi tạo app |
| Gửi message frontend | [frontend/src/hooks/useChat.js](./frontend/src/hooks/useChat.js#L20) `useChat` | Kết nối CopilotKit và xử lý lỗi gửi |
| Parse dữ liệu biểu đồ | [frontend/src/hooks/useChat.js](./frontend/src/hooks/useChat.js#L4) `FORECAST_DATA_PATTERN` | Đọc marker và lấy `forecast_results` |
| Lọc message hiển thị | [frontend/src/hooks/useChat.js](./frontend/src/hooks/useChat.js#L70) `visibleMessages` | Bỏ ToolMessage/assistant message rỗng |
| Nhận diện explanation | [frontend/src/hooks/useChat.js](./frontend/src/hooks/useChat.js#L78) `isExplanation` | Xác định message nào được gắn chart |
| Nút dự đoán/giải thích | [frontend/src/components/ChatMessage.jsx](./frontend/src/components/ChatMessage.jsx#L6) `ChatMessage` | Hiển thị quick actions dưới message |
| Xử lý quick actions | [frontend/src/pages/AgentChatPage.jsx](./frontend/src/pages/AgentChatPage.jsx#L29) `handleQuickAction` | Điền prompt mới hoặc gửi yêu cầu giải thích |
| Biểu đồ SHAP | [frontend/src/components/ShapChart.jsx](./frontend/src/components/ShapChart.jsx#L8) `ShapChart` | Vẽ groups/contributions bằng Recharts |

## 3. Các luồng báo cáo thường gặp

### A. Người dùng hỏi thông thường

1. UI gửi nội dung qua `useChat.sendMessage`.
2. [supervisor.py](./backend/supervisor.py) gọi Agent với toàn bộ messages.
3. Nếu Agent phát tool call, `agent_tools.py` gọi `graph.run_forecast`.
4. Nếu không có `tool_calls`, `route_agent` trả `END`.
5. Frontend hiển thị câu trả lời Markdown.

**Nơi kiểm tra khi lỗi:** `useChat.js` → `graph.py` → `agent_node.py`.

### B. Người dùng yêu cầu dự báo có ngày

1. Gemini tạo tool call `predict_revenue`.
2. `ToolNode` thực thi tool.
3. `predict_revenue` chuẩn hóa ngày và gọi `graph.run_forecast`.
4. Forecast Graph chạy `feature_node` rồi `ml_node`.
5. `ml_node` gọi `ensemble_engine.run_ensemble`.
6. Ensemble tạo cả Revenue và COGS, kèm `groups` và `contributions`.
7. `save_forecast_node` lưu kết quả vào `last_forecast`.
8. Graph quay lại `agent_node` để tạo phản hồi tự nhiên.
9. Agent thêm marker `FORECAST_DATA`; frontend hiển thị kết quả và nút thao tác.

**Nơi kiểm tra khi lỗi:** `agent_tools.py` → `feature_node.py` →
`ml_node.py` → `ensemble_engine.py` → `save_forecast_node.py`.

### C. Người dùng yêu cầu dự báo nhưng thiếu ngày

Agent xử lý trực tiếp trong [supervisor.py](./backend/supervisor.py), hỏi lại
ngày và **không gọi** `predict_revenue`.

**Nơi kiểm tra khi lỗi:** `SYSTEM_PROMPT`, `PREDICTION_WORDS` và lịch sử
messages trong `supervisor.agent_process`.

### D. Người dùng yêu cầu giải thích

1. Frontend gửi câu `Giải thích kết quả dự đoán vừa rồi`.
2. Agent đọc `last_forecast` trong state.
3. Agent không gọi lại `predict_revenue`.
4. Agent tạo lời giải thích và thêm marker `FORECAST_DATA`.
5. `useChat.js` parse marker.
6. `ChatMessage.jsx` truyền dữ liệu vào `ShapChart.jsx`.

**Nơi kiểm tra khi lỗi:** `last_forecast` → `_forecast_marker` →
`FORECAST_DATA_PATTERN` → `forecast_results` → `ShapChart`.

## 4. Công thức ML và SHAP

### Ensemble

Trong [backend/ml_engine/final_model_shap.py](./backend/ml_engine/final_model_shap.py):

```text
TwoPeriod = 0.45 * expm1(P1) + 0.55 * expm1(P2)
Shape = 0.8 * expm1(M2)
Final = 0.5 * TwoPeriod + 0.5 * Shape
```

### Giá trị cơ sở và đóng góp

`shap_one_row` lấy `base_value` trong log-space từ LightGBM. `explain_final`
gộp base của các model, sau đó dùng cùng một hệ số quy đổi để đảm bảo:

```text
base_value + sum(contributions.money) = prediction
```

### Nhóm SHAP

| Nhóm | Feature tiêu biểu |
|---|---|
| Seasonality | Fourier, month, doy, quarter, weekofyear |
| Lunar calendar | days_to_tet ngoài cửa sổ Tết |
| Tet | days_to_tet, tet_before, tet_after |
| Edge-of-month | dom, is_som, is_eom, dom_reverse |
| Weekly | dow, is_weekend |
| Promotion | promo, discount và promotion count |
| Special day | is_special_day |
| Year-parity | is_odd_year |

## 5. Bảng trạng thái quan trọng

| Tên | Nơi tạo/cập nhật | Ý nghĩa |
|---|---|---|
| `messages` | CopilotKit + `ForecastState` | Lịch sử hội thoại trong thread |
| `target_date` | Tool arguments | Ngày cần dự báo |
| `discount_pct` | Tool arguments | Promotion theo phần trăm |
| `fixed_discount` | Tool arguments | Promotion cố định bằng VND |
| `ml_features` | `feature_node.process` | Một dòng feature Two-Period |
| `forecast_results` | `ml_node.process` | Kết quả Revenue/COGS hiện tại |
| `last_forecast` | `save_forecast_node.process` | Kết quả gần nhất để giải thích |
| `FORECAST_DATA` | `agent_node._forecast_marker` | Dữ liệu nội bộ cho frontend vẽ chart |

## 6. Tra cứu theo lỗi

| Triệu chứng | Kiểm tra đầu tiên |
|---|---|
| Agent không trả lời | `backend/.env`, `agent_node.process`, log `RUN_ERROR` |
| Thiếu API key Gemini | `load_dotenv` và `GOOGLE_API_KEY`/`GEMINI_API_KEY` trong [backend/agent_node.py](./backend/agent_node.py#L9) |
| Dự báo không chạy | `SYSTEM_PROMPT`, tool call và `route_agent` |
| Hỏi dự báo nhưng agent hỏi lại | Người dùng chưa cung cấp ngày hoặc ngày không được Gemini trích xuất |
| Chỉ có Revenue, thiếu COGS | `ensemble_engine.MODELS` và `forecast_results.cogs` |
| Số dự báo sai | `feature_node`, thứ tự feature trong model và `ensemble_engine` |
| Biểu đồ không hiện | Marker trong `agent_node`, parser trong `useChat.js`, `ShapChart` |
| Biểu đồ bị gọi/hiện hai lần | Kiểm tra quick action và không thêm tool call riêng cho chart |
| Giải thích dự đoán lại | `last_forecast` trong state và rule giải thích của `SYSTEM_PROMPT` |
| F5 vẫn thấy hội thoại cũ | `threadId` trong [frontend/src/App.jsx](./frontend/src/App.jsx#L9) và checkpoint graph |
| Lỗi ngày lễ/Tết/promotion | [backend/ml_engine/features_shape_13_19.py](./backend/ml_engine/features_shape_13_19.py) và `promotions.csv` |

## 7. Các file cấu hình và dữ liệu

| File | Dùng cho |
|---|---|
| [backend/.env](./backend/.env) | API key Gemini; không commit hoặc đưa giá trị vào báo cáo |
| [backend/requirements.txt](./backend/requirements.txt) | Dependency Python |
| [frontend/package.json](./frontend/package.json) | Dependency và script frontend |
| [backend/calendar_features.parquet](./backend/calendar_features.parquet) | Feature store Two-Period |
| [backend/ml_engine/promotions.csv](./backend/ml_engine/promotions.csv) | Lịch promotion cho Shape |
| [backend/models/](./backend/models/) | Các model LightGBM |
| [diagram.mmd](./diagram.mmd) | Sơ đồ sequence của hệ thống |

## 8. Quy ước khi ghi báo cáo lỗi

Khi mô tả một lỗi, nên ghi đủ:

```text
Luồng: QA / Dự báo / Giải thích / Frontend chart
Input: câu người dùng, ngày, promotion
Kết quả mong đợi:
Kết quả thực tế:
File/symbol nghi ngờ:
Log hoặc ảnh lỗi:
```

Ví dụ:

```text
Luồng: Giải thích
Input: "Giải thích kết quả dự đoán vừa rồi"
Kết quả mong đợi: Một lời giải thích + biểu đồ SHAP, không gọi predict lần hai
File/symbol nghi ngờ: agent_node.process, save_forecast_node.process,
useChat.forecastData, ShapChart
```
