# Code Graph: Kiến Trúc & Luồng Điều Khiển Forecast Chatbot

---

## 1. Toàn Cảnh Kiến Trúc Hệ Thống (Architecture Code Graph)

Sơ đồ thể hiện sự kết nối giữa các tầng: **Frontend (React)** ➔ **API Gateway (FastAPI)** ➔ **Supervisor Agent (LangGraph + Gemini)** ➔ **Forecast Sub-Graph** ➔ **ML Ensemble Engine** ➔ **Storage & Data Sources**.

```mermaid
flowchart TB
    %% Styling
    classDef ui fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef api fill:#ede7f6,stroke:#512da8,stroke-width:2px,color:#311b92;
    classDef agent fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef pipeline fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef ml fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;
    classDef data fill:#f5f5f5,stroke:#616161,stroke-width:2px,color:#212121;

    subgraph FRONTEND [" Tầng Frontend (React + Vite) "]
        APP["App.jsx<br/>(CopilotKit Provider, threadId)"]:::ui
        CHAT_PAGE["AgentChatPage.jsx<br/>(Giao diện hội thoại & Quick Actions)"]:::ui
        HOOK_CHAT["useChat.js<br/>(Parse FORECAST_DATA marker)"]:::ui
        MSG_COMP["ChatMessage.jsx<br/>(Render tin nhắn & Action buttons)"]:::ui
        SHAP_CHART["ShapChart.jsx<br/>(Vẽ biểu đồ SHAP bằng Recharts)"]:::ui

        APP --> CHAT_PAGE
        CHAT_PAGE --> HOOK_CHAT
        CHAT_PAGE --> MSG_COMP
        MSG_COMP --> SHAP_CHART
    end

    subgraph RUNTIME_LAYER [" Tầng CopilotKit Runtime (Express, Node) "]
        EXPRESS["CopilotRuntime<br/>(Express server)"]:::api
    end

    subgraph API_GATEWAY [" Tầng Backend Gateway (FastAPI) "]
        MAIN["backend/main.py<br/>FastAPI Server"]:::api
        COPILOT_EP["Endpoint /api/copilotkit<br/>LangGraphAGUIAgent"]:::api
        ROOT_EP["GET /<br/>Health Check"]:::api

        MAIN --> COPILOT_EP
        MAIN --> ROOT_EP
    end

    subgraph SUPERVISOR_LAYER [" Tầng Agent Giám Sát LangGraph (supervisor.py, có LLM) "]
        SUP_GRAPH["supervisor_graph<br/>StateGraph(ForecastState)<br/>(bộ chạy graph)"]:::agent
        AGENT_NODE["agent_node (agent_process)<br/>Gemini 3.5 Flash + SYSTEM_PROMPT<br/>+ thêm marker FORECAST_DATA ở response cuối"]:::agent
        ROUTE_CHECK{"route_agent<br/>Kiểm tra Tool Calls?"}:::agent
        TOOL_NODE["tool_node (ToolNode)<br/>Thực thi TOOLS"]:::agent
        SAVE_NODE["save_forecast_node<br/>(save_forecast_function.process)"]:::agent
        CHECKPOINTER[("InMemorySaver<br/>Lưu context theo thread_id")]:::agent

        SUP_GRAPH -- "START" --> AGENT_NODE
        AGENT_NODE --> ROUTE_CHECK
        ROUTE_CHECK -- "Có tool_calls" --> TOOL_NODE
        ROUTE_CHECK -- "Không tool_calls" --> END_SUP([KẾT THÚC VÒNG]):::agent
        TOOL_NODE --> SAVE_NODE
        SAVE_NODE --> AGENT_NODE
        SUP_GRAPH <-. lưu và khôi phục state .-> CHECKPOINTER
    end

    subgraph FORECAST_PIPELINE [" Tầng Dịch Vụ Dự Báo (agent_tools.py + graph.py) "]
        TOOL_PREDICT["agent_tools.py<br/>predict_revenue()"]:::pipeline
        F_GRAPH["forecast_graph<br/>StateGraph(ForecastState)"]:::pipeline
        FEAT_NODE["feature_node (process)<br/>Tra cứu & Override Promotion"]:::pipeline
        ML_NODE["ml_node (process)<br/>Gọi Ensemble Engine"]:::pipeline

        TOOL_PREDICT -- "run_forecast" --> F_GRAPH
        F_GRAPH --> FEAT_NODE
        FEAT_NODE --> ML_NODE
        ML_NODE --> END_F([KẾT THÚC DỰ BÁO]):::pipeline
    end

    subgraph ML_SHAP_ENGINE [" Tầng Mô Hình Máy Học & SHAP Engine "]
        ENS_ENGINE["ensemble_engine.py<br/>run_ensemble() & _cached_ensemble()"]:::ml
        FINAL_SHAP["ml_engine/final_model_shap.py<br/>explain_final()"]:::ml

        ENS_ENGINE --> FINAL_SHAP
    end

    subgraph DATA_STORAGE [" Tầng Dữ Liệu & Artifacts "]
        PARQUET_STORE[("calendar_features.parquet<br/>2023 - 2030 Feature Store")]:::data
        MODELS_LGB[("LightGBM Models (.txt)<br/>Revenue & COGS (6 models)")]:::data
    end

    %% Giao tiếp liên tầng
    HOOK_CHAT <== "HTTP / Streaming" ==> EXPRESS
    EXPRESS <== "AG-UI" ==> COPILOT_EP
    COPILOT_EP --> SUP_GRAPH
    TOOL_NODE -- "gọi, nhận JSON" --> TOOL_PREDICT
    ML_NODE --> ENS_ENGINE
    FEAT_NODE -- "đọc" --> PARQUET_STORE
    ENS_ENGINE -- "nạp 6 model" --> MODELS_LGB
```

---

## 2. Đồ Thị Trạng Thái LangGraph Chi Tiết (State Graph)

Dự án gồm **hai đồ thị LangGraph lồng nhau** (Nested / Supervisor Pattern):

### 2.1. Supervisor Graph (`backend/supervisor.py`)
Đóng vai trò điều phối viên hội thoại chính của ứng dụng:

```mermaid
stateDiagram-v2
    [*] --> agent_node: Nhận tin nhắn từ người dùng

    agent_node : agent_node
    agent_node : Gọi LLM Gemini 3.5 Flash với SYSTEM_PROMPT và TOOLS

    state route_agent <<choice>>

    tool_node : tool_node
    tool_node : Thực thi predict_revenue (target_date, discount_pct, fixed_discount)

    save_forecast_node : save_forecast_node
    save_forecast_node : Trích xuất forecast_results từ ToolMessage và cập nhật last_forecast

    Format_Response : Format_Response
    Format_Response : Nếu có last_forecast và câu hỏi liên quan, gắn marker FORECAST_DATA

    agent_node --> route_agent: Đánh giá output của LLM
    route_agent --> tool_node: Nếu phát sinh tool_calls (predict_revenue)
    route_agent --> Format_Response: Nếu KHÔNG có tool_calls

    tool_node --> save_forecast_node
    save_forecast_node --> agent_node: Quay lại Agent với kết quả trong context

    Format_Response --> [*]: Trả lời người dùng (END)
```

### 2.2. Deterministic Forecast Graph (`backend/graph.py`)
Đồ thị con thuần túy xử lý thuật toán và không phụ thuộc LLM:

```mermaid
stateDiagram-v2
    [*] --> feature_node: target_date, discount_pct, fixed_discount

    state feature_node {
        [*] --> Tra_Cuu_Parquet: Đọc calendar_features.parquet
        Tra_Cuu_Parquet --> Kiem_Tra_Override: Có discount_pct hoặc fixed_discount?
        Kiem_Tra_Override --> Ep_Kieu_LightGBM: Chép đè promo_season, discount_type
        Ep_Kieu_LightGBM --> [*]: Xuất ra ml_features (27 features)
    }

    feature_node --> ml_node: ml_features

    state ml_node {
        [*] --> Chay_Ensemble: ensemble_engine.run_ensemble()
        Chay_Ensemble --> Cache_Lookup: _cached_ensemble()
        Cache_Lookup --> Tinh_TwoPeriod_Shape: 2 Phase Two-Period + Shape 13-19
        Tinh_TwoPeriod_Shape --> Tinh_TreeSHAP: explain_final() cho Revenue & COGS
        Tinh_TreeSHAP --> [*]: Xuất ra forecast_results
    }

    ml_node --> [*]: Trả về kết quả hoàn chỉnh
```

---

## 3. Sơ Đồ Phụ Thuộc Mã Nguồn & Lời Gọi Hàm (Call Graph)

Sơ đồ dưới đây chi tiết các lời gọi hàm từ điểm vào (Entry Point) đến các hàm tính toán lõi:

```mermaid
flowchart LR
    subgraph S1 [" main.py "]
        F_INIT["FastAPI()"]
        F_AGENT["LangGraphAGUIAgent(graph=supervisor_graph)"]
        F_EP["add_langgraph_fastapi_endpoint()"]
    end

    subgraph S2 [" supervisor.py "]
        S_AGENT["agent_process(state)"]
        S_LLM["ChatGoogleGenerativeAI.invoke()"]
        S_ROUTE["route_agent(state)"]
        S_MARKER["_forecast_marker(forecast)"]
        S_CTX["_format_forecast_context()"]
    end

    subgraph S3 [" agent_tools.py "]
        T_NORM["_normalize_date(value)"]
        T_PRED["predict_revenue(target_date, ...)"]
    end

    subgraph S4 [" graph.py "]
        G_RUN["run_forecast(target_date, ...)"]
        G_INVOKE["forecast_graph.invoke()"]
    end

    subgraph S5 [" nodes/ "]
        N_FEAT["feature_node.process(state)"]
        N_ML["ml_node.process(state)"]
        N_SAVE["save_forecast_function.process(state)"]
    end

    subgraph S6 [" ensemble_engine.py "]
        E_RUN["run_ensemble()"]
        E_CACHE["_cached_ensemble()"]
        E_TP_FRAME["_two_period_frame()"]
        E_SH_FRAME["_shape_frame()"]
        E_FACTORS["_top_factors()"]
    end

    subgraph S7 [" ml_engine/ "]
        M_SHAPE["features_shape_13_19.build_features()"]
        M_SHAP["final_model_shap.explain_final()"]
        M_ONEROW["final_model_shap.shap_one_row()"]
    end

    %% Connections
    F_AGENT --> S_AGENT
    S_AGENT --> S_LLM
    S_AGENT --> S_ROUTE
    S_AGENT --> S_MARKER
    S_AGENT --> S_CTX

    S_ROUTE --> T_PRED
    T_PRED --> T_NORM
    T_PRED --> G_RUN
    G_RUN --> G_INVOKE

    G_INVOKE --> N_FEAT
    N_FEAT --> N_ML
    N_ML --> E_RUN
    T_PRED --> N_SAVE

    E_RUN --> E_CACHE
    E_CACHE --> E_TP_FRAME
    E_CACHE --> E_SH_FRAME
    E_CACHE --> E_FACTORS
    E_SH_FRAME --> M_SHAPE
    E_CACHE --> M_SHAP
    M_SHAP --> M_ONEROW
```

---

## 4. Sơ Đồ Biến Đổi Dữ Liệu (State & Data Flow Graph)

Trạng thái `ForecastState` được định nghĩa tại `backend/nodes/state.py` và biến đổi tuần tự qua các node:

```mermaid
flowchart TD
    STATE_0["<b>Trạng thái ban đầu:</b><br/>messages: [HumanMessage('Dự đoán ngày 2024-05-20')]"]
    
    STEP_AGENT_1["<b>1. Sau agent_node (lần 1):</b><br/>messages: + [AIMessage(tool_calls=[predict_revenue('2024-05-20')])]"]
    
    STEP_TOOL["<b>2. Sau tool_node:</b><br/>messages: + [ToolMessage(content='json_payload')]"]
    
    STEP_SAVE["<b>3. Sau save_forecast_node:</b><br/>target_date: '2024-05-20'<br/>discount_pct: 0.0<br/>fixed_discount: 0.0<br/>forecast_results: {revenue: {...}, cogs: {...}}<br/>last_forecast: {target_date, forecast_results, ...}"]
    
    STEP_AGENT_2["<b>4. Sau agent_node (lần 2):</b><br/>messages: + [AIMessage('Dự kiến doanh thu... &lt;!--FORECAST_DATA:{...}--&gt;')]"]

    UI_CONSUME["<b>5. Frontend useChat.js & ChatMessage.jsx:</b><br/>- Bóc tách văn bản giải thích tự nhiên hiển thị cho User<br/>- Parse JSON từ FORECAST_DATA<br/>- Truyền forecast_results vào ShapChart.jsx vẽ biểu đồ đóng góp"]

    STATE_0 --> STEP_AGENT_1
    STEP_AGENT_1 --> STEP_TOOL
    STEP_TOOL --> STEP_SAVE
    STEP_SAVE --> STEP_AGENT_2
    STEP_AGENT_2 --> UI_CONSUME
```

---

## 5. Sơ Đồ Thuật Toán Ensemble & SHAP Trees (ML Explainer Graph)

Mô hình dự báo sử dụng kiến trúc kết hợp **Two-Period** (Phase 1 & Phase 2) cùng **Shape Model** để tối ưu hóa độ chính xác:

```mermaid
flowchart TB
    subgraph INPUTS [" Đầu Vào Tính Toán "]
        DATE["target_date"]
        USER_PROMO["Khuyến Mãi (pct / fixed)"]
    end

    subgraph FEAT_BUILD [" Trích Xuất Đặc Trưng "]
        F_TP["27 Two-Period Features<br/>(Lịch + Parquet Store)"]
        F_SP["25 Shape Features<br/>(Lịch + promotions.csv)"]
    end

    subgraph MODEL_PRED [" Các Mô Hình LightGBM Booster "]
        P1["Phase 1 (log1p space)"]
        P2["Phase 2 (log1p space)"]
        M2["Shape Model (log1p space)"]
    end

    subgraph ENSEMBLE_CALC [" Công Thức Hợp Nhất (Ensemble) "]
        CALC_TP["TwoPeriod = 0.45 * expm1(P1) + 0.55 * expm1(P2)"]
        CALC_SP["Shape = 0.80 * expm1(M2)"]
        CALC_FINAL["Final = 0.5 * TwoPeriod + 0.5 * Shape"]
    end

    subgraph SHAP_ENGINE [" Phân Rã & Giải Thích SHAP "]
        SHAP_TREE["TreeSHAP từng model (Base Value + SHAP values)"]
        SHAP_GROUPS["Gộp vào 8 Nhóm Đặc Trưng:<br/>• Seasonality<br/>• Lunar calendar<br/>• Tet<br/>• Edge-of-month<br/>• Weekly<br/>• Promotion<br/>• Special day<br/>• Year-parity"]
        SHAP_SCALE["Quy đổi Log-space sang Tiền tệ (VND):<br/>base_value + sum(contributions.money) = prediction"]
        TOP_FACTORS["Xác định Top Drivers (tác động dương)<br/>& Top Barriers (tác động âm)"]
    end

    INPUTS --> FEAT_BUILD
    DATE --> F_TP & F_SP
    USER_PROMO --> F_TP & F_SP

    F_TP --> P1 & P2
    F_SP --> M2

    P1 & P2 --> CALC_TP
    M2 --> CALC_SP
    CALC_TP & CALC_SP --> CALC_FINAL

    P1 & P2 & M2 --> SHAP_TREE
    SHAP_TREE --> SHAP_GROUPS
    SHAP_GROUPS --> SHAP_SCALE
    SHAP_SCALE --> TOP_FACTORS
```

---

## 6. Bảng Tra Cứu Toàn Bộ Thành Phần Mã Nguồn (Symbol & File Map)

| Module / Lớp | Đường dẫn tệp | Ký hiệu / Hàm trọng tâm | Trách nhiệm |
|---|---|---|---|
| **API Entry** | [`backend/main.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/main.py) | `app`, `agent`, `/api/copilotkit` | Khởi tạo FastAPI server và tích hợp CopilotKit endpoint |
| **Supervisor** | [`backend/supervisor.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/supervisor.py) | `supervisor_graph`, `agent_process`, `route_agent` | Quản lý hội thoại, quyết định QA, gọi tool hoặc giải thích |
| **Agent Tools** | [`backend/agent_tools.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/agent_tools.py) | `predict_revenue`, `_normalize_date` | Chuẩn hóa định dạng ngày và gọi Forecast Pipeline |
| **Forecast Pipeline** | [`backend/graph.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/graph.py) | `forecast_graph`, `run_forecast` | Chuỗi node tất định (deterministic) từ feature đến ML |
| **State Definition** | [`backend/nodes/state.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/nodes/state.py) | `ForecastState` | Schema lưu trữ tin nhắn, tham số và kết quả dự báo |
| **Feature Node** | [`backend/nodes/feature_node.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/nodes/feature_node.py) | `process` | Tra cứu Parquet O(1), override khuyến mãi, chuẩn bị 27 features |
| **ML Node** | [`backend/nodes/ml_node.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/nodes/ml_node.py) | `process` | Đóng gói đầu vào và gọi bộ máy ensemble |
| **Save Memory** | [`backend/save_forecast_function.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/save_forecast_function.py) | `process` | Lưu `last_forecast` vào conversation checkpoint memory |
| **Ensemble Engine** | [`backend/ensemble_engine.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/ensemble_engine.py) | `run_ensemble`, `_cached_ensemble` | Tải 6 mô hình LightGBM, caching và tổng hợp kết quả |
| **Shape Features** | [`backend/ml_engine/features_shape_13_19.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/ml_engine/features_shape_13_19.py) | `build_features` | Tạo 25 đặc trưng Shape (Tết, âm lịch, khuyến mãi CSV) |
| **SHAP Explainer** | [`backend/ml_engine/final_model_shap.py`](file:///c:/Research%20Dataset/forecast-chatbot/backend/ml_engine/final_model_shap.py) | `explain_final`, `shap_one_row`, `GROUPS` | TreeSHAP, phân rã 8 nhóm và quy đổi đóng góp sang tiền tệ |
| **UI Chat Page** | [`frontend/src/pages/AgentChatPage.jsx`](file:///c:/Research%20Dataset/forecast-chatbot/frontend/src/pages/AgentChatPage.jsx) | `AgentChatPage`, `handleQuickAction` | Khung chat, xử lý nút bấm gợi ý và gửi prompt |
| **UI Hook** | [`frontend/src/hooks/useChat.js`](file:///c:/Research%20Dataset/forecast-chatbot/frontend/src/hooks/useChat.js) | `useChat`, `FORECAST_DATA_PATTERN` | Kết nối CopilotKit, bóc tách `FORECAST_DATA` marker |
| **UI Message** | [`frontend/src/components/ChatMessage.jsx`](file:///c:/Research%20Dataset/forecast-chatbot/frontend/src/components/ChatMessage.jsx) | `ChatMessage` | Hiển thị bong bóng chat Markdown và biểu đồ tương ứng |
| **UI SHAP Chart** | [`frontend/src/components/ShapChart.jsx`](file:///c:/Research%20Dataset/forecast-chatbot/frontend/src/components/ShapChart.jsx) | `ShapChart` | Trực quan hóa giá trị cơ sở và các nhóm tác động bằng Recharts |
