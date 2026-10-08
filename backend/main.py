from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from graph import forecast_graph

app = FastAPI(title="Revenue Prediction Agent API (LangGraph + CopilotKit)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến đồ thị thành một AG-UI agent.
agent = LangGraphAGUIAgent(
    name="default",
    description="Chuyên gia dự báo doanh thu và phân tích tác động SHAP bằng LightGBM.",
    graph=forecast_graph,
)

# LangGraphAGUIAgent implements run(), so it must use the AG-UI endpoint.
# CopilotKitRemoteEndpoint is the legacy endpoint and calls agent.execute().
add_langgraph_fastapi_endpoint(app, agent, path="/api/copilotkit")

@app.get("/")
def read_root():
    return {"status": "Backend LangGraph + CopilotKit đang hoạt động tốt!"}
