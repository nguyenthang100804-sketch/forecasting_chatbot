from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from nodes.state import ForecastState
from nodes import intent_node, feature_node, ml_node, nlg_node
from langchain_core.messages import AIMessage

def route_intent(state: ForecastState):
    if state.get("is_prediction_query") == False or state.get("clarification_needed"):
        return "fallback_node"
    return "feature_node"

def fallback_node(state: ForecastState):
    reply = state.get("clarification_needed")
    if not isinstance(reply, str) or not reply.strip():
        if state.get("is_prediction_query") is False:
            reply = "Mình có thể hỗ trợ dự báo doanh thu. Bạn muốn dự báo cho ngày nào?"
        else:
            reply = "Bạn muốn dự báo cho ngày nào? Ví dụ: 25/12/2026 hoặc 'ngày mai'."
    return {"final_reply": reply, "messages": [AIMessage(content=reply)]}

builder = StateGraph(ForecastState)

builder.add_node("intent_node", intent_node.process)
builder.add_node("fallback_node", fallback_node)
builder.add_node("feature_node", feature_node.process)
builder.add_node("ml_node", ml_node.process)
builder.add_node("nlg_node", nlg_node.process)

builder.set_entry_point("intent_node")
builder.add_conditional_edges("intent_node", route_intent)
builder.add_edge("fallback_node", END)
builder.add_edge("feature_node", "ml_node")
builder.add_edge("ml_node", "nlg_node")
builder.add_edge("nlg_node", END)

# AG-UI uses thread_id to load and continue conversations, which requires a
# checkpointer. InMemorySaver is suitable for local development; use a durable
# saver (Postgres/Redis) when deploying multiple workers.
forecast_graph = builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    print("Đang khởi chạy đồ thị test...")
    res = forecast_graph.invoke({"messages": [HumanMessage(content="Dự báo ngày mai giảm 20%")]})
    print("\n[AI Reply]:", res.get("final_reply"))
