from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.edges import route_based_on_intent, route_after_math

from app.agents.nodes.input_nodes import translate_user_input, extract_url_content
from app.agents.nodes.router_node import analyze_intent
from app.agents.nodes.finance_nodes import fetch_ledger_data, run_math_engine
from app.agents.nodes.purchase_nodes import analyze_image, extract_purchase_info, convert_currency, fetch_market_price, analyze_purchase_risk
from app.agents.nodes.response_node import generate_final_response

# 1. Create graph
workflow = StateGraph(AgentState)

# 2. Register all nodes
workflow.add_node("translate_input", translate_user_input)
workflow.add_node("extract_url_content", extract_url_content) 
workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("fetch_ledger_data", fetch_ledger_data)
workflow.add_node("run_math_engine", run_math_engine)
workflow.add_node("analyze_image", analyze_image)
workflow.add_node("extract_purchase_info", extract_purchase_info)
workflow.add_node("convert_currency", convert_currency) 
workflow.add_node("fetch_market_price", fetch_market_price)
workflow.add_node("analyze_purchase_risk", analyze_purchase_risk)
workflow.add_node("generate_final_response", generate_final_response)

# 3. Define data flow (Edges)
workflow.set_entry_point("translate_input")
workflow.add_edge("translate_input", "extract_url_content")   
workflow.add_edge("extract_url_content", "analyze_intent")    

workflow.add_conditional_edges(
    "analyze_intent",
    route_based_on_intent,
    {
        "fetch_ledger_data": "fetch_ledger_data",
        "generate_final_response": "generate_final_response"
    }
)

workflow.add_edge("fetch_ledger_data", "run_math_engine")

workflow.add_conditional_edges(
    "run_math_engine",
    route_after_math,
    {
        "analyze_image": "analyze_image",
        "generate_final_response": "generate_final_response"
    }
)

workflow.add_edge("analyze_image", "extract_purchase_info")
workflow.add_edge("extract_purchase_info", "convert_currency") 
workflow.add_edge("convert_currency", "fetch_market_price") 
workflow.add_edge("fetch_market_price", "analyze_purchase_risk")
workflow.add_edge("analyze_purchase_risk", "generate_final_response") 
workflow.add_edge("generate_final_response", END)

# Export graph
app_workflow = workflow