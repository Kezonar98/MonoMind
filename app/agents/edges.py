from app.agents.state import AgentState

def route_based_on_intent(state: AgentState) -> str:
    """Routes the graph based on the initial intent detected by Node 1."""
    if state.get("intent") in ["get_balance", "analyze_runway", "evaluate_purchase"]:
        return "fetch_ledger_data"
    return "generate_final_response"

def route_after_math(state: AgentState) -> str:
    """Routes the graph after core math processing is complete."""
    if state.get("intent") == "evaluate_purchase":
        return "analyze_image"
    return "generate_final_response"