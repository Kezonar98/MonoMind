from typing import Annotated, TypedDict, Sequence, Optional
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The central memory structure of our LangGraph pipeline.
    This dictionary flows through every node, accumulating data as it goes.
    """
    # Core Chat Data
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str                            # User identifier (e.g., Telegram ID as a string to prevent overflow)
    translated_text: Optional[str]          
    intent: Optional[str]                   
    
    # Financial Data from Database & Math Engine
    extracted_transactions: list[dict]      
    financial_result: Optional[float]       
    metrics: Optional[dict]                 
    
    # Specific Fields for Purchase Evaluation Logic
    purchase_data: Optional[dict]           
    purchase_risk: Optional[dict]           
    market_context: Optional[str]   

    # Vision and URL Contexts
    image_base64: Optional[str]
    vision_context: Optional[str]
    url_context: Optional[str]              # Stores the scraped Markdown text from links