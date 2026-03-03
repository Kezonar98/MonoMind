import os
from langchain_ollama import ChatOllama
from app.models.schemas import PurchaseExtraction

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Fetch the model name directly from the .env file, defaulting to qwen2.5:1.5b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# Fast, deterministic LLM used strictly for routing logic and data extraction
# INCREASED num_ctx to 2048 so it can read web page texts without truncating memory!
llm_router = ChatOllama(
    base_url=OLLAMA_URL,
    model=OLLAMA_MODEL, 
    temperature=0.0, 
    format="json",
    num_predict=50,      
    num_ctx=2048          
)

# Chat LLM used for generating the final human-readable response
llm_chat = ChatOllama(
    base_url=OLLAMA_URL,
    model=OLLAMA_MODEL, 
    temperature=0.0, 
    num_ctx=1024     
)

# Structured LLM forced to output data matching the PurchaseExtraction Pydantic schema
structured_llm = llm_router.with_structured_output(PurchaseExtraction)