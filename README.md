# MonoMind: Edge-AI Financial Assistant

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=for-the-badge&logo=postgresql)
![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-FF4B4B?style=for-the-badge)

MonoMind is an experimental, fully local financial assistant. It explores how to build an AI-driven FinTech application without sending sensitive financial data to cloud providers (like OpenAI) and without allowing the LLM to hallucinate numbers.

The core idea is strict separation of concerns: the LLM is only used for semantic routing and text formatting, while all math and data retrieval are handled deterministically by Python and PostgreSQL.

##  Architecture & Trade-offs

When building this MVP, I focused on solving two main problems: **Data Privacy** and **Hardware Constraints**.

### 1. The Memory Constraint (Running AI on Edge)
Running LLMs locally usually requires significant VRAM/RAM. This project was specifically optimized to run on a machine with less than 2.5 GiB of free RAM.
* **Solution:** We use `llama3.2:1b` via Ollama. 
* **Optimization:** By default, Ollama allocates a huge context window (~8k tokens). Since the router node only needs to classify a single sentence into a JSON intent, I applied strict memory pruning (`num_ctx=512` and `num_predict=50`). This prevented OOM (Out of Memory) kills and kept the routing process blazing fast.

### 2. The LangGraph State Machine
Instead of a standard conversational loop, the AI operates as a directed graph (State Machine):
1. **Semantic Router:** The local LLM reads the input and outputs strictly validated JSON (e.g., `{"intent": "get_balance"}`).
2. **Ledger Fetcher:** Triggers an async SQLAlchemy query to PostgreSQL.
3. **Deterministic Math:** Pure Python calculates the balance. The LLM is explicitly forbidden from doing math to prevent hallucinations.
4. **Presenter:** The LLM receives the calculated facts as a system prompt and formats a human-readable response.

### 3. The Data Layer (Event Sourced Ledger)
Standard CRUD doesn't work well for FinTech. The database follows an **Append-Only Ledger** pattern:
* Balances are never updated via `UPDATE`. They are calculated on the fly by aggregating `DEPOSIT`, `WITHDRAWAL`, and `SUBSCRIPTION` transactions.
* Managed via Alembic migrations with strict B-Tree indexing for fast aggregations.


 Tech Stack
* **API Gateway:** FastAPI (Ports & Adapters pattern)
* **Database:** PostgreSQL (Asyncpg) + Alembic
* **Orchestration:** LangGraph, Langchain
* **Inference:** Local Ollama (`Llama 3.2:1B`)

 Current Status & Next Steps

**What's done:**
- [x] Immutable PostgreSQL Ledger setup.
- [x] Local LLM integration with memory optimization.
- [x] LangGraph routing and deterministic math execution.

**In progress:**
- [ ] Exposing the LangGraph agent through a FastAPI endpoint (`POST /api/v1/chat/`).

**Planned:**
- [ ] Add runway prediction (burn rate analysis).
- [ ] Simple Telegram Bot integration for the client-side UI.

##Running Locally

```bash
# Clone the repository
git clone [https://github.com/yourusername/MonoMind.git](https://github.com/yourusername/MonoMind.git)

# Setup Virtual Environment
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Pull the lightweight local model
ollama pull llama3.2:1b

# Run Database Migrations
alembic upgrade head

# Start the API Gateway
uvicorn main:app --reload --port 8000
