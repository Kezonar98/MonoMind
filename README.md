#  MonoMind: Edge-AI Financial Assistant

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=for-the-badge&logo=postgresql)
![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-FF4B4B?style=for-the-badge)
![Aiogram](https://img.shields.io/badge/Telegram_UI-aiogram-2CA5E0?style=for-the-badge&logo=telegram)

MonoMind is an experimental, fully local financial AI assistant. It explores how to build an AI-driven FinTech application without sending sensitive financial data to cloud providers (like OpenAI) and without allowing the LLM to hallucinate numbers.

The core idea is a strict **separation of concerns**: the LLM is only used for semantic routing and text formatting, while all math, logic, and data retrieval are handled deterministically by Python and PostgreSQL.

## Architecture & Engineering Trade-offs

When building this MVP, I focused on solving two main problems: **Data Privacy** and **Hardware Constraints**.

### 1. The Memory Constraint (Running AI on Edge)
Running LLMs locally usually requires significant VRAM/RAM. This project was specifically optimized to run on a machine with limited memory.
* **Solution:** We use `llama3.2:1b` via Ollama. 
* **RAM Optimization:** I configured a Linux Swap file to prevent OS-level OOM killers during tensor loading. Additionally, context windows were strictly pruned (`num_ctx=512` and `num_predict=50` for the router node) to keep memory footprint minimal.
* **Linguistic Trade-off:** To maximize the 1B model's reasoning capabilities, the AI is strictly instructed to respond in English, bypassing the tokenization issues of Cyrillic languages in ultra-small models.

### 2. The LangGraph State Machine
Instead of a standard conversational loop, the AI operates as a directed graph:
1. **Semantic Router:** Determines if the user wants a balance check or a complex runway analysis.
2. **Ledger Fetcher:** Triggers an async SQLAlchemy query to PostgreSQL.
3. **Deterministic Math:** Pure Python calculates balance, burn rate, and financial runway. *The LLM is explicitly forbidden from doing math.*
4. **Presenter:** The LLM receives calculated facts as a system prompt and formats a human-readable response.

### 3. The Data Layer (Event-Sourced Ledger)
Standard CRUD doesn't work well for FinTech. The database follows an **Append-Only Ledger** pattern:
* Balances are never updated via `UPDATE`. They are calculated on the fly by aggregating `DEPOSIT` and `WITHDRAWAL` transactions.

##  Current Status (MVP Achieved)

**What's done:**
- [x] Immutable PostgreSQL Ledger setup via asyncpg & Alembic.
- [x] Local LLM integration with memory optimization.
- [x] LangGraph routing and deterministic math execution (Balance & Burn Rate).
- [x] Predictive Analytics: Financial Runway calculation.
- [x] FastAPI Gateway exposing the cognitive engine (`POST /api/v1/chat/`).
- [x] Hexagonal Architecture UI: Asynchronous Telegram Bot client connected to the core API.

##  Roadmap & Next Steps

This project is actively evolving. The upcoming milestones are divided into infrastructure, integrations, and advanced cognitive features:

### Phase 1: Infrastructure & DevOps
- [ ] **Dockerization:** Wrap the entire ecosystem (PostgreSQL, FastAPI Core, Telegram UI, and Ollama) into a single `docker-compose.yml` for one-click deployment.

### Phase 2: Real-World Data
- [ ] **Open Banking API Integration:** Connect to real-world bank APIs (e.g., Monobank Webhooks) to automatically sync and ingest live transactions into the PostgreSQL ledger.

### Phase 3: Advanced Cognitive Finance & RAG
- [ ] **Micro-Loan Risk Analysis:** If a user asks, *"Should I get a loan for a bag of chips?"*, the AI will calculate the loan term, assess cash-flow risks, and warn if monthly expenses (including the new loan) exceed income.
- [ ] **Macroeconomic RAG (Retrieval-Augmented Generation):** Integrate a news parser vector database. If a user asks about the Hryvnia (UAH) currency drop, the AI will fetch recent financial news, analyze the causes, and generate scenarios for currency appreciation based on real-world events.

### Phase 4: Experimental
- [ ] **Multilingual Support:** Experiment with fine-tuning or larger models to handle native Ukrainian processing without losing Edge deployment capabilities.

##  Running Locally
*(Instructions will be updated to `docker-compose up` in the next release)*
```bash
# Start the Core Backend
uvicorn main:app

# Start the Telegram UI Client
python app/ui/bot/main.py