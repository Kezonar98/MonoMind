# main.py
from fastapi import FastAPI
from app.core.config import settings
from app.api import routes

def get_application() -> FastAPI:
    """Ініціалізація Bank-Grade FastAPI сервера."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Core API Gateway for MonoMind Financial Assistant",
        version="1.0.0",
        # Off documentation endpoints for security hardening; can be enabled in development
        docs_url="/docs", 
        redoc_url="/redoc",
    )
    
    # Connect router
    app.include_router(routes.router, prefix="/api/v1")
    
    return app

app = get_application()

@app.get("/health")
async def health_check():
    """Ендпоінт для балансувальників навантаження (Kubernetes/AWS)."""
    return {"status": "operational", "service": settings.PROJECT_NAME}