from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from loguru import logger
import sys
import uvicorn

from backend.routers import chat, scrape
from backend.models import HealthResponse, ErrorResponse
from backend.dependencies import get_rag_system

#Configure logging 
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Create FastAPI app
app = FastAPI(
    title="RAG Charity Chatbot API",
    description="API for querying and indexing charity information using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(scrape.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "status": "operational",
        "service": "RAG Charity Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "chat": "/chat/",
            "scrape": "/scrape/"
        }
    }

# Health check endpoint 
@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        rag = get_rag_system()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            components={
                "rag_system": "operational",
                "vector_db": "operational",
                "llm_provider": "operational"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            components={
                "error": str(e)
            }
        )

#Global exception handler   
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error",
            message=f"Internal server error: {str(exc)}",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__": 
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  
        log_level="info"
    )