"""
Configuration management for RAG Charity Chatbot
Loads settings from environment variables and .env file
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 1024

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Vector Database Configuration
    vector_db_type: str = "chroma"  # chroma or pinecone
    chroma_db_path: str = "./data/vector_store"
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "charity-rag-index"

    # Web Scraping Configuration
    headless_browser: bool = True
    browser_timeout: int = 30
    max_pages_to_scrape: int = 50
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

    # PDF Generation
    pdf_output_dir: str = "./data/pdfs"
    pdf_chunk_size: int = 2000

    # Chunking Strategy
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_method: str = "semantic"  # semantic, fixed, or sliding

    # RAG Configuration
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.3

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # Environment
    environment: str = "development"  # development or production

    class Config:
        """Pydantic config"""
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()
