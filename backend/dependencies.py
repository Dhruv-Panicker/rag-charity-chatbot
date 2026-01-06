import os
from functools import lru_cache
from loguru import logger
from dotenv import load_dotenv

from src.rag.rag_system import RAGSystem, RAGConfig
from src.retrieval.retriever import RetrievalConfig
from src.llm.llm_client import LLMConfig

#Load environment variables from .env file
load_dotenv()   

#Get or create RAG system instance (singleton)
@lru_cache()
def get_rag_system() -> RAGSystem: 
    logger.info("Initializing RAG System...")

    #Get API key 
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: 
        logger.error("OPENAI_API_KEY not set in environment variables")
        raise ValueError("OPENAI_API_KEY not set")
    
    #Create configuration 
    config = RAGConfig(
        retrieval_config = RetrievalConfig(
            top_k=5, 
            similarity_threshold=0.3,
            rerank=True,
            debug=False
        ), 
        llm_config=LLMConfig(
            provider="openai", 
            model_name=os.getenv("OPENAI_MODEL"), 
            api_key=api_key, 
            temperature=0.7, 
            max_tokens=1024
        ), 
        max_context_tokens=10000, 
        debug=False
    )

    #Intialize RAG system
    rag = RAGSystem(config)
    logger.info("RAG System initialized successfully")

    return rag 

#Session storage (in-memory)
SESSIONS = {}

#Get or create session data
def get_session(session_id: str) -> dict: 
    if session_id not in SESSIONS: 
        SESSIONS[session_id] = {
            'history': [],
            'created_at': None
        }
    return SESSIONS[session_id]


    