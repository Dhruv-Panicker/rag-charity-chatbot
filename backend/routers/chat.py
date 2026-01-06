from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from loguru import logger
import uuid

from backend.models import ChatRequest, ChatResponse, Source
from backend.dependencies import SESSIONS, get_rag_system, get_session
from src.rag.rag_system import RAGSystem

router = APIRouter(prefix="/chat", tags=["chat"])

#Main chat endpoint, returns ChatResponse with answer and sources
@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, rag: RAGSystem = Depends(get_rag_system)): 
    try: 
        logger.info(f"Received chat request: {request.query}")
        
        #Generate session ID 
        session_id = request.session_id or str(uuid.uuid4())

        #Get or create session 
        session = get_session(session_id)

        #Query RAG system 
        result = rag.query(
            query=request.query, 
            charity_name=request.charity_name, 
            top_k=request.top_k
        )

        #Store in session history
        session['history'].append({
            'query': request.query,
            'response': result.get('response'),
            'timestamp': datetime.now().isoformat() 
        })

        #Format response 
        response = ChatResponse(
            status=result.get('status', 'success'),
            query=request.query,
            response=result.get('response', 'No response generated'),
            sources=[
                Source(**source) for source in result.get('sources', [])
            ],
            retrieved_chunks=result.get('retrieved_chunks', 0),
            processing_time=result.get('processing_time', 0.0),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            session_id=session_id
        )

        logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

#Get conversation history for a session    
@router.get("/history/{session_id}")
async def get_chat_history(session_id: str): 
    try: 
        session = get_session(session_id)
        return {
            "status": "success",
            "session_id": session_id,
            "history": session['history'],
            "message_count": len(session['history'])
        }
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )

#Clear conversation history for a session 
@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str): 
    try: 
        if session_id in SESSIONS: 
            SESSIONS[session_id]['history'] = []
        
        return {
            "status": "success",
            "message": f"History cleared for session {session_id}"
        }
    except Exception as e:
        logger.error(f"History clear error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear history: {str(e)}"
        )
