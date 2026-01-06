from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime


#Request model for chat endpoint 
class ChatRequest(BaseModel): 
    query: str = Field(..., min_length=1, max_length=500, description="User's question")
    charity_name: Optional[str] = Field(None, description="Filter by charity name")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top relevant chunks to retrieve")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")

#Retrieved source/chunk 
class Source(BaseModel): 
    text: str
    similarity: float
    metadata: Dict

#Response model for chat endpoint 
class ChatResponse(BaseModel): 
    status: str
    query: str
    response: str
    sources: List[Source]
    retrieved_chunks: int
    processing_time: float
    timestamp: str
    session_id: Optional[str] = None

#Request model for scraping endpoint 
class ScrapeRequest(BaseModel):
    charity_name: str = Field(..., min_length=1, max_length=200)
    url: str = Field(..., description="URL to scrape")
    scrape_type: str = Field("homepage", description="Type of scraping: homepage, sitemap, specific")

#Response model for scraping endpoint 
class ScrapeResponse(BaseModel): 
    status: str
    charity_name: str
    chunks_indexed: int
    processing_time: float
    message: str

#Health check response 
class HealthResponse(BaseModel): 
    status: str
    timestamp: str
    components: Dict[str, str]

#Error response model 
class ErrorResponse(BaseModel): 
    status: str = "error"
    message: str
    

