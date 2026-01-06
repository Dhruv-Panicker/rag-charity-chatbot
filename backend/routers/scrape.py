from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from loguru import logger
import time

from backend.models import ScrapeRequest, ScrapeResponse
from backend.dependencies import get_rag_system
from src.scraper.web_scraper import WebScraper, HTMLCleaner, SiteMapper
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.embeddings.chunking import ChunkConfig
from src.embeddings.embedding_generator import EmbeddingConfig

router = APIRouter(prefix="/scrape", tags=["scraping"])

#Scrape and index a charity website 
@router.post("/", response_model=ScrapeResponse)
async def scrape_charity(request: ScrapeRequest): 
    try:
        start_time = time.time()
        logger.info(f"Starting scrape for {request.charity_name} at {request.url}")
        
        # Initialize scraper components
        scraper = WebScraper(timeout=30, max_retries=3)
        cleaner = HTMLCleaner()
        
        # Choose scraping method based on type
        if request.scrape_type == "sitemap":
            # Scrape entire site using sitemap/site discovery
            logger.info("Scraping entire site via sitemap...")
            mapper = SiteMapper(request.url, max_pages=50)
            pages = mapper.discover_pages()
            
            all_text = []
            for page_url in pages:
                try:
                    html = scraper.scrape_with_requests(page_url)
                    if html:
                        cleaned_html = cleaner.clean(html)
                        text = cleaner.extract_text(cleaned_html)
                        if text:
                            all_text.append(text)
                    time.sleep(1)  # Respect server load
                except Exception as e:
                    logger.warning(f"Failed to scrape page {page_url}: {e}")
                    continue
            
            result = {
                'content': '\n\n'.join(all_text),
                'text': '\n\n'.join(all_text)
            }
        else:
            # Default: scrape homepage only
            logger.info("Scraping homepage only...")
            html = scraper.scrape_with_requests(request.url)
            if html:
                cleaned_html = cleaner.clean(html)
                text = cleaner.extract_text(cleaned_html)
                result = {'content': text, 'text': text}
            else:
                result = {'content': '', 'text': ''}
        
        #Check if scraping was successful
        if not result or not result.get('text'):
            raise HTTPException(
                status_code=400,
                detail="Failed to scrape content from URL. Check if the URL is valid."
            )
        
        logger.info(f"Scraped {len(result.get('content', ''))} characters of content")

        #Initialize embedding pipeline
        pipeline = EmbeddingPipeline(
            chunk_config=ChunkConfig(
                strategy="fixed", 
                chunk_size=256, 
                overlap=50
            ), 
            embedding_config=EmbeddingConfig()
        )
        #Process and index the scraped content
        content = result['content']
        index_result = pipeline.process_charity(
            charity_name=request.charity_name, 
            document_text=content
        )

        processing_time = time.time() - start_time
        
        response = ScrapeResponse(
            status="success",
            charity_name=request.charity_name,
            chunks_indexed=index_result['stats']['total_chunks'],
            processing_time=processing_time,
            message=f"Successfully scraped and indexed {index_result['stats']['total_chunks']} chunks from {request.charity_name}"
        )

        logger.info(f"Scraping completed in {processing_time:.2f}s - {response.chunks_indexed} chunks indexed")
        return response
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Scraping error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Scraping failed: {str(e)}"
        )
    
#List all indexed charities
@router.get("/charities")
async def list_charities(rag=Depends(get_rag_system)): 
    try: 
        #Get all collections from vector DB 
        collections = rag.vector_db.list_collections()

        #Get count for each collection 
        charity_info = []
        for collection_name in collections: 
            try: 
                collection = rag.vector_db.get_collection(collection_name)
                count = collection.count()
                charity_info.append({
                    "name": collection_name.replace('_', ' ').title(),
                    "collection_name": collection_name,
                    "chunk_count": count
                })
            except Exception as e:
                logger.warning(f"Failed to get info for collection '{collection_name}': {e}")
                charity_info.append({
                    "name": collection_name.replace('_', ' ').title(),
                    "collection_name": collection_name,
                    "chunk_count": 0
                })
        return {
            "status": "success",
            "charities": charity_info, 
            "count": len(charity_info)
        }
    except Exception as e:
        logger.error(f"List of charities error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list charities: {str(e)}"
        )
    






