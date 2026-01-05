from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
from src.embeddings.chunking import DocumentChunker, ChunkConfig
from src.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from src.vector_db.chromadb_handler import ChromaVectorDB


#End-to-end pipeline from PDF text to embedded chunks in vector DB

class EmbeddingPipeline: 
    def __init__(self, chunk_config: ChunkConfig = None, embedding_config: EmbeddingConfig = None, ):
        self.chunk_config = chunk_config or ChunkConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        
        self.chunker = DocumentChunker(self.chunk_config)
        self.embedding_gen = EmbeddingGenerator(self.embedding_config)
        self.vector_db = ChromaVectorDB() 
    
    #process charity returns dictionary with stats and results 
    def process_charity(self, charity_name: str, document_text: str, document_metadata: Dict = None) -> Dict: 
        logger.info(f"Starting embedding pipeline for charity: {charity_name}")

        try: 
            #Chunk the document
            logger.info("Chunking document...")
            chunks = self.chunker.chunk_document(
                document_text, 
                metadata=document_metadata or {}
            )

            #Evaluate chunking results 
            chunks_stats = self.chunker.evaluate_chunking(chunks)
            logger.info(f"Chunking stats: {chunks_stats}")

            #Generate embeddings for chunks 
            logger.info("Generating embeddings...")
            chunks = self.embedding_gen.embed_chunks(chunks)

            #Create colllection in vector DB 
            logger.info("Creating vector DB collection...")
            collection_name = charity_name.lower().replace(" ", "_")
            self.vector_db.create_collection(
                name=collection_name, 
                metadata={
                    'charity_name': charity_name,
                    'chunk_strategy': self.chunk_config.strategy,
                    'chunk_size': self.chunk_config.chunk_size,
                    'embedding_model': self.embedding_config.model_name
                }
            )

            #Add chunks to vector DB
            logger.info("Adding chunks to vector DB...")
            self.vector_db.add_chunks(chunks)

            #Get collection stats
            stats = self.vector_db.get_collection_stats()

            result = {
                'status': 'success',
                'charity_name': charity_name,
                'collection_name': collection_name,
                'chunking_stats': chunks_stats,
                'db_stats': stats,
                'embedding_model_info': self.embedding_gen.get_model_info()
            }

            logger.info(f"Embedding pipeline completed successfully for {charity_name}")
            return result
        except Exception as e:
            logger.error(f"Embedding pipeline failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'charity_name': charity_name,
                'error': str(e)
            }
        
    #Search for relevant chunks for a query 
    def search(self, query: str, charity_name: str, n_results: int = 5) -> List[Dict]: 
        logger.info(f"Searching for: '{query}' in charity: {charity_name}")

        try: 
            #Get the collection 
            query_embedding = self.embedding_gen.embed_query(query)

            #search 
            results = self.vector_db.search(
                query_embedding=query_embedding,
                n_results=n_results,
                name=charity_name
            )

            return results
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []





            
        

        