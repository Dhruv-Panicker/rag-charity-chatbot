from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from loguru import logger
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_db.chromadb_handler import ChromaVectorDB


#Configuration for retrieval 
@dataclass
class RetrievalConfig: 
    top_k: int = 5 #Number of chunks to retrieve
    similarity_threshold: float = 0.3 #Minimum similarity for retrieval
    rerank: bool = True #Rerank results by relevance
    include_metadata: bool = True 
    debug: bool = True 

#Retrieves relevan chunks using semantic search 
class SemanticRetriever: 
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_db: ChromaVectorDB, config: RetrievalConfig = RetrievalConfig()): 
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.config = config

        self.retrieval_logs = [] 


    #Retrieve relevant chunks for a query and returns relevant chunks with metadata 
    def retrieve(self, query: str, charity_name: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict]: 
        if self.config.debug: 
            log_entry = {
                'query': query,
                'charity_name': charity_name,
                'top_k_requested': top_k or self.config.top_k
            }
        
        try: 
            #embed the query 
            query_embedding = self.embedding_generator.embed_query(query)

            if self.config.debug:
                log_entry['embedding_generated'] = True 
            
            #search vector DB
            k = top_k or self.config.top_k

            #Get collection of charity if specified 
            if charity_name: 
                collection_name = charity_name.lower().replace(" ", "_")
                try: 
                    self.vector_db.get_collection(collection_name)
                except: 
                    logger.warning(f"Collection for charity '{charity_name}' not found.")
                    return [] 
            
            results = self.vector_db.search(
                query_embedding=query_embedding,
                n_results=k * 2,  #Retrieve more for reranking
                name=charity_name,
                threshold=0.0
            )

            if self.config.debug: 
                log_entry['intial_results_count'] = len(results)
            
            #Filter by similarity threshold
            filtered_results = [
                r for r in results
                if r['similarity'] >= self.config.similarity_threshold
            ] 

            if self.config.debug:
                log_entry['after_threshold_filter'] = len(filtered_results)
                log_entry['threshold_used'] = self.config.similarity_threshold
            
            #Rerank if enabled
            if self.config.rerank and len(filtered_results) > 1:
                filtered_results = self._rerank_results(
                    filtered_results, 
                    query_embedding
                )
                if self.config.debug: 
                    log_entry['reranked'] = True 
            
            #Take top k
            final_results = filtered_results[:k]

            if self.config.debug: 
                log_entry['final_results_count'] = len(final_results)
                log_entry['status'] = 'success'
                self.retrieval_logs.append(log_entry)
                logger.debug(f"Retrieval log: {log_entry}")
            
            logger.info(
                f"Retrieved {len(final_results)} chunks"
                f"(threshold: {self.config.similarity_threshold})"
            )
            return final_results
        except Exception as e: 
            logger.error(f"Retrieval failed: {e}")
            if self.config.debug: 
                log_entry['status'] = 'error'
                log_entry['error'] = str(e)
                self.retrieval_logs.append(log_entry)

            return []
    
    #Rerank results using cosine similarity with query embedding
    def _rerank_results(self, results: List[Dict], query_embedding: np.ndarray) -> List[Dict]: 
        logger.info("Reranking results...")

        #Sort by cosine similarity
        reranked = sorted(
            results, 
            key=lambda x: x['similarity'],
            reverse=True
        )
        return reranked
    
    #Get stats about retrieval performance
    def get_retrieval_stats(self) -> Dict:
        if not self.retrieval_logs: 
            return {'total_retrievals': 0}
        
        successful = [l for l in self.retrieval_logs if l.get('status') == 'success']
        failed = [l for l in self.retrieval_logs if l.get('status') == 'error']

        avg_results = (
            sum(l.get('final_results_count', 0) for l in successful) / len(successful)
            if successful else 0
        )

        return {
            'total_retrievals': len(self.retrieval_logs),
            'successful_retrievals': len(successful),
            'failed_retrievals': len(failed),
            'avg_results_per_successful': avg_results, 
            'threshold_used': self.config.similarity_threshold
        }
    
    #clear logs
    def clear_lofs(self): 
        self.retrieval_logs = []


#Class that will build context from retreived chunks 
class ContextBuilder: 

    #Build context string from chunks 
    @staticmethod
    def build_context(chunks: List[Dict], max_tokens: int = 2000, include_sources: bool = True ) -> str: 
        if not chunks: 
            return "No relevant information found."
        
        context_parts = [] 
        token_count = 0 

        for i, chunk in enumerate(chunks): 
            chunk_text = chunk['text']

            #estimate tokens 
            chunk_tokens = len(chunk_text.split()) // 0.75  

            if token_count + chunk_tokens > max_tokens:
                logger.warning(f"Context window limit reached after {i - 1} chunks.")
                break 

            #Format chunk with sources 
            if include_sources: 
                metadata = chunk.get('metadata', {})
                source_info = f"[Source: {metadata.get('charity_name', 'Unknown')}]"
                fomatted_chunk = f"{chunk_text}\n{source_info}\n"
            else:
                fomatted_chunk = chunk_text + "\n"

            context_parts.append(fomatted_chunk)
            token_count += chunk_tokens

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Built context with {len(context_parts)} chunks (~{token_count} tokens)")

        return context
    

        