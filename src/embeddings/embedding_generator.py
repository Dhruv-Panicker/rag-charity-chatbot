from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from loguru import logger
from sentence_transformers import SentenceTransformer

#Class for configuration of embedding generation
@dataclass
class EmbeddingConfig:
    model_name : str = "sentence-transformers/all-MiniLM-L6-v2" 
    batch_size : int = 32
    device: str = "cpu"  # or 'cuda' for GPU


#Generates embeddings got text chunks 
class EmbeddingGenerator: 
    def __init__(self, config: EmbeddingConfig = EmbeddingConfig()): 
        self.config = config 
        logger.info(f"Loading embedding model {self.config.model_name}")
        self.model = SentenceTransformer(config.model_name, device=config.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimensions: {self.embedding_dim}")

    #Genreate embeddings for multiple arrays 
    def embed_texts(self, texts: List[str]) -> np.ndarray: 
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        return embeddings
    

    #Add the embeddings to the chunks  returns chunks with embedding field
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:

        texts = [chunk['text'] for chunk in chunks]

        #Genrate embeddings 
        embeddings = self.embed_texts(texts)

        #Add embeddings to chunks 
        for chunk, embedding in zip(chunks, embeddings): 
            chunk['embedding'] = embedding

        logger.info(f"Embedded {len(chunks)} chunks with embeddings")
        return chunks 

    #Embed a single query string returns 1D numpy array of vector space
    def embed_query(self, query: str) -> np.ndarray: 
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float: 
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    #Get information about the model 
    def get_model_info(self) -> Dict: 
        return {
            'model_name': self.config.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.config.device,
            'batch_size': self.config.batch_size
        }
    


     

