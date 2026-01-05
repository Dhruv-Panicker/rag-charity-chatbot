from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from loguru import logger
import chromadb

#Wrapper for chroma vector database 
class ChromaVectorDB:

    def __init__(self, persist_dir: str = "data/chroma_db"): 
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Use PersistentClient for persistent storage to disk 
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        #create or get collection 
        self.collection = None 

    #Create a new collection for a given name 
    def create_collection(self, name: str, metadata: Dict = None) -> None: 
        logger.info(f"Creating ChromaDB collection: {name}")

        #Delete if it already exists 
        try: 
            self.client.delete_collection(name=name)
        except: 
            pass 

        self.collection = self.client.create_collection(name=name, metadata=metadata or {}, get_or_create=True)
        logger.info(f"Collection '{name}' is ready")

    #Get existing collection by name
    def get_collection(self, name: str) -> None: 
        try: 
            self.collection = self.client.get_collection(name=name)
            logger.info(f"Retrieved collection '{name}'")
        except Exception as e:
            logger.error(f"Failed to retrieve collection '{name}': {e}")
            raise
    
    #Add chunks with embeddings to the database 
    def add_chunks(self, chunks: List[Dict]) -> None: 
        if not self.collection: 
            raise ValueError("No collection selected. Call create_collection or get_collection first.")
        
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")

        ids = [] 
        embeddings = [] 
        documents = [] 
        metadata = [] 


        for chunk in chunks: 
            ids.append(chunk['id'])
            embeddings.append(chunk['embedding'].tolist())  # Convert numpy array to list
            documents.append(chunk['text'])
            metadata.append(chunk.get('metadata', {}))

        #Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata
        )
        logger.info(f"Added {len(chunks)} chunks to ChromaDB collection")

    #Search for similar chunks return matching chunks with simlarity scores 
    def search(self, query_embedding: np.ndarray, n_results: int = 5, name: Optional[str] = None, threshold: float = 0.3) -> List[Dict]: 
        if not self.collection: 
            logger.warning("No collection selected")
            return [] 
        
        try: 
            #Build where filter 
            where_filter = None 
            if name: 
                where_filter = {'charity_name': name}
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter, 
                include=['metadatas', 'documents', 'distances']
            )

            #Format results 
            formatted_results = []

            if results['documents'] and results['documents'][0]: 
                for i, (doc, meta, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )): 
                    #convert cosine distance to similarity
                    similarity = 1 - distance

                    if similarity >= threshold: 
                        formatted_results.append({
                            'rank': i + 1,
                            'text': doc,
                            'similarity': similarity,
                            'metadata': meta
                        })
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
        
    
    #Get stats about the collection 
    def get_collection_stats(self) -> Dict: 
        if not self.collection: 
            return {}
        
        try: 
            count = self.collection.count()
            return {
                'collection_name': self.collection.name, 
                'total_chunks': count, 
                'metadata_keys': list(self.collection.metadata.keys())
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    #Delete collection 
    def delete_collection(self, name: str) -> None: 
        try: 
            self.client.delete_collection(name=name)
            logger.info(f"Deleted collection '{name}'")
            self.collection = None
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise
    
    #List all the collections 
    def list_collections(self) -> List[str]: 
        collections = self.client.list_collections()
        return [c.name for c in collections]




