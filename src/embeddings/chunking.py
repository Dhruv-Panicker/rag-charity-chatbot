import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from loguru import logger

#Configuration for document chunking 
@dataclass
class ChunkConfig:
    chunk_size: int = 512 # tokens per chunk 
    overlap: int = 50 #tokens overlap between chunks
    strategy: str = "semantic"  #'fixed', 'semantic', or 'paragraph'
    min_chunk_size: int = 100 

#Class to estimate token counts 
class TokenCounter: 
    #Estimate token count using word count 
    @staticmethod
    def estimate_tokens(text: str) -> int: 
        words = len(text.split())
        return max(1, int(words / 0.75))  
    
    #Count the tokens in a chunk 
    @staticmethod
    def count_tokens_in_chunk(text: str) -> int: 
        return TokenCounter.estimate_tokens(text)
    
#Chunks documents into overlapping segements for RAG 
class DocumentChunker: 

    def __init__(self, config: ChunkConfig = ChunkConfig()): 
        self.config = config 
        self.token_counter = TokenCounter()

    #Chunk text based on the selected strategy, returns list of chunks with metadata
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict]:
        if self.config.strategy == "semantic":
            return self._chunk_semantic(text, metadata)
        elif self.config.strategy == "paragraph":
            return self._chunk_paragraph(text, metadata)
        else:
            return self._chunk_fixed_size(text, metadata)
        
    #Fixed-size chunking based on token count 
    def _chunk_fixed_size(self, text: str, metadata: Dict = None) -> List[Dict]: 
        logger.info(f"Chinking document with fixed strategy size {self.config.chunk_size}")

        words = text.split()
        chunks = [] 
        words_per_chunk = int(self.config.chunk_size * 0.75)
        overlap_words = int(self.config.overlap * 0.75)

        start_idx = 0 
        chunk_id = 0 

        while start_idx < len(words):
            end_idx = min(start_idx + words_per_chunk, len(words))
            chunk_text = ' '.join(words[start_idx:end_idx])

            if self.token_counter.count_tokens_in_chunk(chunk_text) >= self.config.min_chunk_size: 
                chunks.append({
                    'id': f"chunk_{chunk_id}",
                    'text': chunk_text,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'token_count': self.token_counter.count_tokens_in_chunk(chunk_text), 
                    'metadata': metadata or {}
                })
                chunk_id += 1

            # Move to next chunk with overlap
            next_start_idx = end_idx - overlap_words
            
            # If we've reached the end and can't move forward, break to avoid infinite loop
            if next_start_idx >= start_idx and end_idx == len(words):
                break
            
            start_idx = max(next_start_idx, start_idx + 1)  # Ensure we always move forward
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    #Paragraph-based chunking
    def _chunk_paragraph(self, text: str, metadata: Dict = None) -> List[Dict]: 
        logger.info("Chunking document by paragraphs")

        #split by double newlines 
        paragraphs = text.split('\n\n')
        chunks = [] 
        current_chunk = [] 
        current_tokens = 0 
        chunk_id = 0 

        for para in paragraphs: 
            para = para.strip()
            if not para:    
                continue

            para_tokens = self.token_counter.count_tokens_in_chunk(para)

            #If paragraph is too long split it 
            if para_tokens > self.config.chunk_size: 
                #flush current chunk if not empty 
                if current_chunk: 
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'id': f"chunk_{chunk_id}",
                        'text': chunk_text,
                        'token_count': current_tokens,
                        'metadata': metadata or {}
                    })
                    chunk_id += 1
                    current_chunk = []
                    current_tokens = 0

                #Then split long paragraph into fixed-size chunks
                sub_chunks = self._chunk_fixed_size(para, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk['id'] = f"chunk_{chunk_id}"
                    chunks.append(sub_chunk)
                    chunk_id += 1

            # Add paragraphs to current chunk 
            elif current_tokens + para_tokens <= self.config.chunk_size: 
                current_chunk.append(para)
                current_tokens += para_tokens

            #Current chunk is full, save it and start a new one
            else: 
                if current_chunk: 
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'id': f"chunk_{chunk_id}",
                        'text': chunk_text,
                        'token_count': current_tokens,
                        'metadata': metadata or {}
                    })
                    chunk_id += 1

                current_chunk = [para]
                current_tokens = para_tokens
        
        #Add remaining chunks
        if current_chunk: 
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'id': f"chunk_{chunk_id}",
                'text': chunk_text,
                'token_count': current_tokens,
                'metadata': metadata or {}
            })
        
        logger.info(f"Created {len(chunks)} paragraph-based chunks")
        return chunks
    

    
    #Semantic chunking using sentence boundries 
    def _chunk_semantic(self, text: str, metadata: Dict = None) -> List[Dict]:
        logger.info("Chunking document with semantic strategy")

        #split into sentences using regex
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for sentence in sentences: 
            sentence = sentence.strip()
            if not sentence: 
                continue 

            sentence_tokens = self.token_counter.count_tokens_in_chunk(sentence)

            #Add sentence if it fits 
            if current_tokens + sentence_tokens <= self.config.chunk_size:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            #Current chunk full, save and start new chunk
            else: 
                if current_chunk: 
                    chunk_text = ''. join(current_chunk)
                    chunks.append({
                        'id': f"chunk_{chunk_id}",
                        'text': chunk_text,
                        'token_count': current_tokens,
                        'metadata': metadata or {}
                    })   
                    chunk_id += 1

                #start new chunk may include overlap 
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        if current_chunk:
            chunk_text = ''. join(current_chunk)
            chunks.append({
                'id': f"chunk_{chunk_id}",
                'text': chunk_text,
                'token_count': current_tokens,
                'metadata': metadata or {}
            })
            
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    #Split text into sentences using regex
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]: 
        #regex to split on periods, exclamations, question marks followed by space and capital letter but preserves abbreviations
        sentences = re.split(r'(?<=[.!?]) +(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    #Evaluate chunking quality 
    def evaluate_chunking(self, chunks: List[Dict]) -> Dict:
        token_counts = [chunk['token_count'] for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'total_tokens': sum(token_counts)
        }
    



        


            


        
