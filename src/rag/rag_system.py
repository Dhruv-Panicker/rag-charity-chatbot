from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from src.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from src.vector_db.chromadb_handler import ChromaVectorDB
from src.retrieval.retriever import SemanticRetriever, RetrievalConfig, ContextBuilder
from src.llm.llm_client import OpenAIProvider, LLMConfig
from src.llm.prompt_templates import PromptFormatter


#Single message in conversation 
@dataclass
class ConversationMessage: 
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


#Configuration for RAG system 
@dataclass
class RAGConfig: 
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    max_context_tokens: int = 2000
    max_conversation_turns: int = 10
    debug: bool = True 


#RAG system integrating retrieval and LLM for QA
class RAGSystem:
    def __init__(
        self, 
        config: RAGConfig = RAGConfig(), 
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_db: Optional[ChromaVectorDB] = None, 
        llm_provider: Optional[OpenAIProvider] = None
    ): 
        self.config = config 

        #Initialize components 
        self.embedding_gen = embedding_generator or EmbeddingGenerator(EmbeddingConfig())
        self.vector_db = vector_db or ChromaVectorDB()

        if llm_provider is None: 
            self.llm_provider = OpenAIProvider(self.config.llm_config)
        else: 
            self.llm_provider = llm_provider

        #Initialize retriever
        self.retriever = SemanticRetriever(
            self.embedding_gen, 
            self.vector_db,
            config.retrieval_config
        )

        #Conversation history
        self.conversation_history: List[ConversationMessage] = []
        self.session_logs = []

        logger.info("RAG System initialized with OpenAI provider")


    #Process user query and generate response returns response sources and metadata
    def query(self, query: str , charity_name: Optional[str] = None, top_k: Optional[int] = None) -> Dict: 
        logger.info(f"Processing user query: {query}")

        try: 
            session_start = datetime.now()

            #Retrieve relevant chunks 
            logger.info("Retrieving relevant chunks...")
            retrieved_chunks = self.retriever.retrieve(
                query=query, 
                charity_name=charity_name,
                top_k=top_k
            )

            #Build context from retrieved chunks
            logger.info("Building context for LLM...")
            context = ContextBuilder.build_context(
                retrieved_chunks, 
                max_tokens=self.config.max_context_tokens,
            )

            #Format prompts
            logger.info("Formatting prompts...")
            prompts = PromptFormatter.format_rag_prompt(
                query=query, 
                context=context, 
                charity_name=charity_name or "this organization"
            )

            #Generate response from LLM
            logger.info("Generating response from LLM...")
            if retrieved_chunks: 
                response = self.llm_provider.generate(
                    system_prompt=prompts['system'],
                    user_prompt=prompts['user']
                )
            else:
                #No context found, use fallback prompt
                logger.warning("no relevant context found")
                response = PromptFormatter.format_fallback_prompt(
                    query=query, 
                    charity_name=charity_name or "this organization"    
                )
            #Build response object 
            session_end = datetime.now()
            session_duration = (session_end - session_start).total_seconds()

            result = {
                'status': 'success',
                'query': query,
                'response': response,
                'retrieved_chunks': len(retrieved_chunks),
                'sources': [
                    {
                        'text': chunk['text'][:100] + '...',
                        'similarity': chunk['similarity'],
                        'metadata': chunk.get('metadata', {})
                    }
                    for chunk in retrieved_chunks
                ],
                'processing_time': session_duration,
                'timestamp': datetime.now().isoformat()
            }

            #Store in converstaion history
            self.conversation_history.append(
                ConversationMessage(
                    role='user',
                    content=query,
                    metadata={'query_type': 'retrieval'}
                )
            )

            self.conversation_history.append(
                ConversationMessage(
                    role='assistant',
                    content=response,
                     metadata={'sources': len(retrieved_chunks)}
                )
            )
            #Log session
            self.session_logs.append(result)
            logger.info(f"Query processed in {session_duration:.2f}s")

            return result 
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    #Get formatted conversation history 
    def get_conversation_history(self) -> List[Dict]: 
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(), 
            }
            for msg in self.conversation_history
        ]
    
    #Get stats about the session. 
    def get_session_stats(self) -> Dict: 
        successful_queries = [
            l for l in self.session_logs if l.get('status') == 'success'
        ]

        avg_response_time = {
            sum(l.get('processing_time', 0) for l in successful_queries) / len(successful_queries)
            if successful_queries else 0
        }

        avg_sources = {
            sum(l.get('retrieved_chunks', 0) for l in successful_queries) / len(successful_queries)
            if successful_queries else 0
        }

        return {
            'total_queries': len(self.session_logs),
            'successful_queries': len(successful_queries),
            'failed_queries': len(self.session_logs) - len(successful_queries),
            'avg_response_time': avg_response_time,
            'avg_sources_retrieved': avg_sources,
            'conversation_turns': len(self.conversation_history) // 2
        }
    
    #Clear conversation history 
    def clear_conversation_history(self): 
        self.conversation_history = []
        logger.info("Cleared conversation history")
    
    #Get info on RAG system components 
    def get_system_info(self) -> Dict:
        """Get information about RAG system components"""
        return {
            'embedding_model': self.embedding_gen.get_model_info(),
            'llm_provider': self.llm_provider.get_provider_info(),
            'retrieval_config': {
                'top_k': self.config.retrieval_config.top_k,
                'threshold': self.config.retrieval_config.similarity_threshold
            },
            'max_context_tokens': self.config.max_context_tokens
        }







