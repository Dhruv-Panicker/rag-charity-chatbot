import sys
import os
from pathlib import Path
from dotenv import load_dotenv 
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.llm.prompt_templates import PromptFormatter
from src.rag.rag_system import RAGSystem, RAGConfig
from src.rag.conversation_manager import ConversationManager
from src.retrieval.retriever import RetrievalConfig
from src.llm.llm_client import LLMConfig, OpenAIProvider


def test_openai_setup():
    """Test OpenAI provider initialization"""
    print("\n" + "="*80)
    print("TEST 1: OPENAI PROVIDER SETUP")
    print("="*80)
    
    try:
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set")
            return None
        
        # Initialize OpenAI provider
        config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0.7,
            max_tokens=500
        )
        
        provider = OpenAIProvider(config)
        
        print("OpenAI provider initialized successfully")
        print(f"\nOpenAI Configuration:")
        info = provider.get_provider_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return provider
    
    except Exception as e:
        print(f"Failed to initialize OpenAI provider: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_rag_system_setup(openai_provider):
    """Test RAG system initialization with OpenAI"""
    print("\n" + "="*80)
    print("TEST 2: RAG SYSTEM SETUP")
    print("="*80)
    
    try:
        config = RAGConfig(
            retrieval_config=RetrievalConfig(
                top_k=3,
                similarity_threshold=0.3,
                debug=True
            ),
            llm_config=LLMConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.7,
                max_tokens=500
            )
        )
        
        rag = RAGSystem(config, llm_provider=openai_provider)
        
        print("RAG System initialized successfully")
        print(f"\nSystem Configuration:")
        info = rag.get_system_info()
        for component, details in info.items():
            print(f"\n{component}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {details}")
        
        return rag
    
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def test_retrieval_only(rag):
    """Test retrieval with indexed documents"""
    print("\n" + "="*80)
    print("TEST 3: RETRIEVAL-ONLY MODE (WITH INDEXED DOCUMENTS)")
    print("="*80)
    
    try:
        # Test query with Red Cross (indexed charity)
        query = "What educational programs does the charity offer?"
        charity_name = "Red Cross"
        
        print(f"\nQuery: '{query}'")
        print(f"Charity: '{charity_name}'")
        print("(Testing with indexed documents from Chroma)")
        
        result = rag.query(query, charity_name=charity_name)
        
        print(f"\nResult Status: {result.get('status')}")
        print(f"Retrieved Chunks: {result.get('retrieved_chunks', 0)}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        
        if result.get('status') == 'error':
            print(f"Error: {result.get('error')}")
        elif result.get('retrieved_chunks', 0) > 0:
            print(f"\n✅ Retrieved {result.get('retrieved_chunks')} chunks")
            for i, source in enumerate(result.get('sources', []), 1):
                print(f"\nSource {i} (Similarity: {source['similarity']:.2f}):")
                print(f"  {source['text'][:100]}...")
        else:
            print("⚠️  No chunks retrieved")
        
        return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_conversation_manager():
    """Test conversation memory management"""
    print("\n" + "="*80)
    print("TEST 4: CONVERSATION MEMORY MANAGEMENT")
    print("="*80)
    
    try:
        manager = ConversationManager(max_messages=10)
        
        print("\nAdding sample conversation...")
        
        # Simulate conversation
        manager.add_user_message("What does your charity do?")
        manager.add_assistant_message(
            "We provide education and healthcare services to underserved communities."
        )
        manager.add_user_message("How can I volunteer?")
        manager.add_assistant_message(
            "You can contact our volunteer coordinator at volunteers@charity.org"
        )
        
        # Get stats
        stats = manager.get_stats()
        print(f"\nConversation Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Get context
        context = manager.get_conversation_context()
        print(f"\nConversation Context (last 4 messages):")
        print(context)
        
        # Get OpenAI format
        openai_messages = manager.get_messages_for_openai()
        print(f"\nMessages formatted for OpenAI API:")
        for i, msg in enumerate(openai_messages, 1):
            print(f"  {i}. {msg['role'].upper()}: {msg['content'][:50]}...")
        
        print("\n✅ Conversation manager working correctly")
        return manager
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def test_prompt_formatting():
    """Test prompt template formatting"""
    print("\n" + "="*80)
    print("TEST 5: PROMPT FORMATTING")
    print("="*80)
    
    try:     
        query = "What programs do you offer?"
        context = "We offer education and healthcare programs for underserved communities."
        charity_name = "Test Charity"
        
        # Test RAG prompt
        prompts = PromptFormatter.format_rag_prompt(
            query=query,
            context=context,
            charity_name=charity_name
        )
        
        print("\n--- RAG Prompt Format ---")
        print(f"\nSystem Prompt (first 200 chars):")
        print(prompts['system'][:200] + "...")
        
        print(f"\nUser Prompt (first 300 chars):")
        print(prompts['user'][:300] + "...")
        
        # Test fallback prompt
        fallback = PromptFormatter.format_fallback_prompt(
            query="What is the CEO's favorite color?",
            charity_name=charity_name
        )
        
        print(f"\nFallback Prompt (when no context):")
        print(fallback[:200] + "...")
        
        print("\n✅ Prompt formatting working correctly")
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_openai_generation(rag):
    """Test full RAG pipeline with OpenAI generation"""
    print("\n" + "="*80)
    print("TEST 6: FULL RAG PIPELINE (RETRIEVAL + OPENAI GENERATION)")
    print("="*80)
    
    try:
        # Only run if we have a valid RAG system
        if not rag:
            print("⚠️  Skipping: RAG system not initialized")
            return None
        
        print("\nTesting full RAG pipeline with indexed documents...")
        
        # Query: UNICEF
        print("\n--- Query: UNICEF ---")
        query3 = "What education programs does UNICEF support?"
        charity3 = "UNICEF"
        
        print(f"Query: '{query3}'")
        print(f"Charity: '{charity3}'")
        print("Retrieving documents and generating response...")
        
        result3 = rag.query(query3, charity_name=charity3)
        
        print(f"\nStatus: {result3.get('status')}")
        print(f"Retrieved {result3.get('retrieved_chunks', 0)} chunks")
        
        if result3.get('status') == 'success':
            print(f"\n✅ OpenAI Response:")
            print(f"  {result3.get('response')}")
        else:
            print(f"❌ Error: {result3.get('error')}")
        
        return [result3]
    
    except Exception as e:
        print(f"❌ RAG generation failed: {e}")
        print("\nThis could be due to:")
        print("  - Invalid API key")
        print("  - No indexed documents for the charity")
        print("  - Network connection issue")
        import traceback
        traceback.print_exc()
        return None


    
def main():
    print("\n" + "="*80)
    print("RAG SYSTEM WITH OPENAI TESTS")
    print("="*80)
    
    try:
        # Test 1: OpenAI Setup
        openai_provider = test_openai_setup()
        

        # Test 2: RAG System Setup
        rag = test_rag_system_setup(openai_provider)
        
        # Test 3: Retrieval
        if rag:
            test_retrieval_only(rag)
        
        # Test 4: Conversation Manager
        test_conversation_manager()
        
        # Test 5: Prompt Formatting
        test_prompt_formatting()
        
        # Test 6: Full RAG Pipeline 
        if rag:
            test_openai_generation(rag)
    
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


