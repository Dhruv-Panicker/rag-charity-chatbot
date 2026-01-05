import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.chunking import DocumentChunker, ChunkConfig
from src.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from src.embeddings.embedding_pipeline import EmbeddingPipeline


def test_chunking_strategies():
    """Test different chunking strategies"""
    print("\n" + "="*80)
    print("TEST 1: CHUNKING STRATEGIES")
    print("="*80)
    
    sample_text = """
    This is a test document about a charity organization. 
    The organization does important work in the community.
    
    They focus on several key areas:
    - Education and youth development
    - Healthcare and wellness programs
    - Community outreach initiatives
    
    The charity has been active for over 20 years. 
    They serve thousands of people annually.
    Their mission is to create positive change.
    
    You can donate online through their website.
    Volunteers are always welcome.
    Contact them for more information about programs.
    """ * 3  # Repeat to make it longer
    
    strategies = ["fixed", "paragraph", "semantic"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.upper()} ---")
        
        config = ChunkConfig(
            chunk_size=256,
            strategy=strategy,
            overlap=50
        )
        
        chunker = DocumentChunker(config)
        chunks = chunker.chunk_document(sample_text)
        stats = chunker.evaluate_chunking(chunks)
        
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Avg tokens: {stats['avg_tokens']:.1f}")
        print(f"Token range: {stats['min_tokens']} - {stats['max_tokens']}")
        
        print(f"\nFirst chunk preview:")
        print(f"  Text: {chunks[0]['text'][:100]}...")
        print(f"  Tokens: {chunks[0]['token_count']}")

def test_embedding_generation():
    """Test embedding generation"""
    print("\n" + "="*80)
    print("TEST 2: EMBEDDING GENERATION")
    print("="*80)
    
    print("\nLoading embedding model...")
    config = EmbeddingConfig()
    generator = EmbeddingGenerator(config)
    
    print(f"\nModel Info:")
    info = generator.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test embedding texts
    sample_texts = [
        "The charity provides education and healthcare services.",
        "We help vulnerable communities through various programs.",
        "Donate today to support our mission.",
        "Volunteers are essential to our work."
    ]
    
    print(f"\nGenerating embeddings for {len(sample_texts)} texts...")
    embeddings = generator.embed_texts(sample_texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 dimensions): {embeddings[0][:5]}")
    
    # Test similarity
    print(f"\nSimilarity matrix:")
    print("Text 1 vs Text 2:", 
          f"{generator.cosine_similarity(embeddings[0], embeddings[1]):.3f}")
    print("Text 1 vs Text 4:", 
          f"{generator.cosine_similarity(embeddings[0], embeddings[3]):.3f}")
    
def test_full_pipeline():
    """Test end-to-end embedding pipeline"""
    print("\n" + "="*80)
    print("TEST 3: FULL EMBEDDING PIPELINE")
    print("="*80)
    
    # Sample document
    document_text = """
    About Our Charity
    
    We are a non-profit organization dedicated to improving lives in underserved communities.
    Our mission is to provide education, healthcare, and economic opportunities to those in need.
    
    What We Do
    
    Education Programs: We run schools and training centers that provide quality education 
    to children and adults. Our curriculum focuses on practical skills and academic excellence.
    
    Healthcare Services: Our clinics and health programs serve rural and urban communities.
    We provide preventive care, treatment, and health education.
    
    Economic Development: We support small businesses and provide microfinance to entrepreneurs.
    This helps families become self-sufficient.
    
    How to Help
    
    You can support our work through donations, volunteering, or advocacy.
    Every contribution makes a difference in the lives we touch.
    
    Contact us at info@charity.org or visit our website for more information.
    """ * 2  # Repeat for more content
    
    # Create pipeline
    print("\nInitializing embedding pipeline...")
    pipeline = EmbeddingPipeline(
        chunk_config=ChunkConfig(
            chunk_size=512,
            strategy="semantic"
        )
    )
    
    # Process charity
    print("Processing charity document...")
    result = pipeline.process_charity(
        charity_name="Test Charity",
        document_text=document_text,
        document_metadata={
            'source': 'test',
            'date': '2026-01-03'
        }
    )
    
    print(f"\nResult status: {result['status']}")
    print(f"Charity name: {result['charity_name']}")
    print(f"\nChunking stats:")
    for key, value in result['chunking_stats'].items():
        print(f"  {key}: {value}")
    
    print(f"\nDatabase stats:")
    for key, value in result['db_stats'].items():
        print(f"  {key}: {value}")
    
    # Test search
    print(f"\n--- TESTING SEARCH ---")
    queries = [
        "What educational programs do you offer?",
        "How can I donate?",
        "What is your mission?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = pipeline.search(
            query=query,
            charity_name="Test Charity",
            n_results=3
        )
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. [Similarity: {result['similarity']:.3f}]")
                print(f"     {result['text'][:80]}...")
        else:
            print("  No results found")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("EMBEDDING PIPELINE TESTS")
    print("="*80)
    
    try:
        test_chunking_strategies()
        test_embedding_generation()
        test_full_pipeline()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
    
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()



