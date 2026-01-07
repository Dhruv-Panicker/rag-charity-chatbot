#!/usr/bin/env python3
"""
Terminal-based RAG Charity Chatbot CLI
Allows users to scrape charity websites and chat with them using RAG
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.rag_system import RAGSystem, RAGConfig
from src.retrieval.retriever import RetrievalConfig
from src.llm.llm_client import LLMConfig
from src.scraper.web_scraper import WebScraper, HTMLCleaner, SiteMapper
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.embeddings.chunking import ChunkConfig
from src.embeddings.embedding_generator import EmbeddingConfig
from loguru import logger
import time

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>", level="WARNING")

# Load environment variables
load_dotenv()


def print_banner():
    """Print welcome banner"""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║            RAG CHARITY CHATBOT - TERMINAL CLI                  ║
║                                                                ║
║          Query and Chat with Charity Information               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_instructions():
    """Print usage instructions"""
    instructions = """
📚 HOW TO USE:
   1. Enter a charity website URL to scrape
   2. Choose to scrape homepage only or entire site
   3. Start chatting! Ask questions about the charity
   4. Type 'exit' to quit the chatbot

💡 TIPS:
   • First scrape takes a moment - embeddings are being generated
   • Ask specific questions for better results
   • Type 'quit' or 'exit' to stop

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(instructions)


def initialize_rag_system() -> RAGSystem:
    """Initialize and return RAG system"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set in environment variables")
        print("   Please set it: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    config = RAGConfig(
        retrieval_config=RetrievalConfig(
            top_k=5,
            similarity_threshold=0.1, 
            rerank=True,
            debug=False
        ),
        llm_config=LLMConfig(
            provider="openai",
            model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            api_key=api_key,
            temperature=0.7,
            max_tokens=1024
        ),
        max_context_tokens=10000,
        debug=False
    )

    return RAGSystem(config)


def scrape_and_index_charity(url: str, scrape_type: str = "homepage") -> tuple[bool, str]:
    """
    Scrape a charity website and index it
    Returns: (success: bool, collection_name: str)
    """
    print(f"\n🔄 Scraping {url}...")
    print(f"   Mode: {'Homepage only' if scrape_type == 'homepage' else 'Entire site'}")

    try:
        # Initialize scraper
        scraper = WebScraper(timeout=30, max_retries=3)
        cleaner = HTMLCleaner()

        # Scrape based on type
        if scrape_type == "sitemap":
            print("   Discovering pages...", end="", flush=True)
            try:
                mapper = SiteMapper(url, max_pages=50)
                pages = mapper.discover_pages()
                print(f" Found {len(pages)} pages!")
            except Exception as e:
                print(f"\n   ⚠️  Could not discover pages: {e}")
                print("   Falling back to homepage only...")
                pages = [url]

            all_text = []
            successful = 0
            failed = 0

            for i, page_url in enumerate(pages, 1):
                try:
                    html = scraper.get_html(page_url)

                    if html:
                        text = HTMLCleaner.extract_text(html)
                        if text and len(text) > 50:
                            all_text.append(text)
                            successful += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                    
                    time.sleep(0.5)  # Respect server
                except Exception as e:
                    failed += 1
                    continue

            print(f"   ✅ Successfully scraped {successful} pages, {failed} failed")
            
            if not all_text:
                print("❌ Could not scrape any content from the site.")
                print("   The website may have anti-scraping protection.")
                print("   Try using 'Homepage only' mode instead.")
                return False, ""

            content = "\n\n".join(all_text)
            pages_count = successful
        else:
            print("   Scraping homepage...")
            html = scraper.get_html(url)

            if not html:
                print("❌ Could not scrape content from homepage.")
                print("   The website may be blocking scrapers.")
                print("   Please try a different website.")
                return False, ""

            content = HTMLCleaner.extract_text(html)
            pages_count = 1

        if not content or len(content) < 100:
            print("❌ No usable content found (content too short).")
            print("   The website may be blocking scrapers or has minimal text.")
            print("   Please try a different website.")
            return False, ""

        # Extract charity name from URL
        charity_name = url.split("//")[1].split("/")[0].replace("www.", "").split(".")[0].title()

        # Initialize embedding pipeline
        print(f"\n📊 Indexing content...")
        pipeline = EmbeddingPipeline(
            chunk_config=ChunkConfig(strategy="fixed", chunk_size=256, overlap=50),
            embedding_config=EmbeddingConfig()
        )

        # Process and index
        result = pipeline.process_charity(
            charity_name=charity_name,
            document_text=content
        )

        if result.get('status') != 'success':
            print(f"❌ Indexing failed: {result.get('error', 'Unknown error')}")
            return False, ""

        chunking_stats = result.get('chunking_stats', {})
        chunks = chunking_stats.get('total_chunks', 0)
        tokens = chunking_stats.get('total_tokens', 0)
        
        print(f"✅ Successfully indexed {charity_name}")
        print(f"   • Scraped: {pages_count} page(s)")
        print(f"   • Created: {chunks} chunks")
        print(f"   • Total tokens: {tokens}")

        return True, charity_name

    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return False, ""


def chat_with_charity(rag: RAGSystem, charity_name: str):
    """Main chat loop"""
    print(f"\n💬 Chat with {charity_name}")
    print("   (Type 'exit' or 'quit' to stop)\n")
    print("━" * 60)

    conversation_count = 0

    while True:
        try:
            query = input(f"\n❓ You: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break

            # Query RAG system
            print("\n🤔 Thinking...")
            result = rag.query(
                query=query,
                charity_name=charity_name,
                top_k=5
            )

            # Display response
            response = result.get("response", "I couldn't find an answer.")
            sources_count = result.get("retrieved_chunks", 0)
            processing_time = result.get("processing_time", 0)

            print(f"\n🤖 Chatbot:")
            print(f"   {response}")
            print(f"\n   📌 Sources used: {sources_count} chunks")
            print(f"   ⏱️  Response time: {processing_time:.2f}s")
            print("━" * 60)

            conversation_count += 1

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def main():
    """Main entry point"""
    print_banner()
    print_instructions()

    # Initialize RAG system
    print("🚀 Initializing RAG system...")
    rag = initialize_rag_system()
    print("✅ RAG system ready\n")

    while True:
        print("=" * 60)

        # Get charity URL
        url = input("\n🌐 Enter charity website URL (or 'quit' to exit):\n   ").strip()

        if url.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Ask about scrape type
        print("\n📋 Scrape type:")
        print("   [1] Homepage only (faster)")
        print("   [2] Entire site (more comprehensive)")
        scrape_choice = input("   Choose [1 or 2] (default: 1): ").strip()

        scrape_type = "sitemap" if scrape_choice == "2" else "homepage"

        # Scrape and index
        success, charity_name = scrape_and_index_charity(url, scrape_type)

        if success:
            # Start chat
            chat_with_charity(rag, charity_name)
        else:
            print("⚠️  Please try another URL\n")


if __name__ == "__main__":
    main()
