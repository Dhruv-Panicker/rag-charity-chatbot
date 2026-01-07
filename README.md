# RAG Charity Chatbot

A terminal-based retrieval-augmented generation (RAG) chatbot system that scrapes charity websites and provides accurate, context-grounded answers to user queries about charity information.

## Project Overview

This project makes charity information more accessible through intelligent question-answering. Users can scrape a charity website and immediately start asking questions about the organization's mission, programs, donation methods, and more. The system uses semantic search with embeddings and an LLM to generate accurate, sourced answers.

### Key Features

- Web scraping with Requests and Selenium fallback for JavaScript-heavy sites
- Automatic document chunking and semantic embedding using Sentence Transformers
- Vector database storage with Chroma for fast semantic search
- RAG pipeline combining retrieval with OpenAI LLM generation
- Multi-turn conversation support with session management
- Terminal CLI interface for easy interaction
- Minimal similarity threshold (0.15) for comprehensive context retrieval
- Response reranking for quality assurance

## Technology Stack

- **Web Scraping**: BeautifulSoup, Selenium, Requests
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: Chroma (persistent storage)
- **LLM**: OpenAI (GPT-3.5-turbo, GPT-4o, GPT-4o-mini)
- **Backend API**: FastAPI with CORS support
- **Python**: 3.13
- **Logging**: loguru

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Virtual environment (recommended)

### Setup Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-charity-chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
export OPENAI_API_KEY='your-api-key-here'
export OPENAI_MODEL='gpt-3.5-turbo'  # Optional
```

5. Verify installation:
```bash
python -c "from src.rag.rag_system import RAGSystem; print('Installation successful')"
```

## Quick Start

### Running the Terminal Chatbot

```bash
python scripts/cli_chatbot.py
```

This starts an interactive terminal session where you can:

1. Enter a charity website URL
2. Choose to scrape homepage only or entire site
3. Start asking questions about the charity
4. Type 'exit' or 'quit' to end the session



### Running the Backend API

To run the REST API server:

```bash
python scripts/run_backend.py
```

The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs

## API Documentation

### Swagger UI

Access the interactive Swagger documentation at:

```
http://localhost:8000/docs
```

From the Swagger interface you can:
- View all available endpoints
- See request/response schemas
- Try endpoints directly with custom parameters
- View example requests and responses


### Available Endpoints

**Chat Endpoints**
- `POST /chat/` - Send a query and receive an answer with sources
- `GET /chat/history/{session_id}` - Retrieve conversation history for a session
- `DELETE /chat/history/{session_id}` - Clear conversation history

**Scrape Endpoints**
- `POST /scrape/` - Scrape a website and index it into the vector database
- `GET /scrape/charities` - List all indexed charities and their chunk counts

**System Endpoints**
- `GET /health` - Check API health status and component status
- `GET /` - Get service information and available endpoints

### Example API Calls

Query the chatbot:
```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What programs does this charity offer?",
    "charity_name": "Sewacanada",
    "top_k": 5,
    "session_id": "session-123"
  }'
```

List indexed charities:
```bash
curl "http://localhost:8000/scrape/charities"
```

Get health status:
```bash
curl "http://localhost:8000/health"
```

## Running the Backend API

To run the REST API server:

```bash
python scripts/run_backend.py
```

The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs

## System Architecture

### Core Components

**Document Processing**
- Scraper: Retrieves content from websites with JavaScript rendering support
- HTMLCleaner: Removes noise elements and extracts readable text
- Chunker: Splits documents into fixed-size chunks with configurable overlap

**Embedding & Storage**
- EmbeddingGenerator: Uses Sentence Transformers to convert text to 384-dimensional vectors
- ChromaVectorDB: Persists embeddings and enables semantic search
- Collections organized by charity name

**RAG Pipeline**
- SemanticRetriever: Performs vector similarity search with configurable threshold
- PromptFormatter: Constructs system and user prompts with context
- OpenAIProvider: Interfaces with OpenAI API for generation
- RAGSystem: Orchestrates retrieval, formatting, and generation

**API Endpoints**
- GET /health - Server health status
- GET / - Service information
- POST /chat/ - Query the chatbot
- GET /chat/history/{session_id} - Retrieve conversation history
- DELETE /chat/history/{session_id} - Clear conversation history
- GET /scrape/charities - List indexed charities
- POST /scrape/ - Scrape and index a new website

## Configuration

Key parameters can be adjusted in the code:

**Retrieval Configuration**
- `top_k`: Number of chunks to retrieve (default: 5)
- `similarity_threshold`: Minimum relevance score (default: 0.15)
- `rerank`: Enable reranking by similarity (default: True)

**Chunking Configuration**
- `chunk_size`: Fixed size in tokens (default: 256)
- `overlap`: Token overlap between chunks (default: 50)

**LLM Configuration**
- `model_name`: OpenAI model to use
- `temperature`: Response creativity (default: 0.7)
- `max_tokens`: Maximum response length (default: 1024)

## Testing

Run the comprehensive API test suite:

```bash
python scripts/test_backend_api.py
```

Test the RAG system directly:

```bash
python scripts/test_rag_system.py
```

## Project Status

Completed:
- Web scraping with fallback mechanisms
- Document chunking and embedding
- Vector database setup and persistence
- RAG retrieval and generation pipeline
- Multi-turn conversation management
- Terminal CLI interface
- REST API with all endpoints
- Comprehensive testing

Potential Future Enhancements:
- Support for additional document formats (PDFs, images)
- Web UI interface
- Multi-language support
- Fine-tuned models for specific charity types
- Caching layer for repeated queries
- Analytics and usage tracking

## Troubleshooting

**Import Errors**
Ensure virtual environment is activated and all dependencies installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**API Key Issues**
- Verify OPENAI_API_KEY is set: `echo $OPENAI_API_KEY`
- Check API key validity in OpenAI dashboard
- Never commit API keys to version control

**Scraping Failures**
- Website may have anti-bot protection
- Try homepage-only mode for initial test
- Some sites may require JavaScript rendering (automatic fallback attempts this)

**Low Retrieval Performance**
- Current similarity threshold is 0.15 (permissive)
- Try more specific query wording
- Ensure website has substantial text content

**Slow Embedding Generation**
- Initial embedding generation takes time for large documents
- Subsequent queries use cached embeddings
- Batching processes multiple documents efficiently

## Development

### Project Structure

Core modules are organized by function:
- Scraping and ingestion
- Embedding and vectorization
- Vector database operations
- RAG system orchestration
- LLM client integration
- CLI interface
- REST API routes

### Running Tests

```bash
# Test backend API
python scripts/test_backend_api.py

# Test RAG components
python scripts/test_rag_system.py
```

## License

MIT License - see LICENSE file for details

---

Last Updated: January 6, 2026
Status: CLI chatbot fully functional and tested

