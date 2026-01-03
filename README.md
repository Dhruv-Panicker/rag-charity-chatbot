# RAG Charity Chatbot

An intelligent retrieval-augmented generation (RAG) chatbot that automatically scrapes charity websites and provides accurate, context-grounded answers to user queries using the information from those websites.

## Project Overview

This project aims to make charity information more accessible by leveraging modern AI techniques. Instead of navigating complex website structures, users can simply ask questions about a charity's mission, history, donation information, and more—all through an intelligent chatbot powered by retrieval-augmented generation.

### Use Cases
- **Charity Information Retrieval**: Get quick answers about charity missions, history, and impact
- **Donation Guidance**: Find out how to donate and understand fund allocation
- **Program Information**: Learn about specific programs and services offered
- **Contact & Support**: Quick access to frequently asked questions and contact information

## Technology Stack

### Core Components
- **Web Scraping**: BeautifulSoup, Selenium
- **PDF Processing**: PyPDF2, ReportLab
- **LLM Integration**: OpenAI GPT-4, Anthropic Claude
- **Embeddings**: Sentence Transformers
- **Vector Database**: Chroma (development), Pinecone (production)
- **Backend API**: FastAPI
- **Frontend**: React (planned)

### Development Tools
- **Testing**: pytest
- **Code Quality**: black, flake8, mypy
- **Logging**: loguru
- **Data Processing**: pandas, scikit-learn


## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip or conda
- Git

**Required API Keys:**
- `OPENAI_API_KEY`: Get from [OpenAI Platform](https://platform.openai.com)
- `ANTHROPIC_API_KEY`: Get from [Anthropic Console](https://console.anthropic.com)

### Step 5: Verify Installation
```bash
python -c "from config import settings; print('Configuration loaded successfully')"
```

## Development Roadmap

### Phase 1: Foundation & Setup ✅
- [x] Project structure
- [x] Virtual environment
- [x] Dependencies
- [x] Configuration management
- [ ] Git initialization

### Phase 2: Data Ingestion Pipeline
- [ ] Web scraper module
- [ ] PDF generation
- [ ] Document storage system

### Phase 3: Vector Database & Embeddings
- [ ] Document chunking strategies
- [ ] Embedding generation
- [ ] Vector database setup

### Phase 4: RAG Retrieval & LLM Integration
- [ ] Retrieval module
- [ ] Prompt engineering
- [ ] LLM integration

### Phase 5: Web Interface & API
- [ ] FastAPI backend
- [ ] REST endpoints
- [ ] Frontend web interface

### Phase 6: Evaluation & Optimization
- [ ] Retrieval evaluation metrics
- [ ] Generation quality assessment
- [ ] Cost & performance analysis

### Phase 7: Advanced Features
- [ ] Multi-format support
- [ ] Semantic search improvements
- [ ] Feedback loop implementation

### Phase 8: Documentation & Deployment
- [ ] Complete documentation
- [ ] Deployment setup
- [ ] Demo and case studies

## Quick Start (Once Setup Complete)

```python
from config import settings
from src.scraper import WebScraper
from src.ingestion import PDFGenerator
from src.embeddings import EmbeddingModel
from src.rag import RAGChatbot

# 1. Scrape charity website
scraper = WebScraper(settings.headless_browser)
content = scraper.scrape("https://example-charity.org")

# 2. Generate PDF
pdf_gen = PDFGenerator()
pdf_gen.create_pdf(content, "charity_info.pdf")

# 3. Generate embeddings
embedder = EmbeddingModel(settings.embedding_model)
embeddings = embedder.embed_document("charity_info.pdf")

# 4. Initialize RAG chatbot
chatbot = RAGChatbot()
response = chatbot.query("What is this charity about?")
print(response)
```

## Configuration Guide

### Key Settings in `.env`

#### LLM Configuration
- `LLM_MODEL`: Which LLM to use (gpt-4, claude-3-sonnet, etc.)
- `TEMPERATURE`: Response creativity (0.0 = deterministic, 1.0 = creative)
- `MAX_TOKENS`: Maximum response length

#### Embedding Configuration
- `EMBEDDING_MODEL`: Model for generating embeddings
- `EMBEDDING_DIMENSION`: Dimension of embedding vectors

#### Vector Database
- `VECTOR_DB_TYPE`: Choose between `chroma` or `pinecone`
- `CHROMA_DB_PATH`: Local path for Chroma database

#### RAG Configuration
- `TOP_K_RETRIEVAL`: Number of documents to retrieve for context
- `SIMILARITY_THRESHOLD`: Minimum relevance score to include document
- `CHUNK_SIZE`: Size of document chunks for embedding
- `CHUNK_OVERLAP`: Overlap between chunks for context continuity

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_scraper.py -v
```

## Code Quality

```bash
# Format code
black src/

# Check code style
flake8 src/

# Type checking
mypy src/

# Sort imports
isort src/
```


## Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### API Key Issues
- Verify `.env` file exists and is not ignored by git
- Check API key validity in their respective platforms
- Never commit actual `.env` file to git

### Database Errors
- Clear Chroma cache: `rm -rf ./data/vector_store`
- Reset database: `rm ./data/app.db`
- Reinitialize: `python -m src.setup`

## Documentation

- [Architecture Overview](#) - Coming soon
- [API Reference](#) - Coming soon
- [Development Guide](#) - Coming soon


## License

MIT License - see LICENSE file for details

## Contact & Support

For questions or feedback, please open an issue on GitHub or contact the maintainer.

---

**Last Updated**: January 3, 2026  
**Status**: Phase 1 - Foundation Setup Complete ✅
