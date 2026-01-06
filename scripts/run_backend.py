import uvicorn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    print("="*80)
    print("STARTING RAG CHARITY CHATBOT API")
    print("="*80)
    print("\n📡 API will be available at:")
    print("   - Main API: http://localhost:8000")
    print("   - Swagger Docs: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("\n🔧 Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )