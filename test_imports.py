#!/usr/bin/env python3
"""Test that all imports work correctly."""
print("Testing imports...")

try:
    print("  ✓ Importing config...")
    from src.config import OPENAI_API_KEY
    print(f"     API Key loaded: {'Yes' if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20 else 'No'}")
    
    print("  ✓ Importing server components...")
    from src.server.api import app
    print("     Server app imported successfully")
    
    print("  ✓ Importing agents...")
    from src.agents.orchestrator import Orchestrator
    from src.agents.contextor import Contextor
    from src.agents.producer import Producer
    from src.agents.refiner import StyleRefiner
    from src.agents.judge import Judge
    print("     All agents imported successfully")
    
    print("  ✓ Importing retrieval system...")
    from src.retriever.index import HybridRetriever
    from src.retriever.rerank import Reranker
    print("     Retrieval system imported successfully")
    
    print("  ✓ Importing ingestion...")
    from src.ingest.transcript import TranscriptIngester
    print("     Ingestion imported successfully")
    
    print("\n✅ All imports successful! Server should start correctly.")
    print("\nTo start the server, run:")
    print("   python run_server.py")
    
except Exception as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()

