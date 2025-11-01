#!/usr/bin/env python3
"""Script to ingest a transcript."""
import sys
from pathlib import Path
from src.ingest.transcript import TranscriptIngester

def main():
    transcript_path = "transcript_cleaned.txt"
    persona_name = "VirtualHuman"
    
    print(f"📖 Reading transcript from: {transcript_path}")
    transcript = Path(transcript_path).read_text(encoding="utf-8")
    print(f"   Transcript length: {len(transcript)} characters")
    print(f"   Number of words: {len(transcript.split())}")
    print()
    
    print(f"🤖 Starting ingestion for persona: {persona_name}")
    print("   This will make multiple LLM calls and may take a few minutes...")
    print()
    
    try:
        ingester = TranscriptIngester()
        
        print("   Step 1: Chunking transcript...")
        # We'll let the ingest function handle everything
        result = ingester.ingest(
            transcript_path=transcript_path,
            persona_name=persona_name,
        )
        
        print()
        print("✅ Ingestion complete!")
        print(f"   Persona: {result['persona_name']}")
        print(f"   Facts extracted: {result['facts_count']}")
        print(f"   Examples generated: {result['examples_count']}")
        print(f"   Chunks created: {result.get('chunks_count', 'N/A')}")
        print(f"   Status: {result['status']}")
        print()
        print(f"📁 Persona artifacts saved to: persona/{persona_name}/")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

