#!/usr/bin/env python3
"""Test the upload endpoint locally."""
from pathlib import Path
from src.ingest.transcript import TranscriptIngester

# Test with the cleaned transcript
transcript_path = "transcript_cleaned.txt"
persona_name = "VirtualHuman"

print(f"Testing ingestion with: {transcript_path}")
print(f"Persona: {persona_name}")
print()

if not Path(transcript_path).exists():
    print(f"❌ File not found: {transcript_path}")
    exit(1)

try:
    transcript_text = Path(transcript_path).read_text(encoding='utf-8')
    print(f"✓ File read successfully ({len(transcript_text)} chars)")
    print()
    
    print("Starting ingestion...")
    ingester = TranscriptIngester()
    result = ingester.ingest(
        transcript_path=transcript_path,
        persona_name=persona_name,
        transcript_text=transcript_text,
    )
    
    print()
    print("✅ Success!")
    print(f"   Persona: {result['persona_name']}")
    print(f"   Facts: {result['facts_count']}")
    print(f"   Examples: {result['examples_count']}")
    
except Exception as e:
    print()
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


