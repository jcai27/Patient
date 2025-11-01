#!/usr/bin/env python3
"""Example usage of the persona chatbot."""
import requests
import json

BASE_URL = "http://localhost:8000"

def ingest_transcript_example():
    """Example: Ingest a transcript."""
    # Option 1: From file
    with open("example_transcript.txt", "w", encoding="utf-8") as f:
        f.write("""
Interview with Alice:
Q: What's your approach to decision-making?
A: I tend to be methodical. I gather all the facts first, then think through the implications. Sometimes I second-guess myself, but that's okay - it means I care about getting it right.

Q: How do you handle stress?
A: I take deep breaths and break things down into smaller steps. Exercise helps too - going for a run clears my head.

Q: What's important to you in relationships?
A: Honesty above all. I value people who are direct and authentic. Small talk isn't really my thing - I prefer meaningful conversations.
""")
    
    response = requests.post(
        f"{BASE_URL}/ingest/transcript",
        json={
            "transcript_path": "example_transcript.txt",
            "persona_name": "Alice",
        }
    )
    print("Ingest response:", json.dumps(response.json(), indent=2))


def chat_example():
    """Example: Chat with the persona."""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "user_id": "user123",
            "message": "What's your approach to making decisions?",
        }
    )
    print("Chat response:", json.dumps(response.json(), indent=2))
    
    # Get trace
    trace_id = response.json()["trace_id"]
    trace_response = requests.get(f"{BASE_URL}/inspect/trace?trace_id={trace_id}")
    print("\nTrace:", json.dumps(trace_response.json(), indent=2))


if __name__ == "__main__":
    print("Example 1: Ingest transcript")
    print("-" * 50)
    try:
        ingest_transcript_example()
    except Exception as e:
        print(f"Error: {e}")
        print("(Make sure the server is running)")
    
    print("\n\nExample 2: Chat")
    print("-" * 50)
    try:
        chat_example()
    except Exception as e:
        print(f"Error: {e}")
        print("(Make sure the server is running and a persona is loaded)")

