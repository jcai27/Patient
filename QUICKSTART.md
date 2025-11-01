# Quick Start Guide

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   - Copy `.env.example` to `.env`
   - Add your API keys (OpenAI or Anthropic)

3. **Create directories:**
   ```bash
   mkdir -p persona data eval/datasets
   ```

## Running the Server

```bash
python run_server.py
```

The API will be available at `http://localhost:8000`

## Usage

### 1. Ingest a Transcript

Create a transcript file (e.g., `transcript.txt`) and ingest it:

```bash
curl -X POST http://localhost:8000/ingest/transcript \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_path": "transcript.txt",
    "persona_name": "Alice"
  }'
```

Or with inline text:

```bash
curl -X POST http://localhost:8000/ingest/transcript \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_text": "Interview with Alice...",
    "persona_name": "Alice"
  }'
```

This will:
- Chunk the transcript
- Extract canonical facts
- Generate persona profile
- Create style rules and examples
- Build the vector index

### 2. Chat with the Persona

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "What advice do you have for making decisions?"
  }'
```

Response includes:
- `response`: The persona's reply
- `citations`: Source IDs used
- `scores`: Judge scores (factuality, persona, helpfulness, safety)
- `trace_id`: For inspecting the generation process

### 3. Inspect Traces

```bash
curl http://localhost:8000/inspect/trace?trace_id=<trace_id>
```

Shows:
- Retrieval results
- Producer output (neutral draft)
- Contextor output (style pack)
- Refiner output (styled response)
- Judge scores and edits
- Number of revision iterations

### 4. Admin Operations

**Switch persona:**
```bash
curl -X POST http://localhost:8000/admin/persona/switch \
  -H "Content-Type: application/json" \
  -d '{"persona_name": "Bob"}'
```

**Update taboos:**
```bash
curl -X POST http://localhost:8000/admin/taboos?persona_name=Alice \
  -H "Content-Type: application/json" \
  -d '{
    "taboos": ["topic1", "topic2"],
    "redirect_language": "I prefer not to discuss that."
  }'
```

## Evaluation

Use the evaluation harness:

```python
from eval.harness import EvaluationHarness

harness = EvaluationHarness("Alice")
result = harness.evaluate_prompt("What's your approach?")
print(result["judge_scores"])
```

Evaluate a dataset:

```python
results = harness.evaluate_dataset("eval/datasets/test_set.jsonl")
print(f"Persona style pass rate: {results['persona_style_pass_rate']}")
```

## Project Structure

```
/persona/          # Persona artifacts (auto-generated)
/src/
  /retriever/      # Hybrid search (BM25 + embeddings) + reranking
  /agents/         # Producer, Contextor, Refiner, Judge, Orchestrator
  /memory/         # Episodic notes + conversation summaries
  /ingest/         # Transcript ingestion pipeline
  /server/         # FastAPI endpoints
  /data/           # Pydantic models
  /utils/          # LLM client utilities
/eval/             # Evaluation harness
```

## Configuration

Edit `src/config.py` or set environment variables:

- `K_RETRIEVE`: Number of facts to retrieve (default: 5)
- `CONFIDENCE_THRESHOLD`: Low-evidence threshold (default: 0.40)
- `MAX_REVISE_LOOPS`: Max judge revision iterations (default: 2)
- `CHUNK_SIZE_WORDS`: Transcript chunk size (default: 150)
- `CHUNK_OVERLAP_WORDS`: Overlap between chunks (default: 25)

## Troubleshooting

**No personas available:**
- Ingest a transcript first using `/ingest/transcript`

**Empty retrieval results:**
- Check that `canonical_facts.jsonl` exists in the persona directory
- Verify the vector index was built (check for `chroma_db/` directory)

**Low persona style scores:**
- Review and edit `persona_profile.json` and `style_rules.md`
- Add more examples to `examples.jsonl`
- Adjust style parameters in the profile

**API errors:**
- Check that API keys are set in `.env`
- Verify the LLM provider matches your key type (OpenAI vs Anthropic)

