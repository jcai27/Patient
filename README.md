# Persona Chatbot with RAG + Multi-Agent Orchestration

A chatbot that reliably speaks like X person and stays on-brand using small transcripts, retrieval, and a 4-agent loop (Producer, Orchestrator, Contextor, Judge).

## Primary Goals

- **Authentic voice**: Style similarity ≥4/5
- **Factual grounding**: Citations to transcript sources
- **Safe/consistent behavior**: Taboo enforcement, hedging when uncertain
- **Quick iteration**: New persona setup in ≤10 minutes

## Architecture

### Stores
- **Style store**: `persona_profile.json`, `style_rules.md`, `examples.jsonl`, `taboo_list.md`
- **Knowledge store**: `canonical_facts.jsonl` → FAISS/Chroma index
- **Memory**: Per-user episodic notes + rolling conversation summaries

### Agents
1. **Producer**: Neutral factual drafts from retrieved notes
2. **Orchestrator**: Main loop controller (retrieval → produce → style → judge)
3. **Contextor**: Builds Style+Policy Pack (tone, few-shots, taboos)
4. **Judge**: Scores responses (Factuality, Persona, Helpfulness, Safety) and issues edits

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:
```bash
OPENAI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Ingest a transcript
```bash
curl -X POST http://localhost:8000/ingest/transcript \
  -H "Content-Type: application/json" \
  -d '{"transcript_path": "path/to/transcript.txt", "persona_name": "Alice"}'
```

### Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "message": "What advice do you have?"}'
```

### Inspect traces
```bash
curl http://localhost:8000/inspect/trace?session_id=session123
```

## Project Structure

```
/persona/          # Persona artifacts
/src/
  /retriever/      # Hybrid search + reranking
  /agents/         # All 4 agents
  /memory/         # Episodic + summaries
  /server/         # FastAPI endpoints
  /eval/           # Evaluation harness
```

