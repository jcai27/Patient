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

## Detailed Startup Tutorial

1. **Install prerequisites**
   - Ensure Python 3.10+ and `pip` are on your PATH (`python --version`).
   - (Optional) Create and activate a virtualenv: `python -m venv .venv && source .venv/bin/activate` (Windows PowerShell: `.venv\Scripts\Activate.ps1`).

2. **Install project requirements**
   ```bash
   pip install -r requirements.txt
   ```
   This pulls FastAPI, Chroma, sentence-transformers, and all agent dependencies.

3. **Configure API access**
   - Export either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` before running anything that calls the LLM.
   - For bash/zsh:
     ```bash
     export OPENAI_API_KEY=sk-...
     # or
     export ANTHROPIC_API_KEY=sk-ant-...
     ```
   - For PowerShell:
     ```powershell
     $env:OPENAI_API_KEY = "sk-..."
     ```

4. **(Optional) Prime persona artifacts**
   - Place transcript `.txt` files in a convenient location.
   - Run the ingestion helper to generate persona data:
     ```bash
     python src/ingest/transcript.py --persona-name "Alice" --transcript path/to/transcript.txt
     ```
     or use the HTTP endpoint shown below. Ingestion populates `persona/<name>/` with `persona_profile.json`, `style_rules.md`, `canonical_facts.jsonl`, and embeddings.

5. **Verify dependencies load cleanly**
   ```bash
   python test_imports.py
   ```
   This sanity-checks that GPU/CPU libraries and sentence-transformers import without error.

6. **Start the API server**
   ```bash
   python run_server.py
   ```
   The FastAPI app boots on `http://localhost:8000/`. Leave this running while you interact with the system.

7. **Ingest a transcript (if you skipped step 4)**
   ```bash
   curl -X POST http://localhost:8000/ingest/transcript \
     -H "Content-Type: application/json" \
     -d '{"transcript_path": "path/to/transcript.txt", "persona_name": "Alice"}'
   ```
   Watch the server logs for ingestion progress; artifacts appear under `persona/Alice/`.

8. **Start chatting with the persona**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id": "demo-user", "session_id": null, "message": "Hey, can you introduce yourself?"}'
   ```
   The response includes citations, judge scores, and a `trace_id` that you can inspect.

9. **Inspect traces or iterate on persona**
   ```bash
   curl "http://localhost:8000/inspect/trace?trace_id=<trace_id_from_chat>"
   ```
   Adjust `persona/<name>/style_rules.md` or re-run ingestion to refine the voice.

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

