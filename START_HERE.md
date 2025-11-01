# Quick Start: Ingest Transcript & Chat

## Step 1: Start the Server

Open a terminal and run:
```bash
python run_server.py
```

The server will start at `http://localhost:8000`

## Step 2: Ingest Your Transcript

In another terminal (or use Postman/curl), send this request:

**Option A: Using curl**
```bash
curl -X POST http://localhost:8000/ingest/transcript `
  -H "Content-Type: application/json" `
  -d '{\"transcript_path\": \"transcript_cleaned.txt\", \"persona_name\": \"VirtualHuman\"}'
```

**Option B: Using PowerShell Invoke-RestMethod**
```powershell
$body = @{
    transcript_path = "transcript_cleaned.txt"
    persona_name = "VirtualHuman"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/ingest/transcript" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

This will:
- Take 2-5 minutes (multiple LLM calls)
- Create persona/VirtualHuman/ with:
  - `persona_profile.json` - Personality profile
  - `canonical_facts.jsonl` - Extracted facts for RAG
  - `examples.jsonl` - Few-shot examples
  - `style_rules.md` - Style guidelines
  - `taboo_list.md` - Topics to avoid

## Step 3: Start Chatting

Once ingestion completes, chat with the persona:

```powershell
$chatBody = @{
    user_id = "user1"
    message = "How are you handling everything?"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body $chatBody
```

The response will include:
- `response`: The persona's answer
- `citations`: Which facts were used (e.g., [D3], [D7])
- `scores`: Quality scores from Judge
- `trace_id`: For inspecting the full process

## What Happens During Chat (RAG Process):

1. **Query**: "How are you handling everything?"
2. **Retrieval**: Searches canonical_facts.jsonl
   - Finds: "[D12] Doing yoga to keep mind off things"
   - Finds: "[D8] Wife left 3 weeks ago, hardest thing ever"
   - Finds: "[D15] Depressed for months on end"
3. **Generation**: Creates response using those facts
4. **Styling**: Applies VirtualHuman's voice/style
5. **Judge**: Scores quality (factuality, persona match, etc.)
6. **Return**: Response with citations

You can inspect any response using:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/inspect/trace?trace_id=<trace_id>"
```

