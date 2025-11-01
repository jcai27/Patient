# Quick Steps to Get Started

## 1. Make sure server is running
In your terminal, you should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

If not, run:
```bash
python run_server.py
```

## 2. Ingest the transcript
Open a **new PowerShell window** (keep server running) and run:

```powershell
.\ingest_now.ps1
```

Or manually:
```powershell
$body = @{ transcript_path = "transcript_cleaned.txt"; persona_name = "VirtualHuman" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/ingest/transcript" -Method POST -ContentType "application/json" -Body $body
```

**This takes 2-5 minutes** - it's making multiple LLM API calls to:
- Chunk the transcript
- Extract canonical facts (for RAG)
- Generate persona profile
- Create style rules
- Build examples

## 3. Verify ingestion completed
Check if files were created:
```powershell
Get-ChildItem persona/VirtualHuman
```

You should see:
- `persona_profile.json`
- `canonical_facts.jsonl` ← This is your RAG knowledge base!
- `examples.jsonl`
- `style_rules.md`
- `taboo_list.md`

## 4. Start chatting!
```powershell
.\chat.ps1 -Message "How are you handling everything?"
```

Or manually:
```powershell
$body = @{ user_id = "user1"; message = "How are you doing?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method POST -ContentType "application/json" -Body $body
```

## What's happening (RAG System):

1. **You ask**: "How are you handling everything?"
2. **System retrieves** from `canonical_facts.jsonl`:
   - Facts about wife leaving
   - Facts about coping (yoga, etc.)
   - Facts about emotional state
3. **Producer** creates neutral factual response with citations [D3], [D5]
4. **Contextor** builds style pack (tone, hedging, etc.)
5. **Refiner** transforms to VirtualHuman's voice
6. **Judge** scores quality and ensures it's appropriate
7. **Returns** response in persona voice with source citations

## Troubleshooting

**"No personas available" error:**
- Run `.\ingest_now.ps1` first

**"Empty retrieval results":**
- Check `persona/VirtualHuman/canonical_facts.jsonl` exists
- Should have multiple lines with facts

**API errors:**
- Make sure server is still running
- Check `.env` file has your OpenAI API key set

