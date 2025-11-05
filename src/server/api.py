"""FastAPI server for persona chatbot."""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from src.server.schemas import (
    ChatRequest,
    ChatResponse,
    IngestTranscriptRequest,
    IngestTranscriptResponse,
    TraceResponse,
    TabooConfig,
    PersonaSwitchRequest,
    PersonaSwitchResponse,
)
from src.agents.orchestrator import Orchestrator
from src.ingest.transcript import TranscriptIngester
from src.config import PERSONA_DIR
from src.memory.episodic import EpisodicMemory

# Global state (in production, use proper state management)
_current_persona: Optional[str] = None
_orchestrators: Dict[str, Orchestrator] = {}
_traces: Dict[str, Dict[str, Any]] = {}  # trace_id -> trace data
_ingestion_status: Dict[str, Dict[str, Any]] = {}  # persona_name -> status

app = FastAPI(
    title="Persona Chatbot API",
    description="RAG + Multi-Agent Orchestration Chatbot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_orchestrator(persona_name: Optional[str] = None) -> Orchestrator:
    """Get or create orchestrator for persona."""
    global _current_persona, _orchestrators
    
    if persona_name is None:
        persona_name = _current_persona
    
    if persona_name is None:
        # Try to find first persona in directory
        if PERSONA_DIR.exists():
            personas = [d.name for d in PERSONA_DIR.iterdir() if d.is_dir()]
            if personas:
                persona_name = personas[0]
            else:
                raise HTTPException(
                    status_code=404,
                    detail="No personas available. Please ingest a transcript first.",
                )
    
    if persona_name not in _orchestrators:
        try:
            _orchestrators[persona_name] = Orchestrator(persona_name)
            _current_persona = persona_name
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load persona '{persona_name}': {str(e)}",
            )
    
    return _orchestrators[persona_name]


@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = Path("static/index.html")
    if not index_path.exists():
        return HTMLResponse("""
        <html>
            <body>
                <h1>Frontend not found</h1>
                <p>Please create static/index.html</p>
                <p>Current directory: """ + str(Path.cwd()) + """</p>
            </body>
        </html>
        """)
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the persona chatbot."""
    try:
        orchestrator = get_orchestrator()
        
        # Get conversation history from memory (simplified - in production, fetch from DB)
        memory = EpisodicMemory()
        conversation_history = []
        if request.session_id:
            conversation_history = memory.get_conversation_history(request.session_id)
        summary_record = memory.get_summary(request.session_id) if request.session_id else None
        
        result = orchestrator.process_turn(
            user_message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_history=conversation_history,
        )
        
        # Store trace
        _traces[result["trace_id"]] = result["trace"]
        
        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            citations=result["citations"],
            scores=result["scores"],
            revised=result["revised"],
            trace_id=result["trace_id"],
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Chat error: {error_details}")  # Log to server console
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/upload/transcript")
async def upload_transcript(
    file: UploadFile = File(...),
    persona_name: str = Form(...),
):
    """Upload and ingest a transcript file."""
    global _ingestion_status
    
    try:
        # Validate file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")
        
        # Read file content
        content = await file.read()
        transcript_text = content.decode('utf-8')
        
        # Update status
        _ingestion_status[persona_name] = {
            "status": "processing",
            "progress": "Starting ingestion...",
            "persona_name": persona_name,
        }
        
        # Ingest transcript
        ingester = TranscriptIngester()
        result = ingester.ingest(
            transcript_path=file.filename,
            persona_name=persona_name,
            transcript_text=transcript_text,
        )
        
        # Invalidate orchestrator cache for this persona
        global _orchestrators
        if persona_name in _orchestrators:
            del _orchestrators[persona_name]
        
        # Update status
        _ingestion_status[persona_name] = {
            "status": "complete",
            "progress": "Ingestion complete!",
            "persona_name": result["persona_name"],
            "facts_count": result["facts_count"],
            "examples_count": result["examples_count"],
            "chunks_count": result.get("chunks_count", 0),
        }
        
        return {
            "status": "success",
            "persona_name": result["persona_name"],
            "facts_count": result["facts_count"],
            "examples_count": result["examples_count"],
            "message": "Transcript ingested successfully!",
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Upload error: {error_details}")  # Log to server console
        
        _ingestion_status[persona_name] = {
            "status": "error",
            "progress": f"Error: {str(e)}",
            "persona_name": persona_name,
        }
        # Return more detailed error for debugging
        raise HTTPException(
            status_code=500, 
            detail=f"Upload failed: {str(e)}\n\nFull error:\n{error_details}"
        )


@app.get("/ingestion/status/{persona_name}")
async def get_ingestion_status(persona_name: str):
    """Get ingestion status for a persona."""
    if persona_name in _ingestion_status:
        return _ingestion_status[persona_name]
    return {"status": "not_found"}


@app.get("/personas")
async def list_personas():
    """List all available personas."""
    if not PERSONA_DIR.exists():
        return {"personas": []}
    
    personas = []
    for persona_dir in PERSONA_DIR.iterdir():
        if persona_dir.is_dir():
            artifacts = {
                "profile": (persona_dir / "persona_profile.json").exists(),
                "facts": (persona_dir / "canonical_facts.jsonl").exists(),
                "examples": (persona_dir / "examples.jsonl").exists(),
                "style_rules": (persona_dir / "style_rules.md").exists(),
                "taboos": (persona_dir / "taboo_list.md").exists(),
            }
            personas.append({
                "name": persona_dir.name,
                "artifacts": artifacts,
            })
    
    return {"personas": personas}


@app.post("/ingest/transcript", response_model=IngestTranscriptResponse)
async def ingest_transcript(request: IngestTranscriptRequest):
    """Ingest a transcript and generate persona artifacts."""
    try:
        ingester = TranscriptIngester()
        result = ingester.ingest(
            transcript_path=request.transcript_path,
            persona_name=request.persona_name,
            transcript_text=request.transcript_text,
        )
        
        # Invalidate orchestrator cache for this persona
        global _orchestrators
        if request.persona_name in _orchestrators:
            del _orchestrators[request.persona_name]
        
        return IngestTranscriptResponse(
            persona_name=result["persona_name"],
            facts_count=result["facts_count"],
            examples_count=result["examples_count"],
            status=result["status"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inspect/trace", response_model=TraceResponse)
async def inspect_trace(trace_id: str):
    """Inspect a trace for debugging."""
    if trace_id not in _traces:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    trace = _traces[trace_id]
    
    from datetime import datetime
    
    timestamp_str = trace.get("timestamp", datetime.now().isoformat())
    if isinstance(timestamp_str, datetime):
        timestamp_str = timestamp_str.isoformat()
    
    return TraceResponse(
        trace_id=trace["trace_id"],
        session_id=trace["session_id"],
        timestamp=timestamp_str,
        user_message=trace["user_message"],
        final_response=trace.get("final_response", ""),
        retrieval_results=trace.get("retrieval_results", []),
        producer_output=trace.get("producer_output", ""),
        contextor_output=trace.get("contextor_output", {}),
        refiner_output=trace.get("refiner_output", ""),
        judge_scores=trace.get("judge_scores", {}),
        judge_edits=trace.get("judge_edits", []),
        notes_used=trace.get("notes_used", []),
        iterations=trace.get("iterations", 0),
    )


@app.post("/admin/taboos")
async def update_taboos(config: TabooConfig, persona_name: str = "default"):
    """Update taboo list for a persona."""
    try:
        persona_dir = PERSONA_DIR / persona_name
        if not persona_dir.exists():
            raise HTTPException(status_code=404, detail="Persona not found")
        
        taboo_file = persona_dir / "taboo_list.md"
        
        content = "# Taboo List\n\n"
        content += "## Topics\n"
        for taboo in config.taboos:
            content += f"- {taboo}\n"
        content += f"\n## Refusal Language\n{config.redirect_language}\n"
        
        with open(taboo_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Invalidate orchestrator
        global _orchestrators
        if persona_name in _orchestrators:
            del _orchestrators[persona_name]
        
        return {"status": "success", "message": "Taboos updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/persona/switch", response_model=PersonaSwitchResponse)
async def switch_persona(request: PersonaSwitchRequest):
    """Switch to a different persona."""
    try:
        persona_dir = PERSONA_DIR / request.persona_name
        if not persona_dir.exists():
            raise HTTPException(status_code=404, detail="Persona not found")
        
        # Check artifacts
        artifacts_loaded = {
            "profile": (persona_dir / "persona_profile.json").exists(),
            "facts": (persona_dir / "canonical_facts.jsonl").exists(),
            "examples": (persona_dir / "examples.jsonl").exists(),
            "style_rules": (persona_dir / "style_rules.md").exists(),
            "taboos": (persona_dir / "taboo_list.md").exists(),
        }
        
        global _current_persona
        _current_persona = request.persona_name
        
        # Pre-load orchestrator
        get_orchestrator(request.persona_name)
        
        return PersonaSwitchResponse(
            persona_name=request.persona_name,
            status="switched",
            artifacts_loaded=artifacts_loaded,
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Persona switch error: {error_details}")  # Log to server console
        raise HTTPException(status_code=500, detail=f"Persona switch failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Mount static files for frontend
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # Directory might not exist yet


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
