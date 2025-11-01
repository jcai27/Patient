"""Pydantic schemas for API requests and responses."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """Request to chat with the persona."""
    user_id: str
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from persona chatbot."""
    response: str
    session_id: str
    citations: List[str] = Field(default_factory=list)
    scores: Optional[Dict[str, float]] = None
    revised: bool = False
    trace_id: Optional[str] = None


class IngestTranscriptRequest(BaseModel):
    """Request to ingest a transcript."""
    transcript_path: str
    persona_name: str
    transcript_text: Optional[str] = None  # If provided, use instead of file


class IngestTranscriptResponse(BaseModel):
    """Response from transcript ingestion."""
    persona_name: str
    facts_count: int
    examples_count: int
    status: str


class TraceResponse(BaseModel):
    """Trace inspection response."""
    trace_id: str
    session_id: str
    timestamp: str  # ISO format string
    user_message: str
    final_response: str
    retrieval_results: List[Dict[str, Any]] = Field(default_factory=list)
    producer_output: str
    contextor_output: Dict[str, Any] = Field(default_factory=dict)
    refiner_output: str
    judge_scores: Dict[str, float] = Field(default_factory=dict)
    judge_edits: List[str] = Field(default_factory=list)
    notes_used: List[str] = Field(default_factory=list)
    iterations: int = 0


class TabooConfig(BaseModel):
    """Taboo configuration."""
    taboos: List[str]
    redirect_language: str


class PersonaSwitchRequest(BaseModel):
    """Request to switch persona."""
    persona_name: str


class PersonaSwitchResponse(BaseModel):
    """Response from persona switch."""
    persona_name: str
    status: str
    artifacts_loaded: Dict[str, bool]

