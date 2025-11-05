"""Data models for persona artifacts."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class SpeakingStyle(BaseModel):
    """Speaking style configuration."""
    avg_sentence_len: List[int] = Field(default=[12, 18])  # min, max
    hedging_level: int = Field(default=2, ge=0, le=5)
    formality: int = Field(default=3, ge=0, le=5)
    emoji_policy: str = Field(default="none")  # "none", "light", "rich"
    signature_phrases: List[str] = Field(default_factory=list)


class PersonaProfile(BaseModel):
    """Persona profile schema."""
    name: str
    backstory: str
    values: List[str] = Field(default_factory=list)
    topics_of_expertise: List[str] = Field(default_factory=list)
    speaking_style: SpeakingStyle = Field(default_factory=SpeakingStyle)
    taboos_refs: List[str] = Field(default_factory=list)


class CanonicalFact(BaseModel):
    """Canonical fact for RAG."""
    id: str
    text: str
    source: str  # e.g., "interview.min5-7"
    date: Optional[str] = None
    stance: Optional[str] = None  # e.g., "likes", "dislikes", "neutral"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)


class Example(BaseModel):
    """Few-shot example."""
    user: str
    assistant: str
    intent: Optional[str] = None  # e.g., "advice", "storytelling"


class StylePolicyPack(BaseModel):
    """Style+Policy Pack from Contextor."""
    tone: str
    hedging_level: int
    formality: int
    emoji_policy: str
    target_len_tokens: int
    cadence_notes: Optional[str] = None
    follow_up_question_required: bool = True
    signature_moves: List[str] = Field(default_factory=list)
    taboos: List[str] = Field(default_factory=list)
    few_shots: List[Example] = Field(default_factory=list)
    negative_example: Optional[Example] = None


class JudgeScores(BaseModel):
    """Judge scoring output."""
    factuality: float = Field(ge=1.0, le=5.0)
    persona: float = Field(ge=1.0, le=5.0)
    helpfulness: float = Field(ge=1.0, le=5.0)
    safety: float = Field(ge=1.0, le=5.0)
    overall: float = Field(ge=1.0, le=5.0)


class JudgeDecision(BaseModel):
    """Judge decision with optional edits."""
    accept: bool
    scores: JudgeScores
    targeted_edits: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None

