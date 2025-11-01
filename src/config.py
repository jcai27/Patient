"""Configuration settings for the persona chatbot."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PERSONA_DIR = PROJECT_ROOT / "persona"
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if LLM_PROVIDER == "openai":
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
elif LLM_PROVIDER == "anthropic":
    MODEL_NAME = os.getenv("MODEL_NAME", "claude-3-opus-20240229")
else:
    MODEL_NAME = "gpt-4-turbo-preview"

# Retrieval Configuration
K_RETRIEVE = int(os.getenv("K_RETRIEVE", "5"))
K_RETRIEVE_INITIAL = 20  # Initial retrieval before reranking
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))

# Generation Configuration
MAX_REVISE_LOOPS = int(os.getenv("MAX_REVISE_LOOPS", "2"))
LATENCY_BUDGET_SECONDS = int(os.getenv("LATENCY_BUDGET_SECONDS", "30"))

# Style Configuration
STYLE_LENGTH_TARGETS = {
    "advice": (150, 220),
    "chit-chat": (60, 120),
    "storytelling": (200, 350),
    "opinion": (100, 180),
    "default": (100, 200),
}

HEDGING_LEVELS = {
    0: "No hedging - state facts directly",
    1: "Minimal hedging - use 'likely', 'probably'",
    2: "Moderate hedging - use 'I think', 'seems like', 'might'",
    3: "High hedging - use 'I'm not entirely sure', 'could be'",
    4: "Very high hedging - use 'I'm guessing', 'perhaps'",
    5: "Maximum hedging - use 'I don't really know', 'maybe'",
}

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./persona_memory.db")

# Chunking Configuration
CHUNK_SIZE_WORDS = 150  # 120-180 range, target 150
CHUNK_OVERLAP_WORDS = 25  # 20-30 range, target 25

# Vector Store
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")  # "chroma" or "faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

