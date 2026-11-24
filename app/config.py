"""Central configuration, loaded from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Keep Mem0 from sending anonymous usage telemetry.
os.environ.setdefault("MEM0_TELEMETRY", "False")

BASE_DIR = Path(__file__).resolve().parent.parent

# Secrets / connection settings (from .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")

# Local storage
DOCS_DIR = BASE_DIR / "data" / "docs"
QDRANT_PATH = BASE_DIR / "data" / "qdrant"
PARENT_STORE_PATH = BASE_DIR / "data" / "parents"
MEM0_PATH = BASE_DIR / "data" / "mem0"

# Models
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere").strip().lower()
COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
GEMINI_MEMORY_MODEL = os.getenv("GEMINI_MEMORY_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
RERANK_MODEL = "rerank-english-v3.0"

# Parent-document retriever sizes (characters)
PARENT_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 200

# Retrieval / compression
RETRIEVAL_TOP_K = 20  # candidates from hybrid search
RERANK_TOP_N = 5  # documents kept after Cohere ReRank
MIN_RELEVANCE_SCORE = 0.1  # chunks scoring below this are dropped
RRF_K = 60
