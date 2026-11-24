"""Long-term semantic memory backed by Mem0 (facts persisted per user_id)."""
import os

from app import config

# Mem0 creates a global config directory as soon as it is imported. Keep that
# state with the application's ignored, persistent data instead of assuming the
# current user's home directory is writable (important on managed Windows hosts).
os.environ.setdefault("MEM0_DIR", str(config.MEM0_PATH / "client"))

from mem0 import Memory

class SemanticMemory:
    def __init__(self) -> None:
        config.MEM0_PATH.mkdir(parents=True, exist_ok=True)
        self.client = Memory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "user_memories",
                        "path": str(config.MEM0_PATH / "qdrant"),
                        "embedding_model_dims": 768,
                    },
                },
                "llm": {
                    "provider": "gemini",
                    "config": {
                        "model": config.GEMINI_MEMORY_MODEL,
                        "temperature": 0,
                        "api_key": config.GEMINI_API_KEY,
                    },
                },
                "embedder": {
                    "provider": "gemini",
                    "config": {
                        "model": config.GEMINI_EMBEDDING_MODEL,
                        "embedding_dims": 768,
                        "api_key": config.GEMINI_API_KEY,
                    },
                },
                "history_db_path": str(config.MEM0_PATH / "history.db"),
            }
        )

    def get_facts(self, user_id: str) -> list[str]:
        """All semantic facts currently stored for the user."""
        result = self.client.get_all(filters={"user_id": user_id}, top_k=100)
        return [item["memory"] for item in result["results"]]

    def save_interaction(self, user_id: str, session_id: str, query: str, answer: str) -> None:
        """Let Mem0 extract user facts (languages, style preferences...) from the exchange."""
        messages = [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
        self.client.add(messages, user_id=user_id, metadata={"session_id": session_id})
