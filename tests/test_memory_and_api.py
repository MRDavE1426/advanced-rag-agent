"""Offline verification of persistent-memory wiring, persona reuse, and API contracts."""
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app import api, config, ingestion
from app.agent import RAGAgent
from app.memory import SemanticMemory
from mem0 import Memory


class FakeMem0Client:
    """In-process Mem0 stand-in; live provider persistence requires real API keys."""

    def __init__(self):
        self.facts = {}
        self.interactions = []

    def get_all(self, filters, top_k):
        return {"results": [{"memory": fact} for fact in self.facts.get(filters["user_id"], [])][:top_k]}

    def add(self, messages, user_id, metadata):
        self.interactions.append({"messages": messages, "user_id": user_id, "metadata": metadata})
        query = messages[0]["content"].lower()
        facts = self.facts.setdefault(user_id, [])
        if "python" in query and "code-only" in query and "Prefers Python and brief, code-only responses." not in facts:
            facts.append("Prefers Python and brief, code-only responses.")
        return {"results": []}


class FakeLLM:
    def __init__(self):
        self.system_prompts = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.system_prompts.append(prompt)
        if "brief, code-only responses" in prompt:
            answer = "```python\ndef merge_sort(items):\n    return sorted(items)\n```"
        else:
            answer = "Hello! I will remember your Python and concise response preference."
        return SimpleNamespace(content=answer)


class FakeRetriever:
    def invoke(self, query):
        return [Document(page_content="Relevant test context.")]


class FakeCompressor:
    def compress_documents(self, docs, query):
        return docs[:5]


def test_mem0_uses_gemini_and_local_persistent_storage(monkeypatch, tmp_path):
    captured = {}
    client = FakeMem0Client()
    monkeypatch.setattr(config, "MEM0_PATH", tmp_path / "mem0")

    def from_config(settings):
        captured.update(settings)
        return client

    monkeypatch.setattr(Memory, "from_config", from_config)
    memory = SemanticMemory()

    assert captured["vector_store"]["provider"] == "qdrant"
    assert captured["vector_store"]["config"]["path"] == str(tmp_path / "mem0" / "qdrant")
    assert captured["vector_store"]["config"]["embedding_model_dims"] == 768
    assert captured["llm"]["provider"] == "gemini"
    assert captured["llm"]["config"]["model"] == config.GEMINI_MEMORY_MODEL
    assert captured["embedder"]["provider"] == "gemini"
    assert captured["embedder"]["config"]["model"] == config.GEMINI_EMBEDDING_MODEL
    assert captured["embedder"]["config"]["embedding_dims"] == 768
    assert captured["history_db_path"] == str(tmp_path / "mem0" / "history.db")
    assert (tmp_path / "mem0").is_dir()
    memory.save_interaction("usr_99", "session_1", "I use Python", "Noted")
    assert memory.get_facts("usr_99") == []  # test client only extracts the assignment's full preference
    assert client.interactions[0]["metadata"]["session_id"] == "session_1"


def test_cohere_embedding_dimensions_and_collection_are_isolated(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "cohere")
    monkeypatch.setattr(config, "COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    assert ingestion.embedding_dimensions() == 1024
    assert ingestion.collection_name() == "rag_children_cohere"


def test_cohere_embedding_provider_uses_existing_cohere_key(monkeypatch):
    calls = {}
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "cohere")
    monkeypatch.setattr(config, "COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    monkeypatch.setattr(config, "COHERE_API_KEY", "test-key")
    monkeypatch.setattr(ingestion, "CohereEmbeddings", lambda **kwargs: calls.update(kwargs) or "embedder")

    assert ingestion.build_embeddings() == "embedder"
    assert calls == {"model": "embed-english-v3.0", "cohere_api_key": "test-key"}


def test_cohere_embedding_provider_requires_credentials(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "cohere")
    monkeypatch.setattr(config, "COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    monkeypatch.setattr(config, "COHERE_API_KEY", "")

    try:
        ingestion.build_embeddings()
    except RuntimeError as exc:
        assert "COHERE_API_KEY" in str(exc)
    else:
        raise AssertionError("Cohere provider without a key should fail clearly")


def test_local_qdrant_persists_vectors_after_reopening(monkeypatch, tmp_path):
    class DeterministicEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 1023 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 1023

    monkeypatch.setattr(config, "QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "cohere")
    client = ingestion.build_qdrant_client()
    store = ingestion.build_vector_store(client, DeterministicEmbeddings())
    store.add_documents(
        [Document(page_content="persistent vector fixture", metadata={"child_id": "child-1", "parent_id": "parent-1"})],
        ids=[str(uuid.uuid4())],
    )
    client.close()

    reopened_client = ingestion.build_qdrant_client()
    reopened_store = ingestion.build_vector_store(reopened_client, DeterministicEmbeddings())
    try:
        assert ingestion.is_vector_indexed(reopened_store)
        assert reopened_store.similarity_search("persistent vector fixture", k=1)[0].page_content == "persistent vector fixture"
    finally:
        reopened_client.close()


def test_company_multisession_memory_and_chat_api_contract(monkeypatch):
    memory_client = FakeMem0Client()
    memory = SemanticMemory.__new__(SemanticMemory)
    memory.client = memory_client
    llm = FakeLLM()
    agent = RAGAgent(FakeRetriever(), FakeCompressor(), memory, llm)
    monkeypatch.setattr(api, "build_agent", lambda: agent)

    # Exercise the actual FastAPI startup lifecycle with external providers replaced.
    with TestClient(api.app) as client:
        first = client.post(
            "/api/chat",
            json={
                "query": "Hello! I work in Python and prefer brief, code-only responses.",
                "user_id": "usr_99",
                "session_id": "session_1",
            },
        )
        assert first.status_code == 200
        payload = first.json()
        assert payload["user_id"] == "usr_99"
        assert payload["session_id"] == "session_1"
        telemetry = payload["telemetry"]
        assert set(telemetry) == {
            "retrieval_latency_ms",
            "rerank_latency_ms",
            "generation_latency_ms",
            "total_latency_ms",
        }
        assert all(isinstance(value, (int, float)) and value > 0 for value in telemetry.values())

        memories = client.get("/api/users/usr_99/memories")
        assert memories.status_code == 200
        assert "Python" in " ".join(memories.json()["memories"])

        second = client.post(
            "/api/chat",
            json={
                "query": "Write a quick merge sort routine.",
                "user_id": "usr_99",
                "session_id": "session_2",
            },
        )
        assert second.status_code == 200
        assert second.json()["session_id"] == "session_2"
        assert second.json()["response"].startswith("```python")
        assert "brief, code-only responses" in llm.system_prompts[-1]
