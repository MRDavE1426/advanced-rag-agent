"""FastAPI server exposing the chat and memory endpoints."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from app import config
from app.agent import RAGAgent
from app.ingestion import (
    build_docstore,
    build_embeddings,
    build_qdrant_client,
    build_vector_store,
    ingest_documents,
    is_indexed,
    is_vector_indexed,
    load_all_children,
)
from app.memory import SemanticMemory
from app.rerank import build_compressor
from app.retrieval import build_hybrid_retriever


class ChatRequest(BaseModel):
    query: str
    user_id: str
    session_id: str


def build_agent() -> RAGAgent:
    if not config.GEMINI_API_KEY or not config.COHERE_API_KEY:
        raise RuntimeError("GEMINI_API_KEY and COHERE_API_KEY must be set in .env (see .env.example)")

    docstore = build_docstore()
    vector_store = build_vector_store(build_qdrant_client(), build_embeddings())
    if not is_indexed(docstore) or not is_vector_indexed(vector_store):  # handle provider-specific empty indexes
        ingest_documents(config.DOCS_DIR, vector_store, docstore)

    retriever = build_hybrid_retriever(vector_store, docstore, load_all_children(docstore))
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_CHAT_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY,
    )
    return RAGAgent(retriever, build_compressor(), SemanticMemory(), llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent()
    yield


app = FastAPI(title="Advanced RAG Agent", lifespan=lifespan)


@app.post("/api/chat")
def chat(request: ChatRequest):
    return app.state.agent.chat(request.query, request.user_id, request.session_id)


@app.get("/api/users/{user_id}/memories")
def get_memories(user_id: str):
    return {"user_id": user_id, "memories": app.state.agent.memory.get_facts(user_id)}
