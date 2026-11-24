"""RAG agent: retrieve -> rerank -> inject memory profile -> generate, with latency telemetry."""
import time

from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are a helpful assistant.
Use the context below when it is relevant to the question. If it is not relevant, answer from your own knowledge.

Context:
{context}

User profile (facts remembered from earlier sessions):
{profile}

Adapt your answer to the user profile. Follow any stated style preferences exactly
(for example, if the user wants brief, code-only responses, reply with only the code block and no explanation)."""


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


class RAGAgent:
    def __init__(self, retriever, compressor, memory, llm) -> None:
        self.retriever = retriever  # hybrid search -> top 20 parent docs
        self.compressor = compressor  # Cohere ReRank -> top 5 docs
        self.memory = memory
        self.llm = llm

    def chat(self, query: str, user_id: str, session_id: str) -> dict:
        total_start = time.perf_counter()

        t = time.perf_counter()
        candidates = self.retriever.invoke(query)
        retrieval_ms = _ms(t)

        t = time.perf_counter()
        documents = self.compressor.compress_documents(candidates, query)
        rerank_ms = _ms(t)

        facts = self.memory.get_facts(user_id)
        system_prompt = SYSTEM_PROMPT.format(
            context="\n\n".join(doc.page_content for doc in documents) or "(no relevant documents found)",
            profile="\n".join(f"- {fact}" for fact in facts) or "(no stored facts yet)",
        )

        t = time.perf_counter()
        answer = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)]).content
        generation_ms = _ms(t)

        self.memory.save_interaction(user_id, session_id, query, answer)

        return {
            "response": answer,
            "user_id": user_id,
            "session_id": session_id,
            "telemetry": {
                "retrieval_latency_ms": retrieval_ms,
                "rerank_latency_ms": rerank_ms,
                "generation_latency_ms": generation_ms,
                "total_latency_ms": _ms(total_start),
            },
        }
