"""Retrieval fusion tests: dense and sparse results are mocked."""
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.stores import InMemoryStore

from app.retrieval import HybridParentRetriever, reciprocal_rank_fusion


def child(child_id: str, parent_id: str) -> Document:
    return Document(page_content=child_id, metadata={"child_id": child_id, "parent_id": parent_id})


A, B, C, D = child("A", "P1"), child("B", "P1"), child("C", "P2"), child("D", "P3")


def test_rrf_merges_dense_and_sparse_results():
    dense = [A, B, C]
    sparse = [C, A, D]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    # A: 1/61 + 1/62, C: 1/63 + 1/61, B: 1/62, D: 1/63
    assert [d.metadata["child_id"] for d in fused] == ["A", "C", "B", "D"]


def test_rrf_returns_each_document_once():
    fused = reciprocal_rank_fusion([[A, B], [A, B]])

    assert len(fused) == 2


def test_hybrid_retriever_returns_deduplicated_parents():
    docstore = InMemoryStore()
    docstore.mset([(pid, Document(page_content=f"parent {pid}")) for pid in ("P1", "P2", "P3")])
    retriever = HybridParentRetriever(
        dense=RunnableLambda(lambda q: [A, B, C]),
        sparse=RunnableLambda(lambda q: [C, A, D]),
        docstore=docstore,
    )

    parents = retriever.invoke("any query")

    # A and B share parent P1, so P1 appears once
    assert [p.page_content for p in parents] == ["parent P1", "parent P2", "parent P3"]


def test_hybrid_retriever_respects_top_k():
    docstore = InMemoryStore()
    docstore.mset([(pid, Document(page_content=f"parent {pid}")) for pid in ("P1", "P2", "P3")])
    retriever = HybridParentRetriever(
        dense=RunnableLambda(lambda q: [A, C, D]),
        sparse=RunnableLambda(lambda q: [A, C, D]),
        docstore=docstore,
        top_k=2,
    )

    assert len(retriever.invoke("any query")) == 2
