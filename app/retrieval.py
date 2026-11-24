"""Hybrid (dense + sparse) retrieval merged with RRF, returning parent documents."""
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app import config


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = config.RRF_K) -> list[Document]:
    """Merge ranked lists. Each doc scores sum(1 / (k + rank)) over the lists it appears in."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc.metadata["child_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


class HybridParentRetriever(BaseRetriever):
    """Dense + sparse search over child chunks -> RRF -> de-duplicated parent documents."""

    dense: Any  # runnable returning child docs (e.g. Qdrant retriever)
    sparse: Any  # runnable returning child docs (e.g. BM25 retriever)
    docstore: Any  # key-value store: parent_id -> parent Document
    top_k: int = config.RETRIEVAL_TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        dense_docs = self.dense.invoke(query)
        sparse_docs = self.sparse.invoke(query)
        fused_children = reciprocal_rank_fusion([dense_docs, sparse_docs])

        parent_ids: list[str] = []
        for child in fused_children:
            parent_id = child.metadata["parent_id"]
            if parent_id not in parent_ids:
                parent_ids.append(parent_id)

        parents = self.docstore.mget(parent_ids)
        return [p for p in parents if p is not None][: self.top_k]


def build_hybrid_retriever(vector_store, docstore, children: list[Document]) -> HybridParentRetriever:
    dense = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_TOP_K})
    sparse = BM25Retriever.from_documents(children, k=config.RETRIEVAL_TOP_K)
    return HybridParentRetriever(dense=dense, sparse=sparse, docstore=docstore)
