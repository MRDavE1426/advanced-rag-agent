"""Cohere ReRank context compression: top 20 candidates -> top 5 relevant documents."""
from langchain_cohere import CohereRerank

from app import config


class ScoreFilteredCohereRerank(CohereRerank):
    """CohereRerank that also drops chunks below a minimum relevance score."""

    min_score: float = 0.0

    def compress_documents(self, documents, query, callbacks=None):
        reranked = super().compress_documents(documents, query, callbacks)
        return [doc for doc in reranked if doc.metadata["relevance_score"] >= self.min_score]


def build_compressor() -> ScoreFilteredCohereRerank:
    return ScoreFilteredCohereRerank(
        model=config.RERANK_MODEL,
        top_n=config.RERANK_TOP_N,
        cohere_api_key=config.COHERE_API_KEY,
        min_score=config.MIN_RELEVANCE_SCORE,
    )
