"""ReRank score filter tests: the Cohere API call is mocked."""
from unittest.mock import patch

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from app.rerank import ScoreFilteredCohereRerank


def make_compressor(min_score: float) -> ScoreFilteredCohereRerank:
    return ScoreFilteredCohereRerank(
        model="rerank-english-v3.0", top_n=5, cohere_api_key="test-key", min_score=min_score
    )


DOCS = [Document(page_content=f"chunk {i}") for i in range(3)]
MOCK_COHERE_RESULTS = [
    {"index": 2, "relevance_score": 0.92},
    {"index": 0, "relevance_score": 0.40},
    {"index": 1, "relevance_score": 0.01},
]


def test_chunks_below_min_relevance_score_are_dropped():
    with patch.object(CohereRerank, "rerank", return_value=MOCK_COHERE_RESULTS):
        result = make_compressor(min_score=0.1).compress_documents(DOCS, "query")

    assert [d.page_content for d in result] == ["chunk 2", "chunk 0"]
    assert all(d.metadata["relevance_score"] >= 0.1 for d in result)


def test_all_chunks_dropped_when_nothing_is_relevant():
    with patch.object(CohereRerank, "rerank", return_value=MOCK_COHERE_RESULTS):
        result = make_compressor(min_score=0.95).compress_documents(DOCS, "query")

    assert result == []


def test_reranked_order_is_preserved():
    with patch.object(CohereRerank, "rerank", return_value=MOCK_COHERE_RESULTS):
        result = make_compressor(min_score=0.0).compress_documents(DOCS, "query")

    assert [d.metadata["relevance_score"] for d in result] == [0.92, 0.40, 0.01]
