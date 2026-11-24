# Advanced RAG Memory Agent

- **Repository**: [Insert URL]

A FastAPI RAG agent with hybrid dense + sparse search (RRF), parent-document retrieval,
Cohere ReRank context compression (top 20 -> top 5), Mem0 long-term semantic memory,
and per-request latency telemetry.

## How it works

1. **Ingestion** (automatic on first start): every `.md`/`.txt` file in `data/docs/` is split into
   1000-character parents (stored in a local document store, `data/parents/`) and 200-character
   children (embedded with OpenAI and stored in a persistent local Qdrant index, `data/qdrant/`).
2. **Hybrid search**: dense (Qdrant) and sparse (BM25) results over the child chunks are merged with
   Reciprocal Rank Fusion; matching children are mapped back to their de-duplicated parents (top 20).
3. **Cohere ReRank**: the 20 candidates are re-ranked, the top 5 are kept, and chunks below the
   minimum relevance score (`MIN_RELEVANCE_SCORE` in `app/config.py`, default 0.1) are dropped.
4. **Semantic memory**: after each chat, Mem0 extracts user facts. On every request the user's stored
   facts are injected into the system prompt, so new sessions adapt automatically.

## Project structure

```
app/
  config.py      settings and tunables (chunk sizes, top-k, thresholds)
  ingestion.py   parent/child splitting, Qdrant + parent docstore
  retrieval.py   RRF + hybrid parent retriever
  rerank.py      Cohere ReRank compressor with score threshold
  memory.py      Mem0 semantic memory
  agent.py       generation loop + latency telemetry
  api.py         FastAPI app
tests/           pytest suites (RRF fusion, ReRank score filter)
data/docs/       documents to ingest
main.py          uvicorn entry point
```

## Installation

1. Configure the `.env` file: `cp .env.example .env`, then set `OPENAI_API_KEY` and `COHERE_API_KEY`.
   `QDRANT_URL` is optional; leave it empty to use the persistent local index.
2. Install dependencies: `pip install -r requirements.txt` (Python 3.10+)
3. Run FastAPI: `uvicorn main:app --reload`
4. Run tests: `pytest` (no API keys needed, external calls are mocked)

To index your own documents, put `.md`/`.txt` files in `data/docs/` and delete `data/parents/` and
`data/qdrant/` before starting the server.

## API

### `POST /api/chat`

```json
{ "query": "Write a quick merge sort routine.", "user_id": "usr_99", "session_id": "session_2" }
```

Response:

```json
{
  "response": "...",
  "user_id": "usr_99",
  "session_id": "session_2",
  "telemetry": {
    "retrieval_latency_ms": 412.3,
    "rerank_latency_ms": 285.1,
    "generation_latency_ms": 930.7,
    "total_latency_ms": 3120.4
  }
}
```

`total_latency_ms` covers the whole request, including memory lookup and saving.

### `GET /api/users/{user_id}/memories`

Returns `{"user_id": "...", "memories": ["Prefers Python", ...]}`.

An API collection is provided in `RAG_Agent.postman_collection.json` (import into Postman or Bruno).

## Verification workflows

1. **Latency telemetry**: start the server, send a request to `/api/chat`, and check that the
   `telemetry` block has positive `retrieval_latency_ms`, `rerank_latency_ms` and `total_latency_ms`.
2. **Memory persona**: with `user_id=usr_99`, send `Hello! I work in Python and prefer brief, code-only responses.`,
   call `GET /api/users/usr_99/memories` to see the extracted Python fact, then send
   `Write a quick merge sort routine.` with a new `session_id`. The reply should be a code block only.
3. **Test suite**: run `pytest`.
