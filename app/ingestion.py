"""Document ingestion: parent/child splitting, Qdrant indexing and parent storage."""
import uuid
from pathlib import Path

from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app import config

_EMBEDDING_DIMENSIONS = {
    "cohere": 1024,  # embed-english-v3.0
}


def embedding_dimensions() -> int:
    """Return the vector dimension for the configured provider/model pair."""
    provider = config.EMBEDDING_PROVIDER
    if provider not in _EMBEDDING_DIMENSIONS:
        raise ValueError("EMBEDDING_PROVIDER must be 'cohere'")
    if provider == "cohere" and config.COHERE_EMBEDDING_MODEL != "embed-english-v3.0":
        raise ValueError("Only Cohere's embed-english-v3.0 model is supported by this index configuration")
    return _EMBEDDING_DIMENSIONS[provider]


def collection_name() -> str:
    """Keep different embedding spaces in separate Qdrant collections."""
    embedding_dimensions()  # validate the configured model before touching Qdrant
    return "rag_children_cohere" if config.EMBEDDING_PROVIDER == "cohere" else "rag_children"

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.PARENT_CHUNK_SIZE, chunk_overlap=0)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHILD_CHUNK_SIZE, chunk_overlap=0)


def build_docstore():
    """Local document repository that holds the full parent chunks."""
    config.PARENT_STORE_PATH.mkdir(parents=True, exist_ok=True)
    return create_kv_docstore(LocalFileStore(str(config.PARENT_STORE_PATH)))


def build_qdrant_client() -> QdrantClient:
    """Qdrant server if QDRANT_URL is set, otherwise a persistent local index."""
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))


def build_vector_store(client: QdrantClient, embeddings) -> QdrantVectorStore:
    name = collection_name()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=embedding_dimensions(), distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)


def build_embeddings():
    """Construct Cohere document/query embeddings for the local Qdrant index."""
    embedding_dimensions()
    if config.EMBEDDING_PROVIDER == "cohere":
        if not config.COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY is required when EMBEDDING_PROVIDER=cohere")
        return CohereEmbeddings(
            model=config.COHERE_EMBEDDING_MODEL,
            cohere_api_key=config.COHERE_API_KEY,
        )
    raise ValueError("EMBEDDING_PROVIDER must be 'cohere'")


def split_parents(text: str, source: str) -> list[Document]:
    """Split a file into parent chunks with deterministic ids (re-ingestion is idempotent)."""
    parents = parent_splitter.create_documents([text], metadatas=[{"source": source}])
    for i, parent in enumerate(parents):
        parent.metadata["parent_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}"))
    return parents


def split_children(parent: Document) -> list[Document]:
    """Split a parent into small child chunks that point back to it via parent_id."""
    children = child_splitter.create_documents([parent.page_content], metadatas=[dict(parent.metadata)])
    for i, child in enumerate(children):
        child.metadata["child_id"] = f"{parent.metadata['parent_id']}:{i}"
    return children


def load_all_children(docstore) -> list[Document]:
    """Rebuild the child chunks (used by the BM25 sparse index) from the stored parents."""
    parent_ids = list(docstore.yield_keys())
    parents = [p for p in docstore.mget(parent_ids) if p is not None]
    return [child for parent in parents for child in split_children(parent)]


def is_indexed(docstore) -> bool:
    return next(iter(docstore.yield_keys()), None) is not None


def is_vector_indexed(vector_store: QdrantVectorStore) -> bool:
    """Check that the configured embedding collection has vectors to query."""
    return vector_store.client.count(collection_name=vector_store.collection_name, exact=True).count > 0


def ingest_documents(docs_dir: Path, vector_store: QdrantVectorStore, docstore) -> int:
    """Index every .md/.txt file: children -> Qdrant, parents -> local docstore."""
    all_parents: list[Document] = []
    all_children: list[Document] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix not in {".md", ".txt"}:
            continue
        for parent in split_parents(path.read_text(encoding="utf-8"), path.name):
            all_parents.append(parent)
            all_children.extend(split_children(parent))

    if not all_children:
        return 0
    docstore.mset([(p.metadata["parent_id"], p) for p in all_parents])
    child_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, c.metadata["child_id"])) for c in all_children]
    vector_store.add_documents(all_children, ids=child_ids)
    return len(all_parents)
