# app/rag/ingestor.py
"""
simple function to store embeddings of cleaned web search results into ChromDB vectore store.
input: clean_text + metadata
vector store (stored embeddings of input)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# module-level cache so the embedding model is loaded ONCE per process,
# not on every get_vectorstore() call
_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """Function to load existing ChromaDB vectorestore or load new if first run"""
    global _embeddings, _vectorstore

    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )
        _vectorstore = Chroma(
            embedding_function=_embeddings,
            persist_directory=CHROMA_DIR,
        )

    return _vectorstore


def ingest_clean_text(clean_text: list[str], metadata: list[dict]) -> None:
    """
    Function to chunk and store pre-cleaned web results into vectorstore
    """

    if not clean_text or not metadata:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.create_documents(clean_text, metadatas=metadata)

    unique_chunks = []
    chunk_ids = []
    seen_hashes = set()

    for idx, chunk in enumerate(raw_chunks):
        content = chunk.page_content.strip()
        if not content:
            continue

        # Deduplicate identical paragraphs from different scraped URLs
        unique_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if unique_hash not in seen_hashes:
            seen_hashes.add(unique_hash)

        db_id = f"{chunk.metadata.get('query_id', 'unknown')}_{unique_hash}"

        unique_chunks.append(chunk)
        chunk_ids.append(db_id)

    if unique_chunks:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(unique_chunks, ids=chunk_ids)


def cleanup_chroma_memory(query_id: str) -> None:
    "Deletes all vectors associated with a specific query from the disk database."

    if not query_id:
        return

    vectorstore = get_vectorstore()
    try:
        vectorstore.delete(where={"query_id": query_id})
    except Exception as e:
        print(f"Warning: Failed to cleanup memory for {query_id}. Error: {e}")
