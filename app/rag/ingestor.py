# app/rag/ingestor.py
"""
simple funtion to store emgeddings of cleaned web search results into ChromDB vectore store.
input: cleaned_text + metadat
output: vectore store(stored embeddings of input)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# module-level cache so the embedding model is loaded ONCE per process,
# not on every get_vectorstore() call
_embeddings = None
_vectorstore = None


def get_vectorstore() -> Chroma:
    """Function to load existing ChromaDB vectorestore or load new if first run"""
    global _embeddings, _vectorstore

    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"},  # use GTX 1650 instead of CPU
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
    if not clean_text:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents(clean_text, metadatas=metadata)

    # to avoid duplicate vector generation, passing id into add_documents funtion.
    # document having same id will be udated in vector store instead of duplicating.

    id = [
        hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        for chunk in chunks
    ]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks, id=id)
