# app/rag/ingestor.py
"""
simple funtion to store emgeddings of cleaned web search results into ChromDB vectore store.
input: cleaned_text + metadat
output: vectore store(stored embeddings of input)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_vectorstore() -> Chroma:
    """Function to load existing ChromaDB vectorestore or load new if first run"""

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def ingest_clean_text(clean_text: list[str], metadata: list[dict]) -> None:
    """
    Function to chunk and store pre-cleaned web results into vectorstore
    """
    if not clean_text:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(clean_text, metadatas=metadata)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
