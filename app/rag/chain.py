# app/rag/chain.py
"""
A function that returns complete Retrieval chain, contains retrival and augmentation bundled in single function.
input: None
output: RAG chain
"""

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from app.rag.ingestor import get_vectorstore

PROMPT = PromptTemplate(
    input_variables=["context", "input"],
    template="""
    You are a research assistant, use only provided context to answer, always site source URLs at the end of your answer.
    If the context dosen't contain the enough information, say so explicitly.

    context: {context}
    question: {input}
    Answer(with sources):
    """,
)


def get_rag_chain():
    """function that loads persisted vector store from the disk and return Retrieval chain built over it."""
    vectorstore = get_vectorstore()  # loads from disk
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # this chain handles injecting retrieved documents into context variable
    qa_chain = create_stuff_documents_chain(llm=llm_model, prompt=PROMPT)

    # this chain links your vector database with qa chain.
    return create_retrieval_chain(retriever, qa_chain)
