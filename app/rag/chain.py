# app/rag/chain.py
"""
A function that returns complete Retrieval chain, contains retrival and augmentation bundled in single function.
input: None
output: Retrieval chain
"""

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from app.rag.ingestor import get_vectorstore
from config import GROQ_API_KEY

system_prompt = """You are a research assistant, use only provided context to answer, always site source URLs at the end of your answer.
    If the context doesn't contain the enough information, say so explicitly.
    Context: {context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

document_prompt = PromptTemplate.from_template("{page_content}\nSource: {url}")


def build_rag_chain():
    """function that loads persisted vector store from the disk and return Retrieval chain built over it."""
    vectorstore = get_vectorstore()  # loads from disk
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm_model = ChatGroq(
        api_key=GROQ_API_KEY,  # type: ignore
        model="llama-3.3-70b-versatile",
        temperature=0,  # pyright: ignore[reportArgumentType]
    )
    # this chain handles injecting retrieved documents into context variable
    qa_chain = create_stuff_documents_chain(
        llm=llm_model, prompt=PROMPT, document_prompt=document_prompt
    )

    # this chain links your vector database with qa chain.
    return create_retrieval_chain(retriever, qa_chain)
