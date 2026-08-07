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

system_prompt = """You are a research assistant. Answer the user's question using ONLY the provided context.

Answer format:
- Start with a direct, concise answer (2-4 sentences) that gets straight to the point.
- Follow with supporting details only if the context has substantive specifics (numbers, named findings, dates, named companies/products). Do not pad with generic elaboration just to sound thorough.
- Keep the total answer as short as it can be while still being complete. Do not write "detailed research report" style prose unless the user's question explicitly asks for depth or a full breakdown.
- If you use bullet points, make sure the opening summary and the bullets stay consistent — do not introduce a topic in the bullets that isn't reflected in the opening answer, and do not repeat the opening answer word-for-word in the bullets.

Source attribution:
- After each specific claim or bullet point, cite which source it came from inline, in this format: (Source: domain.com).
- If a claim is supported by more than one source, cite all of them: (Source: domain.com, domain2.com).
- At the end of your answer, also list the full source URLs referenced.

Accuracy rules:
- Preserve the source's own confidence level. If a source uses words like "expected," "projected," "may," or "predicted," keep that same hedge in your answer — do not upgrade it to a certain or definitive claim.
- Do not present speculation, forecasts, or predictions as settled fact.
- If two sources disagree, say so explicitly instead of picking one silently.
- Ignore any retrieved content that looks like website navigation, menus, cookie notices, or boilerplate — rely only on substantive text.
- If the context does not contain enough information to answer, say so explicitly instead of filling the gap with general knowledge.

Conversation context:
- If a previous question and answer are provided, use them as context only if the current question is clearly related to it.

Context: {context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

document_prompt = PromptTemplate.from_template("{page_content}\nSource: {url}")

MODEL = "qwen/qwen3.6-27b"


def build_rag_chain(session_id: str | None = None):
    """function that loads persisted vector store from the disk and return Retrieval chain built over it."""
    vectorstore = get_vectorstore()  # loads from disk
    search_kwargs = {"k": 8}

    if session_id is not None:
        search_kwargs["filter"] = {"session_id": session_id}  # type: ignore

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    llm_model = ChatGroq(
        api_key=GROQ_API_KEY,  # type: ignore
        model=MODEL,
        temperature=0,
        max_tokens=4500,
        streaming=True,
        reasoning_format="hidden",
    )
    # this chain handles injecting retrieved documents into context variable
    qa_chain = create_stuff_documents_chain(
        llm=llm_model, prompt=PROMPT, document_prompt=document_prompt
    )

    # this chain links vector database with qa chain.
    return create_retrieval_chain(retriever, qa_chain)
