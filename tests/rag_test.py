"""
pytest to test functionality of RAG chain
"""

from app.rag.ingestor import ingest_clean_text
from app.rag.chain import build_rag_chain

test_strings = [
    "Batman is a DC character features raw martial arts, high tech equipments and billionare lifestyle who lives in bat cave",
    "langchain dont require to specify embedding function for embedding user query because, it already passes through vectorstore",
    """The biggest mistake ML freshers make is loading their resume with theory — CNNs, transformers, loss functions —
    but having zero proof they can build anything that runs in production.
    Companies don't need another person who can explain backpropagation. They need someone who can ship.""",
]
metadata = [
    {
        "url": "batman@example.com",
    },
    {"url": "langchain@example.com"},
    {"url": "resumetips@example.com"},
]

ingest_clean_text(test_strings, metadata=metadata)

chain = build_rag_chain()


def test1():
    result = chain.invoke({"input": "who is batman?"})
    assert "DC" in result["answer"]


def test2():
    result = chain.invoke(
        {
            "input": "why langchain dont require embedding function for embedding user query?"
        }
    )
    print(result["answer"])
    assert "vector store" in result["answer"]


def test3():
    result = chain.invoke({"input": "what is the biggest mistake?"})
    assert "ML fresher" in result["answer"]
