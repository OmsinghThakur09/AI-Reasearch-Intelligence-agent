# scripts/demo_rag.py
from app.rag.ingestor import ingest_clean_text
from app.rag.chain import build_rag_chain

# Sample clean texts (in production these come from pipeline.py via PostgreSQL)
sample_texts = [
    "Solid-state batteries use ceramic electrolytes instead of liquid ones...",
    r"Energy density of lithium-ion cells improved by 40% between 2020 and 2024...",
]

sample_meta = [{"url": "https://example.com/1"}, {"url": "https://example.com/2"}]

ingest_clean_text(sample_texts, sample_meta)

chain = build_rag_chain()

result = chain.invoke({"input": "What are the latest EV battery improvements?"})
print(result["answer"])
