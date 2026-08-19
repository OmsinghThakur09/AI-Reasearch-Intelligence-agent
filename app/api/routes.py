# app/api/routes.py
"FastAPI wrapper of fully functional complete research agent pipeline"

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from sqlalchemy import text
import json

from app.db import queries as db
from app.db.session import init_db, engine
from app.rag.ingestor import get_vectorstore
from app.agent.pipeline import run_research_pipeline, stream_research_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    "Runs once at startup.Loads ChromaDB from disk before accepting requestes"
    init_db()  # create tables if they dont exist
    get_vectorstore()  # loads vectorstore from disk into memory
    yield  # app runs here
    engine.dispose()  # closes all connections in the pool on shutdown


app = FastAPI(title="AI Research Agent", lifespan=lifespan, version="1.0.0")


class ResearchRequest(BaseModel):
    query: str = Field(
        description="The research topic or search query",
        examples=["Quantum computing advancements 2026"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional UUID to track the user session",
    )


class ResearchResponse(BaseModel):
    answer: str
    sources: list["str"]
    query_id: str
    session_id: str


def sse_format(event: str, data) -> str:
    "function to declare custom Server-Sent-Event"
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    "non-streaming endpoint: waits for the full answer, then returns it in one JSON response"
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query can't be empty")

    output = run_research_pipeline(request.query, request.session_id)
    return ResearchResponse(**output)


@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    "streaming token endpoint"
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query can't be empty")

    def event_generator():
        "adapter function for converting plain python dicts into format StreamingResponse expects"
        for event in stream_research_pipeline(request.query, request.session_id):
            yield sse_format(event["event"], event["data"])

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    except Exception:
        raise HTTPException(status_code=503, detail="DataBase Unavailable")

    try:
        get_vectorstore()

    except Exception:
        raise HTTPException(status_code=503, detail="Vectorstore Unavailable")
    return {"status": "ok"}


@app.get("/history")
async def history(limit: int = 20):
    rows = db.get_query_history(limit)
    return [{"id": str(r.id), "query": r.query_text, "status": r.status} for r in rows]
