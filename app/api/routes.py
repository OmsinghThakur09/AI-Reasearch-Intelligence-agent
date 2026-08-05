# app/api/routes.py
"FastAPI wrapper of fully functional complete research agent pipeline"

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sqlalchemy import text
from app.db.session import init_db, engine
from app.rag.ingestor import get_vectorstore
from app.agent.pipeline import run_research_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    "Runs once at startup.Loads ChromaDB from disk before accepting requestes"
    init_db()  # create tables if they dont exist
    get_vectorstore()  # loads vectorstore from disk into memory
    yield  # app runs here
    engine.dispose()  # closes all connections in the pool on shutdown


app = FastAPI(title="AI Research Agent", lifespan=lifespan, version="1.0.0")


class ResearchRequest(BaseModel):
    query: str
    session_id: str | None = None


class ResearchResponse(BaseModel):
    answer: str
    souces: list["str"]
    query_id: str
    session_id: str


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query can't be empty")

    output = run_research_pipeline(request.query, request.session_id)
    return ResearchResponse(**output)


app.get("/health")


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
