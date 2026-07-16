# app/db/model.py

# declaring database schemas using SQLAlchemy, every other component of agent(api, RAG pipeline, etc) will use this to manupulate data from db.
from sqlalchemy import Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, mapped_column
import uuid
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Query(Base):
    __tabelname__ = "queries"
    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = mapped_column(Text)
    status = mapped_column(String(20), default="pending")
    created_at = mapped_column(
        DateTime, server_default=func.now()
    )  # major difference: server_default(sql sever side) vs default(python side)
    # sever_default set default value whenever query is fired but default only set when query fired through python.


class Document(Base):
    __tablename__ = "documents"
    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = mapped_column(UUID(as_uuid=True), ForeignKey("queries.id"))
    url = mapped_column(Text, nullable=False)
    title = mapped_column(Text)
    raw_content = mapped_column(Text)
    clean_text = mapped_column(Text)
    fetched_at = mapped_column(DateTime, server_default=func.now())


class Source(Base):
    __tablename__ = "sources"
    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = mapped_column(UUID(as_uuid=True), ForeignKey("queries.id"))
    url = mapped_column(Text, nullable=False)
    snippet = mapped_column(
        Text
    )  # for storing web-search content preview, prevents saving whole document.


class AgentAction(Base):
    __tablename__ = "agentactions"
    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = mapped_column(UUID(as_uuid=True), ForeignKey("queries.id"))
    step_numbers = mapped_column(Integer)
    action_type = mapped_column(String(50))
    tool_input = mapped_column(Text)
    tool_output = mapped_column(Text)
    created_at = mapped_column(DateTime, server_default=func.now())
