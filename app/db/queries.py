# app/db/queries.py

# script for qureies that will be used again and again in this project
from app.db.models import Query, Document, Source
from app.db.session import Sessionlocal
import uuid


def save_user_query(question: str) -> uuid.UUID:
    with Sessionlocal() as session:
        query = Query(query_text=question)
        session.add(query)
        session.commit()
        session.refresh(query)
        return query.id


def update_query_status(q_id: uuid.UUID, status: str):
    with Sessionlocal() as session:
        session.query(Query).filter(Query.id == q_id).update({"status": status})
        session.commit()


def save_documents(q_id: uuid.UUID, metadatas: list, raw_clean_dict: list):
    with Sessionlocal() as session:
        for metadata, clean_row in zip(metadatas, raw_clean_dict):
            doc = Document(
                query_id=q_id,
                url=metadata["url"],
                title=metadata["title"],
                raw_content=clean_row["raw"],
                clean_text=clean_row["clean"],
            )
            session.add(doc)
        session.commit()


def save_sources(q_id: uuid.UUID, metadatas: list):
    with Sessionlocal() as session:
        for metadata in metadatas:
            source = Source(
                query_id=q_id,
                url=metadata["url"],
                snippet=metadata["content"],
            )
            session.add(source)
        session.commit()
