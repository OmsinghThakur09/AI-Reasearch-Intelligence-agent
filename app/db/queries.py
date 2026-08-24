# app/db/queries.py

# script for qureies that will be used again and again in this project
from app.db.models import Query, Document, Source, AgentAction
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


def update_error_message(q_id: uuid.UUID, error_msg: str):
    with Sessionlocal() as session:
        session.query(Query).filter(Query.id == q_id).update({"error": error_msg})
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
                url=metadata.get("url", ""),
                snippet=metadata.get("content", "")[:100],
            )
            session.add(source)
        session.commit()


def save_agent_actions(search_results: list, query_id: uuid.UUID):
    with Sessionlocal() as session:
        for i, item in enumerate(search_results):
            query = item["query"]

            urls_per_query = []
            for data in item["results"]:
                urls_per_query.append(data["url"])

            session.add(
                AgentAction(
                    query_id=query_id,
                    step_number=i,
                    action_type="search",
                    tool_input=query,
                    tool_output=", ".join(urls_per_query),
                )
            )
        session.commit()


def get_query_history(limit: int = 20) -> list:
    with Sessionlocal() as session:
        return session.query(Query).order_by(Query.created_at.desc()).limit(limit).all()
