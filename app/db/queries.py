# app/db/queries.py

# script for qureies that will be used again and again in this project
from app.db.models import Query
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
