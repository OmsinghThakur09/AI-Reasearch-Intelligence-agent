# app/db/queries.py

# script for qureies that will be used again and again in this project
from app.db.models import Query, Document, Source, AgentAction
from app.db.session import Sessionlocal
import json
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


def save_agent_actions(messages: list, query_id: uuid.UUID):
    with Sessionlocal() as session:
        tool_was_used = False
        for i, msg in enumerate(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_was_used = True
                for call in msg.tool_calls:
                    call_id = call.get("id")
                    urls = []

                    # find the ToolMessage whose tool_call_id matches this call
                    for m in messages:
                        if hasattr(m, "tool_call_id") and m.tool_call_id == call_id:
                            try:
                                parsed = json.loads(m.content)
                                urls = [
                                    item["url"]
                                    for item in parsed.get("metadata", [])
                                    if "url" in item
                                ]
                            except (json.JSONDecodeError, TypeError):
                                urls = []
                            break

                    session.add(
                        AgentAction(
                            query_id=query_id,
                            step_number=i,
                            action_type="search",
                            tool_input=str(call.get("args", "")),
                            tool_output=", ".join(urls),
                        )
                    )

        if not tool_was_used:
            session.add(
                AgentAction(
                    query_id=query_id,
                    step_number=0,
                    action_type="no_tool_used",
                    tool_input=None,
                    tool_output=None,
                )
            )

        session.commit()
