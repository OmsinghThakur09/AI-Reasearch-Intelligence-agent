# app/agent/pipeline.py

from app.db.queries import (
    save_user_query,
    save_documents,
    save_sources,
    update_query_status,
    save_agent_actions,
    update_error_message,
)
from app.agent.search_agent import run_agent
from app.agent.parser import parse_agent_output
from app.utils.cleaner import clean
from app.rag.ingestor import ingest_clean_text
from app.rag.chain import retrieve_by_subqueries, build_llm_call
import uuid

# simple in-memory store for last question/answer per session_id.
SESSION_LAST_QA: dict[str, dict[str, str]] = {}


# generator function
def stream_research_pipeline(query: str, session_id: str | None = None):
    """
    generator function same as before but yields different events instead of complete final answer.
    to stream token one by one from llm to user's web browser we need generator function that able to send data
    as soon as generated and sent by llm.

    output format:
    Yields dicts of the shape {"event": ..., "data": ...}:
        "session" -> {"session_id": ..., "query_id": ...}   (sent first)
        "token"   -> "<piece of the answer>"                (sent repeatedly)
        "done"    -> {"answer", "sources", "query_id", "session_id"}
        "error"   -> "<error message>"
    """
    # step 1: save user query in db and gets UUID
    query_id = save_user_query(query)

    # step 1b: look up previous turn's Q&A for this session, if any
    previous_qa = SESSION_LAST_QA.get(session_id) if session_id else None
    s_id = session_id or str(uuid.uuid4())

    yield {"event": "session", "data": {"session_id": s_id, "query_id": str(query_id)}}

    try:
        # step 2: run search agent
        agent_output, raw_content, sub_queries = run_agent(query, s_id)

        # step 3: parse langgraph's agent output
        sources, raw_docs = parse_agent_output(agent_output)

        # step 4: log every agent tool call to db
        save_agent_actions(agent_output.get("messages", ""), query_id)

        if previous_qa is not None:
            augmented_query = (
                f"Previous question: {previous_qa['question']}\n"
                f"Previous answer: {previous_qa['answer']}\n\n"
                f"Current question: {query}"
            )
        else:
            augmented_query = query

        # step 5: clean raw web page result into clean text
        if len(raw_content) > 0:
            # if escalated node returned full raw web page
            for url, raw in raw_content:
                for item in raw_docs:
                    if url in item.keys():
                        item["url"] = raw

        raw_clean_dict = clean([row["content"] for row in raw_docs])

        # step 6: save documents in db
        save_documents(query_id, raw_docs, raw_clean_dict)

        # step 7: Ingest clean text into ChromaDB
        ingest_clean_text(
            clean_text=[row["clean"] for row in raw_clean_dict],
            metadata=[
                {
                    "url": row["url"],
                    "query_id": str(query_id),
                    "session_id": str(s_id),
                }
                for row in raw_docs
            ],
        )

        # step 8: retreive relevant chunks and send to LLM model for final answer
        context = retrieve_by_subqueries(sub_queries, str(query_id), session_id=s_id)
        rag_chain = build_llm_call()

        answer_parts = []
        for chunk in rag_chain.stream({"input": augmented_query, "context": context}):
            # chunks can be return in the form of langchain message object thats why we need getattr to extract chunk from content block of message object
            token = (
                chunk
                if isinstance(chunk, str)
                else getattr(chunk, "content", str(chunk))
            )
            if not token:
                continue

            answer_parts.append(token)
            yield {"event": "token", "data": token}

        answer = "".join(answer_parts)
        # step 9: save sources in db
        save_sources(query_id, raw_docs)

        # step 10: update status of user query
        update_query_status(query_id, "completed")

        # step 11: save this turn's Q&A as the new "last answer" for this
        # session, overwriting the previous one (we only keep one turn back)
        SESSION_LAST_QA[s_id] = {"question": query, "answer": answer}

        yield {
            "event": "done",
            "data": {
                "answer": answer,
                "sources": sources,
                "query_id": str(query_id),
                "session_id": str(s_id),
            },
        }

    except Exception as e:
        update_query_status(query_id, "failed")
        update_error_message(query_id, str(e))
        yield {"event": "error", "data": str(e)}
        raise


# normal function
def run_research_pipeline(query: str, session_id: str | None = None):
    """
    complete pipeline from user query to final answer.
    """
    final = None
    for event in stream_research_pipeline(query, session_id):
        if event["event"] == "done":
            final = event["data"]

    return final


if __name__ == "__main__":
    query = "Quantum computing advancements 2026"

    result = run_research_pipeline(query)
    print(result["answer"])
    print(result["session_id"])
