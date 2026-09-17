# app/agent/pipeline.py

from app.db.queries import (
    save_user_query,
    save_documents,
    save_sources,
    update_query_status,
    save_agent_actions,
    update_error_message,
)
from app.agent.search_agent import (
    run_agent,
    get_conversation_context,
    save_answer_to_memory,
)
from app.agent.parser import parse_agent_output
from app.utils.cleaner import clean
from app.rag.ingestor import ingest_clean_text
from app.rag.chain import retrieve_by_subqueries, build_llm_call
import uuid


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

    # step 1b: every session is one checkpointed thread in search_agent's
    # graph — resolve the thread id first, then read whatever context that
    # thread already holds (empty string for a brand new session).
    s_id = session_id or str(uuid.uuid4())
    conversation_context = get_conversation_context(s_id)

    yield {"event": "session", "data": {"session_id": s_id, "query_id": str(query_id)}}

    try:
        # step 2: run search agent
        agent_output, raw_content, sub_queries = run_agent(query, s_id)

        # step 3: parse langgraph's agent output
        sources, raw_docs = parse_agent_output(agent_output)

        # step 4: log every agent tool call to db
        save_agent_actions(agent_output.get("search_results", []), query_id)

        augmented_query = (
            f"{conversation_context}\n\nCurrent question: {query}"
            if conversation_context
            else query
        )

        # step 5: clean raw web page result into clean text
        if len(raw_content) > 0:
            # if escalated node returned full raw web page
            for url, raw in raw_content:
                for item in raw_docs:
                    if item["url"] == url:
                        item["content"] = raw

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

        # step 11: write this turn's answer into the same checkpointed thread search_agent.py owns
        save_answer_to_memory(s_id, answer)

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


# if __name__ == "__main__":
#     query = "Detail the performance benchmarks of Retrieval-Aware Fine-Tuning (RAFT) techniques compared to standard RAG pipelines in recent domain-specific evaluations."

#     result = run_research_pipeline(query)
#     print(result["answer"])
#     print(result["session_id"])
