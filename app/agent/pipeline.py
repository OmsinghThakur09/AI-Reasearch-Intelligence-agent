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
from app.agent.parser import parse_agent_output, aianswer_parser
from app.agent.search_tool import web_search_executor
from app.utils.cleaner import clean
from app.rag.ingestor import ingest_clean_text
from app.rag.chain import build_rag_chain
import uuid

# simple in-memory store for last question/answer per session_id.
SESSION_LAST_QA: dict[str, dict[str, str]] = {}


def run_research_pipeline(query: str, session_id: str | None = None):
    """
    complete pipeline from user query to final answer.
    """
    # step 1: save user query in db and gets UUID
    query_id = save_user_query(query)

    # step 1b: look up previous turn's Q&A for this session, if any
    previous_qa = SESSION_LAST_QA.get(session_id) if session_id else None

    try:
        # step 2: run search agent
        s_id = session_id or str(uuid.uuid4())
        agent_output, raw_content = run_agent(query, s_id)

        # step 3: parse langgraph's agent output
        sources, metadatas = parse_agent_output(agent_output)

        # step 4: log every agent tool call to db
        save_agent_actions(agent_output.get("messages", ""), query_id)

        if len(metadatas) <= 0:
            "if agent dont use search tool to answer"

            if session_id is not None:
                "agent skipped search tool but question is follow-up"
                answer = aianswer_parser(agent_output)

                update_query_status(query_id, "completed")

                SESSION_LAST_QA[s_id] = {"question": query, "answer": answer}

                return {
                    "answer": answer,
                    "sources": sources,
                    "query_id": str(query_id),
                    "session_id": str(s_id),
                }

            raw_content, metadatas = web_search_executor(query)
            sources = []
            for item in metadatas:
                sources.append(item["url"])

        # step 4b: build augmented query with previous turn's Q&A (if any).
        # used only for retrieval + final answer generation, NOT for web search,
        # so we don't pollute search terms with old context.
        if previous_qa is not None:
            augmented_query = (
                f"Previous question: {previous_qa['question']}\n"
                f"Previous answer: {previous_qa['answer']}\n"
                f"Current question: {query}"
            )
        else:
            augmented_query = query

        # step 5: clean raw web page result into clean text
        raw_clean_dict = clean(raw_content)

        # step 6: save documents in db
        save_documents(query_id, metadatas, raw_clean_dict)

        # step 7: Ingest clean text into ChromaDB
        ingest_clean_text(
            clean_text=[row["clean"] for row in raw_clean_dict],
            metadata=[
                {
                    "url": row["url"],
                    "session_id": str(s_id),
                }
                for row in metadatas
            ],
        )

        # step 8: retreive relevant chunks and send to LLM model for final answer
        rag_chain = build_rag_chain(str(s_id))
        rag_result = rag_chain.invoke({"input": augmented_query})

        answer = rag_result["answer"]

        # step 9: save sources in db
        save_sources(query_id, metadatas)

        # step 10: update status of user query
        update_query_status(query_id, "completed")

        # step 11: save this turn's Q&A as the new "last answer" for this
        # session, overwriting the previous one (we only keep one turn back)
        SESSION_LAST_QA[s_id] = {"question": query, "answer": answer}

        return {
            "answer": answer,
            "sources": sources,
            "query_id": str(query_id),
            "session_id": str(s_id),
        }

    except Exception as e:
        update_query_status(query_id, "failed")
        update_error_message(query_id, str(e))
        raise


if __name__ == "__main__":
    query = "list down the progress in Quantum Computing in 2026"

    result = run_research_pipeline(query)
    print(result["answer"])
    print(result["session_id"])
