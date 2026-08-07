# app/agent/research_agent.py

"""
this script builds LangGraph's ReAct agent that uses custom Tavily web searching tool to search user query over internet, returns its result.
"""

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from config import GROQ_API_KEY
from app.agent.search_tool import make_search_tool

MODEL = "qwen/qwen3.6-27b"

# persistent connection to a local sqlite file - survives across process restarts
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)


def build_agent():
    """simple function to build react agent"""

    llm_model = ChatGroq(
        api_key=GROQ_API_KEY,  # type: ignore
        model=MODEL,
        temperature=0,
    )
    tool, raw_content = make_search_tool()

    return (
        create_agent(model=llm_model, tools=[tool], checkpointer=memory),
        raw_content,
    )


GLOBAL_AGENT, RAW_CONTENT = build_agent()


def run_agent(query: str, session_id: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    return (
        GLOBAL_AGENT.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,  # type: ignore
        ),
        RAW_CONTENT,
    )


if __name__ == "__main__":
    query = "which automation tools did you mentioned?"

    result, _ = run_agent(query, "10f03787-53c5-48a4-8e78-f767de46c52c")
    print(result)
