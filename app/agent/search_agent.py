# app/agent/research_agent.py

"""
this script builds LangGraph's ReAct agent that uses custom Tavily web searching tool to search user query over internet, returns its result.
"""

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

from config import GROQ_API_KEY
from app.agent.search_tool import make_search_tool

MODEL = "qwen/qwen3.6-27b"

memory = MemorySaver()  # to provide memory to an agent, one shared store


def build_agent():
    """simple function to build react agent"""

    llm_model = ChatGroq(
        api_key=GROQ_API_KEY,  # type: ignore
        model=MODEL,
        temperature=0,
    )
    tool, raw_content = make_search_tool()
    return create_agent(model=llm_model, tools=[tool], checkpointer=memory), raw_content


def run_agent(query: str, session_id: str) -> dict:
    agent, raw_content = build_agent()
    config = {"configurable": {"thread_id": session_id}}
    return (
        agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,  # type: ignore
        ),
        raw_content,
    )


if __name__ == "__main__":
    query = "what is Quantum Computing and How Quantum Computing can be used to cure Cancer?"

    result, _ = run_agent(query, "12jk3")
    print(result)
