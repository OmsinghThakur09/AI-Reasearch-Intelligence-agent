# app/agent/research_agent.py

"""
this script builds LangGraph's ReAct agent that uses Tavily web searching tool to search user query over internet, returns its result.
"""

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from config import GROQ_API_KEY

MODEL = "llama-3.3-70b-versatile"

memory = MemorySaver()  # to provide memory to an agent, one shared store


def build_agent():
    """simple function to build react agent"""

    llm_model = ChatGroq(
        api_key=GROQ_API_KEY,  # type: ignore
        model=MODEL,
        temperature=0,
    )
    tool = TavilySearch(max_result=2, include_raw_content="text")
    return create_agent(model=llm_model, tools=[tool], checkpointer=memory)


def run_agent(query: str, session_id: str) -> dict:
    agent = build_agent()
    config = {"configurable": {"thread_id": session_id}}
    return agent.invoke(
        {"messages": [{"role": "user", "content": query}]}, config=config
    )  # type: ignore


if __name__ == "__main__":
    query = "what was the reason behind chernobyl disaster?"
    result = run_agent(query, "12jk3")
    print(result)
