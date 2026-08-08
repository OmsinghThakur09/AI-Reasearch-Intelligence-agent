# app/agent/search_agent_V2.py

"""
this script builds custom ReAct-style langgraph agent that uses custom Tavily web searching tool to search a user query over web and returns its result.

this version manually starts from web searching user query and then perform its reasoning to descide wether more searching needs or not. it ensures that
agent never skips search tool even for simpler user query.
"""

from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3

from config import GROQ_API_KEY
from app.agent.search_tool import make_search_tool

MODEL = "llama-3.3-70b-versatile"

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)


def build_agent():
    "build a graph that searches web before llm answers"
    tool, raw_content = make_search_tool()
    tools = [tool]

    base_llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL,
        temperature=0,
    )

    # forced variant: only for starting step, must search web because there is only one tool choice
    forced_llm = base_llm.bind_tools(tools, tool_choice="web_search_tool")

    # normal variant: used for every call after the first search task.
    agent_llm = base_llm.bind_tools(tools)

    tool_node = ToolNode(tools)

    def first_search_node(state: MessagesState):
        "A node that always call web_search_tool"
        response = forced_llm.invoke(state["messages"])
        return {"messages": [response]}

    def agent_node(state: MessagesState):
        "A node that acts as a ReAct agent"
        response = agent_llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("search_node", first_search_node)
    graph.add_node("tools", tool_node)
    graph.add_node("agent_node", agent_node)

    graph.add_edge(START, "search_node")  # every run starts here
    graph.add_edge("search_node", "tools")  # tool call is always executed
    graph.add_edge("tools", "agent_node")  # result goes to agent for reasoning

    # after that, normal ReAct behaviour: agent decides to search again or finish
    graph.add_conditional_edges("agent_node", tools_condition)

    compiled = graph.compile(checkpointer=memory)

    return compiled, raw_content


GLOBAL_AGENT, RAW_CONTENT = build_agent()


def run_agent(query: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}

    result = GLOBAL_AGENT.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )

    return result, RAW_CONTENT


if __name__ == "__main__":
    query = "what is meant by multi agent RAG?"
    result, _ = run_agent(query, "01klju33256")

    print(result)
