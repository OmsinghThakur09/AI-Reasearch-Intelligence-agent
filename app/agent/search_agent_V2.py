# app/agent/search_agent_V2.py

"""
this script builds custom ReAct-style langgraph agent, that optimizes user query, uses custom Tavily web searching tool
to search optimized queries over web and returns its result.

Flow:
    START -> optimize_query -> run_search -> agent -> (loop to tools, or end)

this version manually starts from optimizing user query into 1-3 keyword focused sub-queries,
then web search optimized queries then performs its reasoning to descide wether more searching needs or not. it ensures that
agent never skips search tool even for simpler user query.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

import sqlite3
import json
import concurrent.futures
from typing import TypedDict, Annotated

from config import GROQ_API_KEY
from app.agent.search_tool import make_search_tool, web_search_executor

MODEL = "llama-3.3-70b-versatile"

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
MAX_QUERIES = 3


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    optimized_queries: list


OPTIMIZE_SYSTEM_PROMPT = """QUESTION: what is the boiling point of water at sea level
RAW MODEL OUTPUT: {
  "queries": ["boiling point of water at sea level"]
}
PARSED QUERIES: ['boiling point of water at sea level']

QUESTION: who is the current CEO of Microsoft
RAW MODEL OUTPUT: {
  "queries": ["Microsoft current CEO"]
}
PARSED QUERIES: ['Microsoft current CEO']

QUESTION: what is the capital of Japan
RAW MODEL OUTPUT: {
  "queries": ["Japan capital city"]
}
PARSED QUERIES: ['Japan capital city']

QUESTION: explain how photosynthesis works in plants
RAW MODEL OUTPUT: {
  "queries": ["photosynthesis process in plants"]
}
PARSED QUERIES: ['photosynthesis process in plants']

QUESTION: what are the health benefits of drinking green tea
RAW MODEL OUTPUT: {
  "queries": ["green tea health benefits"]
}
PARSED QUERIES: ['green tea health benefits']

QUESTION: compare Python and Rust for backend development
RAW MODEL OUTPUT: {
  "queries": ["Python vs Rust for backend development", "Rust backend development advantages", "Python backend development use cases"]
}
PARSED QUERIES: ['Python vs Rust for backend development', 'Rust backend development advantages', 'Python backend development use cases']

QUESTION: pros and cons of remote work vs office work
RAW MODEL OUTPUT: {
  "queries": ["remote work pros and cons", "office work advantages and disadvantages", "remote vs office work comparison"]
}
PARSED QUERIES: ['remote work pros and cons', 'office work advantages and disadvantages', 'remote vs office work comparison']

QUESTION: difference between machine learning and deep learning, and which one is used in self-driving cars
RAW MODEL OUTPUT: {
  "queries": ["machine learning vs deep learning", "deep learning in self-driving cars"]
}
PARSED QUERIES: ['machine learning vs deep learning', 'deep learning in self-driving cars']

QUESTION: what is the best Claude model for building websites in 2026
RAW MODEL OUTPUT: {
  "queries": ["Claude models for web development 2026", "best Claude model for website building"]
}
PARSED QUERIES: ['Claude models for web development 2026', 'best Claude model for website building']

QUESTION: how do I improve my resume for a software engineering job
RAW MODEL OUTPUT: {
  "queries": ["software engineering resume tips", "resume building for software engineers", "optimizing a software engineering resume"]
}
PARSED QUERIES: ['software engineering resume tips', 'resume building for software engineers', 'optimizing a software engineering resume']

QUESTION: what is quantum computing
RAW MODEL OUTPUT: {
  "queries": ["quantum computing basics"]
}
PARSED QUERIES: ['quantum computing basics']

QUESTION: how does the stock market work
RAW MODEL OUTPUT: {
  "queries": ["stock market basics"]
}
PARSED QUERIES: ['stock market basics']

QUESTION: what are the causes, effects, and solutions for climate change
RAW MODEL OUTPUT: {
  "queries": ["causes of climate change", "effects of climate change", "solutions to climate change"]
}
PARSED QUERIES: ['causes of climate change', 'effects of climate change', 'solutions to climate change']

QUESTION: compare the economic policies of USA, China, and India in 2026
RAW MODEL OUTPUT: {
  "queries": ["US economic policy 2026", "China economic policy 2026", "India economic policy 2026"]
}
PARSED QUERIES: ['US economic policy 2026', 'China economic policy 2026', 'India economic policy 2026']
"""


def build_agent():
    "build the graph: optimize -> concurrent search -> agent (loop or end)"

    tool, raw_content = make_search_tool()
    tools = [tool]

    # planner llm: no tool call only plain llm to optimize user query
    planner_llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0).bind(
        response_format={"type": "json_object"}
    )

    # tool bound llm: used for normal ReAct resoning after first search
    agent_llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0).bind_tools(
        tools
    )

    tool_node = ToolNode(tools)

    def optimize_query_node(state: AgentState):
        "a node for user query optimization before web search"
        user_query = state["messages"][-1].content

        response = planner_llm.invoke(
            [
                SystemMessage(content=OPTIMIZE_SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ]
        )

        parsed = json.loads(response.content)
        queries = parsed.get("queries", [])
        queries = [str(q).strip() for q in queries if str(q).strip()][:MAX_QUERIES]

        if not queries:
            queries = [user_query]

        return {"optimized_queries": queries}

    def search_node(state: AgentState):
        "a node that uses web search tool to search optimize queries concurrently"

        queries = state.get("optimized_queries", [])

        results_by_query = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_query = {
                executor.submit(web_search_executor, q): q for q in queries
            }

            for future in concurrent.futures.as_completed(future_to_query):
                q = future_to_query[future]

                try:
                    raw, metadata = future.result()
                    raw_content.extend(raw)
                    results_by_query.append({"query": q, "results": metadata})

                except Exception as e:
                    results_by_query.append({"query": q, "error": str(e)})

        summary_lines = ["Search results for the optimized queries:\n"]
        for item in results_by_query:
            summary_lines.append(f"query: {item['query']}")

            if "error" in item:
                summary_lines.append(f" (failed to search: {item['error']})")

            else:
                for r in item["results"]:
                    summary_lines.append(f" url: {r['url']}, title: {r['title']}")
                    summary_lines.append(f" content: {r['content']}")

        summary_lines.append("")

        summary_text = "\n".join(summary_lines)
        summary_text += (
            "\nUse the above to answer. If it's insufficient, "
            "you may call web_search_tool again with a more specific query."
        )

        return {"messages": [SystemMessage(content=summary_text)]}

    def agent_node(state: AgentState):
        "A node that acts as a ReAct agent"
        response = agent_llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("optimize_node", optimize_query_node)
    graph.add_node("search_node", search_node)
    graph.add_node("tools", tool_node)
    graph.add_node("agent_node", agent_node)

    graph.add_edge(START, "optimize_node")  # every run starts here
    graph.add_edge(
        "optimize_node", "search_node"
    )  # query optimizes befroe going to search
    graph.add_edge("search_node", "agent_node")  # tool call is always executed
    graph.add_conditional_edges("agent_node", tools_condition)
    graph.add_edge("tools", "agent_node")  # result goes to agent for reasoning

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
    query = (
        "state which of the current claude model is best for building websites in 2026?"
    )

    result, _ = run_agent(query, "98l3677lk")
    print(result)
