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
from datetime import datetime

from config import GROQ_API_KEY
from app.agent.search_tool import make_search_tool, web_search_executor

MODEL = "llama-3.3-70b-versatile"

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
MAX_QUERIES = 3


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    optimized_queries: list


current_date = datetime.now().strftime("%B %d, %Y")

OPTIMIZE_SYSTEM_PROMPT = (
    f"You are a search query planner for a research agent, Today's current date is {current_date}"
    + """Always
use this date as the absolute baseline anchor for the current year and any relative time calculations (e.g., 'yesterday', 'next month').

Given the user's question, output a JSON object with a single key "queries",
containing an array of 1 to 3 focused, keyword-rich search queries.

A "sub-topic" means a distinct entity, item, or thing being discussed —
not a different angle, facet, or aspect of the same single concept.

Rules:
- If the question asks about ONE concept, thing, or entity — even if the
  question uses open-ended phrasing like "explain", "what is", or "how does
  X work" — output exactly ONE query. Do not split a single concept into
  multiple angles or facets of itself.
- Only output more than one query if the question explicitly names or implies
  multiple distinct things (e.g. comparing two items, listing multiple
  entities, or asking about separate causes/effects/solutions/parts of a
  larger topic).
- Respond with ONLY the JSON object. No explanation, no markdown, no extra text.

Example 1 (single concept — one query, even though phrased broadly):
Question: "explain how photosynthesis works in plants"
Output:
{
  "queries": ["photosynthesis process in plants"]
}

Example 2 (single concept — one query):
Question: "what is quantum computing"
Output:
{
  "queries": ["quantum computing explained"]
}

Example 3 (genuinely multiple distinct things — split):
Question: "compare RAG and multi-agent RAG for legal document analysis"
Output:
{
  "queries": ["RAG vs multi-agent RAG comparison", "multi-agent RAG legal document analysis use case"]
}

Example 4 (genuinely multiple distinct things — split):
Question: "what are the causes, effects, and solutions for climate change"
Output:
{
  "queries": ["causes of climate change", "effects of climate change", "solutions to climate change"]
}
"""
)

AGENT_SYSTEM_PROMPT = """You are a research assistant answering the user's question using search results already provided in this conversation.

Before calling web_search_tool again, check the full conversation history for
queries that were already searched. Do not issue a new search query that is a
rephrasing, synonym, or near-duplicate of any prior query — this wastes calls
and returns overlapping results. Only search again if you can name a specific
piece of missing information the existing results do not cover.

If the existing results are enough to answer the question, answer directly
without calling any tool."""


def build_agent():
    "build the graph: optimize -> concurrent search -> agent (loop or end)"

    tool, raw_content, seen_urls = make_search_tool()
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
                    items = future.result()

                    # dedupe by url here, in the main process, against the SAME
                    # seen_urls set used later by web_search_tool in the agent loop,
                    # so a page returned by two different optimized queries (or
                    # re-searched later) is only ever added to raw_content once
                    metadata = []
                    for item in items:
                        url = item["url"]
                        if url and url in seen_urls:
                            continue
                        if url:
                            seen_urls.add(url)

                        raw_content.append(item["raw_content"])
                        metadata.append(
                            {
                                "url": url,
                                "title": item["title"],
                                "content": item["content"],
                            }
                        )

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
            f"\n\nQueries already searched: {queries}\n\n"
            "Use the results above to answer if they are sufficient. "
            "Only call web_search_tool again if there is a specific, unresolved gap "
            "that the above results do not cover. If you do search again, the new "
            "query MUST target a distinctly different angle, fact, or sub-topic — "
            "not a rephrasing or near-duplicate of any query already searched above. "
            "Do not search again just to confirm or restate what you already have."
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
    query = "Detail the performance benchmarks of Retrieval-Aware Fine-Tuning (RAFT) techniques compared to standard RAG pipelines in recent domain-specific evaluations."

    result, raw = run_agent(query, "852ujnrdfc")
    print(len(raw))
    # print(result)
