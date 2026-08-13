# app/agent/search_agent_V2.py

"""
this script builds a langgraph agent that optimizes the user's query, searches
the optimized sub-queries over the web concurrently, then runs a deterministic
(non-LLM) sufficiency check that decides whether the collected content is
enough or whether the top thin sources need their full page fetched.

Flow:
    START -> optimize_node -> search_node -> check_and_escalate_node -> END

no tool-calling loop: the "decide if we have enough" step used to be a
second LLM call (redundant with the final RAG answer generation downstream).
that judgment is now a rule-based check on content length instead, so the
agent still behaves adaptively (search once, then optionally escalate to full
pages for weak sources) without paying for a second, unused LLM-generated
answer every run.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import sqlite3
import json
import concurrent.futures
from typing import TypedDict, Annotated
from datetime import datetime

from config import GROQ_API_KEY
from app.agent.search_tool import web_search_executor, fetch_full_pages

MODEL = "llama-3.3-70b-versatile"

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

MAX_QUERIES = 3
# sufficiency / escalation thresholds
SUFFICIENT_CHARS = (
    1200  # combined snippet length across all sources considered "enough"
)
THIN_CHARS = 250  # a source's snippet shorter than this is considered "thin"
MAX_ESCALATE = 2  # fetch full page for at most this many thin sources per run


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    optimized_queries: list
    search_results: list


current_date = datetime.now().strftime("%B %d, %Y")

OPTIMIZE_SYSTEM_PROMPT = (
    f"You are a search query planner for a research agent, Today's current date is {current_date}"
    + """Always use this date as the absolute baseline anchor for the current year and any relative time calculations (e.g., 'yesterday', 'next month').

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

    Note: check query before optimizing, if query contains keyword or subtopic that doesn't exist at all then dont optimize it.
    """
)


def build_agent():
    "build the graph: optimize -> concurrent search -> deterministic sufficiency/escalation check -> end"

    raw_content = []
    subqueries = []
    seen_urls = set()
    url_index_map: dict[str, int] = (
        {}
    )  # url -> its position in raw_content, for escalation to overwrite later

    # planner llm: plain llm call to optimize user query.
    planner_llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0).bind(
        response_format={"type": "json_object"}
    )

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

        subqueries.extend(queries)
        return {"optimized_queries": queries}

    def search_node(state: AgentState):
        "a node that uses web search (snippet-only, no raw page) concurrently for each optimized query"

        queries = state.get("optimized_queries", [])

        results_by_query = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_query = {
                executor.submit(
                    web_search_executor, q, max_result=5 if len(queries) == 1 else 2
                ): q
                for q in queries
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

                        url_index_map[url] = len(raw_content)
                        # raw_content.append(item["raw_content"])  # "" by default (snippet-only search)
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

        return {"search_results": results_by_query}

    def check_and_escalate_node(state: AgentState):
        "deterministic (non-LLM) sufficiency check: decides if collected snippets are"
        "enough, and if not, fetches the full page for the top few thin sources only."

        results_by_query = state.get("search_results", [])

        all_results = [
            r for item in results_by_query if "results" in item for r in item["results"]
        ]

        # length of all fetched snippets
        total_chars = sum(len(r["content"]) for r in all_results)

        escalated_urls = []
        if total_chars < SUFFICIENT_CHARS:
            thin_sources = [r for r in all_results if len(r["content"]) < THIN_CHARS]
            escalated_urls = [r["url"] for r in thin_sources if r["url"]][:MAX_ESCALATE]

            if escalated_urls:
                fetched = fetch_full_pages(escalated_urls)
                for url, content in fetched.items():
                    idx = url_index_map.get(url)
                    if idx is not None:
                        raw_content.append(content)

        summary_lines = [
            f"Collected {len(all_results)} source(s), {total_chars} total snippet characters."
        ]
        if escalated_urls:
            summary_lines.append(
                f"Snippets were thin, fetched full page content for: {escalated_urls}"
            )
        else:
            summary_lines.append(
                "Snippet content judged sufficient, no full-page fetch needed."
            )

        return {"messages": [SystemMessage(content="\n".join(summary_lines))]}

    graph = StateGraph(AgentState)
    graph.add_node("optimize_node", optimize_query_node)
    graph.add_node("search_node", search_node)
    graph.add_node("check_and_escalate_node", check_and_escalate_node)

    graph.add_edge(START, "optimize_node")
    graph.add_edge("optimize_node", "search_node")
    graph.add_edge("search_node", "check_and_escalate_node")
    graph.add_edge("check_and_escalate_node", END)

    compiled = graph.compile(checkpointer=memory)
    return compiled, raw_content, subqueries


GLOBAL_AGENT, RAW_CONTENT, SUB_QUERIES = build_agent()


def run_agent(query: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}

    result = GLOBAL_AGENT.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )

    return result, RAW_CONTENT, SUB_QUERIES


if __name__ == "__main__":
    query = "What are the primary efficiency limitations identified in Perovskite-Silicon tandem solar cells according to recent materials science literature?"

    result, raw, sub_queries = run_agent(query, "76uy")
    print(result)
    print(len(raw))

"Retrieval-Aware Fine-Tuning evaluation in domain-specific tasks"
"Domain-specific performance of RAFT techniques"
