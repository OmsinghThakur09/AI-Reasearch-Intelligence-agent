# app/agent/search_agent_V2.py

"""
this script builds a langgraph agent that optimizes the user's query, searches
the optimized sub-queries over the web concurrently, then runs a deterministic
(non-LLM) sufficiency check that decides whether the collected content is
enough or whether the top thin sources need their full page fetched.

Flow:
    START -> structural_check_node -> optimize_node -> validity_node -> search_node -> check_and_escalate_node -> END

structural_check_node rejects empty/too-short/gibberish input before
any LLM call is made, and validity_node halts the run right after
optimize_node if the LLM judged the query's topic to not exist at all.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import sqlite3
import json
import re
import concurrent.futures
from typing import TypedDict, Annotated
from datetime import datetime

from config import GROQ_API_KEY
from app.agent.search_tool import web_search_executor, fetch_full_pages

MODEL = "qwen/qwen3.6-27b"

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

MAX_QUERIES = 3
# sufficiency / escalation thresholds
SUFFICIENT_CHARS = (
    1200  # combined snippet length across all sources considered "enough"
)
THIN_CHARS = 250  # a source's snippet shorter than this is considered "thin"
MAX_ESCALATE = 2  # fetch full page for at most this many thin sources per run

MIN_WORDS_COUNT = 3  # user query must contain minimum word count


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    optimized_queries: list
    search_results: list
    topic_valid: bool
    raw_content: list


class EmptyQueryError(Exception):
    """Custom exception class raised by structural_check_node when the raw user query is empty,
    too short, or contains no recognizable words."""


class InvalidQueryError(Exception):
    """custom exception class raised by validity_node when the optimizer's LLM judged the query's
    topic/keyword to not exist at all."""


current_date = datetime.now().strftime("%B %d, %Y")

OPTIMIZE_SYSTEM_PROMPT = """You are a search query planner for a research agent.

Today's date is: {current_date}

Output a JSON object with exactly three keys, IN THIS ORDER: "sub_topics", "valid", "queries".

1. "sub_topics": list every distinct thing needing separate search coverage, BEFORE deciding anything else.
   - A sub-topic is a separate NAMED entity being compared/listed (e.g. countries, products), OR a facet the question explicitly names (e.g. "causes, effects, and solutions").
   - One concept explored broadly ("explain X", "how does X work") is ONE sub-topic, even if open-ended.
   - Max 3 sub-topics; if more are named, keep only the 3 most central.

2. "valid": true if every sub-topic genuinely exists. false if the core topic is fabricated. If false, "queries" must be [].

3. "queries": exactly ONE self-contained query PER sub-topic (1-3 total).
   - Each query must stand alone with no other query for context.
   - NEVER merge sub-topics into one combined query, even for "compare"/"summarize together" requests — synthesis happens later, not in the search.

Respond with ONLY the JSON object.

Example 1 (single concept, broad phrasing — one query):
Question: "explain how photosynthesis works in plants"
{{"sub_topics": ["photosynthesis"], "valid": true, "queries": ["photosynthesis process in plants"]}}

Example 2 (named entities compared — one query PER entity, never combined):
Question: "compare the economics of japan, russia and saudi arabia"
{{"sub_topics": ["Japan", "Russia", "Saudi Arabia"], "valid": true, "queries": ["Japan economy 2026", "Russia economy 2026", "Saudi Arabia economy 2026"]}}

Example 3 (one topic, explicitly-named facets — split by facet):
Question: "what are the causes, effects, and solutions for climate change"
{{"sub_topics": ["causes of climate change", "effects", "solutions"], "valid": true, "queries": ["causes of climate change", "effects of climate change", "solutions to climate change"]}}

Example 4 (topic doesn't exist — invalid):
Question: "explain how the Zorblatt Compression Algorithm reduces latency in neural networks"
{{"sub_topics": ["Zorblatt Compression Algorithm"], "valid": false, "queries": []}}
""".format(current_date=current_date)


def build_agent():
    """build the graph:
    START -> structural_check_node -> optimize_node -> validity_node -> search_node -> check_and_escalate_node -> END
    """

    # planner llm: plain llm call to optimize user query.
    planner_llm = ChatGroq(
        api_key=GROQ_API_KEY, model=MODEL, temperature=0, reasoning_effort="default"
    ).bind(response_format={"type": "json_object"})

    def structural_check_node(state: AgentState):
        "a node that will check for any empty, gibberish, or less than minimum word count query, if found then raises Exception"

        user_query = state["messages"][-1].content.strip()

        if not user_query:
            raise EmptyQueryError("a query cant be empty! plese enter a query")
        if len(user_query) < MIN_WORDS_COUNT:
            raise EmptyQueryError(
                f"query is too short to be meaningful! minimun word count: {MIN_WORDS_COUNT}"
            )

        if not re.search(r"[A-Za-z]{2,}", user_query):
            raise EmptyQueryError("Query does not contain any recognizable words.")

        return {}

    def optimize_query_node(state: AgentState):
        "user query optimization before web search"
        user_query = state["messages"][-1].content

        response = planner_llm.invoke(
            [
                SystemMessage(content=OPTIMIZE_SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ]
        )

        parsed = json.loads(response.content)
        is_valid = bool(parsed.get("valid", True))
        queries = parsed.get("queries", [])
        queries = [str(q).strip() for q in queries if str(q).strip()][:MAX_QUERIES]

        if is_valid and not queries:
            queries = [user_query]

        return {"optimized_queries": queries, "topic_valid": is_valid}

    def validity_check_node(state: AgentState):
        "node to check user query is valid or not, if not then raises exception"

        if not state.get("topic_valid", True):
            raise InvalidQueryError(
                "The topic or keyword in this query does not appear to exist. "
                "Search was not performed."
            )

        return {}

    def search_node(state: AgentState):
        "a node that uses web search (snippet-only, no raw page) concurrently for each optimized query"

        queries = state.get("optimized_queries", [])

        seen_urls = set()
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
                    metadata = []

                    for item in items:
                        url = item["url"]
                        if url and url in seen_urls:
                            continue
                        if url:
                            seen_urls.add(url)

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
        raw_content = []
        if total_chars < SUFFICIENT_CHARS:
            thin_sources = [r for r in all_results if len(r["content"]) < THIN_CHARS]
            escalated_urls = [r["url"] for r in thin_sources if r["url"]][:MAX_ESCALATE]

            if escalated_urls:
                fetched = fetch_full_pages(escalated_urls)
                for url, content in fetched.items():
                    raw_content.append((url, content))

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
    graph.add_node("structural_check_node", structural_check_node)
    graph.add_node("optimize_node", optimize_query_node)
    graph.add_node("validity_node", validity_check_node)
    graph.add_node("search_node", search_node)
    graph.add_node("check_and_escalate_node", check_and_escalate_node)

    graph.add_edge(START, "structural_check_node")
    graph.add_edge("structural_check_node", "optimize_node")
    graph.add_edge("optimize_node", "validity_node")
    graph.add_edge("validity_node", "search_node")
    graph.add_edge("search_node", "check_and_escalate_node")
    graph.add_edge("check_and_escalate_node", END)

    compiled = graph.compile(checkpointer=memory)
    return compiled


GLOBAL_AGENT = build_agent()


def run_agent(query: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}

    result = GLOBAL_AGENT.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )

    raw_content = result.get("raw_content", [])
    sub_queries = result.get("optimized_queries", [])

    return result, raw_content, sub_queries


if __name__ == "__main__":
    query = "explain Lethorium"

    result, raw, sub_queries = run_agent(query, "9887tyfgh")
    print(result)
    print(len(raw))
