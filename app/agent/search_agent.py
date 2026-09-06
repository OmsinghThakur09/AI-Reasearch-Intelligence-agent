# app/agent/search_agent.py

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
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import sqlite3
import json
import re
import time
import concurrent.futures
from typing import TypedDict, Annotated
from datetime import datetime

from config import GROQ_API_KEY
from app.agent.search_tool import web_search_executor, fetch_full_pages

MODEL = "qwen/qwen3.8-27b"

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

MAX_QUERIES = 3
# sufficiency / escalation thresholds
SUFFICIENT_CHARS = (
    10000  # combined snippet length across all sources considered "enough"
)
THIN_CHARS = 1000  # a source's snippet shorter than this is considered "thin"
MAX_ESCALATE = 4  # fetch full page for at most this many thin sources per run

MIN_WORDS_COUNT = 3  # user query must contain minimum word count

# retry settings for the planner LLM's JSON-mode call, which can
# occasionally return invalid/empty JSON (a known flake with hosted
# reasoning models in strict json_object mode)
MAX_OPTIMIZE_RETRIES = 3
OPTIMIZE_RETRY_BACKOFF_SECONDS = 1.5  # multiplied by attempt number


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    optimized_queries: list
    search_results: list
    topic_valid: bool
    time_range: str
    raw_content: list


class EmptyQueryError(Exception):
    """Custom exception class raised by structural_check_node when the raw user query is empty,
    too short, or contains no recognizable words."""


class InvalidQueryError(Exception):
    """custom exception class raised by validity_node when the optimizer's LLM judged the query's
    topic/keyword to not exist at all."""


class OptimizerGenerationError(Exception):
    """raised by optimize_query_node when the planner LLM fails to return valid JSON
    after MAX_OPTIMIZE_RETRIES attempts (e.g. Groq's json_validate_failed / empty
    failed_generation flake)."""


current_date = datetime.now().strftime("%B %d, %Y")

OPTIMIZE_SYSTEM_PROMPT = f"""You are a search query planner for a high-precision research agent.

Today's date is: {current_date}

Output JSON with exactly five keys, in this order: "sub_topics", "valid", "comparison_axis", "time_range", "queries".

if the current question references the previous one via pronouns or implicit comparison, resolve it into a standalone query using the previous question's entities before generating queries/sub_topics.

Each query is used for BOTH a live web search (Tavily) AND a similarity search against a vector store of already-retrieved source text, so every query must work for both jobs at once.

1. "sub_topics": distinct concepts, technical pillars, or entities needing separate coverage.
   - For technical/scientific topics, break the question down into 2-3 core mechanical pillars (e.g., memory optimization, cutting mechanism).
   - If comparing named entities, each sub-topic must cover exactly ONE entity — never merge two entities into one sub-topic.

2. "valid": true if every sub-topic genuinely exists, else false (then comparison_axis and time_range are null, queries is []).

3. "comparison_axis": if comparing 2+ named entities, extract the exact metric or attribute by name (e.g. "annual GDP growth rate", "inference latency benchmark"). Otherwise null.

4. "time_range": choose one of "day", "week", "month", "year" ONLY if the question asks about the current/ongoing state of something (signaled by words like "latest", "current", "now", "this month", or a topic that is inherently fast-moving,
e.g. chip export rules, model releases, live prices) — the range reflects how recently Tavily crawled the page, not any date mentioned in the query. Otherwise, including whenever the question references a specific fixed date, year, or past event
(e.g. "the 2024 survey", "the March 2025 announcement"), set it to null — the source could have been published or crawled at any time and must not be filtered out by recency

5. "queries": exactly 1 or 2 distinct query per sub-topic, written as a dense keyword phrase — NOT a question, NOT a full sentence.
   - For multi-entity comparisons: If comparison_axis is set, append it to each entity's own query — never name two entities in the same query.
   - For single technical topics: Generate a highly targeted keyword query for each isolated sub-topic/pillar.
   - Use 4-7 dense keywords: entity names, technical terms, and official nomenclature, in the vocabulary a source article on the topic would actually use.
   - Do NOT use question words ("what", "how", "why", "compare", "vs") or conversational filler/intent tags ("explained", "overview", "reported", "guide", "summary", "findings").

Respond with ONLY the JSON object.

Example 1 (deep technical topic breakdown):
Q: "Summarize the key architectural changes introduced in the newest open-source LLM models released this year."
{{"sub_topics": ["attention and memory efficiency", "mixture of experts routing", "inference prediction training"], "valid": true, "comparison_axis": null, "time_range":
"year", "queries": ["open source LLM multi head latent attention KV cache", "sparse mixture of experts MoE routing parameters architecture", "native multi token prediction training inference loops"]}}

Example 2 (multi-entity comparison):
Q: "compare the economics of japan, russia and saudi arabia"
{{"sub_topics": ["Japan", "Russia", "Saudi Arabia"], "valid": true, "comparison_axis": "annual GDP growth rate", "time_range": "year", "queries": ["Japan annual GDP growth rate", "Russia annual GDP growth rate", "Saudi Arabia annual GDP growth rate"]}}

Example 3 (quantitative benchmark comparison):
Q: "Detail the performance benchmarks of Retrieval-Aware Fine-Tuning (RAFT) techniques compared to standard RAG pipelines in recent domain-specific evaluations."
{{"sub_topics": ["RAFT domain-specific benchmark results", "standard RAG pipeline benchmark results"], "valid": true, "comparison_axis": "domain-specific evaluation performance metrics", "time_range": "year",
"queries": ["retrieval aware fine tuning RAFT accuracy F1 score benchmark paper", "standard RAG pipeline accuracy F1 score benchmark evaluation paper"]}}

Example 4 (fixed past date/event — no time_range):
Q: "What were the specific findings of the 2024 field survey on soil microbiome diversity in the Sundarbans mangrove forest?"
{{"sub_topics": ["Sundarbans mangrove soil microbiome diversity 2024 survey findings"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["Sundarbans mangrove soil microbiome diversity field survey 2024"]}}
"""


def build_agent():
    """build the graph:
    START -> structural_check_node -> optimize_node -> validity_node -> search_node -> check_and_escalate_node -> END
    """

    # planner llm: plain llm call to optimize user query.
    planner_llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0).bind(
        response_format={"type": "json_object"}
    )

    def structural_check_node(state: AgentState):
        "a node that will check for any empty, gibberish, or less than minimum word count query, if found then raises Exception"
        "also trims message history: since add_messages only appends, a long-lived"
        "session (same thread_id reused across many turns) would otherwise grow"
        "'messages' forever in the sqlite checkpoint. we only ever need the latest"
        "human query here, so every older message is explicitly removed."

        all_messages = state["messages"]
        current_message = all_messages[-1]
        user_query = current_message.content.strip()

        if not user_query:
            raise EmptyQueryError("a query cant be empty! please enter a query")

        if not re.search(r"[A-Za-z]{2,}", user_query):
            raise EmptyQueryError("Query does not contain any recognizable words.")

        if len(user_query.split()) < MIN_WORDS_COUNT:
            raise EmptyQueryError(
                f"query is too short to be meaningful! minimum word count: {MIN_WORDS_COUNT}"
            )

        # drop every message except the current one, so the checkpointed
        # state (and any future prompt built from it) doesn't accumulate
        # unbounded history across turns of the same session
        old_messages = [
            RemoveMessage(id=m.id) for m in all_messages if m.id != current_message.id
        ]

        return {"messages": old_messages} if old_messages else {}

    def optimize_query_node(state: AgentState):
        "user query optimization before web search"
        user_query = state["messages"][-1].content

        last_error: Exception | None = None
        parsed = None

        for attempt in range(1, MAX_OPTIMIZE_RETRIES + 1):
            try:
                response = planner_llm.invoke(
                    [
                        SystemMessage(content=OPTIMIZE_SYSTEM_PROMPT),
                        HumanMessage(content=user_query),
                    ]
                )

                content = (response.content or "").strip()
                if not content:
                    raise ValueError("planner LLM returned an empty response")

                parsed = json.loads(content)
                break  # success, stop retrying

            except Exception as e:
                # covers: groq.BadRequestError (json_validate_failed / empty
                # failed_generation), json.JSONDecodeError, empty-content ValueError
                last_error = e
                if attempt < MAX_OPTIMIZE_RETRIES:
                    time.sleep(OPTIMIZE_RETRY_BACKOFF_SECONDS * attempt)
                    continue

        if parsed is None:
            raise OptimizerGenerationError(
                f"planner LLM failed to produce valid JSON after "
                f"{MAX_OPTIMIZE_RETRIES} attempts: {last_error}"
            )

        is_valid = bool(parsed.get("valid", True))
        queries = parsed.get("queries", [])
        queries = [str(q).strip() for q in queries if str(q).strip()][:MAX_QUERIES]
        time_range = parsed.get("time_range", None)

        if is_valid and not queries:
            queries = [user_query]

        return {
            "optimized_queries": queries,
            "topic_valid": is_valid,
            "time_range": time_range,
        }

    def validity_check_node(state: AgentState):
        "node to check user query is valid or not, if not then raises exception"

        if not state.get("topic_valid", True):
            raise InvalidQueryError(
                "The topic or keyword in this query does not appear to exist. "
                "Search was not performed."
            )

        return {}

    def search_node(state: AgentState):
        queries = state.get("optimized_queries", [])
        time_range = state.get("time_range", "")
        seen_urls = set()
        results_by_query = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(queries) or 1
        ) as executor:
            future_to_query = {
                executor.submit(
                    web_search_executor,
                    q,
                    time_range=time_range,
                    max_result=5 if len(queries) == 1 else 3,
                ): q
                for q in queries
            }

            for future in concurrent.futures.as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    items = future.result()
                    metadata = []
                    for item in items:
                        url = item.get("url")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            metadata.append(item)

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

        return {
            "messages": [SystemMessage(content="\n".join(summary_lines))],
            "raw_content": raw_content,
        }

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


def run_agent(query: str, pre_query: str | None, session_id: str):
    config = {"configurable": {"thread_id": session_id}}

    if pre_query:
        content = f"Previous question: {pre_query}\n\nCurrent question: {query}"
    else:
        content = query

    result = GLOBAL_AGENT.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config=config,
    )

    raw_content = result.get("raw_content", [])
    sub_queries = result.get("optimized_queries", [])

    return result, raw_content, sub_queries


# if __name__ == "__main__":
#     query = "State of the art long-context window mechanisms in large language models"

#     result, raw, sub_queries = run_agent(query, "5657890iohujgvbn")
#     print(result)
#     print(len(raw))
