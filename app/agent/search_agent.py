# app/agent/search_agent.py

"""
this script builds a langgraph agent that optimizes the user's query, searches
the optimized sub-queries over the web concurrently, then runs a deterministic
(non-LLM) sufficiency check that decides whether the collected content is
enough or whether the top thin sources need their full page fetched.

Flow:
    START -> structural_check_node -> manage_history_node -> optimize_node
    -> validity_node -> search_node -> check_and_escalate_node -> END

structural_check_node rejects empty/too-short/gibberish input before
any LLM call is made, and validity_node halts the run right after
optimize_node if the LLM judged the query's topic to not exist at all.

conversation memory lives only in langgraph checkpointer, only last WINDOW_TURNS will be saved
as it is and afterwards previous context will be summarized to keep memory clean and short(rolling context window) .
"""

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    RemoveMessage,
)
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

MODEL = "openai/gpt-oss-20b"

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

MAX_QUERIES = 6
# sufficiency / escalation thresholds
SUFFICIENT_CHARS = (
    10000  # combined snippet length across all sources considered "enough"
)
THIN_CHARS = 1000  # a source's snippet shorter than this is considered "thin"
MAX_ESCALATE = 4  # fetch full page for at most this many thin sources per run

MIN_WORDS_COUNT = 3  # user query must contain minimum word count

WINDOW_TURNS = 4  # how many of the most recent user turns stay as raw messages

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
    conversation_summary: str  # rolling summary of turns pushed out of the window


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


OPTIMIZE_PROMPT_TEMPLATE = """You are a search query planner for a high-precision research agent.

Today's date is: {current_date}

Output JSON with exactly five keys, in this order: "sub_topics", "valid", "comparison_axis", "time_range", "queries".

Input format: the input is either just the user's question, or some conversation context followed by a line starting with "Current question:". The context can contain "Conversation summary so far", "Previous user question" and "Previous answer" lines. Plan ONLY for the current question. Use the context only to resolve references (pronouns like "it" or "their", or phrases like "what about...", "and the other one") into a standalone question using the previous question's entities, before generating sub_topics and queries. If the current question is self-contained or about a different topic, ignore the context completely. Never copy numbers or claims from a previous answer into the queries.

Each query is used for BOTH a live web search (Tavily) AND a similarity search against a vector store of already-retrieved source text, so every query must work for both jobs at once.

1. "sub_topics": distinct concepts, technical pillars, or entities needing separate coverage. Use at most 3 sub_topics; if the question covers more, keep the 3 most important.
   - For technical/scientific topics, break the question down into 2-3 core mechanical pillars (e.g., memory optimization, cutting mechanism).
   - If comparing named entities, each sub-topic must cover exactly ONE entity — never merge two entities into one sub-topic.
   - If the question has several parts, use one sub-topic per part. If the question is very broad ("tell me everything about X"), pick the 3 most important pillars.

2. "valid": true if every sub-topic genuinely exists, else false (then comparison_axis and time_range are null, queries is []).
   - "valid" judges only whether the TOPIC exists, not whether the question's premise is correct. A question with a wrong premise, or a "what if" scenario about real things, is still valid — search for the real facts.
   - Mark false only when a sub-topic is clearly fictional or impossible (e.g. a prize category that does not exist). If you are unsure whether a very recent product, paper, or event exists, mark true — your own knowledge may be out of date.

3. "comparison_axis": if comparing 2+ named entities, extract the exact metric or attribute by name (e.g. "annual GDP growth rate", "inference latency benchmark"). Otherwise null.
   - A comparison across time periods (e.g. 2023 and 2025) counts: each period is its own entity.
   - If the answer needs a calculation (a ratio or a difference), fetch the raw figure for each entity on the SAME metric and period; the calculation happens later.

4. "time_range": choose one of "day", "week", "month", "year" ONLY if the question asks about the current/ongoing state of something (signaled by words like "latest", "current", "now", "this month", or a topic that is inherently fast-moving, e.g. chip export rules, model releases, live prices) — the range reflects how recently Tavily crawled the page, not any date mentioned in the query.
   - Pick the narrowest range that fits: "day" for live or same-day data, "week" for this week's news, "month" for recent developments and frequently updated statistics, "year" for the current state of a fast-moving topic.
   - Otherwise, including whenever the question references a specific fixed date, year, or past event (e.g. "the 2024 survey", "the March 2025 announcement"), set it to null — the source could have been published or crawled at any time and must not be filtered out by recency.
   - For a forecast about a future year, also use null and put the target year in the query.

5. "queries": 1 or 2 distinct queries per sub-topic, written as a dense keyword phrase — NOT a question, NOT a full sentence. Use AT MOST 6 queries in total: extra queries are discarded, so give every sub-topic one query before adding a second query to any sub-topic.
   - For multi-entity comparisons: If comparison_axis is set, append it to each entity's own query — never name two entities in the same query.
   - For single technical topics: Generate a highly targeted keyword query for each isolated sub-topic/pillar.
   - Use 4-7 dense keywords (a multi-word technical term counts as one keyword): entity names, technical terms, and official nomenclature, in the vocabulary a source article on the topic would actually use.
   - Do NOT use question words ("what", "how", "why", "compare", "vs") or conversational filler/intent tags ("explained", "overview", "reported", "guide", "summary", "findings"), unless the word is part of an official term.
   - Ignore instructions about the answer's format, tone or length ("in a table", "in simple terms", "briefly") and any personal backstory — search only the researchable core of the question.
   - For opinion or sentiment questions ("what do people think about X"), aim the queries at surveys, reports and published critiques, because forum and social-media sites are excluded from search.

Respond with ONLY the JSON object.

Example 1 (deep technical topic breakdown):
Q: "Summarize the key architectural changes introduced in the newest open-source LLM models released this year."
{{"sub_topics": ["attention and memory efficiency", "mixture of experts routing", "inference prediction training"], "valid": true, "comparison_axis": null, "time_range": "year", "queries": ["open source LLM multi head latent attention KV cache", "sparse mixture of experts MoE routing parameters architecture", "native multi token prediction training inference loops"]}}

Example 2 (multi-entity comparison):
Q: "compare the economics of japan, russia and saudi arabia"
{{"sub_topics": ["Japan", "Russia", "Saudi Arabia"], "valid": true, "comparison_axis": "annual GDP growth rate", "time_range": "year", "queries": ["Japan annual GDP growth rate", "Russia annual GDP growth rate", "Saudi Arabia annual GDP growth rate"]}}

Example 3 (quantitative benchmark comparison):
Q: "Detail the performance benchmarks of Retrieval-Aware Fine-Tuning (RAFT) techniques compared to standard RAG pipelines in recent domain-specific evaluations."
{{"sub_topics": ["RAFT domain-specific benchmark results", "standard RAG pipeline benchmark results"], "valid": true, "comparison_axis": "domain-specific evaluation performance metrics", "time_range": "year", "queries": ["retrieval aware fine tuning RAFT accuracy F1 score benchmark paper", "standard RAG pipeline accuracy F1 score benchmark evaluation paper"]}}

Example 4 (fixed past date/event — no time_range):
Q: "What were the specific findings of the 2024 field survey on soil microbiome diversity in the Sundarbans mangrove forest?"
{{"sub_topics": ["Sundarbans mangrove soil microbiome diversity 2024 survey findings"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["Sundarbans mangrove soil microbiome diversity field survey 2024"]}}

Example 5 (current numeric statistic, short recency window):
Q: "What is India's current inflation rate?"
{{"sub_topics": ["India consumer price inflation rate"], "valid": true, "comparison_axis": null, "time_range": "month", "queries": ["India CPI inflation rate year-on-year MoSPI"]}}

Example 6 (breaking news, very short recency window):
Q: "Summarize the latest AI regulation news this week"
{{"sub_topics": ["US AI regulation developments", "EU and international AI regulation developments"], "valid": true, "comparison_axis": null, "time_range": "week", "queries": ["US AI regulation bill state law news", "EU AI Act enforcement international policy news"]}}

Example 7 (beginner-level "how does it work" question, wording like "in simple terms" is dropped, official technical terms are used):
Q: "Explain how CRISPR gene editing works in simple terms"
{{"sub_topics": ["CRISPR-Cas9 targeting mechanism", "DNA repair pathways after the cut"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["CRISPR-Cas9 mechanism sgRNA PAM double strand break", "CRISPR gene editing NHEJ HDR DNA repair"]}}

Example 8 (two-product comparison, one product per sub-topic and per query):
Q: "Sony WH-1000XM5 vs Bose QuietComfort Ultra noise cancellation"
{{"sub_topics": ["Sony WH-1000XM5", "Bose QuietComfort Ultra"], "valid": true, "comparison_axis": "noise cancellation performance", "time_range": null, "queries": ["Sony WH-1000XM5 noise cancellation performance", "Bose QuietComfort Ultra noise cancellation performance"]}}

Example 9 (comparison across time, each year is its own entity):
Q: "How did global EV sales in 2023 differ from 2025?"
{{"sub_topics": ["Global EV sales in 2023", "Global EV sales in 2025"], "valid": true, "comparison_axis": "global electric vehicle sales volume", "time_range": null, "queries": ["IEA global electric vehicle sales volume 2023", "IEA global electric vehicle sales volume 2025"]}}

Example 10 (recommendation with constraints, current models matter):
Q: "Best laptop under 80000 rupees for programming"
{{"sub_topics": ["laptop specifications needed for programming", "laptop models available under 80000 INR"], "valid": true, "comparison_axis": null, "time_range": "year", "queries": ["developer laptop 16GB RAM SSD Ryzen", "best laptops under 80000 INR programming review"]}}

Example 11 (detailed review built from many sources, one query per aspect of the product):
Q: "Give me a detailed review of the Sony WH-1000XM6 based on multiple reviews"
{{"sub_topics": ["sound quality and noise cancellation", "comfort, battery life and build quality", "price and drawbacks"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["Sony WH-1000XM6 sound quality noise cancellation review", "Sony WH-1000XM6 comfort battery life build review", "Sony WH-1000XM6 price drawbacks cons review"]}}

Example 12 (question about one specific research paper, fixed past work, so no time_range):
Q: "What are the main ideas of the paper 'Attention Is All You Need'?"
{{"sub_topics": ["transformer architecture in Attention Is All You Need", "translation results of the original transformer"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["Attention Is All You Need Vaswani transformer", "original transformer WMT 2014 English German BLEU"]}}

Example 13 (controversial or conflicting evidence, one query per side so both viewpoints are retrieved):
Q: "Does raising the minimum wage cost jobs?"
{{"sub_topics": ["studies finding job losses from minimum wage increases", "studies finding no employment effect from minimum wage increases"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["minimum wage increase employment loss empirical study", "minimum wage null employment effect Card Krueger"]}}

Example 14 (forecast about a future year, two authoritative forecasters, the target year stays in the query):
Q: "Projected global GDP growth for 2027"
{{"sub_topics": ["global GDP growth projections for 2027"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["IMF WEO latest global growth projection 2027", "World Bank Global Economic Prospects 2027 forecast"]}}

Example 15 (multi-hop question, the middle answer is stable so it is resolved and named in the second query):
Q: "What is the population of the capital city of the country that hosted the 2022 FIFA World Cup?"
{{"sub_topics": ["host country of the 2022 FIFA World Cup and its capital", "population of Doha"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["2022 FIFA World Cup host country capital city", "Doha Qatar population census figures"]}}

Example 16 (follow-up that refers back to the previous turn, rewritten as a standalone comparison):
Q:
Previous user question: Sony WH-1000XM5 vs Bose QuietComfort Ultra noise cancellation
Previous answer: The Sony model has slightly stronger noise cancellation in most tests.
Current question: What about their battery life?
{{"sub_topics": ["Sony WH-1000XM5", "Bose QuietComfort Ultra"], "valid": true, "comparison_axis": "battery life hours", "time_range": null, "queries": ["Sony WH-1000XM5 battery life hours", "Bose QuietComfort Ultra battery life hours"]}}

Example 17 (three questions in one message, one sub-topic and one query for each part):
Q: "What is blockchain, who invented it, and how much energy does it use?"
{{"sub_topics": ["blockchain technology basics", "origin of blockchain and Bitcoin", "blockchain energy consumption"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["blockchain distributed ledger consensus mechanism", "Bitcoin whitepaper Satoshi Nakamoto 2008 origin", "blockchain energy consumption proof-of-work proof-of-stake"]}}

Example 18 (calculation question, the planner only fetches the two raw numbers on the same metric, the math happens later):
Q: "How many times bigger is India's economy than Bangladesh's?"
{{"sub_topics": ["India", "Bangladesh"], "valid": true, "comparison_axis": "nominal GDP in US dollars", "time_range": "year", "queries": ["India nominal GDP US dollars IMF", "Bangladesh nominal GDP US dollars IMF"]}}

Example 19 (hypothetical what-if question, the scenario is imaginary but the topics are real, so valid stays true and searches ground the facts):
Q: "What would happen if the US banned all chip exports?"
{{"sub_topics": ["existing US semiconductor export controls", "revenue dependence of US chip companies on exports", "global chip supply chain effects of export restrictions"], "valid": true, "comparison_axis": null, "time_range": "year", "queries": ["US semiconductor export controls BIS entity list", "US chip companies revenue share international sales", "export restrictions semiconductor global supply chain disruption"]}}

Example 20 (topic that does not exist, so nothing is searched):
Q: "What did the 2019 Nobel Prize in Mathematics winner say in the acceptance speech?"
{{"sub_topics": ["2019 Nobel Prize in Mathematics"], "valid": false, "comparison_axis": null, "time_range": null, "queries": []}}

Example 21 (false premise about a real topic, valid stays true and queries target the real facts):
Q: "Why did Einstein win the Nobel Prize for the theory of relativity?"
{{"sub_topics": ["Einstein Nobel Prize award reason"], "valid": true, "comparison_axis": null, "time_range": null, "queries": ["Einstein 1921 Nobel Prize photoelectric effect citation", "Nobel committee relativity Einstein award controversy"]}}
"""


def _summarize_old_turns(existing_summary: str, old_turns_text: str) -> str:
    "to keep memory short summarizing old turn context older than WINDOW_TURNS recent turns"

    try:
        summarizer_llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0)
        prompt = (
            "Condense the following older research-conversation turns into a short "
            "running memory (3-5 sentences max). Preserve named entities, topics, "
            "and any ongoing comparison. Merge with the existing summary rather "
            "than discarding anything still relevant.\n\n"
            f"Existing summary: {existing_summary or '(none yet)'}\n\n"
            f"Older turns to fold in:\n{old_turns_text}"
        )
        response = summarizer_llm.invoke([HumanMessage(content=prompt)])
        summary = (response.content or "").strip()
        return summary or existing_summary
    except Exception:
        return existing_summary


def build_agent():
    """build the graph:
    START -> structural_check_node -> manage_history_node -> optimize_node
    -> validity_node -> search_node -> check_and_escalate_node -> END
    """

    # planner llm: plain llm call to optimize user query.
    planner_llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0).bind(
        response_format={"type": "json_object"}
    )

    def structural_check_node(state: AgentState):
        "checks for any empty, gibberish, or less-than-minimum-word-count query,"
        "and raises an exception if found. does NOT touch message history — that"
        "is manage_history_node's job now, so this node stays a pure validator."

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

        return {}

    def manage_history_node(state: AgentState):
        "keeps only the last WINDOW_TURNS user turns as raw messages in the"
        "checkpointer. anything older is folded into conversation_summary"
        "instead of being kept verbatim, so a long-lived session (same"
        "thread_id reused across many turns) doesn't grow 'messages' forever"

        all_messages = state["messages"]
        human_indices = [
            i for i, m in enumerate(all_messages) if isinstance(m, HumanMessage)
        ]

        if len(human_indices) <= WINDOW_TURNS:
            return {}

        cutoff_idx = human_indices[-WINDOW_TURNS]
        old_messages = all_messages[:cutoff_idx]

        if not old_messages:
            return {}

        old_turns_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in old_messages
            if isinstance(m, (HumanMessage, AIMessage))
        )

        new_summary = _summarize_old_turns(
            state.get("conversation_summary", ""), old_turns_text
        )

        return {
            "messages": [RemoveMessage(id=m.id) for m in old_messages],
            "conversation_summary": new_summary,
        }

    def optimize_query_node(state: AgentState):
        "user query optimization before web search. builds the planner's input"
        "from checkpointer state only — the running conversation_summary plus"
        "whatever raw turns are still in the window — so this is the single"
        "place that resolves follow-up phrasing. no external dict involved."

        all_messages = state["messages"]
        current_query = all_messages[-1].content
        summary = state.get("conversation_summary", "")

        context_lines = []
        for m in all_messages[:-1]:
            if isinstance(m, HumanMessage):
                context_lines.append(f"Previous user question: {m.content}")
            elif isinstance(m, AIMessage):
                context_lines.append(f"Previous answer: {m.content}")

        context_block = ""
        if summary:
            context_block += f"Conversation summary so far: {summary}\n"
        if context_lines:
            context_block += "\n".join(context_lines) + "\n"

        planner_input = (
            f"{context_block}Current question: {current_query}"
            if context_block
            else current_query
        )

        system_prompt = OPTIMIZE_PROMPT_TEMPLATE.format(
            current_date=datetime.now().strftime("%B %d, %Y")
        )

        last_error: Exception | None = None
        parsed = None

        for attempt in range(1, MAX_OPTIMIZE_RETRIES + 1):
            try:
                response = planner_llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=planner_input),
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
            queries = [current_query]

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
    graph.add_node("manage_history_node", manage_history_node)
    graph.add_node("optimize_node", optimize_query_node)
    graph.add_node("validity_node", validity_check_node)
    graph.add_node("search_node", search_node)
    graph.add_node("check_and_escalate_node", check_and_escalate_node)

    graph.add_edge(START, "structural_check_node")
    graph.add_edge("structural_check_node", "manage_history_node")
    graph.add_edge("manage_history_node", "optimize_node")
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
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    raw_content = result.get("raw_content", [])
    sub_queries = result.get("optimized_queries", [])

    return result, raw_content, sub_queries


def get_conversation_context(session_id: str) -> str:
    "reads this session's checkpointed state and returns a short text block"
    "(running summary + recent turns)"

    config = {"configurable": {"thread_id": session_id}}
    snapshot = GLOBAL_AGENT.get_state(config)

    if not snapshot or not snapshot.values:
        return ""

    messages = snapshot.values.get("messages", [])
    summary = snapshot.values.get("conversation_summary", "")

    lines = []
    if summary:
        lines.append(f"Conversation summary so far: {summary}")
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Previous question: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Previous answer: {m.content}")

    return "\n".join(lines)


def save_answer_to_memory(session_id: str, answer: str) -> None:
    "appends the final generated answer onto this session's checkpointed"
    "thread as an AIMessage."

    config = {"configurable": {"thread_id": session_id}}
    GLOBAL_AGENT.update_state(config, {"messages": [AIMessage(content=answer)]})


# if __name__ == "__main__":
#     query = "State of the art long-context window mechanisms in large language models"

#     result, raw, sub_queries = run_agent(query, "5657890iohujgvbn")
#     print(result)
#     print(len(raw))
