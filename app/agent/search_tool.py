# app/agent/search_tool.py
"""
custom search tool for searching web and only sending 'url', 'title', 'content' to llm model of ReAct agent.
"""

from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from config import TAVILY_API_KEY
from typing import Any

_tavily = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)  # type: ignore


def web_search_executor(query: str):
    "runs one Tavily search and returns a single list of result dicts (url/title/content/raw_content)"
    "returning one combined list (instead of two separate lists) keeps url and raw_content paired together,"
    "which is what the caller needs to dedupe by url before ever appending anything to raw_content"

    raw_results = _tavily.raw_results(
        query=query,
        max_results=2,
        include_raw_content="text",  # type: ignore
    )

    results = raw_results.get("results", [])
    if not results:
        return []

    items = []
    for r in results:
        items.append(
            {
                "url": r.get("url"),
                "title": r.get("title"),
                "content": r.get("content", ""),
                "raw_content": r.get("raw_content", ""),
            }
        )

    return items


def make_search_tool():
    """
    Factory function to create custom search tool
    """
    raw_content = []
    seen_urls = set()  # shared across every call to web_search_tool AND search_node,
    # so a url fetched once (by either path) is never fetched into raw_content again

    @tool("web_search_tool")
    def web_search_tool(search_query: str):
        "custom web searching tool:"
        "this will save document into database"
        "both raw_content and cleaned text"
        "and send all info except raw_content to ReAct agent"

        items = web_search_executor(search_query)

        metadatas = []
        for item in items:
            url = item["url"]
            if url and url in seen_urls:
                continue  # already collected this page earlier in this session, skip it
            if url:
                seen_urls.add(url)

            raw_content.append(item["raw_content"])
            metadatas.append(
                {
                    "url": url,
                    "title": item["title"],
                    "content": item["content"],
                }
            )

        toolmessage: dict[str, Any] = {}
        toolmessage[
            "instruction"
        ] = f"""Found {len(metadatas)} new result(s), full content stored in database for further process,
        Use below summaries to judge whether you have enough information to answer, or need to search again with a more specific query."""

        toolmessage["metadata"] = metadatas

        return toolmessage

    return web_search_tool, raw_content, seen_urls
