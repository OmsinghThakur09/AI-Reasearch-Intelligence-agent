# app/agent/search_tool.py
"""
custom search tool for searching web and only sending 'url', 'title', 'content' to llm model of ReAct agent.
"""

from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from config import TAVILY_API_KEY

_tavily = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)  # type: ignore


def web_search_executor(query: str):
    raw_content = []
    raw_results = _tavily.raw_results(
        query=query,
        max_results=2,
        include_raw_content="text",  # type: ignore
    )

    results = raw_results.get("results", [])
    if not results:
        return "result not found"

    raw_content.extend([r.get("raw_content", "") for r in results])

    metadatas = []
    for item in results:
        metadatas.append(
            {
                "url": item["url"],
                "title": item["title"],
                "content": item.get("content", ""),
            }
        )

    return raw_content, metadatas


def make_search_tool():
    """
    Factory function to create custom search tool
    """
    raw_content = []

    @tool("web_search_tool")
    def web_search_tool(search_query: str):
        "custom web searching tool:"
        "this will save document into database"
        "both raw_content and cleaned text"
        "and send all info except raw_content to ReAct agent"

        raw, metadatas = web_search_executor(search_query)
        raw_content.extend(raw)

        toolmessage = {}
        toolmessage[
            "instruction"
        ] = f"""Found {len(metadatas)} result(s), full content stored in database for further process,
        Use below summaries to judge whether you have enough information to answer, or need to search again with a more specific query."""

        toolmessage["metadata"] = metadatas

        return toolmessage

    return web_search_tool, raw_content
