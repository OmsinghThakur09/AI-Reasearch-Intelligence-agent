# app/agent/search_tool.py
"""
custom search tool for searching web and only sending 'url', 'title', 'content' to llm model of ReAct agent.
"""

from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from app.utils.cleaner import clean
from config import TAVILY_API_KEY
from app.db.queries import save_documents, save_sources
import uuid

_tavily = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)  # type: ignore


def make_search_tool(query_id: uuid.UUID):
    """
    Factory function to create custom search tool for each user query
    """

    @tool("web_search_tool")
    def web_search_tool(search_query: str):
        "custom web searching tool:"
        "this will save document into database"
        "both raw_content and cleaned text"
        "and send all info except raw_content to ReAct agent"

        raw_results = _tavily.raw_results(
            query=search_query, max_results=2, include_raw_content=True
        )

        results = raw_results.get("results", [])
        if not results:
            return "result not found"

        raw_content = [r.get("raw_content", "") for r in results]

        raw_clean_dict = clean(raw_content)

        metadatas = []
        for result in results:
            for item in result:
                metadatas.append(
                    {
                        "url": item["url"],
                        "title": item["title"],
                        "content": item.get("content", ""),
                    }
                )

        save_documents(query_id, metadatas, raw_clean_dict)

        save_sources(query_id, metadatas)

        return (
            f"Found {len(metadatas)} result(s), full content stored in database for further process:\n\n"
            + f"{metadatas}"
            + "\n\nUse these summaries to judge whether you have enough information "
            "to answer, or need to search again with a more specific query."
        )

    return web_search_tool
