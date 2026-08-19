# app/agent/search_tool.py
"""
custom search utilities for web searching.

by default only Tavily's summarized 'content' snippet is fetched (fast).
full raw page content is only fetched on-demand, for specific urls, via
fetch_full_pages() — used when the sufficiency check decides a source's
snippet is too thin to answer from.
"""

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from tavily import TavilyClient
from config import TAVILY_API_KEY

_tavily = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)  # type: ignore
_tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)  # used only for full-page extraction


def web_search_executor(query: str, max_result=2, include_raw_content: bool = False):
    "runs one Tavily search and returns a list of result dicts (url/title/content/raw_content)"
    "include_raw_content defaults to False: normal search only needs the summarized"
    "'content' snippet, not the full page, which keeps every search fast."

    raw_results = _tavily.raw_results(
        query=query,
        max_results=max_result,
        exclude_domains=["youtube.com"],
        include_raw_content="text" if include_raw_content else False,  # type: ignore
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
                "raw_content": r.get("raw_content", "") if include_raw_content else "",
            }
        )

    return items


def fetch_full_pages(urls: list[str]) -> dict[str, str]:
    "fetch full raw page content for specific urls only. called only when the"
    "sufficiency check finds a source's summarized content too thin to answer from."
    "returns {url: raw_content}; missing/failed urls are simply absent from the dict."

    if not urls:
        return {}

    try:
        response = _tavily_client.extract(urls=urls)
    except Exception:
        return {}

    fetched = {}
    for item in response.get("results", []):
        url = item.get("url")
        content = item.get("raw_content", "")
        if url and content:
            fetched[url] = content

    return fetched
