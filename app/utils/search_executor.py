# utils/search_executor.py
"""
simple search executor that searches web via Tavily search api
"""

from tavily import TavilyClient
from config import TAVILY_API_KEY


def search(user_query: str, max_result: int = 3) -> list:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(
        query=user_query, max_results=max_result, include_raw_content="text"
    )

    return response["results"]
