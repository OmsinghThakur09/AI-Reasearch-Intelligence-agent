# app/agent/parser.py

"""
script to parse summarized content and metadata provided by tavily from agent output
"""


def parse_agent_output(agent_output: dict):
    """
    Returns: (source_urls, raw_docs)
    raw_docs: list of {'url':..., 'title': ...., 'content':...}

    Duplicate urls (the same source can be returned by more than one
    sub-query) are only kept once.
    """
    sources = []
    raw_docs = []
    seen_urls = set()

    search_results = agent_output.get("search_results", [])
    if not search_results:
        return []
    for query_block in search_results:
        results = (
            query_block.get("results", []) if isinstance(query_block, dict) else []
        )
        for item in results:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(url)
                raw_docs.append(
                    {
                        "url": url,
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                    }
                )

    return sources, raw_docs  # type: ignore


if __name__ == "__main__":
    from app.agent.search_agent_V2 import run_agent

    query = "How universe was born?"

    result, _, _ = run_agent(query, "089ouiyjghfbc")

    # print(parse_agent_output(result))
    sources, metadata = parse_agent_output(result)
    print(len(sources), len(metadata))
