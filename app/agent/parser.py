# app/agent/parser.py

"""
search agent returns agent.invoke() that contains list of messages (Human message,Tool message, AI answer,.. ),
to store search result and its metadata(url, title, snippet) we need to parse it.
"""

from langchain_core.messages import ToolMessage, SystemMessage
import re
import json


def helper_toolmessage_parser(toolmessage):
    if not toolmessage:
        return None

    parsed_dict = json.loads(toolmessage)
    for key, value in parsed_dict.items():
        if key == "metadata":
            return value


def helper_systemmessage_parser(systemmessage):
    """
    SystemMessage content is plain text, not JSON. It looks like:

    query: <query text>
     url: <url>, title: <title>
     content: <content text>

    repeated for each search result block. This extracts url/title/content
    from each block using regex.
    """
    if not systemmessage:
        return []

    # Each block starts with "url:" and ends right before the next "query:" or "url:" or end of string
    pattern = re.compile(
        r"url:\s*(?P<url>\S+?),\s*title:\s*(?P<title>.*?)\n\s*content:\s*(?P<content>.*?)(?=\n\s*url:|\n\s*query:|\Z)",
        re.DOTALL,
    )

    results = []
    for match in pattern.finditer(systemmessage):
        results.append(
            {
                "url": match.group("url").strip().rstrip(","),
                "title": match.group("title").strip(),
                "content": match.group("content").strip(),
            }
        )
    return results


def parse_agent_output(agent_output: dict):
    """
    Returns: (source_urls, raw_docs)
    raw_docs: list of {'url':..., 'title': ...., 'content':...}
    """
    messages = agent_output.get("messages", [])
    source = []
    raw_docs = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            results = helper_systemmessage_parser(msg.content)
            for item in results:
                source.append(item["url"])
                raw_docs.append(item)

        if isinstance(msg, ToolMessage):
            # Tavilysearch tool returns list of result dicts inside Toolmessage.content

            result = helper_toolmessage_parser(msg.content)
            if result is None:
                return result

            for item in result:
                source.append(item["url"])
                raw_docs.append(
                    {
                        "url": item["url"],
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                    }
                )

    return source, raw_docs  # type: ignore


if __name__ == "__main__":
    from app.agent.search_agent_V2 import run_agent

    query = "Detail the performance benchmarks of Retrieval-Aware Fine-Tuning (RAFT) techniques compared to standard RAG pipelines in recent domain-specific evaluations"

    result, _ = run_agent(query, "0hgrtcgkbhg2")

    # print(parse_agent_output(result))
    sources, metadata = parse_agent_output(result)
    print(len(sources), len(metadata))
