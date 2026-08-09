# app/agent/parser.py

"""
search agent returns agent.invoke() that contains list of messages (Human message,Tool message, AI answer,.. ),
to store search result and its metadata(url, title, snippet) we need to parse it.
"""

from langchain_core.messages import ToolMessage, AIMessage
import uuid
import json


def helper_toolmessage_parser(toolmessage):
    if not toolmessage:
        return None

    parsed_dict = json.loads(toolmessage)
    for key, value in parsed_dict.items():
        if key == "metadata":
            return value


def parse_agent_output(agent_output: dict):
    """
    Returns: (source_urls, raw_docs)
    raw_docs: list of {'url':..., 'title': ...., 'content':...}
    """
    messages = agent_output.get("messages", [])
    source = []
    raw_docs = []

    for msg in messages:
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

    return list(set(source)), raw_docs  # type: ignore


def aianswer_parser(output: dict):
    "to parse only ai message that dont contain any tool calls"
    messages = output["messages"]

    for message in messages:
        if isinstance(message, AIMessage):
            return message.content


if __name__ == "__main__":
    from app.agent.search_agent_V2 import run_agent

    query = "who won last IPL tournament?"
    query_id = uuid.UUID("70b2a0a2-56df-42d3-abe0-d8914a0a392c")

    result, _ = run_agent(query, "124225jk3")

    print(parse_agent_output(result))
