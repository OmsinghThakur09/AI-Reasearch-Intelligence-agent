# app/planner/planner.py
"""
breaking down user query for optimal web search: understanding user query -> decomposing into smaller topics ->
generating relevant multiple queries -> desciding which query to search -> generating optimal search plan in Json format.
this enables smart and optimal web search results by adding planning layer, makes 'Reseach agent' actually Intelligence!
"""

from groq import Groq
from dotenv import load_dotenv
import json
import os

load_dotenv()


def llm_planner(user_query: str) -> dict:

    GROQ_MODEL = "llama-3.3-70b-versatile"

    SYSTEM_PROMPT = """You are a research planning assistant.

    Your job is NOT to answer the question.

    Your job is to create the optimal research plan.

    If the question can be answered accurately using stable, general knowledge, do not provide research plan, simply set research_require": "false" and provide user query as it is.

    If the question depends on recent events, current facts, changing technologies, comparisons, market analysis, or evidence from multiple sources, provide research plan.

    If research is needed, classify it as simple or deep and generate an appropriate search plan.

    Break the problem into research tasks.

    Generate high-quality search queries.

    Assign priorities.

    Return only valid JSON.
    The JSON must have exactly this structure:

    {
    "goal": "...",
    "subtopics": [],
    "research_require":"true" or "false"
    "estimated_depth":(can take any value from "deep" or "simple" descide on the basis of user query and assign value accordingly)
    "searches": [(you can never left this empty if query is not worth research then simple paste user query as it is)
        {
        "query": "...",
        "priority": high or low or medium,
        "reason": "...",
        "expected_info": "..."
        }
    ]
    }

    Generate 5-10 search queries (if only user topic requires deep research otherwise dont) that together would allow another AI system to answer the user's question.

    The queries should be specific, diverse, and non-overlapping.

    Output nothing except valid JSON."""

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        model=GROQ_MODEL,
        response_format={"type": "json_object"},
    )

    return json.loads(str(response.choices[0].message.content))
