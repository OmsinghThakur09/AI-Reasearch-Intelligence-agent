# app/planner/planner.py
"""
classifying user query into 'deep' and 'simple' research depth and desciding whether query topic requires research or not
"""

from groq import Groq
from dotenv import load_dotenv
import json
import os

load_dotenv()


def llm_planner(user_query: str) -> dict:

    GROQ_MODEL = "llama-3.3-70b-versatile"

    SYSTEM_PROMPT = """You are a query topic classifier.
    Your job is to classify query topic rquires research or not.

    If the question can be answered accurately using stable, general knowledge, and cant be devided into subtopic only then simply set research_require": "false" and provide user query as it is.

    If the question depends on recent events, current facts, changing technologies, comparisons, market analysis, or evidence from multiple sources,
    If research is needed, or topic is very broad or contains subtopics then classify it as "simple" or "deep" and simply set research_require": "true"

    for example:
    {
    topic:...
    subtopics: []
    research_require:'true' or 'false'
    reason:...
    estimated_depth:'deep' or 'simple'
    }

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
