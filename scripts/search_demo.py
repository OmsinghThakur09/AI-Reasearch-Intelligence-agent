from app.utils.classifier import llm_planner
from app.utils.search_executor import search
from app.utils.cleaner import clean

query = "what is agentic ai?"
plan = llm_planner(query)
# print(plan)

response = search(query, max_result=5 if plan["research_require"] == "true" else 3)

raw_texts = []
for item in response:
    raw_texts.append(item["raw_content"])

records = clean(raw_texts)
for record in records:
    print(record["clean"])
