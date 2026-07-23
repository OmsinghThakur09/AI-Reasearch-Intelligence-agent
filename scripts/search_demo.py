from app.utils.classifier import llm_planner
from app.utils.search_executor import search

query = "what is agentic ai?"
plan = llm_planner(query)
# print(plan)

response = search(query, max_result=5 if plan["research_require"] == "true" else 3)
for item in response:
    print(item["raw_content"])

# print("-"*100)
# result = search("what is who?")
# for item in result:
#     print(item['url'])
