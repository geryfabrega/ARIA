import os

from dotenv import load_dotenv
import litellm
import jailbreakbench as jbb

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

# 1. Load the JBB-Behaviors dataset
dataset = jbb.read_dataset()
behaviors = dataset.behaviors[:3]
goals = dataset.goals[:3]

print(f"Querying gpt-4o-mini with the first 3 JBB behaviors...\n")

# 2. Query gpt-4o-mini for each behavior
for behavior, goal in zip(behaviors, goals):
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": goal}],
        api_key=api_key,
        max_tokens=150,
    )
    answer = response.choices[0].message.content.strip()

    # 3. Print the results
    print(f"Behavior : {behavior}")
    print(f"Goal     : {goal}")
    print(f"Response : {answer}")
    print("-" * 80)
