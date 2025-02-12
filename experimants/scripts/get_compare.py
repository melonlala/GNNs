import json
import os

result_file = "metagpt/ext/aflow/scripts/optimized/GSM8K/workflows/results.json"
with open(result_file, "r") as f:
    results = json.load(f)

total_cost = 0
num_results = 0
highest_score = 0
for result in results:
    score = result["score"]
    if score > 0:
        cost = result["total_cost"]
        total_cost += cost
        num_results += 1
        if score > highest_score:
            highest_score = score
            round = result["round"]

print(f"Average cost for {num_results} results: {total_cost / num_results}")
print(f"Highest score: {highest_score} in round {round}")