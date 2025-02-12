REVIEW_PROMPT = """
Please review the following response based on clarity, detail, and relevance.
Response: {response}
"""

FINAL_PROMPT = """
Based on the review, provide a final answer to the question ensuring clarity and correctness.
Question: {question}
Review: {review}
"""

CLARIFICATION_PROMPT = """
The previous response seems to lack detail. Please provide additional information.
Question: {question}
"""