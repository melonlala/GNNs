REVIEW_PROMPT = """
Please review the following response and ensure it is accurate, relevant, and comprehensive.
Response: {response}
"""

FINAL_PROMPT = """
Based on the review, provide a final answer to the question, ensuring clarity and correctness.
Question: {question}
Review: {review}
"""

CLARIFICATION_PROMPT = """
The previous response seems to lack detail. Please provide additional information.
Question: {question}
"""

VALIDATION_PROMPT = """
Please ensure the following response is coherent, complete, and meets the necessary criteria. Provide insights or suggestions if possible.
Response: {response}
"""