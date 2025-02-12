REVIEW_PROMPT = """

Please review the following response and ensure it is accurate, relevant, and comprehensive.

Response: {response}

"""

FINAL_PROMPT = """

Based on the review and ensemble results, provide a final answer to the question.

Question: {question}
Review: {review}
Ensemble: {ensemble}

"""