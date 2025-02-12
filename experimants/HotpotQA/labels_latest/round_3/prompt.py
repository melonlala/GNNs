REVIEW_PROMPT = """

Please review the following response and ensure it is accurate and relevant.

Response: {response}

"""

FINAL_PROMPT = """

Based on the review, provide a final answer to the question.

Question: {question}
Review: {review}

"""