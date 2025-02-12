REVIEW_PROMPT = """

Please review the following response in detail, ensuring it is accurate, relevant, and well-supported by evidence.

Response: {response}

"""

FINAL_PROMPT = """

Based on the review and the ensemble results, provide a final answer to the question.

Question: {question}
Review: {review}
Ensemble: {ensemble}

"""