REVIEW_PROMPT = """

Please review the following response and ensure it is accurate and relevant. Provide suggestions for improvement if necessary.

Response: {response}

"""

FINAL_PROMPT = """

Based on the review and ensemble results, provide a final answer to the question.

Question: {question}
Review: {review}
Ensemble: {ensemble}

"""

ENSEMBLE_PROMPT = """

Generate multiple responses to the following question to ensure a variety of perspectives.

Question: {question}

"""