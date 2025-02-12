REVIEW_PROMPT = """

Please review the following response and ensure it is accurate, relevant, and comprehensive.

Response: {response}

"""

FINAL_PROMPT = """

Based on the review and summary, provide a final answer to the question, ensuring clarity and correctness.

Question: {question}
Review: {review}
Summary: {summary}

"""

SUMMARY_PROMPT = """

Please summarize the following review to capture the main points and insights.

Review: {review}

"""

CLARIFICATION_PROMPT = """

Please provide a clarification question based on the following response to ensure a more accurate answer.

Response: {response}

"""