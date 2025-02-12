REVIEW_PROMPT = """

Please review the following response and ensure it is accurate, relevant, and comprehensive.

Response: {response}

"""

FINAL_PROMPT = """

Based on the review and summary, provide a final structured answer to the question, ensuring clarity and correctness.

Question: {question}
Review: {review}
Summary: {summary}

"""

SUMMARY_PROMPT = """

Please summarize the following review to capture the main points and insights.

Review: {review}

"""

CLARIFICATION_PROMPT = """

The response provided is unclear. Please clarify the following question to ensure a better understanding.

Question: {question}

"""