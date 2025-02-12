REVIEW_PROMPT = """

Please review the following response and ensure it is accurate, relevant, and comprehensive.

Response: {response}

"""

FINAL_PROMPT = """

Based on a review of the responses, provide a final answer to the question. Ensure clarity and correctness in your response.

Question: {question}
Review: {review}
Summary: {summary}

"""

SUMMARY_PROMPT = """

Please summarize the following review to capture the main points and insights.

Review: {review}

"""

CLARIFICATION_PROMPT = """

The previous response seems to lack detail. Please provide additional information or clarification regarding the question.

Question: {question}

"""