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

The previous response seems to lack detail. Please provide additional information or clarification regarding the question.

Question: {question}

"""

CONTEXT_PROMPT = """

Please provide detailed context or background information relevant to the following question.

Question: {question}

"""

REVIEW_DETAIL_PROMPT = """

Please review the following summary to ensure it captures the main insights and is detailed enough.

Summary: {summary}

"""