REVIEW_PROMPT = """
Please review the following response and ensure it is accurate, relevant, and comprehensive.
Response: {response}
"""

FINAL_PROMPT = """
Based on the context and summary, provide a final answer to the question, ensuring clarity and correctness.
Question: {question}
Summary: {summary}
"""

SUMMARY_PROMPT = """
Please summarize the context provided to capture the main points and insights.
Context: {context}
"""

CLARIFICATION_PROMPT = """
The previous response seems to lack detail. Please provide additional information or clarification regarding the question.
Question: {question}
"""

CONTEXT_PROMPT = """
Please provide detailed context or background information relevant to the following question.
Question: {question}
"""