CONTEXT_PROMPT = """
Please provide detailed context or background information relevant to the following question.
Question: {question}
"""

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

VALIDATION_PROMPT = """
Please validate the following final answer to ensure it is correct and meets the question's requirements.
Final Answer: {answer}
"""

CONTEXT_INTEGRATION_PROMPT = """
Integrate the context and clarify the response based on the provided details.
Preliminary Answer: {response}
Context: {context}
"""