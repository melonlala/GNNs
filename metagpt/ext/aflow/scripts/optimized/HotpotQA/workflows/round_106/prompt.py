DIRECT_ANSWER_PROMPT = """
If possible, provide a direct answer to the following question without generating context.
Question: {question}
"""

FALLBACK_PROMPT = """
The previous responses were inconclusive. Please provide a fresh perspective or new context on the question.
Question: {question}
"""

CLARIFICATION_PROMPT = """
The previous information was vague or ambiguous. Please clarify the following context.
Context: {context}
"""

VALIDATION_PROMPT = """
Verify the relevance and comprehensiveness of the following context.
Context: {context}
"""

REVIEW_PROMPT = """
Review the following information and summarize it. If there are any ambiguities, please elaborate.
Information: {information}
"""

SUMMARY_PROMPT = """
Generate a concise summary based on the following details.
Details: {details}
"""

CONTEX_PROMPT = """
Provide detailed context for the following problem.
Problem: {problem}
"""