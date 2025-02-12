DIRECT_ANSWER_PROMPT = """
If possible, provide a direct answer to the following question without generating context.

Question: {question}
"""

FALLBACK_PROMPT = """
The previous responses were inconclusive. Please provide a fresh perspective or new context on the question.

Question: {question}
"""