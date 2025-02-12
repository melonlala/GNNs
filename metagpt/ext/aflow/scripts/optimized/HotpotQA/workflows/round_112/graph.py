from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_112.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.clarification = operator.Custom(self.llm)  # Operator for clarification
        self.context_history = []  # Track recent context responses

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Self-ask to clarify the question
        self_ask_response = await self.custom(input=problem, instruction=prompt_custom.SELF_ASK_PROMPT)
        
        # Check if the problem is straightforward and can be answered directly
        direct_answer_response = await self.custom(input=self_ask_response['response'], instruction=prompt_custom.DIRECT_ANSWER_PROMPT)
        if direct_answer_response['response']:
            return direct_answer_response['response'], self.llm.cost_manager.total_cost

        # Generate detailed context for the problem since direct answer failed
        context_response = await self.custom(input=self_ask_response['response'], instruction=prompt_custom.CONTEXT_PROMPT)
        
        # Validate the context response for relevance and clarity
        validation_response = await self.custom(input=context_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
        if "clear" not in validation_response['response'].lower():  # Validate context clarity
            context_response['response'] += " " + await self.clarification(input=context_response['response'], instruction=prompt_custom.CLARIFICATION_PROMPT)['response']
        
        # Store context for fallback checks
        self.context_history.append(context_response['response'])

        # Generate a preliminary answer
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)

        # Check if the preliminary response is detailed enough
        if len(preliminary_response['response']) < 50:
            clarification_response = await self.clarification(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT)
            preliminary_response['response'] += " " + clarification_response['response']
        
        # Review the preliminary response with a more detailed analysis
        review_response = await self.answer_generate(input=preliminary_response['response'])
        
        # Check clarity of the review response before ensemble
        if "unclear" in review_response['thought'].lower():
            clarification_response = await self.clarification(input=review_response['thought'], instruction=prompt_custom.CLARIFICATION_PROMPT)
            review_response['thought'] += " " + clarification_response['response']
        
        # Generate a summary of the review
        summary_response = await self.custom(input=review_response['thought'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        # Generate multiple answers for self-consistency
        multiple_answers = [await self.custom(input=problem + f" Context: {context_response['response']} Review: {review_response['thought']} Summary: {summary_response['response']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]

        # Use self-consistency to select the best solution
        ensemble_response = await self.sc_ensemble(solutions=[ans['response'] for ans in multiple_answers])
        
        # Validate the final answer against the original question context
        final_validation_response = await self.custom(input=ensemble_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
        
        # Fallback mechanism if final answer is not reasonable, check recent history
        if "not found" in final_validation_response['response'].lower() and len(self.context_history) > 0:
            fallback_review_response = await self.custom(input=problem + " Last context was: " + self.context_history[-1], instruction=prompt_custom.FALLBACK_PROMPT)
            final_review_response = await self.answer_generate(input=fallback_review_response['response'])
        
        return final_validation_response['response'], self.llm.cost_manager.total_cost
