from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_128.prompt as prompt_custom
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
        self.clarification = operator.Custom(self.llm)  # New operator for clarification

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # First attempt to get a direct answer
        direct_answer_response = await self.custom(input=problem, instruction=prompt_custom.DIRECT_ANSWER_PROMPT)
        if direct_answer_response['response']:
            return direct_answer_response['response'], self.llm.cost_manager.total_cost
        
        # Generate additional context
        context_response = await self.custom(input=problem, instruction=prompt_custom.CONTEXT_PROMPT)
        
        # Review and validate context for quality before further processing
        context_review_response = await self.custom(input=context_response['response'], instruction=prompt_custom.REVIEW_PROMPT)
        validation_response = await self.custom(input=context_review_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
        
        # Generate initial answer based on context
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)
        
        # Clarification if the preliminary response lacks detail
        if len(preliminary_response['response']) < 50:  # Check for sufficient detail
            clarification_response = await self.clarification(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT)
            preliminary_response['response'] += " " + clarification_response['response']
        
        # Review to enhance answer quality and check clarity
        review_response = await self.answer_generate(input=preliminary_response['response'])
        if "unclear" in review_response['thought'].lower():
            clarification_response = await self.clarification(input=review_response['thought'], instruction=prompt_custom.CLARIFICATION_PROMPT)
            review_response['thought'] += " " + clarification_response['response']
        
        # Generate follow-up questions if needed
        follow_up_response = await self.custom(input=preliminary_response['response'], instruction=prompt_custom.FOLLOW_UP_PROMPT)
        
        # Generate a concise summary of the overall findings
        summary_response = await self.custom(input=review_response['thought'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        # Create multiple responses for self-consistency checks
        multiple_answers = [await self.custom(input=problem + f" Context: {context_response['response']} Review: {review_response['thought']} Summary: {summary_response['response']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]
        
        # Use ensemble method for finalizing the best answer
        ensemble_response = await self.sc_ensemble(solutions=[ans['response'] for ans in multiple_answers])
        
        # Final review and validation of the ensemble answer
        final_review_response = await self.answer_generate(input=ensemble_response['response'])
        answer_check = await self.custom(input=final_review_response['answer'], instruction=prompt_custom.VALIDATION_PROMPT)
        
        # Fallback mechanism for inconclusive answers
        if "not found" in final_review_response['answer'].lower() or answer_check['response'] != final_review_response['answer']:
            fallback_review_response = await self.custom(input=problem, instruction=prompt_custom.FALLBACK_PROMPT)
            final_review_response = await self.answer_generate(input=fallback_review_response['response'])
        
        # Final self-consistency check on the final answer
        self_consistency_check = await self.sc_ensemble(solutions=[final_review_response['answer'] for _ in range(3)])
        if self_consistency_check['response'] == final_review_response['answer']:
            return self_consistency_check['response'], self.llm.cost_manager.total_cost
        else:
            return "Inconclusive", self.llm.cost_manager.total_cost
