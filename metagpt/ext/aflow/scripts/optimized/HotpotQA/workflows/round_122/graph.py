from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_122.prompt as prompt_custom
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
        self.clarification = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Direct answer attempt
        direct_answer_response = await self.custom(input=problem, instruction=prompt_custom.DIRECT_ANSWER_PROMPT)
        if direct_answer_response['response']:
            return direct_answer_response['response'], self.llm.cost_manager.total_cost
        
        # Generate detailed context if direct answer fails
        context_response = await self.custom(input=problem, instruction=prompt_custom.CONTEXT_PROMPT)
        
        # Review context before further processing
        context_review_response = await self.custom(input=context_response['response'], instruction=prompt_custom.REVIEW_PROMPT)
        
        # Validate context response for relevance and comprehensiveness
        validation_response = await self.custom(input=context_review_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
        
        # Generate preliminary answer
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)
        
        # Checking preliminary response detail
        if len(preliminary_response['response']) < 50:  
            clarification_response = await self.clarification(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT)  
            preliminary_response['response'] += " " + clarification_response['response']
        
        # Review detailed analysis of the preliminary response
        review_response = await self.answer_generate(input=preliminary_response['response'])
        
        # Check for clarity in the review response
        if "unclear" in review_response['thought'].lower():
            clarification_response = await self.clarification(input=review_response['thought'], instruction=prompt_custom.CLARIFICATION_PROMPT)
            review_response['thought'] += " " + clarification_response['response']
        
        # Generate summary of the review
        summary_response = await self.custom(input=review_response['thought'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        # Generate multiple answers for self-consistency
        multiple_answers = [await self.custom(input=problem + f" Context: {context_response['response']} Review: {review_response['thought']} Summary: {summary_response['response']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]
        
        # Self-consistency check to select the best solution
        ensemble_response = await self.sc_ensemble(solutions=[ans['response'] for ans in multiple_answers])
        
        # Review ensemble response for accuracy
        final_review_response = await self.answer_generate(input=ensemble_response['response'])
        
        # Fallback mechanism if final answer is not reasonable
        if "not found" in final_review_response['answer'].lower() or validation_response['response'] == 'unsatisfactory':
            fallback_response = await self.custom(input=problem, instruction=prompt_custom.FALLBACK_PROMPT)
            final_review_response = await self.answer_generate(input=fallback_response['response'])
        
        # Final self-consistency check on the answer
        self_consistency_check = await self.sc_ensemble(solutions=[final_review_response['answer'] for _ in range(3)])
        if self_consistency_check['response'] == final_review_response['answer']:
            return self_consistency_check['response'], self.llm.cost_manager.total_cost
        else:
            return "Inconclusive", self.llm.cost_manager.total_cost
