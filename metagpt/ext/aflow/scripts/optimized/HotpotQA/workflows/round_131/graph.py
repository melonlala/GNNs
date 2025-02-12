from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_131.prompt as prompt_custom
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
        direct_answer_response = await self.custom(input=problem, instruction=prompt_custom.DIRECT_ANSWER_PROMPT)
        if direct_answer_response['response']:
            return direct_answer_response['response'], self.llm.cost_manager.total_cost
        
        context_response = await self.custom(input=problem, instruction=prompt_custom.CONTEXT_PROMPT)
        context_review_response = await self.custom(input=context_response['response'], instruction=prompt_custom.REVIEW_PROMPT)
        validation_response = await self.custom(input=context_review_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
        
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)
        
        # Removed length check and directly generate clarification if needed
        clarification_response = await self.clarification(input=preliminary_response['response'], instruction=prompt_custom.CLARIFICATION_PROMPT) 
        preliminary_response['response'] += " " + clarification_response['response']
        
        review_response = await self.answer_generate(input=preliminary_response['response'])
        
        if "unclear" in review_response['thought'].lower():
            clarification_response = await self.clarification(input=review_response['thought'], instruction=prompt_custom.CLARIFICATION_PROMPT)
            review_response['thought'] += " " + clarification_response['response']
        
        summary_response = await self.custom(input=review_response['thought'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        multiple_answers = [await self.custom(input=problem + f" Context: {context_response['response']} Review: {review_response['thought']} Summary: {summary_response['response']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]
        
        ensemble_response = await self.sc_ensemble(solutions=[ans['response'] for ans in multiple_answers])
        
        final_review_response = await self.answer_generate(input=ensemble_response['response'])
        
        if "not found" in final_review_response['answer'].lower():
            fallback_review_response = await self.custom(input=problem, instruction=prompt_custom.FALLBACK_PROMPT)
            final_review_response = await self.answer_generate(input=fallback_review_response['response'])
        
        self_consistency_check = await self.sc_ensemble(solutions=[final_review_response['answer'] for _ in range(3)])
        if self_consistency_check['response'] == final_review_response['answer']:
            # Added a feedback loop to reinforce continuous improvement based on the final answer quality
            feedback_response = await self.custom(input=self_consistency_check['response'], instruction=prompt_custom.REVIEW_PROMPT)
            return feedback_response['response'], self.llm.cost_manager.total_cost
        else:
            return "Inconclusive", self.llm.cost_manager.total_cost
