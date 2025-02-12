from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_99.prompt as prompt_custom
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
        self.context_history = []  # Track recent context responses

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        direct_answer_response = await self.custom(input=problem, instruction=prompt_custom.DIRECT_ANSWER_PROMPT)
        if direct_answer_response['response']:
            return direct_answer_response['response'], self.llm.cost_manager.total_cost
        
        context_response = await self.custom(input=problem, instruction=prompt_custom.CONTEXT_PROMPT)
        
        validation_response = await self.custom(input=context_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
        if "clear" not in validation_response['response'].lower():  # Validate context clarity
            context_response['response'] += " " + await self.clarification(input=context_response['response'], instruction=prompt_custom.CLARIFICATION_PROMPT)['response']
        
        self.context_history.append(context_response['response'])

        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)
        
        # New Review Step
        review_preliminary_response = await self.answer_generate(input=preliminary_response['response'])
        
        if len(preliminary_response['response']) < 50:
            clarification_response = await self.clarification(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT)
            preliminary_response['response'] += " " + clarification_response['response']
        
        review_response = await self.answer_generate(input=review_preliminary_response['thought'])  # Review the quality of preliminary response
        
        if "unclear" in review_response['thought'].lower():
            clarification_response = await self.clarification(input=review_response['thought'], instruction=prompt_custom.CLARIFICATION_PROMPT)
            review_response['thought'] += " " + clarification_response['response']
        
        summary_response = await self.custom(input=review_response['thought'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        multiple_answers = [await self.custom(input=problem + f" Context: {context_response['response']} Review: {review_response['thought']} Summary: {summary_response['response']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]

        ensemble_response = await self.sc_ensemble(solutions=[ans['response'] for ans in multiple_answers])
        
        final_review_response = await self.answer_generate(input=ensemble_response['response'])

        # Modified Fallback Mechanism
        if "not found" in final_review_response['answer'].lower() and len(self.context_history) > 1:
            fallback_contexts = " ".join(self.context_history[-3:])  # Use the last three contexts for fallback
            fallback_review_response = await self.custom(input=problem + " Recent contexts were: " + fallback_contexts, instruction=prompt_custom.FALLBACK_PROMPT)
            final_review_response = await self.answer_generate(input=fallback_review_response['response'])
        
        self_consistency_check = await self.sc_ensemble(solutions=[final_review_response['answer'] for _ in range(3)])
        if self_consistency_check['response'] == final_review_response['answer']:
            return self_consistency_check['response'], self.llm.cost_manager.total_cost
        else:
            return "Inconclusive", self.llm.cost_manager.total_cost
