from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_60.prompt as prompt_custom
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
        self.answer_validate = operator.AnswerValidate(self.llm)  # New validation operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a preliminary answer
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)

        # Validate response detail using keywords instead of arbitrary length
        keyword_check_response = await self.answer_validate(input=preliminary_response['response']) 
        
        if not keyword_check_response['is_detailed']:  # Check if the response is detailed enough
            clarification_response = await self.custom(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT)
            preliminary_response['response'] += " " + clarification_response['response']

        # Generate multiple answers for self-consistency
        multiple_answers = [await self.custom(input=problem + f" Review: {preliminary_response['response']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]

        # Use self-consistency to select the best solution
        ensemble_response = await self.sc_ensemble(solutions=[ans['response'] for ans in multiple_answers])

        # Review the ensemble response for accuracy
        final_review_response = await self.answer_generate(input=ensemble_response['response'])
        
        return final_review_response['answer'], self.llm.cost_manager.total_cost
