from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_14.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple responses using the AnswerGenerate operator
        responses = await self.answer_generate(input=problem)
        
        # Use ScEnsemble to select the best response from generated answers
        ensemble_response = await self.sc_ensemble(solutions=[resp['answer'] for resp in responses])
        
        # Generate a summary of the ensemble response for clarity
        summary_response = await self.custom(input=ensemble_response['response'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        # Review the summary response for quality enhancement
        review_response = await self.custom(input=summary_response['response'], instruction=prompt_custom.REVIEW_PROMPT)
        
        # Return the reviewed response and the total cost
        return review_response['response'], self.llm.cost_manager.total_cost
