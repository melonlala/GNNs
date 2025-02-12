from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_124.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Self-ask mechanism to clarify the problem
        clarification_response = await self.custom(input=f"Please clarify the following problem: {problem}", instruction="")
        
        # Generating solution based on clarified input
        solution = await self.custom(input=clarification_response['response'] + " " + problem, instruction="")
        
        # Review step to validate the solution
        review_input = f"Validate this solution: {solution['response']} for the problem: {problem}"
        review_output = await self.custom(input=review_input, instruction="")
        
        return review_output['response'], self.llm.cost_manager.total_cost
