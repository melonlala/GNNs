from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_142.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Self-ask mechanism to enhance understanding of the problem
        self_ask_response = await self.custom(input="Clarify: " + problem, instruction="")
        
        # Generating the solution using the custom operator
        solution = await self.custom(input=problem + f" Clarified version: {self_ask_response['response']}", instruction="")
        
        # Validating the solution with the Programmer operator
        validation_response = await self.programmer(problem=problem, analysis=solution['response'])
        
        return solution['response'], validation_response['output'], self.llm.cost_manager.total_cost
