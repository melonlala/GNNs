from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_41.prompt as prompt_custom
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
        self.programmer = operator.Programmer()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution
        solution = await self.custom(input=problem, instruction="")
        
        # Validate the solution using a review step
        review = await self.custom(input=solution['response'], instruction="Review this solution for correctness.")
        
        # If the review indicates the solution is valid, proceed to calculate using the Programmer
        if review['response'] == "Valid":
            calculation = await self.programmer(problem=problem, analysis=solution['response'])
            return calculation['output'], self.llm.cost_manager.total_cost
        else:
            return "Solution is invalid.", self.llm.cost_manager.total_cost
