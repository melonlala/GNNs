from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_36.prompt as prompt_custom
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
        # Generate the initial solution
        solution = await self.custom(input=problem, instruction="")
        
        # Use the Programmer operator to perform calculations
        calculation_code = f"def calculate():\n    return {solution['response']}\nresult = calculate()"
        calculation_result = await self.programmer(problem=calculation_code)
        
        # Review the solution before returning
        review = await self.custom(input=f"Review this solution: {calculation_result['output']}", instruction="Is this correct?")
        
        return review['response'], self.llm.cost_manager.total_cost
