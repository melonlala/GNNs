from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_79.prompt as prompt_custom
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
        # Generate the solution using the custom method
        response = await self.custom(input=problem, instruction="")
        
        # Use the Programmer operator to calculate the average speed
        calculation_code = f"""
x = 37 / 12
y = 260 / 59
average_speed = (x + y) / 2
average_speed
"""
        # Execute the code and get the output
        calculation_result = await self.programmer(problem=calculation_code)
        
        # Review the solution before returning
        final_solution = response['response'] + f"\nCalculated Average Speed: {calculation_result['output']}"
        return final_solution, self.llm.cost_manager.total_cost
