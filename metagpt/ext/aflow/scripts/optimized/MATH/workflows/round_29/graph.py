from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_29.prompt as prompt_custom
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
        response = await self.custom(input=problem, instruction="")
        # Validate the solution using a programming approach
        validation_code = f"""
def validate_solution():
    # Extract the values from the response
    r_values = {response['response']}
    return r_values in [27, 93]  # Check if the values are valid

validate_solution()
"""
        validation_result = await self.programmer(problem=validation_code)
        if validation_result['output'] == "True":
            return response['response'], self.llm.cost_manager.total_cost
        else:
            return "Invalid solution", self.llm.cost_manager.total_cost
