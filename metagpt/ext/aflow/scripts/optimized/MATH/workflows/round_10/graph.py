from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_10.prompt as prompt_custom
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
        # Generate code to solve the problem
        code_response = await self.programmer(problem=problem)
        # Validate the generated code
        validation_response = await self.custom(input=code_response['output'], instruction="Validate this code.")
        # If validation is successful, return the solution
        if validation_response['response'] == "Valid":
            solution = await self.custom(input=problem, instruction="")
            return solution['response'], self.llm.cost_manager.total_cost
        else:
            return "Code validation failed.", self.llm.cost_manager.total_cost
