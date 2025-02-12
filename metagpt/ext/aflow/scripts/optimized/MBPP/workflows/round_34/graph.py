from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_34.prompt as prompt_custom
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
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.test_operator = operator.Test(self.llm)  # Added Test operator

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate code solution based on the problem.
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        # Validate the code solution using public test cases.
        test_result = await self.test_operator(problem=problem, solution=solution['response'], entry_point=entry_point)
        if not test_result['result']:  # If tests fail, handle accordingly
            return f"Test failed. Current solution: {solution['response']}", self.llm.cost_manager.total_cost
        return solution['response'], self.llm.cost_manager.total_cost
