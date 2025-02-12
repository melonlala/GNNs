from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_38.prompt as prompt_custom
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
        solution = response['response']
        
        # Validate the solution using a Python code execution
        verification_code = f"def verify_solution():\n    return {solution}\nresult = verify_solution()"
        verification_response = await self.programmer(problem=verification_code)
        
        # Check if the result is correct
        if verification_response['output'] == "True":
            return solution, self.llm.cost_manager.total_cost
        else:
            return "Solution verification failed.", self.llm.cost_manager.total_cost
