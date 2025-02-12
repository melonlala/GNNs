from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_8.prompt as prompt_custom
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
        self.ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions
        solutions = []
        for _ in range(3):  # Generate 3 solutions for ensemble
            solution = await self.custom(input=problem, instruction="")
            solutions.append(solution['response'])
        
        # Use ScEnsemble to select the best solution
        best_solution = await self.ensemble(solutions=solutions, problem=problem)
        
        # Validate the best solution using a Python code execution
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {best_solution['response']}  # Example return"
        validation_result = await self.programmer(problem=validation_code)
        
        # Return the best solution and validation result
        return best_solution['response'], validation_result['output'], self.llm.cost_manager.total_cost
