from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_7.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Added ScEnsemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution
        solution = await self.custom(input=problem, instruction="")
        
        # Review the solution for quality
        review_instruction = f"Review the following solution for correctness: {solution['response']}"
        review_result = await self.custom(input=review_instruction, instruction="")

        # Validate the solution using a Python code execution
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {solution['response']}  # Example return"
        validation_result = await self.programmer(problem=validation_code)
        
        # Return the solution, review result, and validation result
        return solution['response'], review_result['response'], validation_result['output'], self.llm.cost_manager.total_cost
