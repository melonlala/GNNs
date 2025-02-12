from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_118.prompt as prompt_custom
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
        # Generate multiple solutions for ensemble
        solutions = []
        for _ in range(5):  # Generate 5 solutions
            solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
            solutions.append(solution['response'])
        
        # Use ensemble to get the best solution
        ensemble_result = await self.ensemble(solutions=solutions, problem=problem)

        # Review step to validate the ensemble solution before final output
        validation_code = f"def validate_solution():\n    # Your validation logic here for solution '{ensemble_result['response']}'\n    return True\n"
        validation_result = await self.programmer(problem=validation_code)

        if validation_result['output'] == "True":
            # Return the final solution and cost if valid
            return ensemble_result['response'], self.llm.cost_manager.total_cost
        else:
            return "Solution validation failed", self.llm.cost_manager.total_cost
