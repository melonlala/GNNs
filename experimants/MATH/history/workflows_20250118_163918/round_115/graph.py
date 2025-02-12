from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_115.prompt as prompt_custom
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
        # Generate multiple solutions for the ensemble
        solutions = []
        for _ in range(5):  # Generate 5 solutions
            solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
            solutions.append(solution['response'])
        
        # Validate the generated solutions
        solutions_validated = [s for s in solutions if await self.validate_solution(s, problem)]
        
        # If no valid solutions, handle the case gracefully
        if not solutions_validated:
            return "No valid solutions found", self.llm.cost_manager.total_cost
        
        # Use ensemble to get the best solution
        ensemble_result = await self.ensemble(solutions=solutions_validated, problem=problem)

        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost

    async def validate_solution(self, solution: str, problem: str) -> bool:
        """
        Validate if the generated solution is correct or meets criteria
        """
        validation_input = f"Validate the solution: {solution} for problem: {problem}"
        validation_result = await self.custom(input=validation_input, instruction=prompt_custom.VALIDATE_PROMPT)
        return validation_result['response'] == "Valid"
