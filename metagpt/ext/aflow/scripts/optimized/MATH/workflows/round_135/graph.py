from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_135.prompt as prompt_custom
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
        
        # Review and filter valid solutions
        valid_solutions = []
        for sol in solutions:
            validation_code = f"def validate_solution(solution):\n    # Your validation logic here\n    return True if isinstance(solution, (str, int, float)) and len(str(solution)) > 1 else False\n"
            validation_result = await self.programmer(problem=validation_code + f"validate_solution({sol})")
            if validation_result['output'] == 'True':
                valid_solutions.append(sol)

        # Use ensemble to get the best solution from valid ones
        if valid_solutions:
            ensemble_result = await self.ensemble(solutions=valid_solutions, problem=problem)
        else:
            ensemble_result = {'response': 'No valid solutions generated.'}

        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
