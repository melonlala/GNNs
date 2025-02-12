from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_137.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
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
        # Step 1: Validate the input problem
        if not self.validate_problem(problem):
            return "Invalid problem format."

        # Generate multiple solutions for ensemble
        solutions = []
        for _ in range(5):  # Generate 5 solutions
            solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
            solutions.append(solution['response'])

        # Review and score the generated solutions
        validation_code = f"""
        def score_solutions(solutions):
            scores = []
            for solution in solutions:
                correctness = 10 if "correct" in solution else 0
                efficiency = len(solution)  # Assuming shorter solutions are better
                completeness = 5 if "complete" in solution else 0
                score = correctness + efficiency + completeness
                scores.append(score)
            return scores
        """
        scores = await self.programmer(problem=validation_code, analysis=str(solutions))  # Pass solutions for scoring
        
        # Use ensemble to get the best solution based on scores
        ensemble_result = await self.ensemble(solutions=solutions, problem=problem)

        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost

    def validate_problem(self, problem: str) -> bool:
        # Basic check to validate the problem format before processing
        return isinstance(problem, str) and len(problem) > 0 and " " in problem  # Example validation
