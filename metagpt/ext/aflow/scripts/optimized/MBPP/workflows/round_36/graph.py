from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_36.prompt as prompt_custom
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
        self.test_operator = operator.Test(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions for the problem
        solutions = []
        for _ in range(3):  # Generate three different solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])
            print("Generated solution:", solution['response'])
        
        # Implement a review mechanism before ensemble selection
        solutions = [sol for sol in solutions if self.validate_solution(sol)]  # Filter valid solutions
        
        # Add a solution refinement by generating variations based on validation feedback
        refined_solutions = []
        for sol in solutions:
            refined_response = await self.custom(input=sol, instruction="Generate a more optimal version of the provided code.")
            refined_solutions.append(refined_response['response'])
        
        # Use ScEnsemble to select the best solution from the refined solutions
        best_solution = await self.sc_ensemble(solutions=refined_solutions, problem=problem)
        
        # Validate the selected best solution using the Test operator
        test_result = await self.test_operator(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        
        if not test_result['result']:
            print("Test failed. Current solution:", best_solution['response'])
        
        return best_solution['response'], self.llm.cost_manager.total_cost

    def validate_solution(self, solution: str) -> bool:
        return bool(solution)  # Simplistic check for demonstration purposes
