from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_40.prompt as prompt_custom
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
        # Validate the problem encounter and check for string issues
        if not problem or not isinstance(problem, str):
            return "Invalid problem statement provided."
        
        # Add an additional validation step before generating solutions
        if not problem.endswith('?'):
            return "Problem statement must be a question."

        # Generate multiple solutions for the problem
        solutions = []
        for _ in range(4):  
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])
        
        # Review the generated solutions for any common errors
        reviewed_solutions = []
        for solution in solutions:
            review_result = await self.custom(input=f"Review this solution: {solution}", instruction="Please check for errors or improvements.")
            reviewed_solutions.append(review_result['response'])
        
        # Use ScEnsemble to select the best solution from reviewed solutions
        best_solution = await self.sc_ensemble(solutions=reviewed_solutions, problem=problem)
        
        # Validate the selected best solution using the Test operator
        test_results = []
        for sol in best_solution['response']:
            test_result = await self.test_operator(problem=problem, solution=sol, entry_point=entry_point)
            test_results.append(test_result)
        
        # Ensure all test results are stored and return the first successful one
        for result in test_results:
            if result['result']:
                return result['solution'], self.llm.cost_manager.total_cost
        
        return "All tests failed.", self.llm.cost_manager.total_cost
