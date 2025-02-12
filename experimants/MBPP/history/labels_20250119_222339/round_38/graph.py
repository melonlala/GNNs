from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_38.prompt as prompt_custom
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
        solutions = []
        for _ in range(5):  # Generate five different solutions for better diversity
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])

        # Review generated solutions before selecting the best one
        review_results = await self.custom(input="Review these solutions: " + str(solutions), instruction=prompt_custom.REVIEW_PROMPT)
        print("Review results:", review_results['response'])

        # Use ScEnsemble to select the best solution from generated solutions
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)

        # Validate the selected best solution before testing
        if not self.validate_solution(best_solution['response']):
            print("Validation failed. Current solution is invalid:", best_solution['response'])
            return None, self.llm.cost_manager.total_cost
        
        # Validate with additional data checks before testing
        if not self.additional_data_validation(best_solution['response']):
            print("Additional validation failed. Current solution is invalid:", best_solution['response'])
            return None, self.llm.cost_manager.total_cost
        
        # Test the selected best solution
        test_result = await self.test_operator(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        
        if not test_result['result']:
            print("Test failed. Current solution:", best_solution['response'])
        
        return best_solution['response'], self.llm.cost_manager.total_cost

    def validate_solution(self, solution: str) -> bool:
        return "def " in solution and "return" in solution and solution.count('(') == solution.count(')') and solution.count(':') > 0

    def additional_data_validation(self, solution: str) -> bool:
        # Implement any additional data validation checks here
        return True  # For now, we return True as a placeholder
