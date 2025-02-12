from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_12.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions using custom_code_generate
        solutions = []
        for _ in range(3):  # Generate three different solutions
            response = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(response['response'])

        # Review the generated solutions for correctness
        review_responses = []
        for solution in solutions:
            review_response = await self.custom(input=solution, instruction="Review this solution for correctness.")
            review_responses.append(review_response['response'])

        # Use ScEnsemble to select the best reviewed solution
        best_solution = await self.sc_ensemble(solutions=review_responses, problem=problem)

        # Test the best solution with a retry mechanism
        test_result = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        
        if not test_result['result']:  # If tests fail, attempt to generate a new solution
            response = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            new_solution = response['response']
            test_result = await self.test(problem=problem, solution=new_solution, entry_point=entry_point)

            if test_result['result']:  # If new tests pass
                best_solution = new_solution

        if test_result['result']:  # If tests pass
            return best_solution['response'], self.llm.cost_manager.total_cost
        else:
            return "The solution failed the tests.", self.llm.cost_manager.total_cost
