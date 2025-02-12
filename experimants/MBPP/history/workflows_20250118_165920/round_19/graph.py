from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_19.prompt as prompt_custom
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
        if not problem.strip():
            return "Error: Problem statement is empty.", self.llm.cost_manager.total_cost
        
        # Validate the problem statement for length
        if len(problem) < 10:
            return "Error: Problem statement is too short.", self.llm.cost_manager.total_cost
        
        # Generate multiple solutions for the problem
        solutions = []
        for _ in range(3):  # Generate 3 different solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])
        
        # Use ScEnsemble to select the best solution
        ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # Review the selected solution before testing
        review_result = await self.custom(input=ensemble_result['response'], instruction="Review the following code for correctness, completeness, and suggest improvements.")
        
        # Validate the generated solution using the Test operator
        test_result = await self.test_operator(problem=problem, solution=review_result['response'], entry_point=entry_point)
        
        if not test_result['result']:
            # If the test fails, we can log the current solution or attempt to modify it
            print("Test failed. Current solution:", review_result['response'])
        
        return review_result['response'], self.llm.cost_manager.total_cost
