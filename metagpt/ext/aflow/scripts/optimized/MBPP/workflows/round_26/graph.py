from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_26.prompt as prompt_custom
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
        Ensure valid input before proceeding and generate multiple solutions to improve reliability.
        """
        # Input validation
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("The problem input must be a non-empty string.")

        # Generate multiple candidate solutions
        candidate_solutions = []
        for _ in range(3):  # Generate three solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            candidate_solutions.append(solution['response'])
        
        # Ensemble the solutions to get the best one
        ensemble_result = await self.sc_ensemble(solutions=candidate_solutions, problem=problem)
        
        # Review the best ensemble solution
        review_result = await self.custom(input=ensemble_result['response'], instruction="Review the following code for correctness and best practices: ")
        
        # Validate the generated solution using the Test operator
        test_result = await self.test_operator(problem=problem, solution=review_result['response'], entry_point=entry_point)
        
        if not test_result['result']:
            print("Test failed. Current solution:", review_result['response'])
        
        return review_result['response'], self.llm.cost_manager.total_cost
