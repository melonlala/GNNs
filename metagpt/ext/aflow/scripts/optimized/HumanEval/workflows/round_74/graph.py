from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.round_74.prompt as prompt_custom
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
        self.test = operator.Test(self.llm)  # Added Test operator

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions using custom_code_generate
        solutions = []
        for _ in range(3):  # Generate three solutions for ensemble
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])
        
        # Review each generated solution for potential issues with detailed instruction
        reviewed_solutions = []
        for sol in solutions:
            review_result = await self.custom(input=sol, instruction="Review this solution for potential issues and suggest improvements.")
            reviewed_solutions.append(review_result['response'])
        
        # Use ScEnsemble to select the best solution from reviewed solutions
        ensemble_result = await self.sc_ensemble(solutions=reviewed_solutions, problem=problem)
        
        # Validate the selected solution before testing
        validated_solution = await self.custom(input=ensemble_result['response'], instruction="Validate this solution for correctness and completeness, including checks for edge cases.")
        
        # Test the validated solution
        test_result = await self.test(problem=problem, solution=validated_solution['response'], entry_point=entry_point)
        
        return validated_solution['response'], self.llm.cost_manager.total_cost, test_result['result']  # Return test result as well
