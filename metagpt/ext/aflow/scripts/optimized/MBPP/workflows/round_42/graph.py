from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_42.prompt as prompt_custom
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
        self.solution_log = []  # Log all generated solutions

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        
        # Logging the generated solution
        self.solution_log.append(solution['response'])

        # Review the generated solution for common errors
        review_result = await self.custom(input=solution['response'], instruction="Review this code for common errors and suggest improvements.")
        
        # Validate the reviewed solution using the Test operator
        test_result = await self.test_operator(problem=problem, solution=review_result['response'], entry_point=entry_point)
        
        if not test_result['result']:
            print("Test failed. Current solution:", review_result['response'])
            # Attempt to generate alternative solutions and select the best one
            alternative_solutions = [await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="") for _ in range(3)]

            # Rate solutions based on past success/failure rates (hypothetical scoring function)
            rated_solutions = self.rate_solutions(alternative_solutions)

            ensemble_result = await self.sc_ensemble(solutions=[sol['response'] for sol in rated_solutions], problem=problem)
            return ensemble_result['response'], self.llm.cost_manager.total_cost
        
        return review_result['response'], self.llm.cost_manager.total_cost
        
    async def rate_solutions(self, solutions):
        # Hypothetical implementation for rating the solutions based on previous runs.
        return solutions  # Placeholder for potential rating logic.
