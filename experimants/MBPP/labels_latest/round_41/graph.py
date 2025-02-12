from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_41.prompt as prompt_custom
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
        # Generate potential solutions
        initial_solutions = []
        for _ in range(3):
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            initial_solutions.append(solution['response'])

        # Review step to refine solutions for common errors
        refined_solutions = []
        for solution in initial_solutions:
            review = await self.custom(input=solution, instruction="Review for edge cases and potential errors.")
            refined_solutions.append(review['response'])

        # Use the ensemble operator to select the best refined solution
        ensemble_decision = await self.sc_ensemble(solutions=refined_solutions, problem=problem)

        # Validate the selected solution using the Test operator
        test_result = await self.test(problem=problem, solution=ensemble_decision['response'], entry_point=entry_point)

        return test_result['solution'], self.llm.cost_manager.total_cost
