from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_109.prompt as prompt_custom
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
        self.ensemble = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        
        # Review the generated solution for clarity and coherence
        review = await self.custom(input=solution['response'], instruction=prompt_custom.REVIEW_PROMPT)

        # Generate multiple solutions using ensemble method
        ensemble_solutions = await self.ensemble(solutions=[solution['response'], review['response']], problem=problem)
        
        # Validate the best ensemble solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {ensemble_solutions['response']}\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Return the final solution and cost
        return ensemble_solutions['response'], self.llm.cost_manager.total_cost
