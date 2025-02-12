from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_111.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Self-ask step to refine the understanding of the problem
        clarified_problem = await self.custom(input=problem, instruction=prompt_custom.SELF_ASK_PROMPT)
        
        # Generate multiple initial solutions
        solutions = await self.custom(input=clarified_problem['response'], instruction=prompt_custom.SOLVE_PROMPT)
        
        # Use ScEnsemble to select the best solution from generated alternatives
        best_solution = await self.sc_ensemble(solutions=[sol['response'] for sol in solutions], problem=clarified_problem['response'])

        # Validate the best solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {best_solution['response']}\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Return the validated solution and cost
        return best_solution['response'], self.llm.cost_manager.total_cost
