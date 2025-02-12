from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_117.prompt as prompt_custom
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
        # Step 1: Self-ask to refine the problem statement
        refinement = await self.custom(input=f"What are the main aspects to consider for: {problem}?", 
                                        instruction=prompt_custom.SELF_ASK_PROMPT)

        refined_problem = refinement['response']

        # Step 2: Generate multiple solutions
        solutions = []
        for _ in range(3):  # Generate three different solutions
            solution = await self.custom(input=refined_problem, instruction=prompt_custom.SOLVE_PROMPT)
            solutions.append(solution['response'])

        # Step 3: Ensemble the solutions to select the best one
        ensemble_result = await self.sc_ensemble(solutions=solutions, problem=refined_problem)

        # Step 4: Validate the final solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {ensemble_result['response']}\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
