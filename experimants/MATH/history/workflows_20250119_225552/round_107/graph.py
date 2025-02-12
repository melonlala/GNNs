from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_107.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Added ScEnsemble for better solution selection

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Step 1: Generate multiple initial solutions
        solutions = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        
        # Step 2: Validate each generated solution
        validation_codes = [f"def validate_solution(solution): return {sol['response']}\n" for sol in solutions]
        validated_solutions = []
        
        for code in validation_codes:
            validation_result = await self.programmer(problem=code)
            validated_solutions.append(validation_result['output'])  # Collect validated outputs
        
        # Step 3: Use ScEnsemble to select the best solution from validated results
        final_solution = await self.sc_ensemble(solutions=validated_solutions, problem=problem)
        
        # Return the final solution
        return final_solution['response']  # Return the voted solution without cost analysis for clarity
