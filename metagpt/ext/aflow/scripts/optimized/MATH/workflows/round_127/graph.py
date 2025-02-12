from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_127.prompt as prompt_custom
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
        self.ensemble = operator.ScEnsemble(self.llm) 
        self.reviewer = operator.Reviewer(self.llm)  # Added Reviewer operator 

    async def __call__(self, problem: str): 
        solutions = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT) 
        generated_solutions = [solution['response'] for solution in solutions] 

        # Review generated solutions to provide feedback and optimization
        reviewed_solutions = await self.reviewer(solutions=generated_solutions)

        # Use ScEnsemble to determine the best solution 
        ensemble_response = await self.ensemble(solutions=reviewed_solutions, problem=problem) 
        validation_code = f"def validate_solution():\n    return {ensemble_response['response']}\n"
        validation_result = await self.programmer(problem=validation_code) 
        return ensemble_response['response'], self.llm.cost_manager.total_cost
