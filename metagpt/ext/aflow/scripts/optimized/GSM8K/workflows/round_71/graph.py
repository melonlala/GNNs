from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_71.prompt as prompt_custom
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
        # Generate multiple initial solutions
        initial_solutions = await self.custom(input=problem, instruction=prompt_custom.MULTI_SOLUTION_PROMPT)
        
        # Check for significant differences in solutions
        if are_significant_differences(initial_solutions['response']):
            # Review the initial solutions using Programmer for detailed analysis
            review_solutions = await self.programmer(problem=problem, analysis=initial_solutions['response'])
            
            # Validate the reviewed solutions
            if is_solution_valid(review_solutions['output']):
                # Validate the solutions using ScEnsemble
                ensemble_solution = await self.ensemble(solutions=[review_solutions['output']], problem=problem)
                
                # Review the ensemble solution for final validation
                final_review = await self.programmer(problem=problem, analysis=ensemble_solution['response'])
                return final_review['output'], self.llm.cost_manager.total_cost

        return "Re-evaluation needed", self.llm.cost_manager.total_cost

    def are_significant_differences(response: str) -> bool:
        # Implement logic to determine if differences are significant
        return True
    
    def is_solution_valid(output: str) -> bool:
        # Implement logic to check if a solution is valid
        return True
