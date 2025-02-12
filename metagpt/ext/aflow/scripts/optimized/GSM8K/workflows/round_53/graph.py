from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_53.prompt as prompt_custom
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
        # Refine the problem statement before generating solutions
        refined_problem = await self.custom(input="Refine this problem statement: " + problem, instruction="")
        
        # Generate initial solution
        initial_solution = await self.custom(input=refined_problem['response'], instruction=prompt_custom.DETAILED_CALCULATION_PROMPT)

        # Validate the initial solution to ensure accuracy
        validation = await self.custom(input="Validate this solution: " + initial_solution['response'], instruction=prompt_custom.VALIDATE_SOLUTION_PROMPT)
        
        # Review the initial solution using Programmer for detailed analysis
        review_solution = await self.programmer(problem=refined_problem['response'], analysis=validation['response'])
        
        # Validate the solution using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[review_solution['output'], initial_solution['response']], problem=refined_problem['response'])
        
        # Review the ensemble solution for final validation
        final_review = await self.programmer(problem=refined_problem['response'], analysis=ensemble_solution['response'])
        
        return final_review['output'], self.llm.cost_manager.total_cost
