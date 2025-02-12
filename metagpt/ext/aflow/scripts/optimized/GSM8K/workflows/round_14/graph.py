from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_14.prompt as prompt_custom
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
        # Self-ask to refine the problem statement
        refined_problem = await self.custom(input=f"Refine the following problem: {problem}", instruction=SUMMARY_PROMPT)
        
        # Generate multiple initial solutions
        initial_solutions = await self.custom(input=refined_problem['response'], instruction="Generate multiple solutions for the problem.")
        
        # Validate initial solutions first for correctness
        initial_validation = await self.custom(input=initial_solutions['response'], instruction="Validate the initial solutions for correctness and completeness.")
        
        # Additional validation for better accuracy
        final_validation = await self.custom(input=initial_validation['response'], instruction="Review the solutions again for any overlooked issues.")
        
        # Review the validated solutions using Programmer for detailed analysis
        review_solution = await self.programmer(problem=refined_problem['response'], analysis=final_validation['response'])
        
        # Validate the solution using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[review_solution['output'], initial_solutions['response']], problem=refined_problem['response'])
        
        # Review the ensemble solution for final validation
        final_review = await self.programmer(problem=refined_problem['response'], analysis=ensemble_solution['response'])
        
        return final_review['output'], self.llm.cost_manager.total_cost
