from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_22.prompt as prompt_custom
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
        self.refiner = operator.Refiner(self.llm)  # Introduced Refiner

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial interpretations and solutions
        initial_solution = await self.custom(input=problem, instruction="Generate multiple interpretations and solutions for the given problem.")
        
        # Review the initial solution using Programmer for detailed analysis
        review_solution = await self.programmer(problem=problem, analysis=initial_solution['response'])
        
        # Validate the solution using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[review_solution['output'], initial_solution['response']], problem=problem)
        
        # Refine the ensemble solution for clarity
        refined_solution = await self.refiner(output=ensemble_solution['response'])  # New refinement step
        
        # Review the refined solution for final validation
        final_review = await self.programmer(problem=problem, analysis=refined_solution['output'])  
        
        # Final validation step to ensure correctness
        final_validation = await self.programmer(problem=problem, analysis=final_review['output'])
        
        return final_validation['output'], self.llm.cost_manager.total_cost
