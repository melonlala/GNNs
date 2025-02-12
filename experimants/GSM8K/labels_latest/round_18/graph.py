from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_18.prompt as prompt_custom
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
        # Generate multiple initial solutions by altering the instruction slightly
        initial_solutions = await self.custom(input=problem, instruction="Solve the problem and provide the answer.")
        alternative_solutions = await self.custom(input=problem, instruction="Consider different methods to solve the problem.")

        # Gather the solutions for ensemble
        all_solutions = [initial_solutions['response'], alternative_solutions['response']]
        
        # Review the multiple solutions using Programmer for detailed analysis
        review_solutions = await self.programmer(problem=problem, analysis=str(all_solutions))
        
        # Validate using ScEnsemble to consolidate results
        ensemble_solution = await self.ensemble(solutions=review_solutions['output'], problem=problem)
        
        # Final review of the ensemble solution for accuracy
        final_review = await self.programmer(problem=problem, analysis=ensemble_solution['response'])
        
        # Final validation step to ensure correctness
        final_validation = await self.programmer(problem=problem, analysis=final_review['output'])
        
        return final_validation['output'], self.llm.cost_manager.total_cost
