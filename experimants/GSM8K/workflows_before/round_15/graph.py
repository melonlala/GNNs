from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_15.prompt as prompt_custom
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
        # Generate multiple initial solutions to diversify outputs
        initial_solution_1 = await self.custom(input=problem, instruction=prompt_custom.SUMMARY_PROMPT)
        initial_solution_2 = await self.custom(input=problem, instruction=prompt_custom.ALTERNATIVE_PROMPT)
        
        # Review the initial solutions using Programmer for detailed analysis
        review_solution_1 = await self.programmer(problem=problem, analysis=initial_solution_1['response'])
        review_solution_2 = await self.programmer(problem=problem, analysis=initial_solution_2['response'])
        
        # Validate the solutions using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[review_solution_1['output'], review_solution_2['output']], problem=problem)
        
        # Review the ensemble solution for final validation
        final_review = await self.programmer(problem=problem, analysis=ensemble_solution['response'])
        
        # Final validation step to ensure correctness
        final_validation = await self.programmer(problem=problem, analysis=final_review['output'])
        
        return final_validation['output'], self.llm.cost_manager.total_cost
