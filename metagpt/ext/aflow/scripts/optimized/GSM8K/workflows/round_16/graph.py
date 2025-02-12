from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_16.prompt as prompt_custom
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
        # Generate initial solution
        initial_solution = await self.custom(input=problem, instruction=prompt_custom.SOLUTION_PROMPT)
        
        # Review the initial solution for validity
        review_solution = await self.programmer(problem=problem, analysis=initial_solution['response'])
        
        # Analyze the initial and reviewed solution for insights
        insights = await self.custom(input=f"Analyze these solutions: {initial_solution['response']} and {review_solution['output']}", 
                                       instruction=prompt_custom.ANALYSIS_PROMPT)
        
        # Validate the solution using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[review_solution['output'], initial_solution['response']], problem=problem)
        
        # Review the ensemble solution for final validation
        final_review = await self.programmer(problem=problem, analysis=ensemble_solution['response'])
        
        # Final validation step to ensure correctness and provide reasoning
        final_validation = await self.programmer(problem=problem, analysis=final_review['output'])
        
        return {'output': final_validation['output'], 'reasoning': insights['response'], 'cost': self.llm.cost_manager.total_cost}
