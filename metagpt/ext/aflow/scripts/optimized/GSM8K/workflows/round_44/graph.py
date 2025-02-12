from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_44.prompt as prompt_custom
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
        # Generate multiple initial solutions with different instructions
        instructions = [
            "Solve the problem and provide a detailed explanation of your reasoning.",
            "Find a solution and highlight the key steps, any potential errors or insights during the solution process."
        ]
        initial_solutions = await asyncio.gather(*[self.custom(input=problem, instruction=instruction) for instruction in instructions])
        
        # Review the initial solutions using Programmer for detailed analysis
        reviewed_solutions = [await self.programmer(problem=problem, analysis=solution['response']) for solution in initial_solutions]
        
        # Validate the solutions using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[review['output'] for review in reviewed_solutions], problem=problem)
        
        # Review the ensemble solution for final validation
        final_review = await self.programmer(problem=problem, analysis=ensemble_solution['response'])
        
        # Final validation step to ensure correctness
        final_validation = await self.programmer(problem=problem, analysis=final_review['output'])
        
        return final_validation['output'], self.llm.cost_manager.total_cost
