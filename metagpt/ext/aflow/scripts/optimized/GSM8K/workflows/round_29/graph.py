from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_29.prompt as prompt_custom
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
        initial_solutions = await self.custom(input=problem, instruction=prompt_custom.SOLUTION_PROMPT)
        
        # Review the initial solutions using Programmer for detailed analysis
        review_solutions = [await self.programmer(problem=problem, analysis=sol['response']) for sol in initial_solutions['responses']]
        
        # Validate the reviewed solutions using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[sol['output'] for sol in review_solutions], problem=problem)

        # Introduce a validation step to ensure the ensemble solution's correctness
        validation_step = await self.programmer(problem=problem, analysis=ensemble_solution['response'])
        
        # Final step to return only validated output along with the cost
        return validation_step['output'], self.llm.cost_manager.total_cost
