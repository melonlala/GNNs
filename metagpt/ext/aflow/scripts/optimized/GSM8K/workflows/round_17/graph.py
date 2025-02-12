from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_17.prompt as prompt_custom
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
        
        # Review the refined problem to ensure clarity
        reviewed_problem = await self.programmer(problem=refined_problem['response'])
        
        # Generate multiple initial solutions
        initial_solutions = await self.custom(input=reviewed_problem['output'], instruction="")
        
        # Validate the initial solutions using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[initial_solutions['response']], problem=reviewed_problem['output'])

        # Ensemble review for final validation
        final_review = await self.ensemble(solutions=[ensemble_solution['response'], initial_solutions['response']], problem=reviewed_problem['output'])
        
        return final_review['response'], self.llm.cost_manager.total_cost
