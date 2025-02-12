from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_36.prompt as prompt_custom
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
        
        # Generate multiple initial solutions with higher variability
        problem_context = refined_problem['response'] + " Generate alternatives."
        initial_solutions = await self.custom(input=problem_context, instruction="Provide multiple approaches for the problem.")
        
        # Review the initial solutions using Programmer for detailed analysis
        review_solution = await self.programmer(problem=refined_problem['response'], analysis=initial_solutions['response'])
        
        # Validate the initial solutions using ScEnsemble
        initial_ensemble = await self.ensemble(solutions=[initial_solutions['response'], review_solution['output']], problem=refined_problem['response'])

        # Review the ensemble solution for final validation
        final_review = await self.custom(input=initial_ensemble['response'], instruction="Critically assess the combined solution.")
        
        return final_review['response'], self.llm.cost_manager.total_cost
