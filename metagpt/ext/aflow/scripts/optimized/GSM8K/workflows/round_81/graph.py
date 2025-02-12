from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_81.prompt as prompt_custom
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
        # Preliminary analysis of the problem to identify key components
        preliminary_analysis = await self.custom(input=problem, instruction="Analyze the problem and identify key components.")
        
        # Self-review of the preliminary analysis for accuracy
        review_analysis = await self.custom(input=preliminary_analysis['response'], instruction="Review the analysis for any missing components or errors.")
        
        # Generate initial solutions based on the reviewed analysis
        initial_solution = await self.custom(input=review_analysis['response'] + problem, instruction="")
        
        # Validate the consistency of multiple solutions using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[initial_solution['response']], problem=problem)
        
        # Review the ensemble solution for final validation
        final_review = await self.programmer(problem=problem, analysis=ensemble_solution['response'])
        
        return final_review['output'], self.llm.cost_manager.total_cost
