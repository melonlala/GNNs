from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_164.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions using the custom method
        solutions = []
        for _ in range(4):  # Generate four solutions for ensemble
            response = await self.custom(input=problem, instruction="")
            # Validate each individual solution after generation
            validation_response = await self.custom(input=response['response'], instruction="Please review this solution for accuracy.")
            solutions.append(validation_response['response'])
        
        # Use ScEnsemble to select the best solution, ensuring it meets validity
        ensemble_response = await self.ensemble(solutions=solutions, problem=problem)
        
        # Final validation on the ensemble response
        final_validation_response = await self.custom(input=ensemble_response['response'], instruction="Please confirm the validity of this final solution.")
        
        return final_validation_response['response'], self.llm.cost_manager.total_cost
