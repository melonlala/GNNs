from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.round_71.prompt as prompt_custom
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
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Validate the problem statement for common issues
        validation_response = await self.custom(input=problem, instruction=prompt_custom.VALIDATION_PROMPT)
        if not validation_response['response']:
            return "Validation failed: " + validation_response['error_message']

        # Generate multiple solutions using the custom operator
        response1 = await self.custom(input=problem, instruction=prompt_custom.SOLUTION_PROMPT)
        response2 = await self.custom(input=problem, instruction=prompt_custom.SOLUTION_PROMPT)
        
        # Use ScEnsemble to select the best solution
        ensemble_response = await self.sc_ensemble(solutions=[response1['response'], response2['response']], problem=problem)
        
        # Review the selected solution for edge cases and common errors
        review_response = await self.custom(input=ensemble_response['response'], instruction=prompt_custom.REVIEW_PROMPT)
        
        # Validate the reviewed solution before testing
        validated_solution = await self.custom_code_generate(problem=review_response['response'], entry_point=entry_point, instruction="")
        
        return validated_solution['response'], self.llm.cost_manager.total_cost
