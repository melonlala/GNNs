from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.round_50.prompt as prompt_custom
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
        self.test = operator.Test(self.llm)  # Added Test operator

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions using the custom operator
        response1 = await self.custom(problem, instruction="Generate a solution for the problem.")
        response2 = await self.custom(problem, instruction="Provide an alternative solution for the problem.")
        
        # Use ScEnsemble to select the best solution from the generated responses
        ensemble_response = await self.sc_ensemble(solutions=[response1['response'], response2['response']], problem=problem)
        
        # Review the selected solution for correctness and efficiency
        review_response = await self.custom(problem=problem, instruction=f"Review the following solution for correctness and efficiency: {ensemble_response['response']}")
        
        # Validate the reviewed solution before testing
        solution = review_response['response']
        validation_result = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=solution)
        
        # Test the validated solution
        test_result = await self.test(problem=problem, solution=validation_result['response'], entry_point=entry_point)
        
        return test_result['result'], test_result['solution'], self.llm.cost_manager.total_cost
