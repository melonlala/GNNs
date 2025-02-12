from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.round_24.prompt as prompt_custom
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
        self.test_operator = operator.Test(self.llm)  # Added Test operator

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions using the custom operator
        response1 = await self.custom(input=problem, instruction=prompt_custom.SOLUTION_PROMPT)
        response2 = await self.custom(input=problem, instruction=prompt_custom.SOLUTION_PROMPT)
        
        # Review the generated solutions for quality
        review_response1 = await self.custom(input=response1['response'], instruction=prompt_custom.REVIEW_PROMPT)
        review_response2 = await self.custom(input=response2['response'], instruction=prompt_custom.REVIEW_PROMPT)
        
        # Use ScEnsemble to select the best solution
        ensemble_response = await self.sc_ensemble(solutions=[review_response1['response'], review_response2['response']], problem=problem)
        
        # Validate the selected solution before testing
        validated_solution = await self.custom_code_generate(problem=ensemble_response['response'], entry_point=entry_point, instruction="")
        
        # Test the validated solution
        test_result = await self.test_operator.test(problem=problem, solution=validated_solution['response'], entry_point=entry_point)
        
        return validated_solution['response'], self.llm.cost_manager.total_cost, test_result['result']
