from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_9.prompt as prompt_custom
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
        self.test_operator = operator.Test(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        
        # Review the generated solution before testing
        review_result = await self.custom(input=solution['response'], instruction="Review the following code for correctness and best practices: ")
        
        # Validate if the generated solution is a valid function
        if 'def ' not in review_result['response']:
            print("Generated solution is not a valid function.")
            return None, self.llm.cost_manager.total_cost
        
        # Validate the generated solution using the Test operator
        test_result = await self.test_operator(problem=problem, solution=review_result['response'], entry_point=entry_point)
        
        if not test_result['result']:
            # If the test fails, we can log the current solution or attempt to modify it
            print("Test failed. Current solution:", review_result['response'])
        
        return review_result['response'], self.llm.cost_manager.total_cost
