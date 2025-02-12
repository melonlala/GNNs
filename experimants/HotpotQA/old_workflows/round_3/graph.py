from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_3.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

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
        self.answer_generate = operator.AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a detailed answer using the AnswerGenerate operator
        answer_details = await self.answer_generate(input=problem)
        
        # Use the custom method to generate the final response
        solution = await self.custom(input=problem + f" Details: {answer_details['thought']}", instruction="")
        
        # Review the solution to ensure accuracy
        review_solution = await self.custom(input=solution['response'], instruction="Review this answer for accuracy.")
        
        return review_solution['response'], self.llm.cost_manager.total_cost
