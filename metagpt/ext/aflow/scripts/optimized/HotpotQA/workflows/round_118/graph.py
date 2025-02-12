from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_118.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)  # Added AnswerGenerate operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        initial_response = await self.custom(input=problem, instruction="")
        # Generate a review step before final output
        review = await self.answer_generate(input=initial_response['response'])
        # Combine both responses for better context
        final_solution = await self.custom(input=problem + " " + review['thought'], instruction="")
        return final_solution['response'], self.llm.cost_manager.total_cost
