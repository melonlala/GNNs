from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_112.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Step to clarify the problem before processing
        clarification = await self.custom(input=problem, instruction="Please clarify the problem.")
        
        # Automated calculation step using Programmer operator
        calculation = await self.programmer(problem=problem, analysis=clarification['response'])

        # Final solution generation logic based on the computed result
        solution = await self.custom(input=problem + " " + calculation['output'], instruction="Generate a final solution.")
        
        return solution['response'], self.llm.cost_manager.total_cost
