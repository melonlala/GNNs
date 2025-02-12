from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_119.prompt as prompt_custom
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

    async def self_ask(self, problem: str) -> str:
        # Mechanism for clarifying the problem
        ask_response = await self.custom(input=problem, instruction="Clarify the problem for better understanding.")
        return ask_response['response']

    async def analyze_problem(self, clarified_problem: str) -> str:
        # Analyze the clarified problem before solving
        analysis_response = await self.custom(input=clarified_problem, instruction="Analyze the problem for solution strategy.")
        return analysis_response['response']

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        clarified_problem = await self.self_ask(problem)
        analysis = await self.analyze_problem(clarified_problem)
        solution = await self.custom(input=f"{analysis} {problem}", instruction="")
        return solution['response'], self.llm.cost_manager.total_cost
