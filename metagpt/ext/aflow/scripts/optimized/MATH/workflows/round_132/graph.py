from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_132.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.ensemble = operator.ScEnsemble(self.llm)
        self.self_ask = operator.SelfAsk(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Use selfAsk to explore the problem in depth
        exploration_questions = await self.self_ask(input=problem)

        # Generate multiple solutions for ensemble
        solutions = []
        for _ in range(5):  # Generate 5 solutions
            solution = await self.custom(input=problem + " " + exploration_questions['response'], instruction=prompt_custom.SOLVE_PROMPT)
            solutions.append(solution['response'])

        # Combine all solutions to get the best one
        ensemble_result = await self.ensemble(solutions=solutions, problem=problem)

        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
