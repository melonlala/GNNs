from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.round_8.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a step-by-step thought process
        thought_process = await self.answer_generate(input=problem)
        
        # Use the custom method to generate multiple initial solutions
        solutions = await self.custom(input=problem + thought_process['thought'], instruction="Generate multiple solutions for this problem.")
        
        # Use self-consistency to select the best solution
        best_solution = await self.sc_ensemble(solutions=['response1', 'response2', 'response3'])  # Placeholder for actual responses
        
        # Review the best solution before finalizing
        review = await self.custom(input=best_solution['response'], instruction="Review this solution for accuracy.")
        
        return review['response'], self.llm.cost_manager.total_cost
