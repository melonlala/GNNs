from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_16.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.review = operator.Review(self.llm)  # New Review operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a detailed answer using the AnswerGenerate operator
        answer_details = await self.answer_generate(input=problem)
        
        # Use the custom method to generate multiple solutions
        solutions = []
        for _ in range(3):  # Generate three different solutions
            solution = await self.custom(input=problem + f" Details: {answer_details['thought']}", instruction="")
            # Review each solution for accuracy
            reviewed_solution = await self.review(input=solution['response'], instruction="Please review this answer for accuracy.")
            solutions.append(reviewed_solution['response'])  # Append reviewed solution
        
        # Use ScEnsemble to select the best solution based on frequency
        final_solution = await self.sc_ensemble(solutions=solutions)
        
        return final_solution['response'], self.llm.cost_manager.total_cost
