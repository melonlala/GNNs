from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_8.prompt as prompt_custom
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
            solutions.append(solution['response'])
        
        # Use ScEnsemble to select the best solution based on frequency
        final_solution = await self.sc_ensemble(solutions=solutions)
        
        # Review the final solution to ensure accuracy
        review_solution = await self.custom(input=final_solution['response'], instruction="Please review this answer for accuracy and provide any necessary corrections.")
        
        return review_solution['response'], self.llm.cost_manager.total_cost
