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
        self.format = operator.Format(self.llm)
        

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions using the custom operator
        solutions = []
        for _ in range(3):  # Generate three different responses
            solution = await self.custom(input=problem, instruction="")
            solutions.append(solution["response"])
        
        # Review the generated solutions for quality
        reviewed_solutions = []
        for solution in solutions:
            review = await self.custom(input=solution, instruction="Evaluate the quality of this solution.")
            if review["response"] == "Good":
                reviewed_solutions.append(solution)
        
        # Use self-consistency to select the best solution from reviewed solutions
        ensemble_solution = await self.sc_ensemble(reviewed_solutions)
        formatted_solution = await self.format(ensemble_solution["response"])
        return formatted_solution['solution'], self.llm.cost_manager.total_cost
