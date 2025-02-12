from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_13.prompt as prompt_custom
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
        # Generate multiple detailed answers using the AnswerGenerate operator
        answer_details_list = await self.answer_generate(input=problem)
        
        # Use the custom method to generate initial responses for each detailed answer
        initial_solutions = []
        for detail in answer_details_list:
            initial_solution = await self.custom(input=problem + f" Details: {detail['thought']}", instruction="")
            initial_solutions.append(initial_solution['response'])
        
        # Generate summaries for each initial solution for context
        summary_solutions = [await self.answer_generate(input=sol) for sol in initial_solutions]
        
        # Review each initial solution for accuracy, including their summaries
        reviewed_solutions = []
        for i, initial_solution in enumerate(initial_solutions):
            review_solution = await self.custom(input=initial_solution, instruction="Review this answer for accuracy. Summary: " + summary_solutions[i]['answer'])
            reviewed_solutions.append(review_solution['response'])
        
        # Use ScEnsemble to select the best solution from the reviewed responses
        ensemble_solution = await self.sc_ensemble(solutions=reviewed_solutions)
        
        return ensemble_solution['response'], self.llm.cost_manager.total_cost
