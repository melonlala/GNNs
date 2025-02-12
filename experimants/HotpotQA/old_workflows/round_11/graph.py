from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_11.prompt as prompt_custom
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
        initial_solutions = await self.answer_generate(input=problem)
        
        # Use the custom method to generate initial responses based on the detailed answers
        initial_responses = []
        for detail in initial_solutions['thoughts']:  # Assuming 'thoughts' is a list of generated thoughts
            response = await self.custom(input=problem + f" Details: {detail}", instruction="")
            initial_responses.append(response['response'])
        
        # Review the initial solutions for accuracy
        review_responses = []
        for response in initial_responses:
            review_solution = await self.custom(input=response, instruction="Review this answer for accuracy.")
            review_responses.append(review_solution['response'])
        
        # Use ScEnsemble to select the best solution from multiple reviewed responses
        ensemble_solution = await self.sc_ensemble(solutions=review_responses)
        
        return ensemble_solution['response'], self.llm.cost_manager.total_cost
