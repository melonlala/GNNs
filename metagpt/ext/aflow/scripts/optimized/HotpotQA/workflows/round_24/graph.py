from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_24.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble()  # Added ScEnsemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple preliminary answers
        preliminary_responses = await self.custom(input=problem, instruction=prompt_custom.PREVIEW_PROMPT)
        
        # Review the preliminary responses
        review_responses = [await self.answer_generate(input=response['response']) for response in preliminary_responses]
        
        # Use ScEnsemble to select the best solution from reviewed responses
        ensemble_response = await self.sc_ensemble(solutions=[response['thought'] for response in review_responses])
        
        # Finalize the solution based on the ensemble result
        solution = await self.custom(input=problem + f" Review: {ensemble_response['response']}", instruction=prompt_custom.FINAL_PROMPT)
        return solution['response'], self.llm.cost_manager.total_cost
