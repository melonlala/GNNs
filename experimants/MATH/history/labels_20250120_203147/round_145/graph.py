from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_145.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Integrate ScEnsemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Encourage model to clarify and analyze the problem
        self_ask_prompt = "Please clarify the problem and break it down for better understanding."
        clarification_response = await self.custom(input=problem, instruction=self_ask_prompt)

        # Review step before finalizing
        review_prompt = "Please ensure your generated response is correct and meets the requirements."
        reviewed_response = await self.custom(input=clarification_response['response'], instruction=review_prompt)

        # Use ScEnsemble to finalize the response for better reliability
        ensemble_response = await self.sc_ensemble(solutions=[clarification_response['response'], reviewed_response['response']], problem=problem)
        
        return ensemble_response['response'], self.llm.cost_manager.total_cost
