from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_104.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Added ensemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Step 1: Self-ask to clarify the problem
        clarification = await self.custom(input=problem, instruction="Clarify the problem.")
        
        # Step 2: Review step before generating the solution
        instruction = "Use the clarified problem to generate a solution."
        solution = await self.custom(input=clarification['response'], instruction=instruction)

        # Step 3: Use ScEnsemble for result verification
        ensemble_result = await self.sc_ensemble(solutions=[solution['response']], problem=problem)
        
        return ensemble_result['response'], self.llm.cost_manager.total_cost
