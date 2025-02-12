from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_103.prompt as prompt_custom
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
        self.self_ask = operator.SelfAsk(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Step 1: Ask clarifying questions about the problem to improve understanding
        self.ask_result = await self.self_ask(problem)
        
        # Step 2: Generate initial solution based on the problem and self-ask results
        initial_solution = await self.custom(input=self.ask_result['question'], instruction="")
        
        # Step 3: Validate the solution by reviewing it before finalizing the result
        ensemble_solution = await self.sc_ensemble([initial_solution['response']])
        return ensemble_solution['response'], self.llm.cost_manager.total_cost
