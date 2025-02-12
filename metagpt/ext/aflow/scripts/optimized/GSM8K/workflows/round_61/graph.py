from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_61.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)
        self.ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Step 1: Preliminary calculation using Programmer operator
        analysis = await self.programmer(problem=problem)
        
        # Step 2: Collecting context for Custom operator
        combined_input = problem + f"Analysis: {analysis['output']}"
        
        # Step 3: Generate solution using Custom operator
        custom_response = await self.custom(input=combined_input, instruction="")
        
        # Step 4: Reviewing the solution using ScEnsemble operator
        final_solution = await self.ensemble(solutions=[custom_response['response']], problem=problem)

        return final_solution['response'], self.llm.cost_manager.total_cost
