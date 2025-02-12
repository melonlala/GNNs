from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_139.prompt as prompt_custom
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
        self.ensemble = operator.ScEnsemble(self.llm)  # Initialize ScEnsemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Analysis step to improve problem understanding
        analysis = await self.custom(input=problem, instruction="Analyze the problem and generate insights.")
        
        # Generate possible solutions with the Custom operator
        solutions = await self.custom(input=f"{analysis['response']} {problem}", instruction="")
        
        # Use ScEnsemble to verify and select the best solution
        final_solution = await self.ensemble(solutions=[sol['response'] for sol in solutions], problem=problem)
        
        return final_solution['response'], self.llm.cost_manager.total_cost
