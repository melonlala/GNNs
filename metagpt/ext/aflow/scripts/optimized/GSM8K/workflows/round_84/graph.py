from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_84.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Decompose the problem into simpler segments for easier analysis.
        decomposition_analysis = f"Analyze the following components of the problem: {problem}"
        decomposition_response = await self.custom(input=decomposition_analysis, instruction="Please provide a breakdown of this problem into simpler parts.")
        
        # Generate multiple solutions using the custom operator based on the decomposition results.
        initial_solution = await self.custom(input=decomposition_response['response'] + problem, instruction="")
        additional_solution = await self.custom(input=decomposition_response['response'] + problem, instruction="Please provide another way to find the solution.")
        
        # Use self-consistency to validate and select the best solution.
        ensemble_response = await self.sc_ensemble(solutions=[initial_solution['response'], additional_solution['response']], problem=problem)
        
        # Provide analysis with the programmer for better clarity.
        analysis_response = await self.programmer(problem=problem, analysis=ensemble_response['response'])
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
