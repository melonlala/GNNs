from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_56.prompt as prompt_custom
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
        # Generate multiple solutions using the custom operator.
        initial_solution = await self.custom(input=problem, instruction="")
        additional_solution = await self.custom(input=problem, instruction="Please provide another way to find the solution.")
        
        # Use self-consistency to validate and select the best solution.
        ensemble_response = await self.sc_ensemble(solutions=[initial_solution['response'], additional_solution['response']], problem=problem)
        
        # Implement a comparative analysis of the solutions to enhance validation.
        comparison_response = await self.custom(input=f"Compare these solutions: {initial_solution['response']} and {additional_solution['response']}", instruction="What are the differences and which is better?")
        
        # Provide analysis with the programmer for better clarity.
        analysis_response = await self.programmer(problem=problem, analysis=comparison_response['response'])
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
