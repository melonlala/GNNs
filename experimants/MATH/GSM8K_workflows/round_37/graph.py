from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_37.prompt as prompt_custom
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
        # Generate multiple solutions using the custom operator with detailed context.
        response = await self.custom(input=problem, instruction="Please provide a thorough explanation and solution to the following mathematical problem and ensure to consider all possible interpretations.")
        
        # Review and generate additional solutions based on the consistent output
        additional_solution = await self.custom(input=problem, instruction="Can you provide another approach to solving this problem?")
        
        # Use self-consistency to validate and select the best solution.
        ensemble_response = await self.sc_ensemble(solutions=[response['response'], additional_solution['response']], problem=problem)
        
        # Provide analysis with the programmer for better clarity.
        analysis_response = await self.programmer(problem=problem, analysis=ensemble_response['response'])
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
