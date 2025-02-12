from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_64.prompt as prompt_custom
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
        self.sc_output = operator.ScOutput(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions using the custom operator.
        initial_solution = await self.custom(input=problem, instruction="")
        additional_solution = await self.custom(input=problem, instruction="Please provide another way to find the solution.")
        
        # Use self-consistency to validate and select the best solution.
        ensemble_response = await self.sc_ensemble(solutions=[initial_solution['response'], additional_solution['response']], problem=problem)

        # Provide analysis with the programmer for better clarity.
        analysis_response = await self.programmer(problem=problem, analysis=ensemble_response['response'])

        # Final output validation step
        final_output = await self.sc_output(analysis=analysis_response['output'])

        return final_output['validated_output'], self.llm.cost_manager.total_cost
