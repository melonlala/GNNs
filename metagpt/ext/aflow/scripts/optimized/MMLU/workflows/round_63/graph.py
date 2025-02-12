from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.round_63.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.format = operator.Format(self.llm)
        

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a detailed thought process for the problem
        answer_steps = await self.answer_generate(input=problem)
        
        # Use the custom method to generate multiple solutions based on the thought process
        solutions = await self.custom(input=problem + answer_steps['thought'], instruction="Generate multiple solutions.")
        
        # Review the generated solutions before selecting the best one
        reviewed_solutions = await self.custom(input=solutions["response"], instruction="Review these solutions for quality.")
        
        # Use self-consistency ensemble to select the best solution from the reviewed ones
        best_solution = await self.sc_ensemble(reviewed_solutions["response"])
        
        # Format the selected solution
        formatted_solution = await self.format(best_solution["response"])
        return formatted_solution['solution'], self.llm.cost_manager.total_cost
