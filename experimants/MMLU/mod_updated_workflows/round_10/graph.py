from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.round_10.prompt as prompt_custom
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
        
        # Review the thought process for accuracy and completeness
        reviewed_thought = await self.custom(input=answer_steps['thought'], instruction="Please review the following thought process for accuracy and completeness.")
        
        # Use the custom method to generate a solution based on the reviewed thought process
        solution = await self.custom(input=problem + reviewed_thought['response'], instruction="Generate a solution based on the reviewed thought process.")
        
        # Review the generated solution before formatting
        reviewed_solution = await self.custom(input=solution["response"], instruction="Please review this solution for correctness.")
        
        formatted_solution = await self.format(reviewed_solution["response"])
        return formatted_solution['solution'], self.llm.cost_manager.total_cost
