from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_36.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a step-by-step thought process
        thought_process = await self.answer_generate(input=problem)
        
        # Use the thought process to generate multiple potential solutions
        potential_solutions = await self.custom(input=problem + f" Thought process: {thought_process['thought']}", instruction="")
        
        # Review the potential solutions to enhance quality
        review_feedback = await self.custom(input=f"Review the following solutions: {potential_solutions['response']}", instruction="Provide feedback on the quality of these solutions.")
        
        # Use the feedback and potential solutions to select the best one
        final_solution = await self.sc_ensemble(solutions=[potential_solutions['response'], review_feedback['response']])
        
        return final_solution['response'], self.llm.cost_manager.total_cost
