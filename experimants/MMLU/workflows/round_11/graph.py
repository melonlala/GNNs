from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.round_11.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Review the problem statement before generating the thought process
        review_problem = await self.custom(input=problem, instruction="Review the following problem statement for clarity and context.")
        
        # Generate a step-by-step thought process based on the reviewed problem
        thought_process = await self.answer_generate(input=review_problem['response'])
        
        # Review the thought process before final answer generation
        review = await self.custom(input=thought_process['thought'], instruction="Review the following thought process and provide feedback.")
        
        # Generate multiple answers based on the reviewed thought process
        multiple_answers = await self.sc_ensemble(solutions=[thought_process['answer'], review['response']])
        
        # Generate the final answer based on the ensemble results
        solution = await self.custom(input=problem + f" Thought Process Review: {review['response']}, Ensemble Results: {multiple_answers['response']}", instruction="")
        return solution['response'], self.llm.cost_manager.total_cost
