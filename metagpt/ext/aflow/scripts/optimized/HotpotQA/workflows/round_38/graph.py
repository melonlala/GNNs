from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_38.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a step-by-step thought process
        thought_process = await self.answer_generate(input=problem)
        
        # Self-ask to refine the thought process
        refined_thought = await self.custom(input=thought_process['thought'], instruction=prompt_custom.SELF_ASK_PROMPT)
        
        # Review the refined thought process for accuracy
        review = await self.custom(input=refined_thought['response'], instruction=prompt_custom.REVIEW_PROMPT)
        
        # Generate multiple solutions based on the reviewed thought process
        solutions = await self.answer_generate(input=problem + f" Thought process: {review['response']}")
        
        # Use ScEnsemble to select the best solution from the generated answers
        final_solution = await self.sc_ensemble(solutions=[sol['answer'] for sol in solutions])
        
        return final_solution['response'], self.llm.cost_manager.total_cost
