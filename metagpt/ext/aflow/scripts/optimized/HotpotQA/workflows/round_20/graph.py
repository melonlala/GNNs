from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_20.prompt as prompt_custom
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
        self.self_ask = operator.SelfAsk(self.llm)  # New operator for generating clarifying questions

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate clarifying questions to better understand the problem
        clarifying_questions = await self.self_ask(input=problem)
        
        # Generate a step-by-step thought process
        thought_process = await self.answer_generate(input=problem + f" Clarifying questions: {clarifying_questions['response']}")
        
        # Generate a summary of the thought process for context
        summary = await self.custom(input=thought_process['thought'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        # Review the thought process to ensure clarity and suggest improvements
        review = await self.custom(input=thought_process['thought'], instruction=prompt_custom.REVIEW_IMPROVEMENT_PROMPT + summary['response'])
        
        # Generate multiple potential solutions based on the reviewed thought process
        multiple_solutions = await self.answer_generate(input=problem + f" Thought process: {review['response']}")
        
        # Use self-consistency to select the best solution from the generated options
        solution = await self.sc_ensemble(solutions=multiple_solutions['answer'])
        return solution['response'], self.llm.cost_manager.total_cost
